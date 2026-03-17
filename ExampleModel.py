import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
import OpenEXR
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time  # 仅使用原生time模块

# import Utils  # 若未实现可注释，不影响核心功能

TRAIN_DTYPE = torch.float32  # 训练使用 FP32
INFER_DTYPE = torch.float16  # 推理使用 FP16


# ====================================================================
# 🟢 位置编码层（保持不变）
# ====================================================================
class NeRFPositionalEncoding(nn.Module):
    """
    时间位置编码 (EX-1)：使用 E^x - 1 函数作为编码，只占一维。
    """

    def __init__(self, time_dim=1, num_freqs=1, device='cuda:0'):
        super().__init__()
        self.time_dim = time_dim
        self.output_dim = 1  # 最终输出维度：1 维
        self.device = torch.device(device)

    def forward(self, time_coords):
        time_embed = torch.expm1(time_coords)
        return time_embed


# ====================================================================
# 🟢 修改后的FinalModel（新增时间专属MLP，适配增量训练）
# ====================================================================
class FinalModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, feature_height=1024, feature_width=512,
                 hidden_dim1=32, hidden_dim2=32, device="cuda"):
        super(FinalModel, self).__init__()
        self.device = device
        self.height = feature_height * 2
        self.width = feature_width * 2
        self.total_pixels = self.height * self.width
        self.TIME_COL_IDX = 2
        self.SPATIAL_COLS = [0, 1]
        self.output_threshold = 1e-4
        self.MODEL_DTYPE = TRAIN_DTYPE  # 参数存储类型为 FP32 (训练)

        # FeatureMap初始化（保持不变）
        self.featuremap = torch.nn.Parameter(
            torch.empty((1, 4, feature_width, feature_height), dtype=self.MODEL_DTYPE, device=device),
            requires_grad=True
        )

        # 时间编码（保持不变）
        self.time_embed = NeRFPositionalEncoding(device=device)
        self.input_combined_dim = input_dim + 4 + self.time_embed.output_dim

        # MLP Blocks 初始化 (FP32) - 保持原有结构
        self.b1 = nn.Linear(self.input_combined_dim, hidden_dim1).to(self.MODEL_DTYPE).to(device)
        self.b2 = nn.Linear(hidden_dim1, hidden_dim2).to(self.MODEL_DTYPE).to(device)
        self.b3 = nn.Linear(hidden_dim2, hidden_dim2).to(self.MODEL_DTYPE).to(device)
        # Head 初始化 (FP32) - 简化为 Linear
        self.head1 = nn.Linear(hidden_dim1, output_dim).to(self.MODEL_DTYPE).to(device)
        self.head2 = nn.Linear(hidden_dim2, output_dim).to(self.MODEL_DTYPE).to(device)
        self.head3 = nn.Linear(hidden_dim2, output_dim).to(self.MODEL_DTYPE).to(device)

        # ========== 新增：26个时间专属MLP（适配增量训练mode2） ==========
        self.TIME_COUNT = 26  # 对应26个时间点
        self.time_mlps = nn.ModuleList([
            self._build_time_mlp(hidden_dim1, hidden_dim2) for _ in range(self.TIME_COUNT)
        ]).to(device, dtype=self.MODEL_DTYPE)

        # 初始化
        nn.init.normal_(self.featuremap, mean=0.1, std=0.1)

    def _build_time_mlp(self, hidden_dim1, hidden_dim2):
        """构建与主MLP结构一致的时间专属MLP（8→hidden1→hidden2→hidden2→3）"""
        mlp = nn.Sequential(
            nn.Linear(self.input_combined_dim, hidden_dim1),  # b1
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),  # b2
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim2),  # b3
            nn.GELU(),
            nn.Linear(hidden_dim2, 3)  # head3
        )
        return mlp

    def freeze_featuremap(self):
        """冻结FeatureMap（模式二专用）"""
        self.featuremap.requires_grad = False

    def forward(self, x, time_idx=None):
        """
        扩展forward函数，支持指定时间点的MLP推理（适配mode2）
        :param x: 输入坐标 (B, 3)
        :param time_idx: 时间点索引（None=使用共享MLP，int=使用对应时间MLP）
        :return: 输出列表/张量
        """
        total_start = time.perf_counter()
        # 统一输入类型 (训练参数类型)
        x = x.to(self.MODEL_DTYPE)

        # 🌟 关键修改：推理模式 (Evaluation)
        if not self.training:
            with torch.no_grad():
                output = self._forward_internal(x, time_idx)
            return output

        output = self._forward_internal(x, time_idx)
        return output

    def _forward_internal(self, x, time_idx):
        # 输入坐标拆分
        spatial_coords = x[:, self.SPATIAL_COLS]
        time_coords = x[:, [self.TIME_COL_IDX]]
        t_embed = self.time_embed(time_coords)

        # 坐标归一化（GridSample要求[-1,1]）
        spatial_coords_grid_sample = spatial_coords * 2.0 - 1.0
        spatial_grid_coords = spatial_coords_grid_sample.unsqueeze(0).unsqueeze(1).to(torch.float32)

        # GridSample 核心操作
        feature = F.grid_sample(
            self.featuremap,
            spatial_grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        # 维度调整
        feature = feature.squeeze(0).squeeze(1).permute(1, 0)

        # 拼接输入（8维：3坐标 + 4feature + 1时间编码）
        input_combined = torch.cat([x, feature, t_embed], dim=1)

        # ========== 分支逻辑：时间专属MLP or 共享MLP ==========
        if time_idx is not None and 0 <= time_idx < self.TIME_COUNT:
            # 模式二：使用指定时间点的MLP
            output = self.time_mlps[time_idx](input_combined)
            return [output]  # 保持列表格式兼容原有逻辑
        else:
            # 模式一：使用共享MLP（原有逻辑）
            outputs = []
            h1_pre = self.b1(input_combined)
            h1 = F.gelu(h1_pre)
            outputs.append(self.head1(h1))

            h2_pre = self.b2(h1)
            h2 = F.gelu(h2_pre)
            outputs.append(self.head2(h2))

            h3_pre = self.b3(h2)
            h3 = F.gelu(h3_pre)
            outputs.append(self.head3(h3))

            return outputs


# ====================================================================
# 🟢 MLP堆栈（JIT编译版，保持不变）
# ====================================================================
class MLPStack(nn.Module):
    """
    最简单的 MLP 堆栈，forward 方法只返回最终 Head 的输出。
    中间输出通过 Hooks 获取。
    """

    def __init__(self, input_combined_dim, output_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.activation = nn.GELU()
        # 定义所有 Block 和 Head
        self.b1 = nn.Linear(input_combined_dim, hidden_dim1)
        self.b2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.b3 = nn.Linear(hidden_dim2, hidden_dim2)
        self.head3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, input_combined: torch.Tensor) -> torch.Tensor:
        """模型默认只返回最终输出。"""
        h1 = self.activation(self.b1(input_combined))
        h2 = self.activation(self.b2(h1))
        h3 = self.activation(self.b3(h2))
        output = self.head3(h3)
        return output


# ====================================================================
# 🟢 修正后的FinalModel_JIT（与FinalModel结构对齐，适配推理）
# ====================================================================
class FinalModel_JIT(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 feature_height=1024,  # 图像高度（分辨率）
                 feature_width=1024,  # 图像宽度（分辨率）
                 hidden_dim1=16,
                 hidden_dim2=16,
                 device="cuda"):
        super().__init__()

        # 基础配置
        self.device = torch.device(device)
        self.feature_height = feature_height  # FeatureMap高度
        self.feature_width = feature_width  # FeatureMap宽度
        self.TIME_COL_IDX = 2  # 时间维度列索引
        self.SPATIAL_COLS = [0, 1]  # 空间维度列索引
        self.MODEL_DTYPE = INFER_DTYPE  # 推理使用FP16
        self.TIME_COUNT = 26  # 26个时间专属MLP

        # FeatureMap（推理时冻结，不更新）
        self.featuremap = nn.Parameter(
            torch.empty((1, 4, feature_width, feature_height), dtype=self.MODEL_DTYPE, device=self.device),
            requires_grad=False  # 推理时冻结
        )

        print(f"6. featuremap (初始化后) shape: {self.featuremap.shape}")

        # 时间编码层（与FinalModel一致）
        self.time_embed = NeRFPositionalEncoding(device=device)
        self.input_combined_dim = input_dim + 4 + self.time_embed.output_dim

        # 基础MLP堆栈（JIT专用，单头输出）
        self.jit_mlp_stack = MLPStack(
            input_combined_dim=self.input_combined_dim,
            output_dim=output_dim,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2
        ).to(self.device, dtype=self.MODEL_DTYPE)

        # 26个时间专属MLP（与FinalModel结构对齐）
        self.time_mlps = nn.ModuleList([
            self._build_time_mlp(hidden_dim1, hidden_dim2) for _ in range(self.TIME_COUNT)
        ]).to(self.device, dtype=self.MODEL_DTYPE)

    def _build_time_mlp(self, hidden_dim1, hidden_dim2):
        """构建与FinalModel一致的时间专属MLP"""
        mlp = nn.Sequential(
            nn.Linear(self.input_combined_dim, hidden_dim1),  # b1
            nn.GELU(),
            nn.Linear(hidden_dim1, hidden_dim2),  # b2
            nn.GELU(),
            nn.Linear(hidden_dim2, hidden_dim2),  # b3
            nn.GELU(),
            nn.Linear(hidden_dim2, 3)  # head3
        ).to(self.device, dtype=self.MODEL_DTYPE)
        return mlp

    def _forward_shared_mlp(self, input_combined):
        """共享MLP前向（仅返回head3，适配JIT单头输出）"""
        return self.jit_mlp_stack(input_combined)

    @torch.no_grad()  # 推理时禁用梯度
    def forward(self, x, time_idx=None):
        """
        统一forward接口，支持两种模式：
        :param x: 输入坐标 (B, 3)
        :param time_idx: 时间点索引（None=共享MLP，int=时间专属MLP）
        :return: 单张量输出 (B, 3)
        """
        # 输入类型转换（适配FP16推理）
        x = x.to(self.device, dtype=self.MODEL_DTYPE)

        # 输入坐标拆分
        spatial_coords = x[:, self.SPATIAL_COLS]
        # 修复时间坐标取值：使用全部batch的时间值，而非仅第一个
        time_coords = x[:, [self.TIME_COL_IDX]]
        t_embed = self.time_embed(time_coords)

        # 坐标归一化（GridSample要求[-1,1]）
        spatial_coords_grid_sample = spatial_coords * 2.0 - 1.0
        spatial_grid_coords = spatial_coords_grid_sample.unsqueeze(0).unsqueeze(1).to(torch.float32)

        # GridSample 核心操作（与FinalModel一致）
        feature = F.grid_sample(
            self.featuremap,
            spatial_grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        # 维度调整
        feature = feature.squeeze(0).squeeze(1).permute(1, 0)

        # 拼接输入（8维：3坐标 + 4feature + 1时间编码）
        input_combined = torch.cat([x, feature, t_embed], dim=1)

        # ========== 分支逻辑：时间专属MLP or 共享MLP ==========
        if time_idx is not None and 0 <= time_idx < self.TIME_COUNT:
            # 模式二：使用指定时间点的MLP
            output = self.time_mlps[time_idx](input_combined)
        else:
            # 模式一：使用共享MLP（JIT单头版本）
            output = self._forward_shared_mlp(input_combined)

        # JIT模型仅返回单张量（丢弃多余head，节省开销）
        return output