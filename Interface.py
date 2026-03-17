import torch
import os
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from ExampleModel import FinalModel_JIT

# ======================== 全局常量 (适配0-23小时值) ========================
# 训练侧时间点 → 小时映射 (2300 → 23, 1810 → 18.1, 590→5.9 等)
TRAIN_TIME_TO_HOUR = {
    "0": 0.0, "100": 1.0, "200": 2.0, "300": 3.0, "400": 4.0, "500": 5.0,
    "590": 5.9, "600": 6.0, "700": 7.0, "800": 8.0, "900": 9.0, "1000": 10.0,
    "1100": 11.0, "1200": 12.0, "1300": 13.0, "1400": 14.0, "1500": 15.0,
    "1600": 16.0, "1700": 17.0, "1800": 18.0, "1810": 18.1, "1900": 19.0,
    "2000": 20.0, "2100": 21.0, "2200": 22.0, "2300": 23.0
}
# 反向映射：小时 → 训练侧时间字符串
HOUR_TO_TRAIN_TIME = {v: k for k, v in TRAIN_TIME_TO_HOUR.items()}
# 小时 → 时间索引
HOUR_TO_INDEX = {v: i for i, v in enumerate(TRAIN_TIME_TO_HOUR.values())}
# 索引 → 小时/训练时间字符串
INDEX_TO_HOUR = {i: v for i, v in enumerate(TRAIN_TIME_TO_HOUR.values())}
INDEX_TO_TRAIN_TIME = {i: k for i, (k, v) in enumerate(TRAIN_TIME_TO_HOUR.items())}

# 数据类型/文件配置
GLOBAL_DTYPE = torch.float32
INFER_DTYPE = torch.float32
MLP_BASE_FILE_LENGTH = 16384  # 基础MLP文件长度 (含gamma)
MU_INDEX = MLP_BASE_FILE_LENGTH - 1  # gamma存储在最后一位
MLP_FILE_DTYPE = np.float32
TIME_COUNT = 26  # 26个时间点

# 增量MLP文件路径配置
TIME_MLP_BASE_DIR = "./Parameters/time_mlps"
TIME_MLP_FILE_NAME = "mlp_{}.bin"


def inverse_gamma_transform_fixed(x_gamma: torch.Tensor, mu: float) -> torch.Tensor:
    """Gamma反变换：与训练代码完全一致"""
    current_gamma = mu
    if abs(current_gamma - 1.0) < 1e-6:
        return x_gamma
    else:
        inv_gamma = 1.0 / current_gamma
        return torch.pow(x_gamma.clamp(min=0.0), inv_gamma)


class BasicInterface:

    def __init__(self, lightmap_config, device):
        # 基础配置初始化
        self.device = device
        self.mu_value = 0.0
        self.current_time_idx = 0  # 默认使用第0个时间点
        self.available_time_indices = set()  # 记录成功加载的增量MLP索引
        print(f"初始化配置: {lightmap_config}")

        # 解析分辨率/ID信息
        resolution = lightmap_config['resolution']
        self.resolution = resolution
        self.height = resolution['height']
        self.width = resolution['width']
        self.level = lightmap_config["level"]
        self.id = lightmap_config["id"]

        # ======================== 1. 预生成基础坐标 (一次性完成) ========================
        H, W = self.height, self.width
        ys, xs = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        # 基础空间坐标 (shape: [H*W, 2])
        self.base_coords = torch.stack([
            ys.ravel() / (H - 1),
            xs.ravel() / (W - 1)
        ], dim=-1).to(INFER_DTYPE)
        # 预分配完整坐标张量 (shape: [H*W, 3])
        self.full_coords = torch.zeros(
            (self.base_coords.shape[0], 3),
            dtype=INFER_DTYPE,
            device=self.device
        )
        self.full_coords[:, :2] = self.base_coords  # 固定空间维度
        print(f"✅ 基础坐标预生成完成，形状: {self.base_coords.shape}")

        # ======================== 2. 模型初始化 ========================
        self.model = FinalModel_JIT(
            input_dim=3, output_dim=3,
            feature_width=self.width // 2,
            feature_height=self.height // 2,
            hidden_dim1=16, hidden_dim2=16,
            device=self.device
        )

        # ======================== 3. 预加载所有参数 (基础+所有增量MLP) ========================
        self.load_base_params()  # 基础参数必须加载成功，否则抛出异常
        self.preload_all_incremental_mlps()  # 增量MLP加载失败不影响，自动兜底
        print(f"✅ 初始化完成 | 可用增量MLP数量: {len(self.available_time_indices)}/{TIME_COUNT}")

        # 模型最终设置
        self.model.eval()
        self.model.to(self.device)
        self.total_pixels = self.height * self.width

    def load_base_params(self):
        """加载基础参数 (FeatureMap + 基础MLP + Gamma值) - 基础参数必须加载成功"""
        level, id = self.level, self.id

        # 1. 加载FeatureMap (FP16)
        featuremap_file = f"./Parameters/model_{level}_{id}_featuremap_f16.bin"
        if not os.path.exists(featuremap_file):
            raise FileNotFoundError(f"❌ 基础FeatureMap文件缺失: {featuremap_file}\n基础参数是必需的，请检查文件路径")

        try:
            featuremap_array = np.fromfile(featuremap_file, dtype=np.float16)
        except Exception as e:
            raise RuntimeError(f"❌ 加载FeatureMap失败: {e}\n基础参数加载失败，无法继续")

        # 验证并赋值FeatureMap
        fm_param = self.model.featuremap
        fm_size = fm_param.numel()
        if featuremap_array.size != fm_size:
            raise ValueError(f"❌ FeatureMap尺寸不匹配: 期望{fm_size}, 实际{featuremap_array.size}")

        with torch.no_grad():
            reshaped_fm = featuremap_array.reshape(fm_param.shape)
            fm_param.data.copy_(torch.from_numpy(reshaped_fm).to(INFER_DTYPE).to(self.device))
        print(f"✅ 基础FeatureMap加载完成 (FP16, {fm_size}元素)")

        # 2. 加载基础MLP权重 + Gamma值
        base_mlp_file = f"./Parameters/model_{level}_{id}_mlp_f32.bin"
        if not os.path.exists(base_mlp_file):
            raise FileNotFoundError(f"❌ 基础MLP文件缺失: {base_mlp_file}\n基础参数是必需的，请检查文件路径")

        try:
            mlp_array = np.fromfile(base_mlp_file, dtype=MLP_FILE_DTYPE)
        except Exception as e:
            raise RuntimeError(f"❌ 加载基础MLP失败: {e}\n基础参数加载失败，无法继续")

        # 验证MLP文件长度
        if mlp_array.size != MLP_BASE_FILE_LENGTH:
            raise ValueError(f"❌ 基础MLP文件长度错误: 期望{MLP_BASE_FILE_LENGTH}, 实际{mlp_array.size}")

        # 提取Gamma值 (最后一位)
        self.mu_value = float(mlp_array[MU_INDEX])
        print(f"✅ 提取Gamma参数: {self.mu_value:.6f}")

        # 加载基础MLP权重
        self._load_mlp_weights(mlp_array[:MU_INDEX], is_incremental=False)
        print(f"✅ 基础MLP权重加载完成 (FP32)")

    def preload_all_incremental_mlps(self):
        """初始化时预加载所有26个增量MLP参数 (加载失败自动记录，不影响主流程)"""
        level, id = self.level, self.id
        time_mlp_dir = os.path.join(TIME_MLP_BASE_DIR, f"{level}_{id}")

        # 增量MLP目录不存在 - 直接返回，全部使用基础模型
        if not os.path.exists(time_mlp_dir):
            print(f"⚠️ 增量MLP目录不存在: {time_mlp_dir} | 所有推理将使用基础模型")
            return

        loaded_count = 0
        for time_idx in range(TIME_COUNT):
            # 获取对应训练时间字符串
            train_time_str = INDEX_TO_TRAIN_TIME[time_idx]
            incremental_mlp_file = os.path.join(time_mlp_dir, TIME_MLP_FILE_NAME.format(train_time_str))

            # 文件不存在 - 跳过，推理时使用基础模型
            if not os.path.exists(incremental_mlp_file):
                print(f"⚠️ 增量MLP文件缺失: {incremental_mlp_file} (索引{time_idx}) | 该时间点将使用基础模型")
                continue

            try:
                # 加载增量MLP权重
                mlp_array = np.fromfile(incremental_mlp_file, dtype=MLP_FILE_DTYPE)
                self._load_mlp_weights(mlp_array, is_incremental=True, time_idx=time_idx)
                self.available_time_indices.add(time_idx)  # 记录可用索引
                loaded_count += 1
            except Exception as e:
                print(f"⚠️ 加载增量MLP失败 (索引{time_idx}): {str(e)} | 该时间点将使用基础模型")

        print(
            f"✅ 增量MLP预加载完成 | 成功加载: {loaded_count}/{TIME_COUNT} | 可用索引: {sorted(list(self.available_time_indices))}")

    def _load_mlp_weights(self, mlp_weights, is_incremental=False, time_idx=None):
        """通用MLP权重加载函数 (匹配训练代码参数顺序)"""
        # MLP参数顺序: b1.weight → b1.bias → b2.weight → b2.bias → b3.weight → b3.bias → head3.weight → head3.bias
        mlp_param_specs = [
            ('b1', 'weight', (16, 8)),  # 8→16
            ('b1', 'bias', (16,)),  # 16
            ('b2', 'weight', (16, 16)),  # 16→16
            ('b2', 'bias', (16,)),  # 16
            ('b3', 'weight', (16, 16)),  # 16→16
            ('b3', 'bias', (16,)),  # 16
            ('head3', 'weight', (3, 16)),  # 16→3
            ('head3', 'bias', (3,))  # 3
        ]

        param_idx = 0
        with torch.no_grad():
            for (layer_name, param_type, param_shape) in mlp_param_specs:
                param_size = np.prod(param_shape)
                if param_idx + param_size > len(mlp_weights):
                    raise ValueError(
                        f"MLP权重不足: {layer_name}.{param_type} 需要{param_size}个参数，剩余{len(mlp_weights) - param_idx}个")

                # 提取并重塑参数
                param_slice = mlp_weights[param_idx:param_idx + param_size]
                param_idx += param_size
                param_data = param_slice.reshape(param_shape)
                param_tensor = torch.from_numpy(param_data).to(INFER_DTYPE).to(self.device)

                # 区分基础MLP和增量MLP
                if is_incremental and time_idx is not None:
                    if hasattr(self.model, 'time_mlps') and time_idx < len(self.model.time_mlps):
                        time_mlp = self.model.time_mlps[time_idx]
                        layer_map = {'b1': 0, 'b2': 2, 'b3': 4, 'head3': 6}
                        if layer_name in layer_map:
                            mlp_layer = time_mlp[layer_map[layer_name]]
                            getattr(mlp_layer, param_type).data.copy_(param_tensor)
                else:
                    param_path = f'jit_mlp_stack.{layer_name}.{param_type}'
                    if hasattr(self.model, 'jit_mlp_stack'):
                        param_instance = self.model.get_parameter(param_path)
                        param_instance.data.copy_(param_tensor)

    def _get_time_idx_by_hour(self, hour_value):
        """根据小时值快速匹配时间索引 (无文件IO)"""
        # 标准化小时值
        hour_value = float(hour_value)
        hour_value = max(0.0, min(23.0, hour_value))

        # 匹配最接近的时间索引
        closest_hour = min(HOUR_TO_TRAIN_TIME.keys(), key=lambda x: abs(x - hour_value))
        time_idx = HOUR_TO_INDEX[closest_hour]

        return time_idx, closest_hour

    @torch.no_grad()
    @autocast(dtype=torch.float16)
    def reconstruct(self, current_time):
        """
        核心推理函数 (完善兜底逻辑)
        :param current_time: 小时值 (如 0, 1, 5.9, 18.1, 23)
        """
        # 1. 快速匹配时间索引 (无文件读取)
        time_idx, closest_hour = self._get_time_idx_by_hour(current_time)
        self.current_time_idx = time_idx

        # 2. 仅更新时间维度坐标 (空间坐标复用)
        time_norm = closest_hour / 24.0
        self.full_coords[:, 2] = time_norm  # 批量赋值，无需torch.full
        print(f"✅ 时间维度更新完成 | 输入小时: {current_time} → 匹配小时: {closest_hour} | 归一化值: {time_norm:.4f}")

        # 3. 模型推理 (完善兜底逻辑)
        if time_idx in self.available_time_indices:
            # 增量MLP可用 - 使用增量MLP推理
            print(f"🔵 使用增量MLP推理 | 时间索引: {time_idx}")
            x_gamma = self.model(self.full_coords, time_idx=time_idx)
        else:
            # 增量MLP不可用 - 降级使用基础模型
            print(f"🟡 增量MLP不可用 (索引{time_idx}) | 自动降级使用基础模型推理")
            x_gamma = self.model(self.full_coords)

        # 4. Gamma反变换
        self.result = inverse_gamma_transform_fixed(x_gamma, self.mu_value)
        return self.result

    @torch.no_grad()
    def get_result(self):
        """返回格式化结果 (1, 3, H, W)"""
        return self.result.reshape(
            self.height, self.width, 3
        ).permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    @autocast(dtype=torch.float16)
    def random_test(self, coord):
        """单坐标测试函数 (完善兜底逻辑)"""
        # 解析坐标
        y_idx = int(coord[0, 0])
        x_idx = int(coord[0, 1])
        hour_value = float(coord[0, 2])

        # 快速匹配时间索引
        time_idx, closest_hour = self._get_time_idx_by_hour(hour_value)

        # 生成单坐标
        H, W = self.height, self.width
        coord_tensor = torch.tensor([
            y_idx / (H - 1),
            x_idx / (W - 1),
            closest_hour / 24.0
        ], dtype=INFER_DTYPE, device=self.device).unsqueeze(0)

        # 推理 (完善兜底逻辑)
        if time_idx in self.available_time_indices:
            print(f"🔵 使用增量MLP测试 | 时间索引: {time_idx}")
            x_gamma = self.model(coord_tensor, time_idx=time_idx)
        else:
            print(f"🟡 增量MLP不可用 (索引{time_idx}) | 自动降级使用基础模型测试")
            x_gamma = self.model(coord_tensor)

        # Gamma反变换
        return inverse_gamma_transform_fixed(x_gamma, self.mu_value)


def get(lightmap_config, device):
    """接口入口函数"""
    return BasicInterface(lightmap_config, device)