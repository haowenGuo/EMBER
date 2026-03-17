import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
from typing import Dict, Any, Tuple

# 导入路径可能需要根据您的项目结构调整
TRAIN_DTYPE = torch.float32
INFER_DTYPE = torch.float16


# ====================================================================
# 🟢 激活函数 - Swish-Like (近似)
# ====================================================================
def swish_like_activation(x):
    """SHADER中使用近似 Swish 激活函数: x / (1 + exp(-1.702 * x))"""
    beta = 1.702
    return x / (1.0 + torch.exp(-beta * x))


def swish_like_activation_np(x):
    """NumPy版本 Swish-Like 激活函数"""
    beta = 1.702
    x = x.astype(np.float32)
    return x / (1.0 + np.exp(-beta * x))


# ====================================================================
# 🟢 MLP堆栈 (带中间输出存储)
# ====================================================================
class MLPStack(nn.Module):
    def __init__(self, input_combined_dim, output_dim, hidden_dim):
        super().__init__()
        self.activation = swish_like_activation
        self.b1 = nn.Linear(input_combined_dim, hidden_dim)  # 8 -> 16
        self.b2 = nn.Linear(hidden_dim, hidden_dim)  # 16 -> 16
        self.b3 = nn.Linear(hidden_dim, hidden_dim)  # 16 -> 16
        self.head = nn.Linear(hidden_dim, output_dim)  # 16 -> 3
        self.intermediate_outputs = {}

    def forward(self, input_combined: torch.Tensor) -> torch.Tensor:
        # L1: 8 -> 16
        h1_pre_act = self.b1(input_combined)
        h1 = self.activation(h1_pre_act)
        self.intermediate_outputs['L1_output'] = h1.squeeze(0).cpu().numpy()

        # L2: 16 -> 16
        h2_pre_act = self.b2(h1)
        h2 = self.activation(h2_pre_act)
        self.intermediate_outputs['L2_output'] = h2.squeeze(0).cpu().numpy()

        # L3: 16 -> 16
        h3_pre_act = self.b3(h2)
        h3 = self.activation(h3_pre_act)
        self.intermediate_outputs['L3_output'] = h3.squeeze(0).cpu().numpy()

        # L_Out: 16 -> 3
        output = self.head(h3)
        self.intermediate_outputs['L4_output_raw'] = output.squeeze(0).cpu().numpy()

        return output


# ====================================================================
# 🟢 JIT编译模型
# ====================================================================
class FinalModel_JIT(nn.Module):
    def __init__(self,
                 input_dim=3,
                 output_dim=3,
                 feature_height=256,
                 feature_width=512,
                 hidden_dim=16,
                 device="cuda"):
        super().__init__()
        self.device = device
        self.TIME_COL_IDX = 2
        self.MODEL_DTYPE = INFER_DTYPE
        self.featuremap = nn.Parameter(
            torch.empty((1, 4, feature_width, feature_height), dtype=self.MODEL_DTYPE, device=device),
            requires_grad=True
        )
        self.input_combined_dim = input_dim + 4 + 1
        self.jit_mlp_stack = MLPStack(
            input_combined_dim=self.input_combined_dim, output_dim=output_dim,
            hidden_dim=hidden_dim
        )


# ===================== 2. 核心工具函数 =====================

def load_raw_bin_to_model(raw_bin_path: str, model: FinalModel_JIT, device: str) -> Dict[str, Any]:
    """从原始BIN文件加载 MLP 参数和最后的 mu，并返回包含权重的字典。"""
    params_array = np.fromfile(raw_bin_path, dtype=np.float32)
    param_idx = 0
    mlp_weights = {}

    print(f"\n📥 加载原始 MLP BIN 文件: {raw_bin_path} (总长度: {len(params_array)})")

    # 1. 计算 MLP 权重总数
    mlp_total_params = sum(
        param.numel() for name, param in model.named_parameters()
        if "jit_mlp_stack" in name
    )

    # 2. 读取 mu (文件最后一个元素)
    mu_value = params_array[-1]
    mlp_weights["mu"] = mu_value
    print(f"  - 从文件最后一个索引读取到 Mu (Gamma): {mu_value:.4f}")

    # 3. 加载 MLP 权重并存储到字典 (FP32)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "jit_mlp_stack" in name:
                param_size = param.numel()
                param_shape = param.shape

                param_data = params_array[param_idx:param_idx + param_size]

                param.data = torch.from_numpy(param_data.reshape(param_shape)).to(device)
                param_idx += param_size

                # 存储到字典 (使用原始权重名称作为键，方便后续对比)
                key = name.split("jit_mlp_stack.")[1].replace(".", "_")
                mlp_weights[key] = param.data.cpu().numpy().astype(np.float32)
            elif "featuremap" in name:
                pass

    if param_idx == mlp_total_params:
        print("✅ MLP 参数加载完成")
    else:
        print(f"⚠️ 警告: MLP 参数加载不完整！已加载 {param_idx} 元素，预期 {mlp_total_params} 元素。")

    return mlp_weights


def convert_mlp_to_shader_format(mlp_weights: Dict[str, np.ndarray], shader_bin_path: str) -> np.ndarray:
    """
    转换MLP权重为SHADER格式，并将mu放在 4095索引 处。
    同时，执行权重还原和对比，以校验重排逻辑是否正确。
    """
    shader_params = np.zeros((1024, 4), dtype=np.float32)

    # ------------------ L1: Linear(8 → 16) ------------------
    layer1_weight = mlp_weights["b1_weight"]  # (16, 8)
    shader_l1_weights = np.zeros_like(layer1_weight)  # 用于还原校验

    for i in range(4):
        for j in range(2):
            out_s, out_e = i * 4, (i + 1) * 4
            in_s, in_e = j * 4, (j + 1) * 4
            weight_block = layer1_weight[out_s:out_e, in_s:in_e]  # 4x4
            base_idx = (i * 2 + j) * 4
            np.copyto(shader_params[base_idx:base_idx + 4, :], weight_block)

            # **L1 权重还原校验**
            shader_l1_weights[out_s:out_e, in_s:in_e] = weight_block

    # L1 偏置 (16 维)
    layer1_bias = mlp_weights["b1_bias"]
    bias_base_idx = 176
    for i in range(4):
        shader_params[bias_base_idx + i] = layer1_bias[i * 4:(i + 1) * 4]

    # L1 权重对比
    l1_max_diff = np.max(np.abs(layer1_weight - shader_l1_weights))
    print("layer1_weight ", layer1_weight)
    print("shader_l1_weights ", shader_l1_weights)
    print(f"\n--- 🔬 L1 权重校验 (b1_weight) ---")
    print(f"  PyTorch shape: {layer1_weight.shape}")
    print(f"  Shader block count: 8 blocks (4x4)")
    print(f"  L1 权重重排最大差异: {l1_max_diff:.6e}")

    # ------------------ L2: Linear(16 → 16) ------------------
    layer2_weight = mlp_weights["b2_weight"]  # (16, 16)
    shader_l2_weights = np.zeros_like(layer2_weight)  # 用于还原校验

    for i in range(4):
        for j in range(4):
            out_s, out_e = i * 4, (i + 1) * 4
            in_s, in_e = j * 4, (j + 1) * 4
            weight_block = layer2_weight[out_s:out_e, in_s:in_e]  # 4x4
            base_idx = (8 + i * 4 + j) * 4
            np.copyto(shader_params[base_idx:base_idx + 4], weight_block)

            # **L2 权重还原校验**
            shader_l2_weights[out_s:out_e, in_s:in_e] = weight_block

    # L2 偏置 (16 维)
    layer2_bias = mlp_weights["b2_bias"]
    bias_base_idx = 180
    for i in range(4):
        shader_params[bias_base_idx + i] = layer2_bias[i * 4:(i + 1) * 4]

    # L2 权重对比
    l2_max_diff = np.max(np.abs(layer2_weight - shader_l2_weights))
    print(f"\n--- 🔬 L2 权重校验 (b2_weight) ---")
    print(f"  PyTorch shape: {layer2_weight.shape}")
    print(f"  Shader block count: 16 blocks (4x4)")
    print(f"  L2 权重重排最大差异: {l2_max_diff:.6e}")
    # ------------------ (L3, L_Out, Mu 逻辑不变, 校验逻辑类似 L1/L2) ------------------

    # ------------------ L3: Linear(16 → 16) ------------------
    layer3_weight = mlp_weights["b3_weight"]
    for i in range(4):
        for j in range(4):
            out_s, out_e = i * 4, (i + 1) * 4
            in_s, in_e = j * 4, (j + 1) * 4
            weight_block = layer3_weight[out_s:out_e, in_s:in_e]
            base_idx = (24 + i * 4 + j) * 4
            np.copyto(shader_params[base_idx:base_idx + 4], weight_block)

    # L3 偏置 (16 维)
    layer3_bias = mlp_weights["b3_bias"]
    bias_base_idx = 184
    for i in range(4):
        shader_params[bias_base_idx + i] = layer3_bias[i * 4:(i + 1) * 4]

    # ------------------ L_Out: Linear(16 → 3) ------------------
    layer4_weight = mlp_weights["head_weight"]  # (3, 16)
    layer4_bias = mlp_weights["head_bias"]  # (3)

    # 权重 (3, 16) 补齐到 (4, 16)
    layer4_weight_pad = np.pad(layer4_weight, ((0, 1), (0, 0)), mode="constant")
    for j in range(4):
        in_s, in_e = j * 4, (j + 1) * 4
        weight_block = layer4_weight_pad[:, in_s:in_e]
        base_idx = (40 + j) * 4
        np.copyto(shader_params[base_idx:base_idx + 4], weight_block)

    # L_Out 偏置 (3 维) 补齐到 (4)
    bias_base_idx = 188
    shader_params[bias_base_idx] = np.pad(layer4_bias, (0, 1), mode="constant")

    # ------------------ 放置 MU (索引 4095) ------------------
    mu_value = mlp_weights["mu"]
    shader_params[1023, 0] = mu_value
    print(f"\n📢 **重要:** Mu (Gamma) 值 {mu_value:.4f} 已放置到 SHADER 数组的索引 4095 (即 [1023, 0]) 处。")

    # 保存
    flat_data = shader_params.flatten()
    with open(shader_bin_path, "wb") as f:
        f.write(struct.pack(f"{len(flat_data)}f", *flat_data))

    print(f"✅ SHADER兼容BIN已保存: {shader_bin_path} (大小: {len(flat_data) * 4} 字节)")
    return shader_params


# 全局变量用于存储 SHADER L_Out 原始输出，以便在 validate_results 中对比
SHADER_L4_RAW_OUTPUT: np.ndarray = np.zeros(3, dtype=np.float32)


def simulate_shader_inference(shader_bin_path: str, input_y: float, input_x: float, input_time: float,
                              input_feature: np.ndarray) -> np.ndarray:
    """
    🟢 修改后：直接从 SHADER BIN 文件路径读取 MLP 参数，模拟 SHADER 推理，
    打印中间输出，并更新全局变量 SHADER_L4_RAW_OUTPUT。
    """
    global SHADER_L4_RAW_OUTPUT
    # 1. 构造 model_input (8 维)
    time_coord_expm1 = np.expm1(input_time)
    model_input = np.zeros((2, 4), dtype=np.float32)
    # 输入顺序: (y, x, time, F0)
    model_input[0] = [input_y, input_x, input_time, input_feature[0]]
    # 输入顺序: (F1, F2, F3, expm1(time))
    model_input[1] = [input_feature[1], input_feature[2], input_feature[3], time_coord_expm1]

    # 2. 从 SHADER BIN 文件加载权重 (关键修改)
    if not os.path.exists(shader_bin_path):
        print(f"❌ 错误: SHADER BIN 文件 {shader_bin_path} 不存在，无法模拟。")
        # 返回一个包含 NaN 的数组以表示失败
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    shader_params_flat = np.fromfile(shader_bin_path, dtype=np.float32)
    # 假设 SHADER BIN 文件的大小始终是 1024 * 4 = 4096 个 FP32 元素
    if len(shader_params_flat) != 4096:
        print(f"❌ 错误: SHADER BIN 文件长度不正确，预期 4096 元素，实际 {len(shader_params_flat)}。")
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    shader_params = shader_params_flat.reshape(1024, 4)

    print("\n--- 💻 SHADER 模拟 (中间输出，权重来自 BIN 文件) ---")

    # ------------------ L1: Linear(8 → 16) ------------------
    Output1 = np.zeros((4, 4), dtype=np.float32)
    bias_base_idx1 = 176
    for i in range(4):
        for j in range(2):
            base_idx = (i * 2 + j) * 4
            weight_matrix = shader_params[base_idx:base_idx + 4]
            # 注意：np.dot(Matrix, Vector) 实现了所需的矩阵乘法 W*x
            Output1[i] += np.dot(weight_matrix, model_input[j])
        Output1[i] += shader_params[bias_base_idx1 + i]
        Output1[i] = swish_like_activation_np(Output1[i])

    L1_output_flat = Output1.flatten()
    print(f"  L1 Output (16D, Max: {np.max(L1_output_flat):.6e}, Min: {np.min(L1_output_flat):.6e})")

    # ------------------ L2: Linear(16 → 16) ------------------
    Output0 = np.zeros((4, 4), dtype=np.float32)
    bias_base_idx2 = 180
    for i in range(4):
        for j in range(4):
            base_idx = (8 + i * 4 + j) * 4
            weight_matrix = shader_params[base_idx:base_idx + 4]
            Output0[i] += np.dot(weight_matrix, Output1[j])
        Output0[i] += shader_params[bias_base_idx2 + i]
        Output0[i] = swish_like_activation_np(Output0[i])

    L2_output_flat = Output0.flatten()
    print(f"  L2 Output (16D, Max: {np.max(L2_output_flat):.6e}, Min: {np.min(L2_output_flat):.6e})")

    # ------------------ L3: Linear(16 → 16) ------------------
    Output1_new = np.zeros((4, 4), dtype=np.float32)
    bias_base_idx3 = 184
    for i in range(4):
        for j in range(4):
            base_idx = (24 + i * 4 + j) * 4
            weight_matrix = shader_params[base_idx:base_idx + 4]
            Output1_new[i] += np.dot(weight_matrix, Output0[j])
        Output1_new[i] += shader_params[bias_base_idx3 + i]
        Output1_new[i] = swish_like_activation_np(Output1_new[i])

    L3_output_flat = Output1_new.flatten()
    print(f"  L3 Output (16D, Max: {np.max(L3_output_flat):.6e}, Min: {np.min(L3_output_flat):.6e})")

    # ------------------ L_Out: Linear(16 → 3) ------------------
    Output4_xyz = np.zeros(3, dtype=np.float32)
    bias_base_idx4 = 188
    for j in range(4):
        base_idx = (40 + j) * 4
        weight_matrix = shader_params[base_idx:base_idx + 4]
        mul_result_4d = np.dot(weight_matrix, Output1_new[j])
        Output4_xyz += mul_result_4d[:3]
    Output4_xyz += shader_params[bias_base_idx4][:3]

    SHADER_L4_RAW_OUTPUT = Output4_xyz  # 更新全局变量
    print(f"  L_Out Output Raw (3D): {Output4_xyz}")

    # 最后的 Gamma 校正
    mu = shader_params[1023, 0]
    inv_gamma = 1.0 / mu

    if np.any(Output4_xyz < 0):
        print(f"🚨 **警告：L_Out 原始输出包含负数！** Min: {np.min(Output4_xyz):.4e}")

    # 确保 Gamma 反变换前进行钳制 (如您在 SHADER 中所做)
    # 钳制是强制性的，防止 pow(负数, 非整数) 产生 NaN
    Output4_xyz = np.maximum(0.0, Output4_xyz)

    # 避免 mu=0 导致的除零，或 mu 异常导致的 pow 错误
    if mu == 0 or np.isclose(mu, 0):
        print("🚨 **严重警告：Mu 值接近或等于零，Gamma 反变换可能导致无穷大。**")
        final_output = Output4_xyz  # 返回未校正结果
    else:
        final_output = np.power(Output4_xyz, inv_gamma)

    print(f"Gamma 校正 (mu={mu:.4f}, inv_gamma={inv_gamma:.4f}) 后输出: {final_output}")

    return final_output


def validate_results(model: FinalModel_JIT, raw_bin_path: str, shader_bin_path: str, device: str,
                     mlp_weights: Dict[str, Any]) -> bool:
    """
    验证结果，执行完整的 PyTorch 模型推理并与 SHADER 模拟结果进行对比。
    """
    # 固定的测试输入
    test_y = 0.1
    test_x = 0.8
    test_time = 12.0 / 24.0
    test_feature_numpy = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    print(f"\n===== 📊 推理结果校验 (Input y={test_y}, x={test_x}, t={test_time}) =====")

    # --- 1. PyTorch 模型推理 ---
    model.eval()
    with torch.no_grad():
        x_input_tensor = torch.tensor([[test_y, test_x, test_time]], dtype=TRAIN_DTYPE, device=device)
        time_coords = x_input_tensor[:, [model.TIME_COL_IDX]]
        time_coord_expm1 = torch.expm1(time_coords)
        feature_tensor = torch.from_numpy(test_feature_numpy).unsqueeze(0).to(TRAIN_DTYPE).to(device)
        input_combined_tensor = torch.cat([x_input_tensor, feature_tensor, time_coord_expm1], dim=1)

        # 调用 MLPStack (填充 intermediate_outputs)
        pytorch_output_raw_tensor = model.jit_mlp_stack(input_combined_tensor)
        L4_raw_pt = pytorch_output_raw_tensor.squeeze(0).cpu().numpy()  # L_Out 原始输出 (校正前)

        # 打印 PyTorch 中间输出
        print("\n--- 💡 PyTorch MLP (中间输出) ---")

        L1_pt = model.jit_mlp_stack.intermediate_outputs['L1_output']
        L2_pt = model.jit_mlp_stack.intermediate_outputs['L2_output']
        L3_pt = model.jit_mlp_stack.intermediate_outputs['L3_output']

        print(f"  L1 Output (16D, Max: {np.max(L1_pt):.6e}, Min: {np.min(L1_pt):.6e})")
        print(f"  L2 Output (16D, Max: {np.max(L2_pt):.6e}, Min: {np.min(L2_pt):.6e})")
        print(f"  L3 Output (16D, Max: {np.max(L3_pt):.6e}, Min: {np.min(L3_pt):.6e})")
        print(f"  L_Out Output Raw (3D): {L4_raw_pt}")

        # 应用 Gamma 校正
        # 注意：此处使用字典中的 mu 值，确保与 SHADER 模拟中的 mu (来自 BIN 文件)进行校验
        mu = mlp_weights.get("mu", 2.2)
        inv_gamma = 1.0 / mu
        # PyTorch 也要进行钳制，以保证对比的一致性
        pytorch_output_clamped = np.maximum(0.0, L4_raw_pt)
        pytorch_output = np.power(pytorch_output_clamped, inv_gamma)

        print(f"\n  ✅ PyTorch 最终输出 (mu={mu:.4f}): {pytorch_output}")

    # --- 2. SHADER 模拟推理 ---
    # 调用 simulate_shader_inference，它会从 SHADER BIN 文件中读取权重
    shader_output_final = simulate_shader_inference(
        shader_bin_path,
        input_y=test_y,
        input_x=test_x,
        input_time=test_time,
        input_feature=test_feature_numpy
    )
    L4_raw_shader = SHADER_L4_RAW_OUTPUT  # 获取 SHADER 模拟的 L_Out 原始输出

    # --- 3. 结果对比 ---
    abs_diff = np.abs(pytorch_output - shader_output_final)
    max_abs_diff = np.max(abs_diff)

    # 原始输出对比 (定位重排错误的关健)
    max_raw_diff = np.max(np.abs(L4_raw_pt - L4_raw_shader))

    # 相对误差
    epsilon = 1e-6
    rel_diff = abs_diff / (np.abs(pytorch_output) + epsilon)
    max_rel_diff = np.max(rel_diff)

    print(f"\n===== 🔬 最终对比结果 =====")
    print(f"  PyTorch 输出:  {pytorch_output}")
    print(f"  SHADER 模拟: {shader_output_final}")
    print(f"  L_Out Raw 最大差异: {max_raw_diff:.6e}")
    print(f"  最终输出最大绝对误差: {max_abs_diff:.6e}")
    print(f"  最终输出最大相对误差: {max_rel_diff:.6e}")

    tolerance = 1e-4

    if max_raw_diff < tolerance:
        print(f"✅ 权重校验成功！L_Out Raw 最大差异 ({max_raw_diff:.6e}) 小于容忍度 ({tolerance:.6e})。")
        return True
    else:
        print(
            f"❌ 权重校验失败！L_Out Raw 最大差异 ({max_raw_diff:.6e}) 超过容忍度 ({tolerance:.6e})。问题出在 **MLP 权重重排、BIN文件存储或读取逻辑**。")
        return False


# *** 辅助函数 (未修改) ***
def load_featuremap_from_bin(featuremap_bin_path: str, model: FinalModel_JIT, feature_width: int, feature_height: int,
                             device: str):
    try:
        flat_fm = np.fromfile(featuremap_bin_path, dtype=np.float16)
        fm_shape = (1, 4, feature_width, feature_height)
        expected_size = np.prod(fm_shape)
        if len(flat_fm) != expected_size: return None
        fm_data_numpy = flat_fm.reshape(fm_shape)
        with torch.no_grad():
            fm_tensor = torch.from_numpy(fm_data_numpy).to(model.MODEL_DTYPE).to(device)
            if hasattr(model, 'featuremap') and model.featuremap is not None:
                model.featuremap.data.copy_(fm_tensor)
                print(f"✅ Featuremap 从文件 {featuremap_bin_path} 加载并设置到模型 (Shape: {model.featuremap.shape})")
                return fm_data_numpy
            else:
                return None
    except Exception as e:
        print(f"❌ 错误: 加载 Featuremap 文件 {featuremap_bin_path} 时发生异常: {e}")
        return None


def save_featuremap_for_ue(featuremap: np.ndarray, featuremap_save_path: str):
    if featuremap.ndim != 4: return
    featuremap_ue = np.transpose(featuremap, (0, 3, 2, 1))
    flat_data = featuremap_ue.astype(np.float16).flatten()
    os.makedirs(os.path.dirname(featuremap_save_path), exist_ok=True)
    with open(featuremap_save_path, "wb") as f:
        f.write(flat_data.tobytes())
    print(f"✅ Featuremap 已保存 (FP16): {featuremap_save_path} (大小: {len(flat_data) * 2} 字节)")


def validate_featuremap_remapping(original_fm: np.ndarray, featuremap_save_path: str):
    """
    读取已保存的 UE 格式的 Featuremap BIN 文件，执行逆转置，并与原始 Featuremap 对比。

    原始 PyTorch/NumPy 形状: [1, C, W, H]
    UE 保存形状 (save_featuremap_for_ue 中转置后): [1, H, W, C]
    实际保存的文件格式: [H * W * C] (平坦的 FP16 数据)

    原始转置: (0, 3, 2, 1) -> (Batch, H, W, C)
    逆转置: (0, 3, 2, 1) -> (Batch, C, W, H)
    """
    if original_fm is None:
        print("❌ 原始 Featuremap 为空，无法进行校验。")
        return

    # 1. 确定原始维度 (注意：W/H 互换以匹配 PyTorch [C, W, H])
    B, C, W, H = original_fm.shape
    expected_size = C * W * H

    print(f"\n--- 🔬 Featuremap 存储/读取校验 ({featuremap_save_path}) ---")

    try:
        # 2. 从 BIN 文件加载平坦的 FP16 数据
        flat_fm_ue = np.fromfile(featuremap_save_path, dtype=np.float16)

        if len(flat_fm_ue) != expected_size:
            print(f"❌ 校验失败：文件大小不匹配。预期 {expected_size} 个元素，实际 {len(flat_fm_ue)} 个。")
            return

        # 3. 还原形状: [H * W * C] -> [H, W, C] (忽略 Batch 维度)
        # 按照 save_featuremap_for_ue 的逻辑，平坦化后的顺序是 H*W*C 的连续数据
        restored_fm_hwc = flat_fm_ue.reshape((H, W, C))

        # 4. 执行逆转置，还原到 PyTorch 格式 [B, C, W, H]
        # 原始转置：[B, C, W, H] -> [B, H, W, C] (维度 (0, 3, 2, 1))
        # 逆转置：[H, W, C] -> [C, W, H]
        restored_fm_cwh = np.transpose(restored_fm_hwc, (2, 1, 0))  # 逆转置 (C, W, H)

        # 重新加上 Batch 维度 [1, C, W, H]
        restored_fm = np.expand_dims(restored_fm_cwh, axis=0)

        # 5. 对比
        # 原始 Featuremap 是 FP16
        original_fm_fp16 = original_fm.astype(np.float16)
        print(restored_fm_hwc)
        print(original_fm_fp16)

        abs_diff = np.abs(original_fm_fp16 - restored_fm)
        max_abs_diff = np.max(abs_diff)

        if max_abs_diff < 1e-4:
            print(f"✅ Featuremap 校验成功！最大绝对差异: {max_abs_diff:.6e} (小于容忍度 1e-4)。")
        else:
            print(f"❌ Featuremap 校验失败！最大绝对差异: {max_abs_diff:.6e} (大于容忍度 1e-4)。")
            # 打印差异最大的点
            max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            print(f"  - 差异点索引 (B, C, W, H): {max_idx}")
            print(f"  - 原始值: {original_fm_fp16[max_idx]:.6e}")
            print(f"  - 还原值: {restored_fm[max_idx]:.6e}")

    except Exception as e:
        print(f"❌ 校验 Featuremap 文件 {featuremap_save_path} 时发生异常: {e}")


# ===================== 3. 主执行流程 =====================
if __name__ == '__main__':
    data_set_list = ['Data_HPRC']
    device = "cpu"
    mlp_weights = {}  # 全局字典，用于在函数间传递权重数据，包括 mu

    for data_set in data_set_list:
        dataset_path = f'../../../Data/{data_set}'
        config_file = 'config.json'

        if not os.path.exists(os.path.join(dataset_path, config_file)):
            print(f"错误：配置文件 {os.path.join(dataset_path, config_file)} 不存在。跳过。")
            continue

        with open(os.path.join(dataset_path, config_file), 'r', encoding='utf-8') as f:
            data = json.load(f)

        for lightmap_config in data['lightmap_list']:
            resolution = lightmap_config['resolution']
            height = resolution['height']
            width = resolution['width']
            level = lightmap_config["level"]
            id = lightmap_config["id"]

            print(f"\n========================================================")
            print(f"📌 当前加载模型ID: {id}, Level: {level}")
            print(f"========================================================")

            hidden_dim = 16
            feature_width = resolution['width'] // 2
            feature_height = resolution['height'] // 2

            model = FinalModel_JIT(
                input_dim=3,
                output_dim=3,
                feature_width=feature_width,
                feature_height=feature_height,
                hidden_dim=hidden_dim,
                device=device
            )

            # --- 📌 文件路径定义 ---
            RAW_BIN_PATH = f"./Parameters/model_{level}_{id}_mlp_f32.bin"
            Featuremap_PATH = f"./Parameters/model_{level}_{id}_featuremap_f16.bin"
            SHADER_BIN_PATH = f"./Parameters/UE/Neural_Texture_L_{level}_{id}_mlp_params.bin"
            Featuremap_Save_PATH = f"./Parameters/UE2/Neural_Texture_L_{level}_{id}_featuremap.bin"

            DEVICE = "cpu"

            # 1. 加载 MLP 参数 (并读取最后的 Mu)
            if not os.path.exists(RAW_BIN_PATH):
                print(f"错误：原始 MLP BIN 文件 {RAW_BIN_PATH} 不存在。跳过重排。")
                continue
            try:
                # 这一步仍然必须执行，用于 PyTorch 模型推理和 mu 值的获取
                mlp_weights = load_raw_bin_to_model(RAW_BIN_PATH, model, device=DEVICE)
            except Exception as e:
                print(f"❌ 加载 MLP 文件失败: {e}")
                continue

            # 2. 加载 Featuremap
            featuremap_numpy = load_featuremap_from_bin(Featuremap_PATH, model, feature_width, feature_height,
                                                        device=DEVICE)

            # 3. 导出 Featuremap (维度转换)
            if featuremap_numpy is not None:
                save_featuremap_for_ue(featuremap_numpy, Featuremap_Save_PATH)

            # 4. 转换为 SHADER MLP 格式 (执行 L1/L2 权重对比，并将结果写入 SHADER_BIN_PATH)
            os.makedirs(os.path.dirname(SHADER_BIN_PATH), exist_ok=True)
            shader_params = convert_mlp_to_shader_format(mlp_weights, SHADER_BIN_PATH)

            # 5. 验证结果 (执行中间输出对比)
            # 现在 validate_results 内部会调用 simulate_shader_inference，
            # simulate_shader_inference 会从 SHADER_BIN_PATH 中读取权重
            validate_results(model, RAW_BIN_PATH, SHADER_BIN_PATH, device=DEVICE, mlp_weights=mlp_weights)