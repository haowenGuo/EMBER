import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Tuple, Dict, Any
import Utils  # 改为官方Utils6
from ExampleModel import FinalModel  # 引用修改后的FinalModel
import OpenEXR
TRAIN_DTYPE = torch.float32
INFER_DTYPE = torch.float32
"""
进阶训练阶段2：
可以把前面的训练看做训练阶段一，核心目的是训练出一个较好的FEETREUEMAP存储贴图时序信息，
训练阶段二的目的就是固定住Featuremap,然后训练多个MLP，每个MLP是一个时间段的专家，
专门针对该时间段完成解码。完成训练阶段1可以获得一个Featureemap和一个基础MLP网络用于推理。
完成训练阶段2可以认为得到一个混合专家架构，通过时间这一信息，把对应的数据送给对应的MLP专家进行解码。
由于我的模型是8->16->16->3的极小模型，分别针对26个时间训练26个专家总参数量也只有10MB，推理速度只增加了一个IF ELSE的开销。
在我的训练阶段二中，使用的基础模型和训练阶段一一样都是8->16->16->3的小模型，针对26个时刻都训练一个专家MLP，在训练阶段二中，固定住训练阶段一得到的FEATUREMAP，只训练MLP权重和偏置。
为了工程方便，代码很多都是和训练阶段一是一样的，只进行了一部分的修改。
"""

# 设置设备和精度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True

# 定义26个时间点常量
TIME_POINTS = ["0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900",
               "1000", "1100", "1200", "1300", "1400", "1500", "1600", "1700", "1800",
               "1810", "1900", "2000", "2100", "2200", "2300"]
TIME_COUNT = len(TIME_POINTS)  # 26个时间点

# ====================================================================
# 模型组件 B：FixedMu (GAMMA) 及其变换 (保持不变)
# ====================================================================
class FixedMu(nn.Module):
    def __init__(self, initial_mu: float = 0.5):
        super(FixedMu, self).__init__()
        log_mu_initial = torch.tensor(np.log(initial_mu), dtype=torch.float32)
        self.register_buffer('log_mu', log_mu_initial)

    @property
    def mu(self):
        current_gamma = torch.exp(self.log_mu)
        return torch.clamp(current_gamma, min=1e-6)

    def set_mu(self, new_mu: float):
        new_mu = max(1e-6, new_mu)
        log_mu_val = torch.tensor(np.log(new_mu), dtype=torch.float32)
        self.log_mu.data = log_mu_val.to(self.log_mu.device)

def gamma_transform_fixed(x: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x
    else:
        return torch.pow(x.clamp(min=0.0), current_gamma)

def inverse_gamma_transform_fixed(x_gamma: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    current_gamma = mu_param.mu
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x_gamma
    else:
        inv_gamma = 1.0 / current_gamma
        return torch.pow(x_gamma.clamp(min=0.0), inv_gamma)

# ====================================================================
#  训练辅助函数
# ====================================================================
def set_trainable(model, layer_names, head_names, requires_grad):
    """模式一的参数可训练性设置"""
    for name, param in model.named_parameters():
        param.requires_grad = False

    always_trainable_layers = ["featuremap", "pos_embed", "encoding_fusion"]

    for name, param in model.named_parameters():
        is_always_trainable = any(layer_name in name for layer_name in always_trainable_layers)
        if is_always_trainable:
            param.requires_grad = True

    for name, param in model.named_parameters():
        is_target_layer = any(layer_name in name for layer_name in layer_names)
        is_target_ln = any(f"{layer_name}_ln" in name for layer_name in layer_names)
        if is_target_layer or is_target_ln:
            param.requires_grad = requires_grad

    for name, param in model.named_parameters():
        is_target_head = any(head_name in name for head_name in head_names)
        if is_target_head:
            param.requires_grad = requires_grad

def set_time_mlp_trainable(model, time_idx, requires_grad=True):
    """模式二：仅设置指定时间点的MLP可训练"""
    # 冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 仅启用指定时间点的MLP
    if 0 <= time_idx < len(model.time_mlps):
        for param in model.time_mlps[time_idx].parameters():
            param.requires_grad = requires_grad

    # FeatureMap始终冻结
    model.featuremap.requires_grad = False

def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def load_pretrained_featuremap(model, featuremap_path, level, id):
    """加载预训练的FeatureMap（模式二专用）"""
    # 读取featuremap二进制文件
    featuremap_file = os.path.join(featuremap_path, f"model_{level}_{id}_featuremap_f16.bin")
    if not os.path.exists(featuremap_file):
        raise FileNotFoundError(f"FeatureMap文件不存在: {featuremap_file}")

    # 读取并重塑featuremap
    featuremap_data = np.fromfile(featuremap_file, dtype=np.float16)
    feature_shape = model.featuremap.shape
    featuremap_tensor = torch.from_numpy(featuremap_data.reshape(feature_shape)).to(device, TRAIN_DTYPE)

    # 替换并冻结featuremap
    model.featuremap.data = featuremap_tensor
    model.featuremap.requires_grad = False
    print(f"已加载并冻结FeatureMap: {featuremap_file}")

    # 读取gamma值
    mlp_weight_file = os.path.join(featuremap_path, f"model_{level}_{id}_mlp_f32.bin")
    if os.path.exists(mlp_weight_file):
        mlp_weights = np.fromfile(mlp_weight_file, dtype=np.float32)
        gamma_value = mlp_weights[-1]  # gamma存储在最后一位
        return gamma_value
    return 0.5  # 默认值

def save_time_mlp_to_bin(mlp_model, save_path):
    """
    将时间专属MLP的权重保存为BIN文件（适配新增b3层）
    格式：b1.weight -> b1.bias -> b2.weight -> b2.bias -> b3.weight -> b3.bias -> head3.weight -> head3.bias
    """
    # 获取MLP的所有参数（按固定顺序，适配Sequential结构）
    param_order = [
        (0, 'weight'),  # b1.weight
        (0, 'bias'),    # b1.bias
        (2, 'weight'),  # b2.weight
        (2, 'bias'),    # b2.bias
        (4, 'weight'),  # b3.weight（新增）
        (4, 'bias'),    # b3.bias（新增）
        (6, 'weight'),  # head3.weight
        (6, 'bias'),    # head3.bias
    ]

    param_list = []
    total_params = 0

    # 遍历所有参数并转换为numpy数组
    for layer_idx, param_type in param_order:
        try:
            layer = mlp_model[layer_idx]
            param = getattr(layer, param_type)
            param_np = param.detach().cpu().numpy().flatten().astype(np.float32)
            param_list.append(param_np)
            total_params += len(param_np)
        except Exception as e:
            print(f"警告：读取MLP层{layer_idx}参数{param_type}失败: {e}")
            continue

    # 拼接所有参数
    if param_list:
        all_params = np.concatenate(param_list)
        all_params.tofile(save_path)
        print(f"已保存MLP权重到BIN文件: {save_path}")
        print(f"MLP参数总数: {total_params} | 文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")
    else:
        print(f"警告：MLP参数为空，未保存文件: {save_path}")

def get_time_specific_data_for_eval(total_coords: torch.Tensor, lightmap_data_original: torch.Tensor,
                                    mask_data: np.ndarray, resolution: Dict, time_idx: int):
    """为指定时间点获取专属的评估数据"""
    pixels_per_time = resolution['height'] * resolution['width']
    start_idx = time_idx * pixels_per_time
    end_idx = (time_idx + 1) * pixels_per_time

    if start_idx >= total_coords.shape[0]:
        print(f"警告：时间点{time_idx}超出数据范围")
        return None, None, None

    end_idx = min(end_idx, total_coords.shape[0])

    time_coords = total_coords[start_idx:end_idx]
    time_lightmap = lightmap_data_original[start_idx:end_idx]
    time_mask = mask_data[time_idx:time_idx + 1]

    return time_coords, time_lightmap, time_mask

def evaluate_metrics(model: nn.Module, mu_container: FixedMu, total_coords: torch.Tensor,
                     lightmap_data_original: torch.Tensor, mask_data: np.ndarray, resolution: Dict,
                     time_count: int, args: argparse.Namespace, train_mode: str = "mode1",
                     time_idx: int = None) -> Tuple[float, float, float]:
    """评估模型：适配新增b3层的输出（head3）"""
    device = total_coords.device

    if train_mode == "mode2" and time_idx is not None and 0 <= time_idx < TIME_COUNT:
        eval_coords, eval_lightmap, eval_mask = get_time_specific_data_for_eval(
            total_coords, lightmap_data_original, mask_data, resolution, time_idx
        )
        if eval_coords is None:
            return 0.0, 0.0, 0.0
        lightmap_data = eval_lightmap.to(device)
        mask_data_eval = eval_mask
        time_count_eval = 1
        current_time_indices = [time_idx]
    else:
        eval_coords = total_coords
        lightmap_data = lightmap_data_original.to(device)
        mask_data_eval = mask_data
        time_count_eval = time_count
        current_time_indices = list(range(time_count_eval))

    with torch.no_grad():
        model.eval()

        # 核心修改：评估索引改为2（对应head3）
        EVAL_HEAD_INDEX = 2
        EVAL_BATCH_SIZE = args.batch_size
        num_batches = (eval_coords.shape[0] + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

        pred_gamma_list = []
        for i in range(num_batches):
            batch_start = i * EVAL_BATCH_SIZE
            batch_end = min(batch_start + EVAL_BATCH_SIZE, eval_coords.shape[0])
            batch_data = eval_coords[batch_start:batch_end]

            if train_mode == "mode2" and time_idx is not None and 0 <= time_idx < TIME_COUNT:
                pred_all = model(batch_data[:, :3], time_idx=time_idx)
                final_pred_tensor_gamma = pred_all[0] if isinstance(pred_all, list) else pred_all
            else:
                pred_all = model(batch_data[:, :3])
                final_pred_tensor_gamma = pred_all[EVAL_HEAD_INDEX] if isinstance(pred_all, list) else pred_all

            pred_gamma_list.append(final_pred_tensor_gamma)

        pred_gamma = torch.cat(pred_gamma_list, dim=0)

        if pred_gamma.shape[0] != eval_coords.shape[0]:
            print(f"警告：预测结果数量({pred_gamma.shape[0]})与评估数据数量({eval_coords.shape[0]})不匹配")
            if pred_gamma.shape[0] > eval_coords.shape[0]:
                pred_gamma = pred_gamma[:eval_coords.shape[0]]
            else:
                padding = torch.zeros(eval_coords.shape[0] - pred_gamma.shape[0], pred_gamma.shape[1],
                                      device=pred_gamma.device, dtype=pred_gamma.dtype)
                pred_gamma = torch.cat([pred_gamma, padding], dim=0)

        pred_linear = inverse_gamma_transform_fixed(pred_gamma, mu_container)

        pred = pred_linear.reshape(time_count_eval, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)
        lightmap_data_tensor = lightmap_data.reshape(time_count_eval, resolution['height'], resolution['width'],
                                                     3).permute(0, 3, 1, 2)

        psnr_list = []
        ssim_list = []
        lpips_list = []

        part_size = 256
        H, W = resolution['height'], resolution['width']
        rows = (H + part_size - 1) // part_size
        cols = (W + part_size - 1) // part_size

        for eval_time_idx in range(time_count_eval):
            original_time_idx = current_time_indices[eval_time_idx]

            pred_frame = pred[[eval_time_idx]]
            gt_frame = lightmap_data_tensor[[eval_time_idx]]
            mask_frame = mask_data_eval[eval_time_idx % len(mask_data_eval)]

            pred_frame_np = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_frame_np[mask_frame <= 0] = 0
            pred_frame = torch.from_numpy(pred_frame_np).to(pred.dtype).to(device).permute(2, 0, 1).unsqueeze(0)

            for i in range(rows):
                for j in range(cols):
                    start_row = i * part_size
                    end_row = min((i + 1) * part_size, H)
                    start_col = j * part_size
                    end_col = min((j + 1) * part_size, W)

                    gt_part = gt_frame[:, :, start_row:end_row, start_col:end_col]
                    pred_part = pred_frame[:, :, start_row:end_row, start_col:end_col]
                    mask_part = mask_frame[start_row:end_row, start_col:end_col]
                    valid_mask = mask_part >= 127

                    if (np.any(valid_mask) and gt_part.max().item() != 0):
                        # 替换为实际的Utils函数
                        psnr_list.append(Utils.cal_psnr(gt_part, pred_part, mask_part))
                        ssim_list.append(Utils.cal_ssim(gt_part, pred_part))
                        lpips_list.append(Utils.cal_lpips(gt_part, pred_part))

        avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
        avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
        avg_lpips = np.mean(lpips_list) if lpips_list else 0.0

        if train_mode == "mode2" and time_idx is not None:
            print(
                f"时间点{time_idx}（{TIME_POINTS[time_idx]}）评估结果 | PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f}, LPIPS={avg_lpips:.3f}")
        else:
            print(f"全时间点评估结果 | PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.3f}, LPIPS={avg_lpips:.3f}")

        return avg_psnr, avg_ssim, avg_lpips

def prepare_time_specific_data(total_data: torch.Tensor, resolution: Dict, time_idx: int):
    """为指定时间点准备专属训练数据"""
    if time_idx < 0 or time_idx >= TIME_COUNT:
        print(f"警告：无效的时间索引{time_idx}")
        return torch.tensor([], device=total_data.device)

    pixels_per_time = resolution['height'] * resolution['width']
    start_idx = time_idx * pixels_per_time
    end_idx = (time_idx + 1) * pixels_per_time

    if start_idx >= total_data.shape[0]:
        print(f"警告：时间点{time_idx}数据索引超出范围（{start_idx} >= {total_data.shape[0]}）")
        return torch.tensor([], device=total_data.device)

    end_idx = min(end_idx, total_data.shape[0])
    time_data = total_data[start_idx:end_idx]

    if time_data.shape[0] > 0:
        time_data = time_data[torch.randperm(time_data.shape[0])]

    print(f"时间点{time_idx}（{TIME_POINTS[time_idx]}）：准备训练数据 {start_idx}:{end_idx}，共{time_data.shape[0]}样本")
    return time_data

# ====================================================================
#  MU (GAMMA) 搜索函数（仅模式一使用）
# ====================================================================
def mu_search(model: nn.Module, mu_container: FixedMu, total_data: torch.Tensor, total_coords: torch.Tensor,
              lightmap_data_original: torch.Tensor, mask_data: np.ndarray, resolution: Dict, time_count: int,
              args: argparse.Namespace) -> float:
    gamma_candidates = [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                        0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]

    SEARCH_ITERS = 100
    SEARCH_LR = args.lr_p4

    stage_layers = ["b1", "b2", "b3"]  # 新增b3
    stage_heads = ["head3"]             # 改为head3
    SEARCH_HEAD_INDEX = 2              # 改为2

    best_gamma = gamma_candidates[0]
    best_metric = -float('inf')
    initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n--- GAMMA Search({len(gamma_candidates)}) ---")

    for gamma_val in gamma_candidates:
        print(f"\n> Testing GAMMA={gamma_val:.2f}")

        model.load_state_dict(initial_model_state)
        mu_container.set_mu(gamma_val)
        set_trainable(model, stage_layers, stage_heads, True)

        trainable_params = get_trainable_params(model)
        optimizer = optim.Adam(trainable_params, lr=SEARCH_LR)
        criterion = nn.L1Loss()

        batch_start = 0
        data_size = total_data.shape[0]
        required_size = SEARCH_ITERS * args.batch_size
        current_data = total_data[torch.randperm(data_size)][:min(data_size, required_size)]

        for it in range(SEARCH_ITERS):
            model.train()
            batch_end = min(batch_start + args.batch_size, current_data.shape[0])

            if batch_start >= batch_end:
                batch_start = 0
                current_data = total_data[torch.randperm(data_size)][:min(data_size, required_size)]
                continue

            batch_data = current_data[batch_start:batch_end]

            input_coords = batch_data[:, :3]
            target_linear = batch_data[:, 3:]
            target_gamma = gamma_transform_fixed(target_linear, mu_container)

            preds_gamma = model(input_coords)
            if isinstance(preds_gamma, list) and len(preds_gamma) > SEARCH_HEAD_INDEX:
                loss = criterion(preds_gamma[SEARCH_HEAD_INDEX], target_gamma)
            elif isinstance(preds_gamma, torch.Tensor):
                loss = criterion(preds_gamma, target_gamma)
            else:
                print(f"警告：搜索迭代{it}输出异常，跳过")
                batch_start = batch_end
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start = batch_end
            if batch_start >= current_data.shape[0]:
                batch_start = 0
                current_data = total_data[torch.randperm(data_size)][:min(data_size, required_size)]

        avg_psnr, avg_ssim, avg_lpips = evaluate_metrics(
            model, mu_container, total_coords, lightmap_data_original, mask_data, resolution, time_count, args
        )
        combined_metric = avg_psnr + 100 * avg_ssim - 100 * avg_lpips

        print(
            f"GAMMA={gamma_val:.2f} | Metric: {combined_metric:.4f} (PSNR:{avg_psnr:.2f}, SSIM:{avg_ssim:.3f}, LPIPS:{avg_lpips:.3f})")

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_gamma = gamma_val

    print(f"\n- Best GAMMA: {best_gamma:.2f} (Combined Metric: {best_metric:.4f}) ---")
    model.load_state_dict(initial_model_state)
    return best_gamma

# ====================================================================
# 训练阶段函数
# ====================================================================
def train_stage(model, mu_container: FixedMu, stage_layers, stage_heads, head_index, stage_name, total_data, args,
                criterion, total_iterations, current_lr, scaler=None):
    """模式一：分层训练共享MLP（适配head3）"""
    set_trainable(model, stage_layers, stage_heads, True)
    trainable_params = get_trainable_params(model)

    print(f"\n--- Start stage {stage_name}: finetune ---")
    print(f" Layers: {stage_layers}, Head: {stage_heads}, HeadIdx: {head_index}")

    optimizer = optim.Adam(trainable_params, lr=current_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=max(1, args.patience // 2),
                                  threshold=1e-5, threshold_mode='abs', min_lr=1e-6)

    batch_start = 0
    train_loss_history = []
    data_size = total_data.shape[0]

    for it in range(total_iterations):
        model.train()
        batch_end = min(batch_start + args.batch_size, data_size)

        if batch_start >= batch_end:
            batch_start = 0
            total_data = total_data[torch.randperm(data_size)]
            continue

        batch_data = total_data[batch_start:batch_end]

        input_coords = batch_data[:, :3]
        target_linear = batch_data[:, 3:]
        target_gamma = gamma_transform_fixed(target_linear, mu_container)

        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_gamma = model(input_coords)
                if isinstance(preds_gamma, list) and len(preds_gamma) > head_index:
                    stage_loss = criterion(preds_gamma[head_index], target_gamma)
                elif isinstance(preds_gamma, torch.Tensor):
                    stage_loss = criterion(preds_gamma, target_gamma)
                else:
                    raise ValueError("模型输出格式异常")
            scaler.scale(stage_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            preds_gamma = model(input_coords)
            if isinstance(preds_gamma, list) and len(preds_gamma) > head_index:
                stage_loss = criterion(preds_gamma[head_index], target_gamma)
            elif isinstance(preds_gamma, torch.Tensor):
                stage_loss = criterion(preds_gamma, target_gamma)
            else:
                raise ValueError("模型输出格式异常")
            optimizer.zero_grad()
            stage_loss.backward()
            optimizer.step()

        train_loss_history.append(stage_loss.item())

        batch_start = batch_end
        if batch_start >= data_size:
            batch_start = 0
            total_data = total_data[torch.randperm(data_size)]

        if (it + 1) % 50 == 0:
            if len(train_loss_history) >= 5:
                avg_recent_loss = np.mean(train_loss_history[-5:])
                scheduler.step(avg_recent_loss)

        if (it + 1) % 1000 == 0:
            print(f"{stage_name} it {it + 1}/{total_iterations} | loss: {stage_loss.item():.6f}")

    print(f"--- stage {stage_name} finished ---")

def train_full_model(model, mu_container: FixedMu, stage_name, total_data, args, criterion, total_iterations,
                     current_lr, scaler=None):
    """模式一：全模型微调（适配head3）"""
    for name, param in model.named_parameters():
        param.requires_grad = True

    print(f"\n--- Start stage {stage_name}: full finetune ({total_iterations} it) ---")

    all_params = list(model.parameters())
    optimizer = optim.Adam(all_params, lr=current_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=args.patience, threshold=1e-5,
                                  min_lr=1e-6)

    batch_start = 0
    best_loss = float('inf')
    best_model_params = None
    patience_counter = 0
    train_loss_history = []

    LOSS_HEAD_INDEX = 2  # 改为2（对应head3）
    data_size = total_data.shape[0]

    for it in range(total_iterations):
        model.train()
        batch_end = min(batch_start + args.batch_size, data_size)

        if batch_start >= batch_end:
            batch_start = 0
            total_data = total_data[torch.randperm(data_size)]
            continue

        batch_data = total_data[batch_start:batch_end]

        input_coords = batch_data[:, :3]
        target_linear = batch_data[:, 3:]
        target_gamma = gamma_transform_fixed(target_linear, mu_container)

        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_gamma = model(input_coords)
                if isinstance(preds_gamma, list) and len(preds_gamma) > LOSS_HEAD_INDEX:
                    loss = criterion(preds_gamma[LOSS_HEAD_INDEX], target_gamma)
                elif isinstance(preds_gamma, torch.Tensor):
                    loss = criterion(preds_gamma, target_gamma)
                else:
                    raise ValueError("模型输出格式异常")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            preds_gamma = model(input_coords)
            if isinstance(preds_gamma, list) and len(preds_gamma) > LOSS_HEAD_INDEX:
                loss = criterion(preds_gamma[LOSS_HEAD_INDEX], target_gamma)
            elif isinstance(preds_gamma, torch.Tensor):
                loss = criterion(preds_gamma, target_gamma)
            else:
                raise ValueError("模型输出格式异常")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_history.append(loss.item())

        batch_start = batch_end
        if batch_start >= data_size:
            batch_start = 0
            total_data = total_data[torch.randperm(data_size)]

        if (it + 1) % 50 == 0:
            if len(train_loss_history) >= 5:
                scheduler.step(np.mean(train_loss_history[-5:]))

        if (it + 1) % 2000 == 0:
            print(
                f"{stage_name} it {it + 1}/{total_iterations} | loss: {loss.item():.6f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

        if (it + 1) % args.val_interval == 0:
            recent_losses = train_loss_history[-10:] if len(train_loss_history) >= 10 else train_loss_history
            avg_current_loss = np.mean(recent_losses) if recent_losses else float('inf')

            if avg_current_loss < best_loss - 1e-4:
                best_loss = avg_current_loss
                best_model_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"Best found: {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience * 3:
                print(f"Early stopping at {it + 1}")
                break

    if best_model_params is not None:
        model.load_state_dict(best_model_params)
        print("Loaded best model.")

def train_time_specific_mlp(model, mu_container: FixedMu, time_idx, time_data, level, id, args, criterion, scaler=None):
    """模式二：训练单个时间点的专属MLP（适配新增b3层）"""
    if time_data.shape[0] == 0:
        print(f"警告：时间点{TIME_POINTS[time_idx]}无训练数据，跳过训练")
        return

    set_time_mlp_trainable(model, time_idx, True)
    trainable_params = get_trainable_params(model)

    if not trainable_params:
        print(f"警告：时间点{TIME_POINTS[time_idx]}没有可训练参数！")
        return

    lr = args.lr_time
    optimizer = optim.Adam(trainable_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=args.patience,
                                  threshold=1e-5, min_lr=1e-6)

    print(f"\n--- 训练时间点 {TIME_POINTS[time_idx]} (idx:{time_idx}) 的专属MLP ---")
    print(f"  训练数据量：{time_data.shape[0]} 样本")
    print(f"  批次大小：{args.batch_size} | 迭代数：{args.iters_time}")

    batch_start = 0
    train_loss_history = []
    data_size = time_data.shape[0]
    best_loss = float('inf')
    best_mlp_state = None

    for it in range(args.iters_time):
        model.train()
        batch_end = min(batch_start + args.batch_size, data_size)

        if batch_start >= batch_end:
            batch_start = 0
            time_data = time_data[torch.randperm(data_size)] if data_size > 0 else time_data
            continue

        batch_data = time_data[batch_start:batch_end]

        input_coords = batch_data[:, :3]
        target_linear = batch_data[:, 3:]
        target_gamma = gamma_transform_fixed(target_linear, mu_container)

        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_gamma = model(input_coords, time_idx=time_idx)
                if isinstance(preds_gamma, list) and len(preds_gamma) > 0:
                    stage_loss = criterion(preds_gamma[0], target_gamma)
                elif isinstance(preds_gamma, torch.Tensor):
                    stage_loss = criterion(preds_gamma, target_gamma)
                else:
                    raise ValueError("时间MLP输出格式异常")
            scaler.scale(stage_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            preds_gamma = model(input_coords, time_idx=time_idx)
            if isinstance(preds_gamma, list) and len(preds_gamma) > 0:
                stage_loss = criterion(preds_gamma[0], target_gamma)
            elif isinstance(preds_gamma, torch.Tensor):
                stage_loss = criterion(preds_gamma, target_gamma)
            else:
                raise ValueError("时间MLP输出格式异常")
            optimizer.zero_grad()
            stage_loss.backward()
            optimizer.step()

        train_loss_history.append(stage_loss.item())

        batch_start = batch_end
        if batch_start >= data_size:
            batch_start = 0
            time_data = time_data[torch.randperm(data_size)] if data_size > 0 else time_data

        if (it + 1) % 10 == 0:
            if len(train_loss_history) >= 5:
                avg_recent_loss = np.mean(train_loss_history[-5:])
                scheduler.step(avg_recent_loss)

        if (it + 1) % 1000 == 0:
            print(
                f"Time {TIME_POINTS[time_idx]} | Iter {it + 1}/{args.iters_time} | Loss: {stage_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if stage_loss.item() < best_loss:
            best_loss = stage_loss.item()
            best_mlp_state = {k: v.detach().cpu().clone() for k, v in model.time_mlps[time_idx].state_dict().items()}

    if best_mlp_state is not None:
        model.time_mlps[time_idx].load_state_dict(best_mlp_state)
        print(f"时间点 {TIME_POINTS[time_idx]} 最佳Loss: {best_loss:.6f}")

    os.makedirs(f"./Parameters/time_mlps/{level}_{id}", exist_ok=True)
    save_path = f"./Parameters/time_mlps/{level}_{id}/mlp_{TIME_POINTS[time_idx]}.bin"
    save_time_mlp_to_bin(model.time_mlps[time_idx], save_path)

def parse_times(time_str):
    return int(time_str) / 100.0

# ====================================================================
# 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument("--batch_size", type=int, default=200000)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dataset", type=str, default='../../../Data/Data_HPRC')
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--mixed_precision", action="store_true")

    # 模式选择
    parser.add_argument("--train_mode", type=str, default="mode2", choices=["mode1", "mode2"],
                        help="mode1: 训练基础模型和FeatureMap; mode2: 增量训练时间专属MLP")

    # 模式一参数
    parser.add_argument("--lr_p1", type=float, default=5e-3)
    parser.add_argument("--iters_p1", type=int, default=20000)
    parser.add_argument("--lr_p2", type=float, default=5e-3)
    parser.add_argument("--iters_p2", type=int, default=20000)
    parser.add_argument("--lr_p3", type=float, default=5e-3)
    parser.add_argument("--iters_p3", type=int, default=20000)
    parser.add_argument("--lr_p4", type=float, default=5e-3)
    parser.add_argument("--iters_p4", type=int, default=80000)

    # 模式二参数
    parser.add_argument("--lr_time", type=float, default=1e-2, help="时间MLP的学习率")
    parser.add_argument("--iters_time", type=int, default=7000, help="每个时间MLP的训练迭代数")
    parser.add_argument("--featuremap_path", type=str, default="./Parameters", help="预训练FeatureMap路径")

    parser.add_argument("--noise_threshold", type=float, default=1e-6)
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)
    if args.train_mode == "mode2":
        os.makedirs(f"./Parameters/time_mlps", exist_ok=True)

    config_file = 'config.json'

    try:
        with open(os.path.join(args.dataset, config_file), 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {os.path.join(args.dataset, config_file)}")
        return

    total_psnr, total_ssim, total_lpips = [], [], []
    initial_mu = 0.5
    mu_container = FixedMu(initial_mu=initial_mu).to(device)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and torch.cuda.is_available() else None

    for lightmap in config['lightmap_list']:
        print(f"\n===== Processing {lightmap['level']}_{lightmap['id']} =====")

        id = lightmap['id']
        if id < 50: continue

        lightmap_names = lightmap['lightmaps']
        mask_names = lightmap['masks']
        resolution = lightmap['resolution']

        # 加载数据（适配26个时间点）
        lightmap_in_different_time = []
        for time_str in TIME_POINTS:
            lightmap_path = os.path.join(args.dataset, "Data", lightmap_names[time_str])
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))
        lightmap_data_original = torch.from_numpy(np.concatenate(lightmap_in_different_time, axis=0)).to(
            TRAIN_DTYPE).to(device)

        mask_in_different_time = []
        for time_str in TIME_POINTS:
            mask_path = os.path.join(args.dataset, "Data", mask_names[time_str])
            mask_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_in_different_time.append(mask_bin.reshape(resolution['height'], resolution['width']))
        mask_data = np.stack(mask_in_different_time, axis=0)
        time_count = TIME_COUNT

        # 生成坐标数据
        xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
        coords = np.stack([ys / (resolution['height'] - 1), xs / (resolution['width'] - 1)], axis=-1).reshape(-1, 2)
        coords = torch.from_numpy(coords).to(TRAIN_DTYPE).to(device)

        total_coords = []
        for time_idx, time_str in enumerate(TIME_POINTS):
            alpha = torch.full((resolution['width'] * resolution['height'], 1),
                               (parse_times(time_str) / 24)).to(TRAIN_DTYPE).to(device)
            coords_with_time = torch.cat([coords, alpha], dim=-1)
            total_coords.append(coords_with_time)
        total_coords = torch.cat(total_coords, dim=0)

        # 组合坐标和光照数据
        total_data = torch.cat([total_coords, lightmap_data_original], dim=-1)

        # 初始化模型（适配新增b3层）
        model = FinalModel(
            input_dim=3, output_dim=3, feature_width=resolution['width'] // 2, feature_height=resolution['height'] // 2,
            hidden_dim1=args.hidden_dim, hidden_dim2=args.hidden_dim, device=device
        ).to(device)

        # ===================== 模式一：训练基础模型和FeatureMap =====================
        if args.train_mode == "mode1":
            print("\n===================== 模式一：训练基础模型和FeatureMap =====================")

            # 1. 搜索 Gamma
            best_mu = mu_search(
                model, mu_container, total_data, total_coords, lightmap_data_original, mask_data,
                resolution, time_count, args
            )
            mu_container.set_mu(best_mu)

            # 2. 分层训练共享MLP（适配b3层）
            train_stage(model, mu_container, stage_layers=["b1"], stage_heads=["head1"], head_index=0, stage_name="P1",
                        total_data=total_data, args=args, criterion=criterion,
                        total_iterations=args.iters_p1, current_lr=args.lr_p1, scaler=scaler)

            train_stage(model, mu_container, stage_layers=["b1", "b2", "b3"], stage_heads=["head3"], head_index=2,
                        stage_name="P2", total_data=total_data, args=args, criterion=criterion,
                        total_iterations=args.iters_p2, current_lr=args.lr_p2, scaler=scaler)

            # 3. 全模型微调
            train_full_model(model, mu_container, stage_name="Final", total_data=total_data, args=args,
                             criterion=criterion, total_iterations=args.iters_p4, current_lr=args.lr_p4, scaler=scaler)

            # 4. 保存基础模型参数（适配b3层）
            final_mu = mu_container.mu.item()
            full_state_dict = model.state_dict()

            featuremap_tensor = full_state_dict.get('featuremap', torch.empty(0))
            featuremap_array_f32 = featuremap_tensor.detach().cpu().numpy().flatten().astype(np.float32)

            # 新增b3参数
            mlp_param_keys_in_order = [
                'b1.weight', 'b1.bias',
                'b2.weight', 'b2.bias',
                'b3.weight', 'b3.bias',  # 新增b3
                'head3.weight', 'head3.bias'  # 改为head3
            ]

            mlp_params_list = [full_state_dict[key].detach().cpu().numpy().flatten().astype(np.float32)
                               for key in mlp_param_keys_in_order]
            all_mlp_weights_f32 = np.concatenate(mlp_params_list)

            MLP_FILE_LENGTH = 4096
            mlp_final_array_f32 = np.zeros(MLP_FILE_LENGTH, dtype=np.float32)
            mlp_final_array_f32[:all_mlp_weights_f32.size] = all_mlp_weights_f32
            mlp_final_array_f32[-1] = final_mu

            # 保存文件
            output_featuremap_f32_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f32.bin"
            featuremap_array_f32.tofile(output_featuremap_f32_filename)

            output_featuremap_f16_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f16.bin"
            featuremap_array_f32.astype(np.float16).tofile(output_featuremap_f16_filename)

            output_mlp_filename = f"./Parameters/model_{lightmap['level']}_{id}_mlp_f32.bin"
            mlp_final_array_f32.tofile(output_mlp_filename)

            print(f"Saved base model params for {lightmap['level']}_{id}")

        # ===================== 模式二：增量训练时间专属MLP =====================
        else:
            print("\n===================== 模式二：增量训练时间专属MLP =====================")

            # 1. 加载预训练FeatureMap并冻结
            try:
                gamma_value = load_pretrained_featuremap(model, args.featuremap_path, lightmap['level'], id)
                mu_container.set_mu(gamma_value)
            except Exception as e:
                print(f"加载FeatureMap失败: {e}")
                continue

            # 2. 冻结FeatureMap
            model.freeze_featuremap()

            # 3. 为每个时间点训练专属MLP
            for time_idx in range(TIME_COUNT):
                time_data = prepare_time_specific_data(total_data, resolution, time_idx)
                if time_data.shape[0] == 0:
                    print(f"跳过空数据的时间点：{TIME_POINTS[time_idx]}")
                    continue

                train_time_specific_mlp(
                    model, mu_container, time_idx, time_data, lightmap['level'], id, args, criterion, scaler
                )

                psnr, ssim, lpips = evaluate_metrics(
                    model, mu_container, total_coords, lightmap_data_original, mask_data,
                    resolution, time_count, args, train_mode="mode2", time_idx=time_idx
                )
                print(f"时间点 {TIME_POINTS[time_idx]} | PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                total_lpips.append(lpips)

        # 整体评估
        if args.train_mode == "mode1":
            avg_psnr, avg_ssim, avg_lpips = evaluate_metrics(
                model, mu_container, total_coords, lightmap_data_original, mask_data,
                resolution, time_count, args, train_mode="mode1"
            )
            print(f"基础模型评估 | PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
            total_psnr.append(avg_psnr)
            total_ssim.append(avg_ssim)
            total_lpips.append(avg_lpips)

    # 打印最终指标
    print(f"\nmetrics of total data set ---------------")
    print(f"PSNR of all lightmaps: {np.mean(total_psnr):.4f}")
    print(f"SSIM of all lightmaps: {np.mean(total_ssim):.4f}")
    print(f"LPIPS of all lightmaps: {np.mean(total_lpips):.4f}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()