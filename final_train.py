import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Tuple, Dict, Any
import Utils 
from ExampleModel import FinalModel  
# 假设 OpenEXR 模块
try:
    import OpenEXR 
except ImportError:
    print("Warning: OpenEXR module not found. Functionality related to EXR files may fail if used.")


# 设置设备和精度
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
TRAIN_DTYPE = torch.float32


# ====================================================================
# 🔴 模型组件 B：FixedMu (GAMMA) 及其变换
# ====================================================================
class FixedMu(nn.Module):
    """
    包装 MU 参数的容器。MU 现在代表 GAMMA (gamma)。
    """

    def __init__(self, initial_mu: float = 0.5): # 初始值改为 GAMMA 常用值
        super(FixedMu, self).__init__()
        # 存储 log(gamma) 或特殊标记。
        # 对于 GAMMA，我们通常搜索 (0, 1] 范围，不需要 log(0) 的特殊标记。
        log_mu_initial = torch.tensor(np.log(initial_mu), dtype=torch.float32)

        self.register_buffer('log_mu', log_mu_initial)

    @property
    def mu(self):
        """返回当前的 MU 值 (即 GAMMA 值)"""
        # 返回 exp(log_mu)
        current_gamma = torch.exp(self.log_mu)
        # 确保 GAMMA > 0 
        return torch.clamp(current_gamma, min=1e-6)

    def set_mu(self, new_mu: float):
        """手动设置 MU 值 (用于搜索)，如果 new_mu=0，则设置一个极小值"""
        new_mu = max(1e-6, new_mu) # 确保 GAMMA > 0
        log_mu_val = torch.tensor(np.log(new_mu), dtype=torch.float32)
        self.log_mu.data = log_mu_val.to(self.log_mu.device)


def gamma_transform_fixed(x: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    """
    应用 Gamma 变换：Y = X ** gamma。
    MU (gamma) 必须 > 0。
    """
    current_gamma = mu_param.mu
    
    # MU=1 (gamma=1) 时，Y = X (线性编码)。
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x
    else:
        # X ** gamma
        # 使用 torch.pow 安全计算，并确保输入数据 (x) 非负
        return torch.pow(x.clamp(min=0.0), current_gamma)


def inverse_gamma_transform_fixed(x_gamma: torch.Tensor, mu_param: FixedMu) -> torch.Tensor:
    """
    应用 Gamma 反变换：X = Y ** (1 / gamma)。
    MU (gamma) 必须 > 0。
    """
    current_gamma = mu_param.mu
    
    # MU=1 (gamma=1) 时，X = Y (线性解码)。
    if torch.isclose(current_gamma, torch.tensor(1.0).to(current_gamma.device)):
        return x_gamma
    else:
        # Y ** (1 / gamma)
        inv_gamma = 1.0 / current_gamma
        # 使用 torch.pow 安全计算，并确保输入数据 (x_gamma) 非负
        return torch.pow(x_gamma.clamp(min=0.0), inv_gamma)


# --------------------------------------------------------------------


# ====================================================================
# 🔴 训练辅助函数
# ====================================================================
def set_trainable(model, layer_names, head_names, requires_grad):
    """设置指定层列表的 requires_grad 属性"""
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


def get_trainable_params(model):
    """返回当前所有可训练的参数"""
    return [p for p in model.parameters() if p.requires_grad]


def evaluate_metrics(model: nn.Module, mu_container: FixedMu, total_coords: torch.Tensor,
                     lightmap_data_original: torch.Tensor, mask_data: np.ndarray, resolution: Dict, time_count: int,
                     args: argparse.Namespace) -> Tuple[float, float, float]:
    """评估模型在给定 MU (GAMMA) 值下的 PSNR, SSIM, LPIPS，采用官方分块和 Mask 阈值。"""

    # 确保 lightmap_data_original 在设备上，且是 FP32 (TRAIN_DTYPE)
    device = total_coords.device
    lightmap_data = lightmap_data_original.to(device)
    
    with torch.no_grad():
        model.eval()

        EVAL_HEAD_INDEX = 2  # 官方评估使用的 Head 3 (索引 2)
        EVAL_BATCH_SIZE = args.batch_size
        num_batches = (total_coords.shape[0] + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

        # 变量名修改以反映 GAMMA 域
        pred_gamma_list = []
        # 1. 模型推理（批量）
        for i in range(num_batches):
            batch_start = i * EVAL_BATCH_SIZE
            batch_end = min(batch_start + EVAL_BATCH_SIZE, total_coords.shape[0])
            batch_data = total_coords[batch_start:batch_end]

            pred_all = model(batch_data[:, :3])
            # 变量名修改以反映 GAMMA 域
            final_pred_tensor_gamma = pred_all[EVAL_HEAD_INDEX] if isinstance(pred_all, list) else pred_all
            pred_gamma_list.append(final_pred_tensor_gamma)

        # 变量名修改
        pred_gamma = torch.cat(pred_gamma_list, dim=0)

        # 2. ⭐️ 核心修改：反 GAMMA 变换
        pred_linear = inverse_gamma_transform_fixed(pred_gamma, mu_container)

        # 3. 重新 reshape 用于评估 (B, C, H, W)
        pred = pred_linear.reshape(time_count, resolution['height'], resolution['width'], 3).permute(0, 3, 1, 2)
        lightmap_data_tensor = lightmap_data.reshape(time_count, resolution['height'], resolution['width'], 3).permute(
            0, 3, 1, 2)

        psnr_list = []
        ssim_list = []
        lpips_list = []

        part_size = 256
        H, W = resolution['height'], resolution['width']
        rows = (H + part_size - 1) // part_size
        cols = (W + part_size - 1) // part_size

        # 4. 官方分块循环和 Mask 逻辑 (保持原样)
        for time_idx in range(time_count):
            pred_frame = pred[[time_idx]]
            gt_frame = lightmap_data_tensor[[time_idx]]
            mask_frame = mask_data[time_idx] # (H, W) numpy array

            # 官方逻辑 1: 将重建结果的无效像素设为 0 (Mask <= 0)
            pred_frame_np = pred_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() # (H, W, 3) numpy for masking
            pred_frame_np[mask_frame <= 0] = 0
            
            # 将修正后的 np 数组转回 torch tensor (1, 3, H, W)
            pred_frame = torch.from_numpy(pred_frame_np).to(pred.dtype).to(device).permute(2, 0, 1).unsqueeze(0)


            # 官方逻辑 2: 分块计算指标
            for i in range(rows):
                for j in range(cols):
                    start_row = i * part_size
                    end_row = min((i + 1) * part_size, H)
                    start_col = j * part_size
                    end_col = min((j + 1) * part_size, W)

                    # 提取分块
                    gt_part = gt_frame[:, :, start_row:end_row, start_col:end_col]
                    pred_part = pred_frame[:, :, start_row:end_row, start_col:end_col]
                    mask_part = mask_frame[start_row:end_row, start_col:end_col]
                    
                    # 官方逻辑 3 & 4: 核心有效像素阈值和跳过条件 (保持原样)
                    valid_mask = mask_part >= 127

                    if (np.any(valid_mask) and gt_part.max().item() != 0):
                        # ⚠️ 假设 Utils 模块的函数 (Utils1.cal_psnr等) 是存在的
                        # 注意: 这里的 Utils1 应该是您代码中的 Utils 模块
                        # 我将变量名改回常用的 Utils 命名方式，请根据您的实际导入调整
                        
                        psnr_list.append(Utils.cal_psnr(gt_part, pred_part, mask_part))
                        ssim_list.append(Utils.cal_ssim(gt_part, pred_part))
                        lpips_list.append(Utils.cal_lpips(gt_part, pred_part))

        return np.mean(psnr_list), np.mean(ssim_list), np.mean(lpips_list)

# ====================================================================
# 🔴 MU (GAMMA) 搜索函数 (使用 GAMMA 范围)
# ====================================================================
def mu_search(model: nn.Module, mu_container: FixedMu, total_data: torch.Tensor, total_coords: torch.Tensor,
              lightmap_data_original: torch.Tensor, mask_data: np.ndarray, resolution: Dict, time_count: int,
              args: argparse.Namespace) -> float:
    """
    使用 P4 配置和精细化遍历搜索寻找最佳 MU 值 (现在代表 GAMMA)。
    指标：Combined Metric = PSNR + SSIM - LPIPS。
    """
    # ⭐️ GAMMA 候选值 (范围集中在 (0, 1] 以实现提亮/非线性编码)
    gamma_candidates = [0.05, 0.1,0.125, 0.15, 0.175,0.2,0.225, 0.25,0.275, 0.3, 0.35,0.4,0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 # 1.0 作为线性基准
    ]

    SEARCH_ITERS = 100
    SEARCH_LR = args.lr_p4

    # P4 阶段配置：所有 MLP 层和 Head 4
    stage_layers = ["b1", "b2", "b3"]
    stage_heads = ["head3"]

    best_gamma = gamma_candidates[0]
    best_metric = -float('inf')

    # 备份初始模型状态
    initial_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n--- 🧪 GAMMA 精细遍历搜索开始 ({len(gamma_candidates)}个候选值, {SEARCH_ITERS} 迭代/GAMMA) ---")

    for gamma_val in gamma_candidates:
        print(f"\n> 尝试 GAMMA={gamma_val:.2f}")

        # 1. 恢复模型状态和设置 GAMMA
        model.load_state_dict(initial_model_state)
        # mu_container 现在存储 gamma_val
        mu_container.set_mu(gamma_val)
        set_trainable(model, stage_layers, stage_heads, True)

        trainable_params = get_trainable_params(model)
        optimizer = optim.Adam(trainable_params, lr=SEARCH_LR)
        criterion = nn.L1Loss()

        # 2. 预训练
        batch_start = 0
        data_size = total_data.shape[0]
        required_size = SEARCH_ITERS * args.batch_size
        current_data = total_data[torch.randperm(data_size)][:min(data_size, required_size)]

        for it in range(SEARCH_ITERS):
            model.train()
            batch_end = min(batch_start + args.batch_size, current_data.shape[0])
            batch_data = current_data[batch_start:batch_end]

            input_coords = batch_data[:, :3]
            target_linear = batch_data[:, 3:]
            # ⭐️ 替换为 GAMMA 变换
            target_gamma = gamma_transform_fixed(target_linear, mu_container)

            preds_gamma = model(input_coords)
            # 损失现在基于 GAMMA 域
            loss = criterion(preds_gamma[2], target_gamma)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start = batch_end
            if batch_start >= current_data.shape[0]:
                batch_start = 0
                current_data = total_data[torch.randperm(data_size)][:min(data_size, required_size)]

        # 3. 评估 (评估 Head 4)
        avg_psnr, avg_ssim, avg_lpips = evaluate_metrics(
            model, mu_container, total_coords, lightmap_data_original, mask_data, resolution, time_count, args
        )

        # 综合指标: PSNR + SSIM - LPIPS
        combined_metric = avg_psnr + 100*avg_ssim - 100*avg_lpips

        print(
            f"GAMMA={gamma_val:.2f} 预训练后 | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | Combined: {combined_metric:.4f}")

        if combined_metric > best_metric:
            best_metric = combined_metric
            best_gamma = gamma_val

    print(f"\n--- 🏆 GAMMA 搜索完成。最佳 GAMMA 值: {best_gamma:.2f} (Combined Metric: {best_metric:.4f}) ---")

    # 4. 恢复初始模型状态
    model.load_state_dict(initial_model_state)

    return best_gamma


# ====================================================================
# 🔴 训练阶段函数 (分层微调)
# ====================================================================
def train_stage(model, mu_container: FixedMu, stage_layers, stage_heads, head_index, stage_name, total_data, args,
                criterion,
                total_iterations,
                current_lr, scaler=None):
    """执行单个分层微调阶段 (MU (GAMMA) 固定)"""

    set_trainable(model, stage_layers, stage_heads, True)
    trainable_params = get_trainable_params(model)

    print(f"\n--- 开始阶段 {stage_name}: 分层微调 ---")
    print(f"目标训练层 (Block): {stage_layers}, Head: {stage_heads}, GAMMA 固定: {mu_container.mu.item():.2f}")

    optimizer = optim.Adam(trainable_params, lr=current_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=max(1, args.patience // 2),
                                  threshold=1e-4, threshold_mode='abs', min_lr=1e-6)

    batch_start = 0
    train_loss_history = []
    LOSS_RECORD_STEP = 50
    LOSS_HISTORY_LEN = 5
    data_size = total_data.shape[0]

    for it in range(total_iterations):
        model.train()
        batch_end = min(batch_start + args.batch_size, data_size)
        batch_data = total_data[batch_start:batch_end]

        input_coords = batch_data[:, :3]
        target_linear = batch_data[:, 3:]
        # ⭐️ 替换为 GAMMA 变换
        target_gamma = gamma_transform_fixed(target_linear, mu_container)

        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_gamma = model(input_coords)
                stage_loss = criterion(preds_gamma[head_index], target_gamma)
            scaler.scale(stage_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()  # AMP 梯度清零位置
        else:
            preds_gamma = model(input_coords)
            stage_loss = criterion(preds_gamma[head_index], target_gamma)
            optimizer.zero_grad()
            stage_loss.backward()
            optimizer.step()

        batch_start = batch_end
        if batch_start >= data_size:
            batch_start = 0
            # 重新打乱数据
            total_data = total_data[torch.randperm(data_size)]

        if (it + 1) % LOSS_RECORD_STEP == 0:
            train_loss_history.append(stage_loss.item())
            if len(train_loss_history) >= LOSS_HISTORY_LEN:
                avg_recent_loss = np.mean(train_loss_history[-LOSS_HISTORY_LEN:])
                scheduler.step(avg_recent_loss)

        if (it + 1) % 1000 == 0:
            current_lr_print = optimizer.param_groups[0]['lr']
            print(
                f"{stage_name} 迭代 {it + 1}/{total_iterations} | 损失: {stage_loss.item():.6f} | LR: {current_lr_print:.6f}")

    print(f"--- 阶段 {stage_name} 训练完成 ---")


def train_full_model(model, mu_container: FixedMu, stage_name, total_data, args, criterion, total_iterations,
                     current_lr,
                     scaler=None):
    """用于 P4 最终全局微调的全模型训练函数 (MU (GAMMA) 固定)"""

    # 解冻所有参数
    for name, param in model.named_parameters():
        param.requires_grad = True

    print(f"\n--- 开始阶段 {stage_name}: 全局微调 ({total_iterations} 迭代) ---")
    print(f"GAMMA 固定: {mu_container.mu.item():.2f}")

    all_params = list(model.parameters())
    optimizer = optim.Adam(all_params, lr=current_lr)

    full_patience = args.patience
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=full_patience,
        threshold=1e-4, threshold_mode='abs', min_lr=2e-6
    )

    batch_start = 0
    best_loss = float('inf')
    best_model_params = None
    patience_counter = 0
    train_loss_history = []
    LOSS_RECORD_STEP = 50
    LOSS_HISTORY_LEN = 5
    LOSS_HEAD_INDEX = 2  # P4 阶段的损失通常基于 Head 3 (索引 2) (GAMMA 域)
    data_size = total_data.shape[0]

    for it in range(total_iterations):
        model.train()
        batch_end = min(batch_start + args.batch_size, data_size)
        batch_data = total_data[batch_start:batch_end]

        input_coords = batch_data[:, :3]
        target_linear = batch_data[:, 3:]
        # ⭐️ 替换为 GAMMA 变换
        target_gamma = gamma_transform_fixed(target_linear, mu_container)

        if args.mixed_precision:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_gamma = model(input_coords)
                loss = criterion(preds_gamma[LOSS_HEAD_INDEX], target_gamma)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            preds_gamma = model(input_coords)
            loss = criterion(preds_gamma[LOSS_HEAD_INDEX], target_gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_start = batch_end
        if batch_start >= data_size:
            batch_start = 0
            total_data = total_data[torch.randperm(data_size)]

        if (it + 1) % LOSS_RECORD_STEP == 0:
            train_loss_history.append(loss.item())
            if len(train_loss_history) >= LOSS_HISTORY_LEN:
                avg_recent_loss = np.mean(train_loss_history[-LOSS_HISTORY_LEN:])
                scheduler.step(avg_recent_loss)

        if (it + 1) % 1000 == 0:
            current_lr_print = optimizer.param_groups[0]['lr']
            print(
                f"{stage_name} 迭代 {it + 1}/{total_iterations} | 损失: {loss.item():.6f} | 学习率: {current_lr_print:.6f}")

        # 早停逻辑 (每隔 val_interval 检查一次)
        if stage_name == "P4" and (it + 1) % args.val_interval == 0:
            recent_losses = train_loss_history[-10:] if len(train_loss_history) >= 10 else train_loss_history
            avg_current_loss = np.mean(recent_losses) if recent_losses else float('inf')

            if avg_current_loss < best_loss - 1e-4:
                best_loss = avg_current_loss
                best_model_params = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"✅ P4 更新最优模型！当前最优损失: {best_loss:.6f}")
            else:
                patience_counter += 1
                print(f"❌ P4 无有效损失下降，计数器: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience*3:
                print(f"\n⚠️ P4 早停触发！连续 {args.patience} 个评估周期无损失提升")
                break

    print(f"--- 阶段 {stage_name} 训练完成 ---")

    if best_model_params is not None:
        print(f"加载最优模型（损失: {best_loss:.6f}）")
        model.load_state_dict(best_model_params)

def parse_times(time_str):
    return int(time_str) / 100.0
# ====================================================================
# 🔴 主函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=200000)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--dataset", type=str, default='../Data/Data_HPRC')
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--lr_p1", type=float, default=5e-3)
    parser.add_argument("--iters_p1", type=int, default=5000)
    parser.add_argument("--lr_p2", type=float, default=5e-3)
    parser.add_argument("--iters_p2", type=int, default=5000)
    parser.add_argument("--lr_p3", type=float, default=5e-3)
    parser.add_argument("--iters_p3", type=int, default=5000)
    parser.add_argument("--lr_p4", type=float, default=5e-3)
    parser.add_argument("--iters_p4", type=int, default=100000)
    parser.add_argument("--noise_threshold", type=float, default=1e-6)
    args = parser.parse_args()

    # ⚠️ 占位符：确保 FinalModel 和 Utils 在此范围内可用
    if 'FinalModel' not in globals() or 'Utils' not in globals():
        print("Error: FinalModel or Utils placeholders are missing. Please define them.")
        return

    os.makedirs(f"./Parameters", exist_ok=True)
    os.makedirs(f"./ResultImages", exist_ok=True)

    config_file = 'config.json'
    times = ["0", "100", "200", "300", "400", "500", "590", "600", "700", "800", "900", "1000", "1100", "1200", "1300",
             "1400", "1500", "1600", "1700", "1800", "1810", "1900", "2000", "2100", "2200", "2300"]
    time_count = len(times) + 1

    try:
        with open(os.path.join(args.dataset, config_file), 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {os.path.join(args.dataset, config_file)}")
        return

    total_psnr = []
    total_ssim = []
    total_lpips = []

    # ⭐️ 初始 MU (GAMMA) 值改为 GAMMA 常用值
    initial_mu = 0.5 
    mu_container = FixedMu(initial_mu=initial_mu).to(device)
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    for lightmap in config['lightmap_list']:
        print(f"\n===== 训练 lightmap {lightmap['level']}_{lightmap['id']} =====")

        id = lightmap['id']
        if (id < 1150): continue
        lightmap_names = lightmap['lightmaps']
        mask_names = lightmap['masks']
        resolution = lightmap['resolution']

        lightmap_in_different_time = []
        for time_idx in range(time_count):
            lightmap_path = os.path.join(args.dataset, "Data", lightmap_names[times[time_idx % (time_count - 1)]])
            lightmap_bin = np.fromfile(lightmap_path, dtype=np.float32)
            lightmap_in_different_time.append(lightmap_bin.reshape(-1, 3))

        lightmap_data_original = torch.from_numpy(np.concatenate(lightmap_in_different_time, axis=0)).to(
            TRAIN_DTYPE).to(device)
        lightmap_data_linear = lightmap_data_original  # 线性 RGB

        mask_in_different_time = []
        for time_idx in range(time_count):
            mask_path = os.path.join(args.dataset, "Data", mask_names[times[time_idx % (time_count - 1)]])
            mask_bin = np.fromfile(mask_path, dtype=np.int8)
            mask_in_different_time.append(mask_bin.reshape(resolution['height'], resolution['width']))
        mask_data = np.concatenate(mask_in_different_time, axis=0).reshape(time_count, resolution['height'],
                                                                           resolution['width'])
        
        # ⭐️ 重置 FixedMu 容器 (为每个 lightmap 重新搜索)
        mu_container.set_mu(initial_mu)

        xs, ys = np.meshgrid(np.arange(resolution['width']), np.arange(resolution['height']))
        coords = np.stack([ys / (resolution['height'] - 1), xs / (resolution['width'] - 1)], axis=-1).reshape(-1, 2)
        coords = torch.from_numpy(coords).to(TRAIN_DTYPE).to(device)
        total_coords = []
        for time_idx in range(time_count):
            current_time = times[time_idx] if time_idx < len(times) else "2400"
            alpha = torch.full((resolution['width'] * resolution['height'], 1), (parse_times(current_time) / 24)).to(
                TRAIN_DTYPE).to(device)
            coords_with_time = torch.cat([coords, alpha], dim=-1)
            total_coords.append(coords_with_time)
        total_coords = torch.cat(total_coords, dim=0)

        total_data = torch.cat([total_coords, lightmap_data_linear], dim=-1)
        total_data = total_data[torch.randperm(total_data.shape[0])]


        # 重新实例化模型 (MU搜索后重置模型参数)
        model = FinalModel(
            input_dim=3, output_dim=3, feature_width=resolution['width']//2 , feature_height=resolution['height'] //2,
            hidden_dim1=args.hidden_dim, hidden_dim2=args.hidden_dim, device=device
        ).to(device)


        # ====================================================================
        # ⭐️ MU (GAMMA) 搜索
        # ====================================================================
        best_mu = mu_search(
            model, mu_container, total_data, total_coords, lightmap_data_original, mask_data, resolution, time_count,
            args
        )

        # 锁定 FixedMu 容器并设置模型
        mu_container.set_mu(best_mu)
        print(f"\n--- 🚀 正式训练开始。锁定 GAMMA 值: {mu_container.mu.item():.4f} ---")


        # ----------------------------------------------------------------------
        # P1-P4 阶段训练
        # ----------------------------------------------------------------------
        train_stage(model, mu_container, stage_layers=["b1"], stage_heads=["head1"], head_index=0, stage_name="P1",
                    total_data=total_data, args=args, criterion=criterion,
                    total_iterations=args.iters_p1, current_lr=args.lr_p1, scaler=scaler)

        train_stage(model, mu_container, stage_layers=["b1", "b2"], stage_heads=["head2"], head_index=1,
                    stage_name="P2",
                    total_data=total_data, args=args, criterion=criterion,
                    total_iterations=args.iters_p2, current_lr=args.lr_p2, scaler=scaler)

        train_stage(model, mu_container, stage_layers=["b1", "b2", "b3"], stage_heads=["head3"], head_index=2,
                    stage_name="P3",
                    total_data=total_data, args=args, criterion=criterion,
                    total_iterations=args.iters_p3, current_lr=args.lr_p3, scaler=scaler)

        train_full_model(model, mu_container, stage_name="P4", total_data=total_data, args=args,
                         criterion=criterion, total_iterations=args.iters_p4, current_lr=args.lr_p4, scaler=scaler)

        # ====================================================================
        # ⭐️ 参数存储逻辑：FeatureMap (FP32/FP16) + MLP/MU (FP32)
        # ====================================================================
        print(f"\n===== 训练结束，拆分模型参数并保存 (FeatureMap双精度, MLP FP32) =====")
        final_mu = mu_container.mu.item()
        print(f"🌟 最终使用的 GAMMA 变换参数 MU: {final_mu:.6f}")

        full_state_dict = model.state_dict()
        id_str = f"{lightmap['level']}_{id}"

        # 1. 提取 FeatureMap 参数 (用于 FeatureMap 文件)
        featuremap_tensor = full_state_dict.get('featuremap', torch.empty(0))
        if featuremap_tensor.numel() == 0:
            featuremap_array_f32 = np.array([], dtype=np.float32)
            print("⚠️ 警告: 'featuremap' 键缺失或为空!")
        else:
            featuremap_array_f32 = featuremap_tensor.detach().cpu().numpy().flatten().astype(np.float32)

        # 2. 提取 MLP 权重 (FP32)
        mlp_param_keys_in_order = [
            'b1.weight', 'b1.bias',
            'b2.weight', 'b2.bias',
            'b3.weight', 'b3.bias',
            'head3.weight', 'head3.bias',
        ]

        # 确保所有权重都是 FP32
        mlp_params_list = [full_state_dict[key].detach().cpu().numpy().flatten().astype(np.float32)
                           for key in mlp_param_keys_in_order if key in full_state_dict]
        all_mlp_weights_f32 = np.concatenate(mlp_params_list)

        # 3. 构建 MLP 权重文件 (固定长度 4096, FP32)
        MLP_FILE_LENGTH = 16384
        if all_mlp_weights_f32.size >= MLP_FILE_LENGTH:
            raise ValueError(f"MLP 权重 ({all_mlp_weights_f32.size}) 超出了固定长度 {MLP_FILE_LENGTH} 的限制！")

        mlp_final_array_f32 = np.zeros(MLP_FILE_LENGTH, dtype=np.float32)
        mlp_final_array_f32[:all_mlp_weights_f32.size] = all_mlp_weights_f32

        # 4. 存储 MU (GAMMA) 值到索引 4095
        MU_INDEX = MLP_FILE_LENGTH - 1
        mlp_final_array_f32[MU_INDEX] = final_mu

        # 5. 保存文件

        # ⭐️ 文件 A1: FeatureMap (FP32)
        output_featuremap_f32_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f32.bin"
        featuremap_array_f32.tofile(output_featuremap_f32_filename)
        print(f"✅ FeatureMap 参数 (FP32) 已保存至 {output_featuremap_f32_filename} ({featuremap_array_f32.size} 元素)")

        # ⭐️ 文件 A2: FeatureMap (FP16)
        output_featuremap_f16_filename = f"./Parameters/model_{lightmap['level']}_{id}_featuremap_f16.bin"
        featuremap_array_f32.astype(np.float16).tofile(output_featuremap_f16_filename)
        print(f"✅ FeatureMap 参数 (FP16) 已保存至 {output_featuremap_f16_filename} ({featuremap_array_f32.size} 元素)")

        # ⭐️ 文件 B: MLP 权重 + MU (GAMMA) (FP32, 固定长度 4096)
        output_mlp_filename = f"./Parameters/model_{lightmap['level']}_{id}_mlp_f32.bin"
        mlp_final_array_f32.tofile(output_mlp_filename)
        print(f"✅ MLP 权重 (含 MU/GAMMA @ 4095) 已保存至 {output_mlp_filename} ({mlp_final_array_f32.size} 元素, FP32)")

        # ----------------------------------------------------------------------
        # 推理和评估逻辑 (用于记录总指标)
        # ----------------------------------------------------------------------
        avg_psnr, avg_ssim, avg_lpips = evaluate_metrics(
            model, mu_container, total_coords, lightmap_data_original, mask_data, resolution, time_count, args
        )
        print(f"最终评估指标: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips:.4f}")
        total_psnr.append(avg_psnr)
        total_ssim.append(avg_ssim)
        total_lpips.append(avg_lpips)

    print(f"\nmetrics of total data set ---------------")
    print(f"PSNR of all lightmaps: {np.mean(total_psnr):.4f}")
    print(f"SSIM of all lightmaps: {np.mean(total_ssim):.4f}")
    print(f"LPIPS of all lightmaps: {np.mean(total_lpips):.4f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    main()