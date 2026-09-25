import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path, obj):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_root_from_script(script_path):
    return Path(script_path).resolve().parents[1]


def format_messages_fallback(messages, add_generation_prompt=False):
    lines = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("ASSISTANT:")
    return "\n".join(lines)


def count_parameters(model):
    model = unwrap_model(model)
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def latest_jsonl_value(path, key):
    last = None
    if not os.path.exists(path):
        return None
    for row in read_jsonl(path):
        if key in row:
            last = row[key]
    return last


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def distributed_enabled():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if distributed_enabled():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank():
    if distributed_enabled():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "distributed": False,
            "device": device,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
        }

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return {
        "distributed": True,
        "device": device,
        "rank": get_rank(),
        "local_rank": local_rank,
        "world_size": get_world_size(),
    }


def barrier():
    if distributed_enabled():
        dist.barrier()


def cleanup_distributed():
    if distributed_enabled():
        dist.destroy_process_group()


def reduce_scalar(value, device, average=True):
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=device)
    tensor = tensor.detach().float().to(device)
    if distributed_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= dist.get_world_size()
    return tensor.item()


def all_gather_objects(obj):
    if not distributed_enabled():
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


class SequentialShardSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.indices = list(range(len(dataset)))[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
