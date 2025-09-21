import os
import torch
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Setup logging
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)


def save_checkpoint(model, dir_path, model_name, epoch):
    os.makedirs(dir_path, exist_ok=True)
    ckpt_path = os.path.join(dir_path, f"{model_name}_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    ckpts = sorted(
        [f for f in os.listdir(dir_path) if f.startswith(model_name+"_") and f.endswith(".pth") and "_best" not in f],
        key=lambda x: int(x.rstrip(".pth").split("_")[-1])
    )
    while len(ckpts) > 3:
        os.remove(os.path.join(dir_path, ckpts.pop(0)))
