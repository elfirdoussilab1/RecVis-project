import os
import sys
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
import argparse

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from src.data.embs import ImageDataset
from src.model.blip.blip_embs import blip_embs

def get_blip_config(model="base"):
    config = dict()
    if model == "base":
        config["pretrained"] = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth "
        )
        config["vit"] = "base"
        config["batch_size_train"] = 32
        config["batch_size_test"] = 16
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 4
        config["init_lr"] = 1e-5
    elif model == "large":
        config["pretrained"] = (
            "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth"
        )
        config["vit"] = "large"
        config["batch_size_train"] = 16
        config["batch_size_test"] = 32
        config["vit_grad_ckpt"] = True
        config["vit_ckpt_layer"] = 12
        config["init_lr"] = 5e-6

    config["image_size"] = 384
    config["queue_size"] = 57600
    config["alpha"] = 0.4
    config["k_test"] = 256
    config["negative_all_rank"] = True

    return config

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

@torch.no_grad()
def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()

    setup(rank, world_size)
    
    dataset = ImageDataset(image_dir=args.image_dir, save_dir=args.save_dir)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers, sampler=sampler)
    
    config = get_blip_config(args.model_type)
    model = blip_embs(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    ).to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    model.eval()

    for imgs, video_ids in tqdm(loader, desc=f"Rank {rank}"):
        imgs = imgs.to(rank)
        img_embs = model.module.visual_encoder(imgs)
        img_feats = F.normalize(model.module.vision_proj(img_embs[:, 0, :]), dim=-1).cpu()
        for img_feat, video_id in zip(img_feats, video_ids):
            torch.save(img_feat, args.save_dir / f"{video_id}.pth")

    cleanup()
    if rank == 0:
        print(f"All embeddings saved for {args.image_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to image directory")
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--model_type", type=str, default="base", choices=["base", "large"])

    args = parser.parse_args()

    subdirectories = [subdir for subdir in args.image_dir.iterdir() if subdir.is_dir()]
    if len(subdirectories) == 0:
        args.save_dir = args.image_dir.parent / f"blip-embs-{args.model_type}"
        args.save_dir.mkdir(exist_ok=True)
        main(args)
    else:
        for subdir in subdirectories:
            args.image_dir = subdir
            args.save_dir = subdir.parent.parent / f"blip-embs-{args.model_type}" / subdir.name
            args.save_dir.mkdir(exist_ok=True, parents=True)
            main(args)
