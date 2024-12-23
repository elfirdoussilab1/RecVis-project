# Author: Aymane El Firdoussi
# Inscpired from: Lucas Ventura

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from multiprocessing import Pool

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.append(project_root)

from src.data.embs import ImageDataset
from src.model.blip.blip_embs import blip_embs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

@torch.no_grad()
def main(args):
    dataset = ImageDataset(
        image_dir=args.image_dir,
        save_dir=args.save_dir,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    print("Creating model")
    config = get_blip_config(args.model_type)
    model = blip_embs(
        pretrained=config["pretrained"],
        image_size=config["image_size"],
        vit=config["vit"],
        vit_grad_ckpt=config["vit_grad_ckpt"],
        vit_ckpt_layer=config["vit_ckpt_layer"],
        queue_size=config["queue_size"],
        negative_all_rank=config["negative_all_rank"],
    )

    # Enabling Parallel data
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    for imgs, video_ids in tqdm(loader):
        imgs = imgs.to(device)
        img_embs = model.visual_encoder(imgs)
        img_feats = F.normalize(model.vision_proj(img_embs[:, 0, :]), dim=-1).cpu()

        # Save the embeddings in parallel
        with Pool(args.num_workers) as pool:
            pool.map(save_embedding, [(img_feat, video_id, args.save_dir) for img_feat, video_id in zip(img_feats, video_ids)])
    print(f"All Embeddings Saved for {args.image_dir}")

def save_embedding(img_feat_video_id_tuple):
    img_feat, video_id, save_dir = img_feat_video_id_tuple
    torch.save(img_feat, save_dir / f"{video_id}.pth")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Path to image directory"
    )
    parser.add_argument("--save_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--model_type", type=str, default="large", choices=["base", "large"]
    )
    args = parser.parse_args()

    subdirectories = [subdir for subdir in args.image_dir.iterdir() if subdir.is_dir()]
    if len(subdirectories) == 0:
        args.save_dir = args.image_dir.parent / f"blip-embs-{args.model_type}"
        args.save_dir.mkdir(exist_ok=True)
        main(args)
    else:
        for subdir in subdirectories:
            args.image_dir = subdir
            args.save_dir = (
                subdir.parent.parent / f"blip-embs-{args.model_type}" / subdir.name
            )
            args.save_dir.mkdir(exist_ok=True, parents=True)
            main(args)
