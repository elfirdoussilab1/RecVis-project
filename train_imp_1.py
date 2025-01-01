import datetime
import shutil
import time
from pathlib import Path
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from src.tools.utils import calculate_model_params
from src.model.blip.blip_imp_1 import BLIP_Imp_1
from src.test.blip.utils import eval_recall
from src.data.cirr import *
from src.tools.scheduler import CosineSchedule

key = '7c2c719a4d241a91163207b8ae5eb635bc0302a4' # Add key here
wandb.login(key=key)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

def train(model, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"[{100.0 * batch_idx / len(train_loader):.0f}%]\tLoss: {loss.item():.6f}")
        
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
@torch.no_grad()
def evaluate_imp_1(model, data_loader):
    # model is the BLIP-Large model ! (gotten with model.model)
    print("Computing features for evaluation...")
    start_time = time.time()

    tar_img_feats = []
    query_feats = []
    captions = []
    pair_ids = []

    for batch in data_loader:
        ref_img = batch["ref_img"].to(device)
        tar_feat = batch["tar_img_feat"].to(device)
        caption = batch["edit"]
        pair_id = batch["pair_id"]

        pair_ids.extend(pair_id.cpu().numpy().tolist())
        captions.extend(caption)

        # Compute comb embedding
        query_feat = model.compute_comb(ref_img, caption)

        query_feats.append(query_feat.cpu())

        # Encode the target image
        tar_img_feats.append(tar_feat.cpu())

    query_feats = torch.cat(query_feats, dim=0)
    tar_img_feats = torch.cat(tar_img_feats, dim=0)

    query_feats = F.normalize(query_feats, dim=-1)
    tar_img_feats = F.normalize(tar_img_feats, dim=-1)

    sim_q2t = (query_feats @ tar_img_feats.t()).cpu().numpy()

    ref_img_ids = [data_loader.dataset.pairid2ref[pair_id] for pair_id in pair_ids]
    tar_img_ids = [data_loader.dataset.pairid2tar[pair_id] for pair_id in pair_ids]

    # Add zeros where ref_img_id == tar_img_id
    for i in range(len(ref_img_ids)):
        for j in range(len(tar_img_ids)):
            if ref_img_ids[i] == tar_img_ids[j]:
                sim_q2t[i][j] = -10

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    eval_result = eval_recall(sim_q2t)
    print(eval_result)

    wandb.log(
        {
            "val/R1": eval_result["R1"],
            "val/R5": eval_result["R5"],
            "val/R10": eval_result["R10"],
            "val/R_mean": eval_result["R_mean"],
        }
    )

# Paths
annotation = {"train": "annotation/cirr/cap.rc2.train.json", 
              "val": "annotation/cirr/cap.rc2.val.json"}
img_dirs = {"train": "datasets/CIRR/images/train", 
            "val": "datasets/CIRR/images/dev"}
emb_dirs = {"train": "datasets/CIRR/blip-embs-large/train", 
            "val": "datasets/CIRR/blip-embs-large/dev"}

def main(args):
    lr = args.lr
    batch_size = args.batch_size
    max_epochs = args.epochs
    num_workers = args.num_workers

    wandb.init(
            # set the wandb project where this run will be logged
            project=f"CoVR-base",
            name = f"Imp-1: batch-size = {batch_size}, lr = {lr}, optimizer = SGD"
        )
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Path.cwd())

    L.seed_everything(1234, workers=True)

    # Dataset
    data = CIRRDataModule(batch_size=batch_size, num_workers=num_workers, 
                          annotation=annotation, img_dirs= img_dirs, emb_dirs=emb_dirs)

    loader_train = data.train_dataloader()
    loader_val = data.val_dataloader()

    model = BLIP_Imp_1(device).to(device)
    calculate_model_params(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("The number of trainabale params : ", n_trainable)
    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9)

    scheduler = CosineSchedule(min_lr= 0, init_lr= lr, decay_rate=None, max_epochs= max_epochs)

    print("Start training")
    start_time = time.time()
    for epoch in range(max_epochs):
        scheduler(optimizer, epoch)

        columns = shutil.get_terminal_size().columns
        print("-" * columns)
        print(f"Epoch {epoch + 1}/{max_epochs}".center(columns))

        train(model, loader_train, optimizer, epoch)

        # Saving the current W
        model.save(epoch= epoch)

        print("Evaluate")
        model.eval()
        evaluate_imp_1(model, loader_val)
        model.save()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

    # Evaluate on Test data
    # columns = shutil.get_terminal_size().columns
    # print("-" * columns)
    # print(f"Testing on cirr".center(columns))

    # data = CIRRTestDataModule(batch_size= batch_size, 
    #                           annotation="annotation/cirr/cap.rc2.test1.json",
    #                           img_dirs = "datasets/CIRR/images/test1",
    #                           emb_dirs = "datasets/CIRR/blip-embs-large/test1",
    #                           num_workers= num_workers
    #                           )
    # test_loader = data.test_dataloader()

    # test = instantiate(cfg.test[dataset].test)
    # test(model, test_loader)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default= 1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    main(args)
