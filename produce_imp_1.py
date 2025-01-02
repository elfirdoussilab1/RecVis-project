# In this file, we will produce a surface plot (heatmap) of how the recall varies with w_1 and w_2.
import numpy as np
import torch
from tqdm.auto import tqdm

import datetime
import shutil
from pathlib import Path
import lightning as L

from src.model.blip.blip_imp_1 import BLIP_Imp_1
from src.data.cirr import *
from train_imp_1 import evaluate_imp_1


print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(Path.cwd())

L.seed_everything(1234, workers=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

# Paths
annotation = {"train": "annotation/cirr/cap.rc2.train.json", 
              "val": "annotation/cirr/cap.rc2.val.json"}
img_dirs = {"train": "datasets/CIRR/images/train", 
            "val": "datasets/CIRR/images/dev"}
emb_dirs = {"train": "datasets/CIRR/blip-embs-large/train", 
            "val": "datasets/CIRR/blip-embs-large/dev"}

def main(args):
    # Blip Model and data loader
    batch_size = args.batch_size
    num_workers = args.num_workers
    N = args.N

    # Dataset
    data = CIRRDataModule(batch_size=batch_size, num_workers=num_workers, 
                            annotation=annotation, img_dirs= img_dirs, emb_dirs=emb_dirs)
    
    loader_val = data.val_dataloader()

    model = BLIP_Imp_1(device).to(device)

    # Weights: W[0] applies to q, W[1] applies to t and W[2] = 1 - (W[0] + W[1])
    W_0 = np.linspace(0, 1, N)
    W_1 = np.linspace(0, 1, N)

    # Initialize lists to store valid points and results
    w0_valid = []
    w1_valid = []
    R_1 = []
    R_5 = []
    R_mean = []

    print("Start computing the recalls")
    columns = shutil.get_terminal_size().columns
    print("-" * columns)

    for i, w_0 in enumerate(W_0):
        print(f"Step {i} / {N} ")
        for j, w_1 in enumerate(W_1):
            print(f"Sub-Step {j} / {N} ")
            if w_0 + w_1 <= 1: # constraint
                w0_valid.append(w_0)
                w1_valid.append(w_1)
                w_2 = 1 - (w_0 + w_1)

                # Update model.W with new values
                new_W = torch.tensor([w_0, w_1, w_2], dtype=torch.float, device= device)
                # print the value of the tensor just 
                if j % 10 == 0:
                    print("New W : ", new_W)
                model.W.data = new_W
                assert torch.allclose(model.W.data, new_W, atol=1e-6), "W was not updated correctly!"

                # Evaluation
                eval_results = evaluate_imp_1(model, loader_val, disable_tqdm=False)
                R_1.append(eval_results["R1"])
                R_5.append(eval_results["R5"])
                R_mean.append(eval_results["R_mean"])

    print("Finished computing the recalls!")

    # Convert lists to arrays
    w0_valid = np.array(w0_valid)
    w1_valid = np.array(w1_valid)
    R_1 = np.array(R_1)
    R_5 = np.array(R_5)
    R_mean = np.array(R_mean)

    # Store results
    np.savez("heatmap.npz", x=w0_valid, y=w1_valid, r1=R_1, r2=R_5, r3=R_mean)
    print("Recalls saved successfully! ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--N", type=int, default=100)

    args = parser.parse_args()
    main(args)