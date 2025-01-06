# In this file, we will evaluate BLIP-Base
import torch
from src.test.blip.utils import eval_recall
from src.data.cirr import *
from src.model.blip.blip_cir import *
from src.model.blip.loss import *
import csv

# Paths
annotation = {"train": "annotation/cirr/cap.rc2.train.json", 
              "val": "annotation/cirr/cap.rc2.val.json"}
img_dirs = {"train": "datasets/CIRR/images/train", 
            "val": "datasets/CIRR/images/dev"}
emb_dirs = {"train": "datasets/CIRR/blip-embs-large/train", 
            "val": "datasets/CIRR/blip-embs-large/dev"}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device : ", device)

@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()

    print("Computing features for evaluation...")

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

        ref_img_embs = model.visual_encoder(ref_img)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        text = model.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
        query_embs = model.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_feat = query_embs.last_hidden_state[:, 0, :]
        query_feat = F.normalize(model.text_proj(query_feat), dim=-1)
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

    eval_result = eval_recall(sim_q2t)
    return eval_result

def main(args):
    # Dataset
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Dataset
    data = CIRRDataModule(batch_size=batch_size, num_workers=num_workers, 
                            annotation=annotation, img_dirs= img_dirs, emb_dirs=emb_dirs)
    loader_val = data.val_dataloader()

    # Model: BLIP-Base last checkpoint
    ckpt_path = 'outputs/cirr/blip-base/blip-base-coco/tv-False_loss-hnnce_lr-1e-05/base/ckpt_4.ckpt'
    loss = HardNegativeNCE(alpha = 1, beta= 0.5)
    blip = BLIPCir(loss, vit = "base", vit_ckpt_layer = 4).to(device)
    model = blip_cir(blip, ckpt_path)

    # Evaluation
    eval_result = evaluate(model, loader_val)

    # Store results in a csv file
    print(eval_result)
    # Write dictionary to CSV
    with open("eval_blip_base.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data.keys())  # Write header
        writer.writerow(data.values())  # Write values

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)
