import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.blip.blip_cir import *
from src.model.blip.loss import *
import csv

# This code resembles to Blip_Imp_1, but instead of doing the mean of t and q, we do the median (or some other aggregation)

class BLIP_Imp_2(nn.Module):
    def __init__(self, device):
        super().__init__()
        ckpt_path = 'outputs/cirr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-0.0001/base/ckpt_5.ckpt'
        loss = HardNegativeNCE(alpha = 1, beta= 0.5)
        blip_large = BLIPCir(loss)
        self.blip = blip_cir(blip_large, ckpt_path).to(device)

        # Freezing weights of the model
        for param in self.blip.parameters():
            param.requires_grad = False
        
        # Creating the 3 weights that will be used to compute the combined embedding
        self.W = nn.Parameter(torch.tensor([0, 0, 1], dtype=torch.float, device= device), requires_grad= True)
        self.device = device

    def compute_comb(self, ref_img, caption):
        # Computes the mixed embedding
        # Query embedding: q
        ref_img_embs = self.blip.visual_encoder(ref_img) # q
        q = F.normalize(self.blip.vision_proj(ref_img_embs), dim=-1) # shape (B, 577, 256)
        q = q.median(dim = 1).values # shape (B, 256)

        # Text encoding
        text = self.blip.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(self.device) # sequence of tokens of modification text "t" (tensor of integers)

        text_output = self.blip.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
        t = text_output.last_hidden_state
        t = F.normalize(self.blip.text_proj(t), dim=-1) # shape (B, 31, 256)
        t = t.median(dim = 1).values

        # Produce the multimodal embedding: f(q,t)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(self.device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.blip.tokenizer.enc_token_id
        query_si_embs = self.blip.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_si_feat = query_si_embs.last_hidden_state[:, 0, :]
        f = F.normalize(self.blip.text_proj(query_si_feat), dim=-1) # f(q, t) shape (B, 256)

        # Improvement: Combining the three embeddings!
        # stack q, t and f
        emb_stack = torch.stack([q, t, f], dim=0)  # Shape: (3, B, 256)
        comb = torch.einsum('i,ibj->bj', self.W, emb_stack) # shape (B, 256)
        #comb = self.W[0] * q + self.W[1] * t + self.W[2] * f
        return comb

    def forward(self, batch):
        device = self.device
        ref_img = batch["ref_img"].to(device) # shape (B, 3, 384, 384)
        caption = batch["edit"] # B lists of strings, each of different length
        tar_img_feat = batch["tar_img_feat"].to(device) # shape (B, 256)

        # Target image encoding: h
        tar_img_feat = tar_img_feat.to(device) 
        tar_img_feat = F.normalize(tar_img_feat, dim=-1) # h(v) shape (B, 256)

        # Combination of embeddings
        comb = self.compute_comb(ref_img, caption)
        
        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.blip.si_ti_weight > 0:
            si_ti_loss = self.blip.loss(comb, tar_img_feat, self.blip.temp)
            loss += si_ti_loss * self.blip.si_ti_weight

        # Caption retrieval loss, only for WebVid-CoVR and CC-CoIR
        if self.blip.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]
            si_tc_loss = self.blip.loss(comb, tar_txt_feat, self.blip.temp)
            loss += si_tc_loss * self.blip.si_tc_weight

        return loss
    
    def save(self, path = "outputs/imp-2/weights.csv", epoch = None):
        l = [f"epoch {epoch} "]
        w = self.W.detach().cpu().numpy()
        l = l + list(w)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(l)  # Write a single row
