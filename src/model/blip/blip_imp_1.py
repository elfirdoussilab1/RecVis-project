import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.blip.blip_cir import *
from src.model.blip.loss import *
import csv

class BLIP_Imp_1(nn.Module):
    def __init__(self, device):
        super().__init__()
        ckpt_path = 'outputs/cirr/blip-large/blip-l-coco/tv-False_loss-hnnce_lr-0.0001/base/ckpt_5.ckpt'
        loss = HardNegativeNCE(alpha = 1, beta= 0.5)
        blip = BLIPCir(loss)
        self.model = blip_cir(blip, ckpt_path).to(device)

        # Freezing weights of the model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Creating the 3 weights that will be used to compute the combined embedding
        self.W = nn.Parameter(torch.ones(3), requires_grad= True, device = device)
        self.device = device

    def forward(self, batch):
        device = self.device
        ref_img = batch["ref_img"].to(device)
        caption = batch["edit"].to(device)
        tar_img_feat = batch["tar_img_feat"].to(device)

        # Query embedding: q
        ref_img_embs = self.model.visual_encoder(ref_img) # q
        q = F.normalize(self.model.vision_proj(ref_img_embs), dim=-1) # vectors of size embed_dim=256
        print("Shape of q : ", q.shape)

        # Target image encoding: h
        tar_img_feat = tar_img_feat.to(device) 
        tar_img_feat = F.normalize(tar_img_feat, dim=-1) # h(v)

        # Text encoding
        text = self.model.tokenizer(
            caption,
            padding="longest",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device) # sequence of tokens of modification text "t" (tensor of integers)

        text_output = self.model.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
        t = text_output.last_hidden_state
        t = F.normalize(self.model.text_proj(t), dim=-1) # vectors of size embed_dim=256
        print("Shape of t : ", t.shape)

        # Produce the multimodal embedding: f(q,t)
        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        # Shift encoder
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.model.tokenizer.enc_token_id
        query_si_embs = self.model.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )
        query_si_feat = query_si_embs.last_hidden_state[:, 0, :]
        f = F.normalize(self.model.text_proj(query_si_feat), dim=-1) # f(q, t)
        print("Shape of f : ", f.shape)

        # Improvement: Combining the three embeddings!
        comb = self.W[0] * q + self.W[1] * t + self.W[2] * f

        # s=source, t=target, i=image, c=caption, w=weight
        loss = 0
        if self.model.si_ti_weight > 0:
            si_ti_loss = self.model.loss(comb, tar_img_feat, self.temp)
            loss += si_ti_loss * self.model.si_ti_weight

        # Caption retrieval loss, only for WebVid-CoVR and CC-CoIR
        if self.model.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]
            si_tc_loss = self.model.loss(comb, tar_txt_feat, self.model.temp)
            loss += si_tc_loss * self.model.si_tc_weight

        return loss
    
    def save(self, path = "outputs/imp-1/weights.csv", epoch = None):
        l = [f"epoch {epoch} "]
        w = self.W.detach().cpu().numpy()
        l = l + list(w)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(l)  # Write a single row
