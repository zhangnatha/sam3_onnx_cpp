import torch
import torch.nn as nn
import os
import sys

# Ultralytics 导入
try:
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model, build_interactive_sam3
    from ultralytics.models.sam.modules.encoders import PromptEncoder
    import ultralytics.models.sam.sam3.vitdet as vitdet
    import ultralytics.models.sam.modules.utils as sam_utils
    from ultralytics.models.sam.sam3.geometry_encoders import Prompt
except ImportError:
    print("Error: Could not import necessary Ultralytics modules. Ensure ultralytics is installed and updated.")
    sys.exit(1)

# =========================================================
# Monkey-patches
# =========================================================

# 1. RoPE 补丁 (替换为实数算术以兼容 ONNX)
def apply_rotary_enc_patch(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_re, xq_im = xq_reshaped[..., 0], xq_reshaped[..., 1]
    
    xk_re, xk_im = None, None
    if xk.shape[-2] != 0:
        xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xk_re, xk_im = xk_reshaped[..., 0], xk_reshaped[..., 1]

    if torch.is_complex(freqs_cis):
        f_re, f_im = freqs_cis.real, freqs_cis.imag
    else:
        f_re, f_im = freqs_cis[..., 0], freqs_cis[..., 1]

    def broadcast_freqs(f, target_shape):
        ndim = len(target_shape)
        while f.ndim < ndim: f = f.unsqueeze(0)
        return f

    f_re = broadcast_freqs(f_re, xq_re.shape)
    f_im = broadcast_freqs(f_im, xq_im.shape)

    xq_out_re = xq_re * f_re - xq_im * f_im
    xq_out_im = xq_re * f_im + xq_im * f_re
    xq_out = torch.stack([xq_out_re, xq_out_im], dim=-1).reshape(*xq.shape)
    
    if xk_re is None: return xq_out.type_as(xq), xk
        
    xk_out_re = xk_re * f_re - xk_im * f_im
    xk_out_im = xk_re * f_im + xk_im * f_re
    xk_out = torch.stack([xk_out_re, xk_out_im], dim=-1).reshape(*xk.shape)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

vitdet.apply_rotary_enc = apply_rotary_enc_patch
sam_utils.apply_rotary_enc = apply_rotary_enc_patch
print("RoPE monkey-patched.")

# 2. PromptEncoder 补丁 (避免 ONNX 的布尔索引问题)
def _embed_points_patch(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
    points = points + 0.5
    if pad:
        padding_point = torch.zeros((points.shape[0], 1, 2), dtype=points.dtype, device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), dtype=labels.dtype, device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    
    point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
    
    mask_not_a_point = (labels == -1).unsqueeze(-1).to(point_embedding.dtype)
    mask_0 = (labels == 0).unsqueeze(-1).to(point_embedding.dtype)
    mask_1 = (labels == 1).unsqueeze(-1).to(point_embedding.dtype)
    mask_2 = (labels == 2).unsqueeze(-1).to(point_embedding.dtype)
    mask_3 = (labels == 3).unsqueeze(-1).to(point_embedding.dtype)
    
    point_embedding = point_embedding * (1 - mask_not_a_point) + mask_not_a_point * self.not_a_point_embed.weight.view(1, 1, -1)
    point_embedding = point_embedding + mask_0 * self.point_embeddings[0].weight.view(1, 1, -1)
    point_embedding = point_embedding + mask_1 * self.point_embeddings[1].weight.view(1, 1, -1)
    point_embedding = point_embedding + mask_2 * self.point_embeddings[2].weight.view(1, 1, -1)
    point_embedding = point_embedding + mask_3 * self.point_embeddings[3].weight.view(1, 1, -1)
    
    return point_embedding

PromptEncoder._embed_points = _embed_points_patch
print("PromptEncoder monkey-patched.")


# =========================================================
# 包装类 (Wrapper Classes)
# =========================================================

# --- 1. 交互式管道 (标准 SAM) ---
class SAM3Encoder(nn.Module):
    def __init__(self, int_model):
        super().__init__()
        self.model = int_model

    def forward(self, img):
        backbone_out = self.model.image_encoder.forward_image_sam2(img)
        fpn = backbone_out["backbone_fpn"]
        vpe = backbone_out["vision_pos_enc"]
        
        # 显式投影 (Required by SAM2 architecture)
        feat0 = self.model.sam_mask_decoder.conv_s0(fpn[0])
        feat1 = self.model.sam_mask_decoder.conv_s1(fpn[1])
        feat2 = fpn[2]
        
        new_backbone_out = {
            "backbone_fpn": [feat0, feat1, feat2],
            "vision_pos_enc": vpe
        }
            
        _, vision_feats, vision_pos_embeds, feat_sizes = self.model._prepare_backbone_features(new_backbone_out)
        
        pix_feat = self.model._prepare_memory_conditioned_features(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict={},
            num_frames=1,
            track_in_reverse=False
        )
        
        hr0 = vision_feats[0].permute(1, 2, 0).view(vision_feats[0].size(1), vision_feats[0].size(2), *feat_sizes[0])
        hr1 = vision_feats[1].permute(1, 2, 0).view(vision_feats[1].size(1), vision_feats[1].size(2), *feat_sizes[1])
        
        return pix_feat, hr0, hr1

class SAM3Decoder(nn.Module):
    def __init__(self, int_model):
        super().__init__()
        self.model = int_model

    def forward(self, pix_feat, high_res_0, high_res_1, point_coords, point_labels):
        point_inputs = {"point_coords": point_coords, "point_labels": point_labels}
        results = self.model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=None, 
            high_res_features=[high_res_0, high_res_1],
            multimask_output=True
        )
        # results[0]: low_res_multimasks [B, 3, 288, 288] (or similar)
        # results[1]: high_res_multimasks [B, 3, 1008, 1008]
        # results[2]: ious [B, 3]
        return results[0], results[2]

# --- 2. Grounding 管道 (文本分割) ---
class SAM3LanguageEncoder(nn.Module):
    def __init__(self, sem_model):
        super().__init__()
        self.language_backbone = sem_model.backbone.language_backbone
    def forward(self, tokens):
        text_attention_mask = (tokens != 0).bool()
        inputs_embeds = self.language_backbone.encoder.token_embedding(tokens)
        _, text_memory = self.language_backbone.encoder(tokens)
        text_attention_mask = text_attention_mask.ne(1)
        text_memory = text_memory.transpose(0, 1)
        text_memory_resized = self.language_backbone.resizer(text_memory)
        return text_attention_mask, text_memory_resized, inputs_embeds.transpose(0, 1)

class SAM3GroundingEncoder(nn.Module):
    def __init__(self, sem_model):
        super().__init__()
        self.model = sem_model.backbone
    def forward(self, img):
        out = self.model.forward_image(img)
        return *out["backbone_fpn"], *out["vision_pos_enc"]

class SAM3GroundingDecoder(nn.Module):
    def __init__(self, sem_model):
        super().__init__()
        self.model = sem_model
        
    def forward(self, feat0, feat1, feat2, vpe0, vpe1, vpe2, lang_mask, lang_feat, box_coords, box_labels, box_masks):
        backbone_out = {
            "backbone_fpn": [feat0, feat1, feat2],
            "vision_pos_enc": [vpe0, vpe1, vpe2],
            "language_mask": lang_mask,
            "language_features": lang_feat
        }
        
        batch_prompts = box_coords.size(0)
        
        geometric_prompt = Prompt(
            box_embeddings=box_coords,
            box_labels=box_labels,
            box_mask=box_masks
        )
        
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            text_ids=torch.arange(batch_prompts, device=box_coords.device, dtype=torch.long),
            geometric_prompt=geometric_prompt
        )
        
        return (
            outputs["pred_boxes_xyxy"], 
            outputs["pred_logits"], 
            outputs["pred_masks"],
            outputs["presence_logit_dec"]
        )

# =========================================================
# 导出函数 (Export Functions)
# =========================================================

# --- 交互式管道导出 ---
def export_interactive_encoder(int_model):
    print("Exporting Interactive Encoder (1008x1008)...")
    encoder = SAM3Encoder(int_model).eval()
    dummy_img = torch.randn(1, 3, 1008, 1008)

    torch.onnx.export(
        encoder, (dummy_img,), os.path.join(os.path.dirname(__file__), "sam3_encoder.onnx"),
        input_names=["images"],output_names=["pix_feat", "high_res_0", "high_res_1"],
        opset_version=14, dynamo=False
    )
    print("Interactive Encoder exported.")

def export_interactive_decoder(int_model):
    print("Exporting Interactive Decoder...")
    encoder = SAM3Encoder(int_model).eval()
    dummy_img = torch.randn(1, 3, 1008, 1008)
        
    with torch.no_grad():
        pix_feat, hr0, hr1 = encoder(dummy_img)

    dummy_point_coords = torch.randn(1, 2, 2)
    dummy_point_labels = torch.randint(0, 2, (1, 2)).int()

    decoder = SAM3Decoder(int_model).eval()
    torch.onnx.export(
        decoder,
        (pix_feat, hr0, hr1, dummy_point_coords, dummy_point_labels),
        os.path.join(os.path.dirname(__file__), "sam3_decoder.onnx"),
        input_names=["pix_feat", "high_res_0", "high_res_1", "point_coords", "point_labels"],
        output_names=["masks", "ious"],
        opset_version=14, dynamo=False,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
            "masks": {2: "height", 3: "width"},
            "ious": {1: "num_masks"}
        }
    )
    print("Interactive Decoder exported.")

# --- Grounding 管道导出 ---
def export_language_encoder(sem_model):
    print("Exporting Language Encoder...")
    lang_encoder = SAM3LanguageEncoder(sem_model).eval()
    dummy_tokens = torch.randint(0, 49408, (1, 32)).long()
    torch.onnx.export(
        lang_encoder, (dummy_tokens,), os.path.join(os.path.dirname(__file__), "sam3_language_encoder.onnx"),
        input_names=["tokens"], output_names=["text_attention_mask", "text_memory", "text_embeds"],
        opset_version=14, dynamo=False,
        dynamic_axes={"tokens": {0: "batch_size"}, "text_attention_mask": {0: "batch_size"}, 
                      "text_memory": {1: "batch_size"}, "text_embeds": {1: "batch_size"}}
    )
    print("Language Encoder exported.")

def export_grounding_encoder(sem_model):
    print("Exporting Grounding Encoder (1008x1008)...")
    gr_encoder = SAM3GroundingEncoder(sem_model).cpu().eval()
    dummy_img = torch.randn(1, 3, 1008, 1008).cpu()
    torch.onnx.export(
        gr_encoder, (dummy_img,), os.path.join(os.path.dirname(__file__), "sam3_grounding_encoder.onnx"),
        input_names=["images"], output_names=["feat0", "feat1", "feat2", "vpe0", "vpe1", "vpe2"],
        opset_version=14, dynamo=False
    )
    print("Grounding Encoder exported.")

def export_grounding_decoder(sem_model):
    print("Exporting Grounding Decoder...")
    gr_encoder = SAM3GroundingEncoder(sem_model).cpu().eval()
    dummy_img = torch.randn(1, 3, 1008, 1008).cpu()
    with torch.no_grad():
        feat0, feat1, feat2, vpe0, vpe1, vpe2 = gr_encoder(dummy_img)
        lang_mask = torch.zeros(1, 32, dtype=torch.bool)
        lang_feat = torch.randn(32, 1, 256)
        box_coords = torch.randn(1, 1, 4)
        box_labels = torch.ones(1, 1).long()
        box_masks = torch.zeros(1, 1).bool()

    decoder = SAM3GroundingDecoder(sem_model).eval()
    torch.onnx.export(
        decoder, 
        (feat0, feat1, feat2, vpe0, vpe1, vpe2, lang_mask, lang_feat, box_coords, box_labels, box_masks), 
        os.path.join(os.path.dirname(__file__), "sam3_grounding_decoder.onnx"), 
        input_names=["feat0", "feat1", "feat2", "vpe0", "vpe1", "vpe2", "lang_mask", "lang_feat", "box_coords", "box_labels", "box_masks"], 
        output_names=["boxes", "scores", "masks", "presence"], 
        opset_version=14, dynamo=False, 
        dynamic_axes={
            "feat0": {0: "batch", 2: "h0", 3: "w0"},
            "lang_mask": {0: "num_prompts"}, 
            "lang_feat": {1: "num_prompts"},
            "box_coords": {0: "num_prompts", 1: "batch"},
            "box_labels": {0: "num_prompts", 1: "batch"},
            "box_masks": {0: "batch", 1: "num_prompts"},
            "boxes": {0: "num_prompts", 1: "num_queries"},
            "scores": {0: "num_prompts", 1: "num_queries"},
            "masks": {0: "num_prompts", 1: "num_queries"},
            "presence": {0: "num_prompts"}
        }
    )
    print("Grounding Decoder exported.")


def main():
    stage = os.environ.get("EXPORT_STAGE", "all")
    model_path = os.path.join(os.path.dirname(__file__), "sam3.pt")
    
    print(f"Loading models from {model_path}...")
    
    if stage == "interactive" or stage == "all":
        int_model = build_interactive_sam3(model_path)
        int_model.eval()
        export_interactive_encoder(int_model)
        export_interactive_decoder(int_model)
        
    if stage == "grounding" or stage == "all":
        sem_model = build_sam3_image_model(model_path)
        sem_model.eval()
        export_language_encoder(sem_model)
        export_grounding_encoder(sem_model)
        export_grounding_decoder(sem_model)
        
    print("All exports completed.")

if __name__ == "__main__":
    main()
