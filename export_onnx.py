#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from mmdet.apis import init_detector
from mmengine.config import Config

# Disable GPU to avoid memory overflow during conversion/loading
os.environ["CUDA_VISIBLE_DEVICES"] = ""

config_file = "configs/pretrain_v1/yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
checkpoint_file = "pretrained_weights/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth"

cfg = Config.fromfile(config_file)
model = init_detector(cfg, checkpoint_file, device="cpu")
model.eval()

text_enhancer = model.neck.text_enhancer
dummy_img = torch.randn(1, 3, 640, 640)
dummy_img_feats = model.backbone.image_model(dummy_img)
dummy_text = "label"
projected_shapes = []


class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.backbone.text_model.model.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs[0]               # [1, 512]
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        text_feat = text_feat.unsqueeze(0)   # [1, 1, 512]
        return text_feat                     # return 3D tensor
    
wrapper_te = TextEncoderWrapper(model).eval()

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
encoding = tokenizer(
    [dummy_text],            # Single text only
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)
input_ids = encoding["input_ids"]             # [1, 77]
attention_mask = encoding["attention_mask"]   # [1, 77]

for i, (x, proj) in enumerate(zip(dummy_img_feats, text_enhancer.projections)):
    with torch.no_grad():
        out = proj(x)
        projected_shapes.append(out.shape[-2:])
for i, (H, W) in enumerate(projected_shapes):
    kernel_h = (H + 1) // 3
    stride_h = kernel_h
    pad_h = max((kernel_h * 3 - H + 1) // 2, 0)
    kernel_w = (W + 1) // 3
    stride_w = kernel_w
    pad_w = max((kernel_w * 3 - W + 1) // 2, 0)
    text_enhancer.image_pools[i] = nn.MaxPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=(pad_h, pad_w))

class VisualBackbone(nn.Module):
    def __init__(self, image_model):
        super().__init__()
        self.image_model = image_model

    def forward(self, image):
        feat_list = self.image_model(image)
        return feat_list[0], feat_list[1], feat_list[2]

backbone = VisualBackbone(model.backbone.image_model).eval()
dummy_image = torch.randn(1, 3, 640, 640)

class NeckHead(nn.Module):
    def __init__(self, neck, head_module):
        super().__init__()
        self.neck = neck
        self.head_module = head_module

    def forward(self, feat_0, feat_1, feat_2, text_feats, txt_masks):
        fused_feats = self.neck([feat_0, feat_1, feat_2], text_feats)
        cls_scores, bbox_preds = self.head_module(fused_feats, text_feats, txt_masks)
        return cls_scores[0], cls_scores[1], cls_scores[2], bbox_preds[0], bbox_preds[1], bbox_preds[2]

post_prediction = NeckHead(model.neck, model.bbox_head.head_module).eval()
dummy_feat0 = torch.randn(1, 320, 80, 80)
dummy_feat1 = torch.randn(1, 640, 40, 40)
dummy_feat2 = torch.randn(1, 640, 20, 20)
dummy_text_feats = torch.randn(1, 1, 512)
dummy_txt_masks = torch.ones(1, 1)

with torch.no_grad():
    torch.onnx.export(
        wrapper_te,
        (input_ids, attention_mask),
        "/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/text_encoder.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["text_feats"],
        opset_version=12,
        do_constant_folding=True
    )
    torch.onnx.export(
    backbone,
    dummy_image,
    "/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/visual_encoder.onnx",
    input_names=["image"],
    output_names=["feat_0", "feat_1", "feat_2"],
    opset_version=12,
    do_constant_folding=True,
    )

    torch.onnx.export(
        post_prediction,
        (dummy_feat0, dummy_feat1, dummy_feat2, dummy_text_feats, dummy_txt_masks),
        "/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/post_prediction.onnx",
        input_names=["feat_0", "feat_1", "feat_2", "text_feats", "txt_masks"],
        output_names=["cls_score_0", "cls_score_1", "cls_score_2", "bbox_pred_0", "bbox_pred_1", "bbox_pred_2"],
        opset_version=12,
        do_constant_folding=True,
    )