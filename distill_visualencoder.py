import torch
import timm
import torch.nn as nn
import os
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


# 1. 加载 MobileViT
student = timm.create_model(
    'mobilevit_s',
    pretrained=True,
    features_only=True,
    out_indices=(1, 2, 3)
)
student.eval()

# 2. 在拿到 feats 后，先下采样，再用 1×1 conv 映射通道
down = nn.MaxPool2d(2, 2)
heads = nn.ModuleList([
    nn.Conv2d(in_c, out_c, kernel_size=1)
    for in_c, out_c in zip(student.feature_info.channels(), [320, 640, 640])
])
# 初始化权重（可选）
for head in heads:
    nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='relu')
    if head.bias is not None:
        nn.init.zeros_(head.bias)

dummy_img = torch.randn(1,3,640,640)
with torch.no_grad():
    feats = student(dummy_img)      # 原始 feats: [1]→64×160×160, [2]→96×80×80, [3]→128×40×40

# 下采样到目标尺寸
f0 = down(feats[0])  # 160→80
f1 = down(feats[1])  #  80→40
f2 = down(feats[2])  #  40→20

# 1×1 conv 映射到目标通道
dummy_feat0 = heads[0](f0)  # [1,320,80,80]
dummy_feat1 = heads[1](f1)  # [1,640,40,40]
dummy_feat2 = heads[2](f2)  # [1,640,20,20]

print(dummy_feat0.shape)
print(dummy_feat1.shape)
print(dummy_feat2.shape)


# inference time test
import time
time_start = time.time()
dummy_img = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    feats = student(dummy_img)  # 原始 feats: [1]→64×160×160, [2]→96×80×80, [3]→128×40×40
time_end = time.time()
print(f"Inference time: {time_end - time_start:.4f} seconds for one forward pass.")