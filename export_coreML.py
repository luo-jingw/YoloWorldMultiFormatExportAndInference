#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_coreML.py

Export YOLO-World model to three separate CoreML models:
1. text_encoder.mlpackage - Text encoding model
2. visual_encoder.mlpackage - Visual feature extraction model  
3. post_prediction.mlpackage - Post-processing detection head model
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from transformers import AutoTokenizer
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import MODELS

# —— Global Configuration —— #
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage
device = torch.device("cpu")

# Model configuration paths
CFG_PATH = "/home/ljw/python_proj/computer_vision/YOLO-World/configs/pretrain_v1/yolo_world_x_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
CKPT_PATH = "/home/ljw/python_proj/computer_vision/YOLO-World/pretrained_weights/yolo_world_x_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_cc3mlite_train_pretrained-8cf6b025.pth"

# CoreML output directory and filenames
COREML_DIR = "coreML_models"
TEXT_ENCODER_COREML = os.path.join(COREML_DIR, "text_encoder.mlpackage")
VISUAL_ENCODER_COREML = os.path.join(COREML_DIR, "visual_encoder.mlpackage")
POST_PREDICTION_COREML = os.path.join(COREML_DIR, "post_prediction.mlpackage")

# Test text example
TEST_TEXT = "bottle"

# Create output directory if it doesn't exist
os.makedirs(COREML_DIR, exist_ok=True)
print(f"CoreML models will be saved to: {COREML_DIR}")

# —— Step 0: Load and build YOLO-World PyTorch model —— #
print("Loading YOLO-World model...")
cfg = Config.fromfile(CFG_PATH)
cfg.work_dir = "."
cfg.load_from = CKPT_PATH

runner = Runner.from_cfg(cfg)
runner.call_hook("before_run")
runner.load_or_resume()

model = MODELS.build(cfg.model)
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Model loaded successfully!")

# —— Step 1: Define TextEncoderWrapper —— #
class TextEncoderWrapper(torch.nn.Module):
    """
    Extract CLIP text encoder from YOLO-World:
      Input:
        - input_ids      : Tensor[1, 77] (int64)
        - attention_mask : Tensor[1, 77] (int64)
      Output:
        - text_feats     : Tensor[1, 1, 512] (float32, L2 normalized)
    """
    def __init__(self, full_model):
        super().__init__()
        self.text_encoder = full_model.backbone.text_model.model.eval()

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs[0]                   # [1, 512]
        text_feat = text_feat.unsqueeze(1)       # [1, 1, 512]
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        return text_feat                         # [1, 1, 512]

# —— Step 2: Define VisualEncoderWrapper —— #
class VisualEncoderWrapper(torch.nn.Module):
    """
    Extract visual encoder from YOLO-World:
      Input:
        - image : Tensor[1, 3, 640, 640] (float32)
      Output:
        - feat_0 : Tensor[1, 320, 80, 80] (float32)
        - feat_1 : Tensor[1, 640, 40, 40] (float32)
        - feat_2 : Tensor[1, 640, 20, 20] (float32)
    """
    def __init__(self, full_model):
        super().__init__()
        self.image_model = full_model.backbone.image_model

    def forward(self, image):
        img_feats = self.image_model(image)  # List of 3 feature maps
        return img_feats[0], img_feats[1], img_feats[2]  # feat_0, feat_1, feat_2

# —— Step 3: Define PostPredictionWrapper —— #
class PostPredictionWrapper(torch.nn.Module):
    """
    Extract post-processing from YOLO-World:
      Input:
        - feat_0     : Tensor[1, 320, 80, 80] (float32)
        - feat_1     : Tensor[1, 640, 40, 40] (float32)
        - feat_2     : Tensor[1, 640, 20, 20] (float32)
        - text_feats : Tensor[1, 1, 512]      (float32)
        - txt_masks  : Tensor[1, 1]           (float32)
      Output:
        6 tensors (float32):
          - cls_score_0 : [1, 1, 80, 80]
          - cls_score_1 : [1, 1, 40, 40]
          - cls_score_2 : [1, 1, 20, 20]
          - bbox_pred_0 : [1, 4, 80, 80]
          - bbox_pred_1 : [1, 4, 40, 40]
          - bbox_pred_2 : [1, 4, 20, 20]
    """
    def __init__(self, full_model):
        super().__init__()
        self.neck = full_model.neck
        self.head_module = full_model.bbox_head.head_module

        # Hardcode pooling layers to ensure consistent dimensions during CoreML inference
        pools = self.neck.text_enhancer.image_pools
        pools[0] = torch.nn.MaxPool2d(kernel_size=27, stride=27, padding=1)
        pools[1] = torch.nn.MaxPool2d(kernel_size=13, stride=13, padding=1)
        pools[2] = torch.nn.MaxPool2d(kernel_size=7,  stride=7,  padding=1)

    def forward(self, feat_0, feat_1, feat_2, text_feats, txt_masks):
        img_feats = [feat_0, feat_1, feat_2]
        fused_feats = self.neck(img_feats, text_feats)
        cls_scores, bbox_preds = self.head_module(fused_feats, text_feats, txt_masks)
        return (
            cls_scores[0], cls_scores[1], cls_scores[2],
            bbox_preds[0], bbox_preds[1], bbox_preds[2]
        )

# —— Step 4: Convert TextEncoderWrapper to CoreML —— #
def convert_text_encoder_to_coreml():
    """Convert TextEncoderWrapper to CoreML (.mlpackage)."""
    print("\n→ Converting TextEncoder to CoreML...")

    wrapper_te = TextEncoderWrapper(model).eval()
    
    # Create example inputs for tracing
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoding = tokenizer(
        [TEST_TEXT],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    input_ids = encoding["input_ids"].to(device)          # Tensor[1,77], int64
    attention_mask = encoding["attention_mask"].to(device) # Tensor[1,77], int64

    # Convert to TorchScript first
    traced_model = torch.jit.trace(wrapper_te, (input_ids, attention_mask))

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids",      shape=input_ids.shape,      dtype=np.int64),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape, dtype=np.int64)
        ],
        outputs=[
            ct.TensorType(name="text_feats", dtype=np.float32)
        ],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    mlmodel.save(TEXT_ENCODER_COREML)
    print(f"[OK] TextEncoder CoreML model saved as '{TEXT_ENCODER_COREML}'")

# —— Step 5: Convert VisualEncoderWrapper to CoreML —— #
def convert_visual_encoder_to_coreml():
    """Convert VisualEncoderWrapper to CoreML (.mlpackage)."""
    print("\n→ Converting VisualEncoder to CoreML...")

    wrapper_ve = VisualEncoderWrapper(model).eval()

    # Create dummy input for tracing
    dummy_image = torch.randn((1, 3, 640, 640), dtype=torch.float32)

    # Convert to TorchScript first
    traced_model = torch.jit.trace(wrapper_ve, (dummy_image,))

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image", shape=dummy_image.shape, dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="feat_0", dtype=np.float32),
            ct.TensorType(name="feat_1", dtype=np.float32),
            ct.TensorType(name="feat_2", dtype=np.float32)
        ],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    mlmodel.save(VISUAL_ENCODER_COREML)
    print(f"[OK] VisualEncoder CoreML model saved as '{VISUAL_ENCODER_COREML}'")

# —— Step 6: Convert PostPredictionWrapper to CoreML —— #
def convert_post_prediction_to_coreml():
    """Convert PostPredictionWrapper to CoreML (.mlpackage)."""
    print("\n→ Converting PostPrediction to CoreML...")

    wrapper_pp = PostPredictionWrapper(model).eval()

    # Create dummy inputs for tracing
    dummy_feat_0 = torch.randn((1, 320, 80, 80), dtype=torch.float32)
    dummy_feat_1 = torch.randn((1, 640, 40, 40), dtype=torch.float32)
    dummy_feat_2 = torch.randn((1, 640, 20, 20), dtype=torch.float32)
    dummy_text_feats = torch.randn((1, 1, 512), dtype=torch.float32)
    dummy_txt_masks = torch.ones((1, 1), dtype=torch.float32)

    # Convert to TorchScript first
    traced_model = torch.jit.trace(wrapper_pp, (
        dummy_feat_0, dummy_feat_1, dummy_feat_2, 
        dummy_text_feats, dummy_txt_masks
    ))

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="feat_0",     shape=dummy_feat_0.shape,     dtype=np.float32),
            ct.TensorType(name="feat_1",     shape=dummy_feat_1.shape,     dtype=np.float32),
            ct.TensorType(name="feat_2",     shape=dummy_feat_2.shape,     dtype=np.float32),
            ct.TensorType(name="text_feats", shape=dummy_text_feats.shape, dtype=np.float32),
            ct.TensorType(name="txt_masks",  shape=dummy_txt_masks.shape,  dtype=np.float32)
        ],
        outputs=[
            ct.TensorType(name="cls_score_0", dtype=np.float32),
            ct.TensorType(name="cls_score_1", dtype=np.float32),
            ct.TensorType(name="cls_score_2", dtype=np.float32),
            ct.TensorType(name="bbox_pred_0", dtype=np.float32),
            ct.TensorType(name="bbox_pred_1", dtype=np.float32),
            ct.TensorType(name="bbox_pred_2", dtype=np.float32)
        ],
        source="pytorch",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    mlmodel.save(POST_PREDICTION_COREML)
    print(f"[OK] PostPrediction CoreML model saved as '{POST_PREDICTION_COREML}'")

# —— Main Program —— #
if __name__ == "__main__":
    print("="*80)
    print("YOLO-World CoreML Export Script")
    print("="*80)
    
    try:
        # Convert all three models
        convert_text_encoder_to_coreml()
        convert_visual_encoder_to_coreml()
        convert_post_prediction_to_coreml()
        
        print("\n" + "="*80)
        print("[SUCCESS] All models exported to CoreML successfully!")
        print(f"Models saved in: {COREML_DIR}")
        print("Files created:")
        print(f"  - {TEXT_ENCODER_COREML}")
        print(f"  - {VISUAL_ENCODER_COREML}")
        print(f"  - {POST_PREDICTION_COREML}")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] CoreML export failed: {str(e)}")
        import traceback
        traceback.print_exc()
