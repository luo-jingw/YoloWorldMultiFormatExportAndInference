#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tflite_inference.py

This script loads TFLite models and performs complete YOLO-World inference pipeline.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import CLIPTokenizer
import torch
from torchvision.ops import nms
import cv2
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from typing import Tuple, List

# ------------------------------------------------------------------------------
# Timing decorator
# ------------------------------------------------------------------------------
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        timer.times[func.__name__].append(end - start)
        return result
    return wrapper

timer.times = defaultdict(list)

def print_time_analysis():
    print("\n=== Time Analysis ===")
    for module, times in timer.times.items():
        avg_time = sum(times) / len(times)
        print(f"{module:20s}: {avg_time:.4f}s (avg of {len(times)} runs)")

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
MIN_THRESH = 0.05
NORM_THRESH = 0.25
NMS_IOU_THRESH = 0.5
TARGET_SIZE = (640, 640)

# ------------------------------------------------------------------------------
# Load TFLite interpreters for text, visual, and post-processing models
# ------------------------------------------------------------------------------
MODEL_DIR = "tflite_models_with_p"
interpreter_text   = tf.lite.Interpreter(os.path.join(MODEL_DIR, "text_encoder.tflite"))
interpreter_visual = tf.lite.Interpreter(os.path.join(MODEL_DIR, "visual_encoder.tflite"))
interpreter_post   = tf.lite.Interpreter(os.path.join(MODEL_DIR, "post_prediction_with_decode.tflite"))

interpreter_text.allocate_tensors()
interpreter_visual.allocate_tensors()
interpreter_post.allocate_tensors()

# ------------------------------------------------------------------------------
# Initialize CLIP tokenizer for text preprocessing
# ------------------------------------------------------------------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ------------------------------------------------------------------------------
# Preprocessing functions
# ------------------------------------------------------------------------------
@timer
def preprocess_image(image_path, target_size=(640, 640)):
    """
    Load and preprocess real image.
    Args:
        image_path: path to the image file
        target_size: target size (width, height)
    Returns:
        tuple of (original_image_rgb, preprocessed_tensor)
    """
    print(f"Loading image: {image_path}")
    
    # Load image using OpenCV and convert BGR to RGB
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Original image size: {image_rgb.shape[:2]}")
    
    # Convert to PIL and resize
    pil_image = Image.fromarray(image_rgb)
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    print(f"Resized image to: {pil_image.size}")
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format
    image_array = image_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    
    # Add batch dimension
    image_tensor = np.expand_dims(image_array, axis=0)  # (1, C, H, W)
    print(f"Final image tensor shape: {image_tensor.shape}")
    
    return image_rgb, image_tensor

@timer
def preprocess_text(text):
    """
    Tokenize and preprocess real text using CLIP tokenizer.
    Args:
        text: input text string
    Returns:
        tuple of (input_ids, attention_mask) both of shape (1, 77)
    """
    print(f"Processing text: '{text}'")
    
    # Tokenize text
    tokens = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="np"
    )
    
    input_ids = tokens["input_ids"].astype(np.int64)
    attention_mask = tokens["attention_mask"].astype(np.int64)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    return input_ids, attention_mask

# ------------------------------------------------------------------------------
# Post-processing functions
# ------------------------------------------------------------------------------
@timer
def post_process_decoded_outputs(outputs: List[np.ndarray],
                                image_rgb: np.ndarray,
                                min_thresh: float = MIN_THRESH,
                                nms_thresh: float = NMS_IOU_THRESH
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """处理包含解码的TFLite模型输出"""
    
    print(f"[INFO] Processing {len(outputs)} decoded outputs from TFLite model")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
        if output.size > 0:
            print(f"    Min/Max values: {output.min():.3f}/{output.max():.3f}")
    
    # 检查是否是2个输出（解码后的格式：boxes, scores）
    if len(outputs) == 2:
        print("[INFO] Using decoded outputs from model")
        
        boxes_output = outputs[0]    # [total_anchors, 4]
        scores_output = outputs[1]   # [total_anchors]
        
        print(f"[INFO] Decoded boxes shape: {boxes_output.shape}")
        print(f"[INFO] Decoded scores shape: {scores_output.shape}")
        
        # 过滤低置信度检测
        valid_mask = scores_output > min_thresh
        boxes = boxes_output[valid_mask]
        scores = scores_output[valid_mask]
        
        print(f"[INFO] After filtering: {len(boxes)} boxes remaining")
        
        if len(boxes) == 0:
            return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
        
        # 转换为torch tensor
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()
        
        # 应用NMS（在640x640坐标系下）
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=nms_thresh)
        boxes_tensor = boxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        
        print(f"[INFO] After NMS: kept {len(boxes_tensor)} boxes")
        
        # 缩放到原始图像尺寸
        if len(boxes_tensor) > 0:
            H_orig, W_orig = image_rgb.shape[:2]
            scale_x = W_orig / TARGET_SIZE[0]
            scale_y = H_orig / TARGET_SIZE[1]
            print(f"[INFO] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
            
            boxes_tensor[:, [0, 2]] *= scale_x  # x1, x2
            boxes_tensor[:, [1, 3]] *= scale_y  # y1, y2
        
        # 创建标签（单类别检测，所有标签都是0）
        labels_tensor = torch.zeros(len(boxes_tensor), dtype=torch.long)
        
        return boxes_tensor, scores_tensor, labels_tensor
    
    else:
        print(f"[ERROR] Expected 2 outputs for decoded model, got {len(outputs)}")
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

# ------------------------------------------------------------------------------
# Visualization functions
# ------------------------------------------------------------------------------
@timer
def visualize(image_rgb: np.ndarray,
             boxes: torch.Tensor,
             scores: torch.Tensor,
             labels: torch.Tensor,
             class_names: List[str],
             save_path: str = "detection_result.jpg") -> None:
    """Draw detection results on the original RGB image and save/display them."""
    img = image_rgb.copy()
    H, W = img.shape[:2]
    font_scale = max(0.5, min(W, H) / 800 * 1.0)
    thickness = max(1, int(min(W, H) / 400))
    
    print(f"[DEBUG] Image dimensions: {W}x{H}")
    print(f"[DEBUG] Processing {len(boxes)} boxes for visualization...")

    valid_boxes = 0
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.tolist()
        print(f"[DEBUG] Box {i+1}: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}], score: {score:.3f}")
        
        # 检查bbox是否在图像范围内
        if x1 >= W or y1 >= H or x2 <= 0 or y2 <= 0:
            print(f"[WARN] Box {i+1} is completely outside image bounds, skipping...")
            continue
            
        # 检查bbox是否有效
        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Box {i+1} has invalid dimensions, skipping...")
            continue
        
        # 裁剪bbox到图像范围内
        x1_clipped = max(0, min(int(x1), W-1))
        y1_clipped = max(0, min(int(y1), H-1))
        x2_clipped = max(0, min(int(x2), W-1))
        y2_clipped = max(0, min(int(y2), H-1))
        
        # 如果裁剪后的bbox太小，跳过
        if (x2_clipped - x1_clipped) < 2 or (y2_clipped - y1_clipped) < 2:
            print(f"[WARN] Box {i+1} is too small after clipping, skipping...")
            continue
        
        print(f"[DEBUG] Clipped box {i+1}: [{x1_clipped}, {y1_clipped}, {x2_clipped}, {y2_clipped}]")
        
        label_text = f"{class_names[cls_id]} {score:.2f}"
        
        # 绘制边界框
        cv2.rectangle(img, (x1_clipped, y1_clipped), (x2_clipped, y2_clipped), (0, 255, 0), thickness)
        
        # 绘制标签
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 确保文本在图像范围内
        text_y = max(text_h + baseline + 4, y1_clipped)
        text_x = max(0, min(x1_clipped, W - text_w - 4))
        
        # 绘制文本背景
        cv2.rectangle(
            img,
            (text_x, text_y - text_h - baseline - 4),
            (text_x + text_w + 4, text_y),
            (0, 255, 0),
            -1
        )
        
        # 绘制文本
        cv2.putText(
            img,
            label_text,
            (text_x + 2, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )
        
        valid_boxes += 1
    
    print(f"[INFO] Drew {valid_boxes} valid boxes out of {len(boxes)} total boxes")

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Detection result saved as {save_path}")

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Detection Result - {valid_boxes} valid boxes")
    plt.show()

# ------------------------------------------------------------------------------
# Main detection pipeline
# ------------------------------------------------------------------------------
@timer
def detect_objects(image_path: str, text: str, save_path: str = "result_tflite.jpg"):
    """Main detection pipeline with integrated decoding"""
    print(f"\n{'='*80}")
    print(f"TFLite Detection Pipeline (with integrated decoding)")
    print(f"Image: {image_path}")  
    print(f"Text: '{text}'")
    print(f"{'='*80}")
    
    try:
        # 1. Preprocess image and text
        image_rgb, image_tensor = preprocess_image(image_path)
        input_ids, attention_mask = preprocess_text(text)
        
        # 2. Text encoder inference
        print("\n[Step 1] Running Text Encoder...")
        start_text = time.perf_counter()
        input_details = interpreter_text.get_input_details()
        for detail in input_details:
            name = detail['name']
            if 'input_ids' in name:
                interpreter_text.set_tensor(detail['index'], input_ids)
            elif 'attention_mask' in name:
                interpreter_text.set_tensor(detail['index'], attention_mask)
        
        interpreter_text.invoke()
        text_outputs = []
        output_details = interpreter_text.get_output_details()
        for detail in output_details:
            text_outputs.append(interpreter_text.get_tensor(detail['index']))
        
        text_feats = text_outputs[0]  # Shape: (1, 1, 512)
        timer.times['text_encoder'].append(time.perf_counter() - start_text)
        print(f"Text features shape: {text_feats.shape}")
        
        # 3. Visual encoder inference
        print("\n[Step 2] Running Visual Encoder...")
        start_visual = time.perf_counter()
        input_details = interpreter_visual.get_input_details()
        for detail in input_details:
            if 'image' in detail['name']:
                interpreter_visual.set_tensor(detail['index'], image_tensor)
        
        interpreter_visual.invoke()
        visual_outputs = []
        output_details = interpreter_visual.get_output_details()
        for detail in output_details:
            visual_outputs.append(interpreter_visual.get_tensor(detail['index']))
        
        feat_2 = visual_outputs[0]  # Shape: (1, 640, 20, 20)
        feat_1 = visual_outputs[1]  # Shape: (1, 640, 40, 40)
        feat_0 = visual_outputs[2]  # Shape: (1, 320, 80, 80)
        timer.times['visual_encoder'].append(time.perf_counter() - start_visual)
        
        print(f"Visual features shapes:")
        print(f"  feat_0: {feat_0.shape}")
        print(f"  feat_1: {feat_1.shape}")
        print(f"  feat_2: {feat_2.shape}")
        
        # 4. Post-processing inference with integrated decoding
        print("\n[Step 3] Running Post-processing with integrated decoding...")
        start_post = time.perf_counter()
        txt_masks = np.ones((1, 1), dtype=np.float32)
        
        input_details = interpreter_post.get_input_details()
        input_mapping = {
            'feat_0': feat_0,
            'feat_1': feat_1,
            'feat_2': feat_2,
            'text_feats': text_feats,
            'txt_masks': txt_masks
        }
        
        for detail in input_details:
            name = detail['name']
            key = name.replace('serving_default_', '').replace(':0', '')
            if key in input_mapping:
                interpreter_post.set_tensor(detail['index'], input_mapping[key])
        
        interpreter_post.invoke()
        post_outputs = []
        output_details = interpreter_post.get_output_details()
        for detail in output_details:
            post_outputs.append(interpreter_post.get_tensor(detail['index']))
        
        timer.times['post_network'].append(time.perf_counter() - start_post)
        
        # 5. 处理解码后的输出
        print("\n[Step 4] Processing decoded outputs...")
        boxes, scores, labels = post_process_decoded_outputs(
            post_outputs,
            image_rgb,
            min_thresh=MIN_THRESH,
            nms_thresh=NMS_IOU_THRESH
        )
        
        # 6. Visualize results
        print("\n[Step 5] Visualizing results...")
        visualize(
            image_rgb   = image_rgb,
            boxes       = boxes,
            scores      = scores,
            labels      = labels,
            class_names = [text],
            save_path   = save_path
        )
        
        print(f"\n✓ TFLite pipeline with integrated decoding executed successfully!")
        return boxes, scores, labels
        
    except Exception as e:
        print(f"\n✗ TFLite pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run detection pipeline
    image_path = "./sample_images/desk.png"
    text_prompt = "silver laptop"
    save_path = "result_tflite.png"
    
    if os.path.exists(image_path):
        start_total = time.perf_counter()
        boxes, scores, labels = detect_objects(image_path, text_prompt, save_path)
        total_time = time.perf_counter() - start_total
        timer.times['total_pipeline'].append(total_time)
        
        if boxes is not None:
            print(f"\nDetected {len(boxes)} objects:")
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                print(f"Object {i+1} - position: {box.tolist()}, confidence: {score:.3f}")
        
        print_time_analysis()
    else:
        print(f"\nWarning: Image file not found: {image_path}")
        print("Please ensure the image file exists at the specified path.")
    
    print("\n" + "="*60)
    print("TFLite inference with integrated decoding completed successfully!")
    print("="*60)