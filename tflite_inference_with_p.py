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
def post_process_with_decode(outputs: List[np.ndarray],
                           image_rgb: np.ndarray,
                           min_thresh: float = MIN_THRESH,
                           norm_thresh: float = NORM_THRESH,
                           nms_thresh: float = NMS_IOU_THRESH
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-processing for TFLite model with integrated decoding"""
    
    # 包含解码的模型输出格式：[boxes, scores, labels]
    # 或者类似的已解码格式，需要根据实际模型输出调整
    
    # 假设模型输出是已经解码的boxes和scores
    if len(outputs) == 3:
        # 格式：[boxes, scores, labels] 或 [boxes, scores, valid_detections]
        boxes_output = outputs[0]  # shape: (batch, max_detections, 4)
        scores_output = outputs[1]  # shape: (batch, max_detections)
        
        # 如果有第三个输出（可能是有效检测数量或标签）
        if outputs[2].shape[-1] == 1:
            # 第三个输出是有效检测数量
            valid_detections = int(outputs[2][0, 0])
            labels_output = np.zeros((1, scores_output.shape[1]), dtype=np.int64)
        else:
            # 第三个输出是标签
            labels_output = outputs[2].astype(np.int64)
            valid_detections = scores_output.shape[1]
    else:
        # 如果输出格式不同，回退到原始后处理
        return post_process_reference(outputs, image_rgb, min_thresh, norm_thresh, nms_thresh)
    
    # 移除batch维度
    boxes = boxes_output[0]      # shape: (max_detections, 4)
    scores = scores_output[0]    # shape: (max_detections,)
    labels = labels_output[0] if len(outputs) == 3 else np.zeros(scores.shape[0], dtype=np.int64)
    
    # 过滤低置信度检测
    valid_mask = scores > min_thresh
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    
    # 转换为torch tensor
    boxes_tensor = torch.from_numpy(boxes).float()
    scores_tensor = torch.from_numpy(scores).float()
    labels_tensor = torch.from_numpy(labels).long()
    
    # 应用NMS（如果模型内部没有应用）
    if len(boxes_tensor) > 0:
        keep = nms(boxes_tensor, scores_tensor, iou_threshold=nms_thresh)
        boxes_tensor = boxes_tensor[keep]
        scores_tensor = scores_tensor[keep]
        labels_tensor = labels_tensor[keep]
    
    print(f"[INFO] After filtering and NMS: kept {len(boxes_tensor)} boxes")
    
    # 缩放boxes到原始图像尺寸
    if len(boxes_tensor) > 0:
        H_orig, W_orig = image_rgb.shape[:2]
        scale_x = W_orig / TARGET_SIZE[0]
        scale_y = H_orig / TARGET_SIZE[1]
        boxes_tensor[:, [0, 2]] *= scale_x
        boxes_tensor[:, [1, 3]] *= scale_y
    
    return boxes_tensor, scores_tensor, labels_tensor

@timer
def post_process_reference(outputs: List[np.ndarray],
                         image_rgb: np.ndarray,
                         min_thresh: float = MIN_THRESH,
                         norm_thresh: float = NORM_THRESH,
                         nms_thresh: float = NMS_IOU_THRESH
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference post-processing implementation matching ONNX version"""
    
    # Reorder outputs to match ONNX format: [cls_20, cls_40, cls_80, bbox_20, bbox_40, bbox_80]
    # TFLite outputs: [bbox_40, bbox_80, cls_80, bbox_20, cls_40, cls_20]
    cls_20 = torch.from_numpy(outputs[5]).float()   # cls_20x20
    cls_40 = torch.from_numpy(outputs[4]).float()   # cls_40x40
    cls_80 = torch.from_numpy(outputs[2]).float()   # cls_80x80
    bbox_20 = torch.from_numpy(outputs[3]).float()  # bbox_20x20
    bbox_40 = torch.from_numpy(outputs[0]).float()  # bbox_40x40
    bbox_80 = torch.from_numpy(outputs[1]).float()  # bbox_80x80
    
    cls_outputs = [cls_20, cls_40, cls_80]
    bbox_outputs = [bbox_20, bbox_40, bbox_80]
    
    all_boxes = []
    all_cls_probs = []
    
    for level, (cls_score, bbox_pred) in enumerate(zip(cls_outputs, bbox_outputs)):
        # Compute sigmoid probability per class, reshape to (H*W, C)
        cls_prob = torch.sigmoid(cls_score)[0].permute(1, 2, 0).reshape(-1, cls_score.shape[1])
        all_cls_probs.append(cls_prob)

        # Decode bounding boxes
        bbox = bbox_pred[0]  # shape (4,H,W)
        tx, ty, tr, tb = bbox[0], bbox[1], bbox[2], bbox[3]
        H, W = tx.shape
        stride = TARGET_SIZE[0] / H  # 640 / H

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x_center = (grid_x + 0.5) * stride
        y_center = (grid_y + 0.5) * stride

        x1 = x_center - tx * stride
        y1 = y_center - ty * stride
        x2 = x_center + tr * stride
        y2 = y_center + tb * stride

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)  # shape (H*W, 4)
        all_boxes.append(boxes)

    if not all_boxes:
        return (torch.empty((0,4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long))

    all_boxes_tensor     = torch.cat(all_boxes, dim=0)     # shape (T,4)
    all_cls_probs_tensor = torch.cat(all_cls_probs, dim=0) # shape (T,C)

    final_boxes  = []
    final_scores = []
    final_labels = []

    num_classes = all_cls_probs_tensor.shape[1]
    for cls_id in range(num_classes):
        raw_scores = all_cls_probs_tensor[:, cls_id]
        if raw_scores.max() < min_thresh:
            continue

        min_s, max_s = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-6)
        mask = norm_scores > norm_thresh
        if mask.sum() == 0:
            continue

        boxes_cls  = all_boxes_tensor[mask]
        scores_cls = raw_scores[mask]
        keep = nms(boxes_cls, scores_cls, iou_threshold=nms_thresh)
        final_boxes.append(boxes_cls[keep])
        final_scores.append(scores_cls[keep])
        final_labels.append(torch.full_like(scores_cls[keep], cls_id, dtype=torch.long))

    if final_boxes:
        final_boxes  = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        print(f"[INFO] After NMS: kept {final_boxes.shape[0]} boxes")
    else:
        print("[WARN] No boxes passed filtering.")
        final_boxes  = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)

    # Scale boxes back to original image size
    H_orig, W_orig = image_rgb.shape[:2]
    scale_x = W_orig / TARGET_SIZE[0]
    scale_y = H_orig / TARGET_SIZE[1]
    scaled_boxes = final_boxes.clone()
    if scaled_boxes.numel() > 0:
        scaled_boxes[:, [0, 2]] *= scale_x
        scaled_boxes[:, [1, 3]] *= scale_y

    return scaled_boxes, final_scores, final_labels

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

    for box, score, cls_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        label_text = f"{class_names[cls_id]} {score:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w + 4, y1),
            (0, 255, 0),
            -1
        )
        # Draw text
        cv2.putText(
            img,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Detection result saved as {save_path}")

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Detection Result")
    plt.show()

# ------------------------------------------------------------------------------
# Main detection pipeline
# ------------------------------------------------------------------------------
@timer
def detect_objects(image_path: str, text: str, save_path: str = "result_tflite.jpg"):
    """Main detection pipeline matching ONNX version structure"""
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
        
        # 5. Post-process decoded outputs
        print("\n[Step 4] Post-processing decoded outputs...")
        boxes, scores, labels = post_process_with_decode(
            post_outputs,
            image_rgb,
            min_thresh=MIN_THRESH,
            norm_thresh=NORM_THRESH,
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
