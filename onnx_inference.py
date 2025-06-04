import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
import numpy as np
from typing import Tuple, List
import torch
from torchvision.ops import nms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
from collections import defaultdict
# Add timing decorator
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

# 1. Initialize model and tokenizer
session_text = ort.InferenceSession("onnx_models/text_encoder.onnx")
session_visual = ort.InferenceSession("onnx_models/visual_encoder.onnx") 
session_post = ort.InferenceSession("onnx_models/post_prediction.onnx")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

MIN_THRESH              = 0.05    # Original score max threshold
NORM_THRESH            = 0.25     # Normalized threshold
NMS_IOU_THRESH         = 0.5     # NMS IOU threshold

TARGET_SIZE            = (640, 640)  # Fixed input size
@timer
def post_process_reference(outputs: Tuple[np.ndarray, ...],
                         image_rgb: np.ndarray,
                         min_thresh: float = MIN_THRESH,
                         norm_thresh: float = NORM_THRESH,
                         nms_thresh: float = NMS_IOU_THRESH
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference post-processing implementation
    Args:
        outputs: (cls_20x20, cls_40x40, cls_80x80, bbox_20x20, bbox_40x40, bbox_80x80)
    """
    all_boxes = []
    all_cls_probs = []
    
    cls_outputs = outputs[:3]    # 20x20, 40x40, 80x80
    bbox_outputs = outputs[3:]   # 20x20, 40x40, 80x80
    
    for level, (cls_score_np, bbox_pred_np) in enumerate(zip(cls_outputs, bbox_outputs)):
        cls_score = torch.from_numpy(cls_score_np).float()
        bbox_pred = torch.from_numpy(bbox_pred_np).float()

        cls_prob = torch.sigmoid(cls_score)[0].permute(1, 2, 0).reshape(-1, cls_score.shape[1])
        all_cls_probs.append(cls_prob)  # list of [H*W, C]

        bbox = bbox_pred[0]
        tx, ty, tr, tb = bbox[0], bbox[1], bbox[2], bbox[3]

        H, W = tx.shape
        stride = TARGET_SIZE[0] / H  # 640/H

        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x_center = (grid_x + 0.5) * stride
        y_center = (grid_y + 0.5) * stride

        x1 = x_center - tx * stride
        y1 = y_center - ty * stride
        x2 = x_center + tr * stride
        y2 = y_center + tb * stride

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)  # [H*W, 4]
        all_boxes.append(boxes)

    if not all_boxes:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    all_boxes_tensor     = torch.cat(all_boxes, dim=0)     # [T, 4]
    all_cls_probs_tensor = torch.cat(all_cls_probs, dim=0) # [T, C]

    final_boxes = []
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

        boxes_cls = all_boxes_tensor[mask]
        scores_cls = raw_scores[mask]

        keep = nms(boxes_cls, scores_cls, iou_threshold=nms_thresh)
        final_boxes.append(boxes_cls[keep])
        final_scores.append(scores_cls[keep])
        final_labels.append(torch.full_like(scores_cls[keep], cls_id, dtype=torch.long))

    if final_boxes:
        final_boxes  = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        print(f"[INFO] After NMS: {final_boxes.shape[0]} boxes kept")
    else:
        print("[WARN] No boxes passed filtering.")
        final_boxes  = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)

    # Scale back to original image
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
    """Draw detection results on the original RGB image and save/display it
    """
    img = image_rgb.copy()
    H, W = img.shape[:2]
    font_scale = max(0.5, min(W, H) / 800 * 1.0)
    thickness = max(1, int(min(W, H) / 400))

    for box, score, cls_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().tolist()
        label_text = f"{class_names[cls_id]} {score:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # Draw text background
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )
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

    # Convert to BGR for saving (OpenCV requires BGR format)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Detection result saved as {save_path}")

    # Keep RGB format for display
    plt.figure(figsize=(10, 8))
    plt.imshow(img)  # matplotlib uses RGB format by default
    plt.axis("off")
    plt.title("Detection Result")
    plt.show()


@timer
def preprocess_image(image_path):
    """Image preprocessing: load and transform to tensor"""
    # Read image and convert to RGB (Note: OpenCV reads in BGR format)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    pil_image = Image.fromarray(image_rgb)
    transform = transforms.Compose([
        transforms.Resize(TARGET_SIZE),
        transforms.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).numpy()
    input_tensor = input_tensor.astype(np.float32)
    
    return image_rgb, input_tensor

# 3. Text preprocessing function
def preprocess_text(text):
    """Text preprocessing: tokenize and prepare masks"""
    tokens = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="np"
    )
    return tokens["input_ids"], tokens["attention_mask"]

# 4. Main inference function
def detect_objects(image_path: str, text: str, save_path: str = "dog.jpeg"):
    """Main detection pipeline"""
    # Preprocess image
    image_rgb, input_tensor = preprocess_image(image_path)
    
    # Preprocess text
    start_text = time.perf_counter()
    input_ids, attention_mask = preprocess_text(text)
    text_feats = session_text.run(
        ["text_feats"],
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64)
        }
    )[0]
    timer.times['text_encoder'].append(time.perf_counter() - start_text)
    
    # Run visual encoder
    start_visual = time.perf_counter()
    feat_0, feat_1, feat_2 = session_visual.run(
        ["feat_0", "feat_1", "feat_2"],
        {"image": input_tensor}
    )
    timer.times['visual_encoder'].append(time.perf_counter() - start_visual)
    
    # Run post-processing network
    start_post = time.perf_counter()
    txt_masks = np.ones((1, 1), dtype=np.float32)
    outputs = session_post.run(
        ["cls_score_0", "cls_score_1", "cls_score_2", 
         "bbox_pred_0", "bbox_pred_1", "bbox_pred_2"],
        {
            "feat_0": feat_0,
            "feat_1": feat_1,
            "feat_2": feat_2,
            "text_feats": text_feats,
            "txt_masks": txt_masks
        }
    )
    timer.times['post_network'].append(time.perf_counter() - start_post)
    
    # Post-process
    boxes, scores, labels = post_process_reference(
        outputs,
        image_rgb,
        min_thresh=MIN_THRESH,
        norm_thresh=NORM_THRESH,
        nms_thresh=NMS_IOU_THRESH
    )
    
    # Visualize
    visualize(
        image_rgb=image_rgb,
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=[text],
        save_path=save_path
    )
    
    return boxes, scores, labels

if __name__ == "__main__":
    image_path = "sample_images/bottles.png"  
    text = "bottle"
    save_path = "result_onnx.png"
    
    start_total = time.perf_counter()
    boxes, scores, labels = detect_objects(
        image_path=image_path,
        text=text,
        save_path=save_path
    )
    total_time = time.perf_counter() - start_total
    timer.times['total_pipeline'].append(total_time)
    
    print(f"\nDetected {len(boxes)} objects:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"Object {i+1} - position: {box.tolist()}, confidence: {score:.3f}")
    
    print_time_analysis()
