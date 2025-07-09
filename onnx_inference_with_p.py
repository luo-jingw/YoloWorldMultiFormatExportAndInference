import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer
import numpy as np
from typing import List
import torch
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

# Initialize ONNX Runtime sessions
session_text = ort.InferenceSession("/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/text_encoder.onnx")
session_visual = ort.InferenceSession("/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/visual_encoder.onnx")
session_post = ort.InferenceSession("/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/onnx_models/post_prediction.onnx")

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
TARGET_SIZE = (640, 640)

# Detection parameters
MIN_THRESH = 0.05
NORM_THRESH = 0.25
NMS_IOU_THRESH = 0.5

@timer
def visualize(image_rgb: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
             labels: np.ndarray, class_names: List[str], save_path: str = "detection_result.jpg") -> None:
    """Draw detection results and save image"""
    img = image_rgb.copy()
    H, W = img.shape[:2]
    font_scale = max(0.5, min(W, H) / 800 * 1.0)
    thickness = max(1, int(min(W, H) / 400))

    # Ensure numpy arrays
    if torch.is_tensor(boxes): boxes = boxes.numpy()
    if torch.is_tensor(scores): scores = scores.numpy()
    if torch.is_tensor(labels): labels = labels.numpy()

    # Filter valid detections
    valid_mask = scores > 0
    valid_boxes = boxes[valid_mask]
    valid_scores = scores[valid_mask]
    valid_labels = labels[valid_mask]
    
    print(f"[DEBUG] Drawing {len(valid_boxes)} boxes on image {H}x{W}")
    
    for i, (box, score, cls_id) in enumerate(zip(valid_boxes, valid_scores, valid_labels)):
        x1, y1, x2, y2 = box.astype(float)
        
        # 注意：这里的box坐标已经是原图尺寸的坐标，不需要再次缩放
        print(f"[DEBUG] Box {i}: ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        
        # Clamp to image boundaries
        x1_final = max(0, min(int(x1), W-1))
        y1_final = max(0, min(int(y1), H-1))
        x2_final = max(0, min(int(x2), W-1))
        y2_final = max(0, min(int(y2), H-1))
        
        # Ensure box has area
        if x2_final <= x1_final or y2_final <= y1_final:
            print(f"[WARN] Box {i} has no area after clamping: ({x1_final},{y1_final},{x2_final},{y2_final})")
            continue
            
        print(f"[DEBUG] Box {i}: final=({x1_final},{y1_final},{x2_final},{y2_final})")
        
        label_text = f"{class_names[int(cls_id)]} {score:.2f}"

        # Draw bounding box with bright color
        cv2.rectangle(img, (x1_final, y1_final), (x2_final, y2_final), (0, 255, 0), thickness)
        
        # Draw text background
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Ensure text position is within image
        text_y = max(text_h + baseline + 4, y1_final)
        cv2.rectangle(img, (x1_final, text_y - text_h - baseline - 4), (x1_final + text_w + 4, text_y), (0, 255, 0), -1)
        cv2.putText(img, label_text, (x1_final + 2, text_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        
        print(f"[DEBUG] Drew box {i} successfully")

    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Detection result saved as {save_path}")
    
    # Display with matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Detection Result - {len(valid_boxes)} objects detected")
    
    # Add text annotation showing detection info
    if len(valid_boxes) > 0:
        info_text = f"Detected {len(valid_boxes)} objects\n"
        for i, (box, score, cls_id) in enumerate(zip(valid_boxes, valid_scores, valid_labels)):
            info_text += f"Object {i+1}: {class_names[int(cls_id)]} {score:.2f}\n"
        plt.figtext(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.show()

@timer
def preprocess_image(image_path):
    """Preprocess image to tensor"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image_rgb)
    transform = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor()])
    input_tensor = transform(pil_image).unsqueeze(0).numpy().astype(np.float32)
    
    return image_rgb, input_tensor

def preprocess_text(text):
    """Preprocess text to tokens"""
    tokens = tokenizer([text], padding="max_length", truncation=True, max_length=77, return_tensors="np")
    return tokens["input_ids"], tokens["attention_mask"]

@timer
def custom_nms_filter(boxes, cls_probs, 
                     min_thresh=0.05, norm_thresh=0.25, nms_thresh=0.5):
    """自定义NMS过滤，在Python中实现"""
    from torchvision.ops import nms
    
    final_boxes = []
    final_scores = []
    final_labels = []
    
    num_classes = cls_probs.shape[1]
    for cls_id in range(num_classes):
        raw_scores = cls_probs[:, cls_id]
        
        # 应用最小阈值过滤
        if raw_scores.max() < min_thresh:
            continue
        
        # 归一化分数
        min_s, max_s = raw_scores.min(), raw_scores.max()
        norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-6)
        mask = norm_scores > norm_thresh
        
        if mask.sum() == 0:
            continue
        
        boxes_cls = boxes[mask]
        scores_cls = raw_scores[mask]
        
        # 应用NMS
        keep = nms(boxes_cls, scores_cls, iou_threshold=nms_thresh)
        final_boxes.append(boxes_cls[keep])
        final_scores.append(scores_cls[keep])
        final_labels.append(torch.full_like(scores_cls[keep], cls_id, dtype=torch.long))
    
    if final_boxes:
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
    else:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)
    
    return final_boxes, final_scores, final_labels

def detect_objects(image_path: str, text: str, save_path: str = "result_onnx_with_p.jpg"):
    """Main detection pipeline using ONNX with integrated decoding"""
    print(f"\nDetecting '{text}' in {image_path} using ONNX with integrated decoding")
    
    # Preprocess
    image_rgb, input_tensor = preprocess_image(image_path)
    input_ids, attention_mask = preprocess_text(text)
    
    # Text encoding
    start_text = time.perf_counter()
    text_feats = session_text.run(
        None,
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64)
        }
    )[0]
    timer.times['text_encoder'].append(time.perf_counter() - start_text)
    
    # Visual encoding
    start_visual = time.perf_counter()
    visual_outputs = session_visual.run(
        None,
        {"image": input_tensor}
    )
    feat_0, feat_1, feat_2 = visual_outputs[0], visual_outputs[1], visual_outputs[2]
    timer.times['visual_encoder'].append(time.perf_counter() - start_visual)
    
    # Post-processing with integrated decoding
    start_post = time.perf_counter()
    txt_masks = np.ones((1, 1), dtype=np.float32)
    post_outputs = session_post.run(
        None,
        {
            "feat_0": feat_0,
            "feat_1": feat_1, 
            "feat_2": feat_2,
            "text_feats": text_feats,
            "txt_masks": txt_masks
        }
    )
    
    # 输出格式：[cls_sigmoid_0, cls_sigmoid_1, cls_sigmoid_2, bbox_decoded_0, bbox_decoded_1, bbox_decoded_2]
    cls_outputs = post_outputs[:3]    # sigmoid后的分类分数
    bbox_outputs = post_outputs[3:]   # 解码后的bbox坐标(TARGET_SIZE尺寸)
    
    timer.times['post_network_decode'].append(time.perf_counter() - start_post)
    
    # 处理每个尺度的输出
    start_nms = time.perf_counter()
    all_boxes = []
    all_cls_probs = []
    
    for level, (cls_score_np, bbox_pred_np) in enumerate(zip(cls_outputs, bbox_outputs)):
        # 分类分数已经是sigmoid后的结果
        cls_prob = torch.from_numpy(cls_score_np).float()[0].permute(1, 2, 0).reshape(-1, cls_score_np.shape[1])
        all_cls_probs.append(cls_prob)
        
        # bbox已经是解码后的绝对坐标，形状为[1, 4, H, W]
        bbox = torch.from_numpy(bbox_pred_np).float()[0]  # [4, H, W]
        boxes = bbox.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]
        all_boxes.append(boxes)
    
    if not all_boxes:
        final_boxes = torch.empty((0, 4))
        final_scores = torch.empty((0,))
        final_labels = torch.empty((0,), dtype=torch.long)
    else:
        all_boxes_tensor = torch.cat(all_boxes, dim=0)
        all_cls_probs_tensor = torch.cat(all_cls_probs, dim=0)
        
        final_boxes, final_scores, final_labels = custom_nms_filter(
            all_boxes_tensor, 
            all_cls_probs_tensor,
            min_thresh=MIN_THRESH,
            norm_thresh=NORM_THRESH,
            nms_thresh=NMS_IOU_THRESH
        )
    
    timer.times['custom_nms'].append(time.perf_counter() - start_nms)
    
    # Results
    valid_detections = len(final_boxes)
    print(f"[INFO] Detected {valid_detections} objects using ONNX with integrated decoding")
    
    # 将bbox从TARGET_SIZE缩放到原始图像尺寸
    if valid_detections > 0:
        H_orig, W_orig = image_rgb.shape[:2]
        scale_x = W_orig / TARGET_SIZE[0]
        scale_y = H_orig / TARGET_SIZE[1]
        
        print(f"[DEBUG] Scaling boxes from {TARGET_SIZE} to {W_orig}x{H_orig} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
        
        # 缩放bbox坐标到原图尺寸
        final_boxes_scaled = final_boxes.clone()
        final_boxes_scaled[:, [0, 2]] *= scale_x  # x坐标缩放
        final_boxes_scaled[:, [1, 3]] *= scale_y  # y坐标缩放
        
        print(f"[DEBUG] Box scaling - before: {final_boxes[0].tolist()}, after: {final_boxes_scaled[0].tolist()}")
    else:
        final_boxes_scaled = final_boxes
    
    # Visualize (使用缩放后的坐标)
    visualize(image_rgb, final_boxes_scaled.numpy(), final_scores.numpy(), 
              final_labels.numpy(), [text], save_path)
    
    return final_boxes_scaled, final_scores, final_labels

if __name__ == "__main__":
    image_path = "/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/sample_images/desk.png"  
    text = "laptop computer"
    save_path = "/home/ljw/python_proj/computer_vision/YOLO-World/demo/YoloWorldMultiFormatExportAndInference/result_onnx_with_p.png"
    
    start_total = time.perf_counter()
    boxes, scores, labels = detect_objects(image_path, text, save_path)
    timer.times['total_pipeline'].append(time.perf_counter() - start_total)
    
    # Print detection details
    print(f"\nDetection details:")
    valid_mask = scores > 0
    for i, (box, score, label) in enumerate(zip(boxes[valid_mask], scores[valid_mask], labels[valid_mask])):
        print(f"Object {i+1} - position: {box.tolist()}, confidence: {score:.3f}, class: {int(label)}")
    
    print_time_analysis()