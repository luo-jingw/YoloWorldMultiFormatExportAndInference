# YOLO-World Multi-Format Export and Inference

This directory contains a comprehensive framework for exporting and running inference with YOLO-World models in multiple formats: **ONNX**, **TFLite**, and **CoreML**. The project demonstrates how to split a complex multi-modal model into manageable components and deploy them across different platforms.

## Project Structure

```
YoloWorldMultiFormatExportAndInference/
├── export_*.py              # Model export scripts
├── *_inference.py           # Inference pipelines  
├── *_models_analysis.py     # Model analysis tools
├── */models/               # Exported model files
├── sample_images/          # Test images
├── result_*.png           # Detection results
└── *.txt                  # Analysis reports
```

## Quick Start

### 1. Export Models

Export YOLO-World to three different formats:

```bash
# Export to ONNX format
python export_onnx.py

# Export to TFLite format  
python export_tflite.py

# Export to CoreML format (requires coremltools)
python export_coreML.py
```

### 2. Run Inference

Test inference with different formats:

```bash
# ONNX inference
python onnx_inference.py

# TFLite inference
python tflite_inference.py

# Note: CoreML inference only works on macOS
```

### 3. Analyze Models

Inspect model structure and specifications:

```bash
# Analyze ONNX models
python onnx_models_analysis.py

# Analyze TFLite models  
python tflite_models_analysis.py

# Analyze CoreML models
python coreml_models_analysis.py
```

## Architecture Overview

YOLO-World is split into **three independent models** for better modularity and deployment flexibility:

### 1. Text Encoder (`text_encoder.*`)
- **Input**: Tokenized text (input_ids, attention_mask)
- **Output**: Text features [1, 1, 512] 
- **Function**: Encodes object class descriptions using CLIP text encoder

### 2. Visual Encoder (`visual_encoder.*`)
- **Input**: Image tensor [1, 3, 640, 640]
- **Output**: Multi-scale visual features (feat_0, feat_1, feat_2)
- **Function**: Extracts visual features at different scales

### 3. Post-Processing (`post_prediction.*`)
- **Input**: Visual features + text features + text masks
- **Output**: Classification scores + bounding box predictions
- **Function**: Fuses features and generates detections

## Supported Formats

| Format | Platform | Inference Support | Model Analysis |
|--------|----------|------------------|----------------|
| **ONNX** | Cross-platform | Full | Complete |
| **TFLite** | Mobile/Edge | Full | Complete |
| **CoreML** | Apple Ecosystem | macOS only | Structure only |

## Sample Images

The `sample_images/` directory contains test images:
- `bottles.png` - Multiple bottle detection
- `bus.jpg` - Vehicle detection
- `desk.png` - Office objects
- `zidane.jpg` - Person detection

## Performance Analysis

Each inference script includes timing analysis:

```
=== Time Analysis ===
text_encoder        : 0.0234s (avg of 1 runs)
visual_encoder      : 0.1456s (avg of 1 runs) 
post_network        : 0.0891s (avg of 1 runs)
post_process_reference: 0.0123s (avg of 1 runs)
total_pipeline      : 0.2704s (avg of 1 runs)
```

## Configuration

### Detection Parameters
```python
MIN_THRESH = 0.05      # Minimum confidence threshold
NORM_THRESH = 0.25     # Normalized score threshold  
NMS_IOU_THRESH = 0.5   # NMS IoU threshold
TARGET_SIZE = (640, 640) # Input image size
```

### Model Paths
```python
# ONNX models
onnx_models/
├── text_encoder.onnx
├── visual_encoder.onnx  
└── post_prediction.onnx

# TFLite models
tflite_models/
├── text_encoder.tflite
├── visual_encoder.tflite
└── post_prediction.tflite

# CoreML models  
coreML_models/
├── text_encoder.mlpackage
├── visual_encoder.mlpackage
└── post_prediction.mlpackage
```

## Dependencies

### Core Requirements
```bash
pip install torch torchvision
pip install onnx onnxruntime
pip install tensorflow
pip install transformers
pip install opencv-python
pip install matplotlib
pip install pillow
```

### Optional (for CoreML)
```bash
pip install coremltools  # macOS recommended
```

### MMDetection Setup
```bash
# Install MMDetection and dependencies
pip install mmdet mmengine mmcv
```

## Usage Examples

### Basic Detection
```python
from onnx_inference import detect_objects

# Detect bottles in image
boxes, scores, labels = detect_objects(
    image_path="sample_images/bottles.png",
    text="bottle", 
    save_path="result.png"
)

print(f"Found {len(boxes)} bottles")
```

### Batch Processing
```python
import os
from pathlib import Path

# Process all images in directory
for img_path in Path("sample_images").glob("*.png"):
    detect_objects(
        image_path=str(img_path),
        text="bottle",
        save_path=f"result_{img_path.stem}.png"
    )
```

## Model Analysis Reports

Each analysis script generates detailed reports:

- `onnx_model_info.txt` - ONNX model specifications
- `tflite_model_info.txt` - TFLite model details
- `coreml_model_info.txt` - CoreML model structure

## Platform Notes

### Windows/Linux
- ONNX inference works perfectly
- TFLite inference works perfectly  
- CoreML limited to model analysis only

### macOS
- All formats fully supported
- CoreML inference available
- Native Apple Silicon optimization

### Mobile/Edge Devices
- TFLite optimized for mobile deployment
- Quantization support available
- Consider model size constraints

## Troubleshooting

### Common Issues

1. **CoreML warnings on Linux/Windows**
   ```
   Failed to load libcoremlpython
   ```
   - **Solution**: This is expected behavior, model analysis still works

2. **TFLite shape mismatches**
   - **Solution**: Check input tensor shapes in analysis reports

3. **ONNX provider warnings**
   - **Solution**: Install appropriate ONNX Runtime providers

### Performance Optimization

1. **GPU Acceleration**
   ```python
   # ONNX with GPU
   providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
   session = ort.InferenceSession("model.onnx", providers=providers)
   ```

2. **TFLite Delegates**
   ```python
   # TFLite with GPU delegate
   interpreter = tf.lite.Interpreter(
       model_path="model.tflite",
       experimental_delegates=[tf.lite.experimental.load_delegate('libedgetpu.so.1')]
   )
   ```

## License

This project follows the same license as YOLO-World. Please refer to the main repository for licensing details.

## Contributing

Contributions are welcome! Please focus on:
- Additional export formats
- Mobile optimization
- Performance improvements
- Bug fixes and documentation

## Support

For issues related to:
- **Model export**: Check export script configurations
- **Inference errors**: Verify input/output shapes in analysis reports  
- **Platform compatibility**: See platform notes above
- **Performance**: Review timing analysis outputs

---

*This framework demonstrates the power of modular AI model deployment across multiple platforms and formats.*
