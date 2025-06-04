import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_onnx_to_tflite(onnx_path: str, saved_model_dir: str, tflite_path: str):
    # 1) 加载 ONNX
    print(f"[INFO] Loading ONNX model from '{onnx_path}'...")
    onnx_model = onnx.load(onnx_path)

    # 2) 使用 onnx-tf 将 ONNX 转为 TensorFlow Graph (pfrep)
    print(f"[INFO] Converting ONNX to TensorFlow SavedModel at '{saved_model_dir}'...")
    tf_rep = prepare(onnx_model)         # 这一步会把 NCHW 自动转换为 NHWC（TensorFlow 默认）
    # 如果 saved_model_dir 已存在，删除后重建
    if os.path.isdir(saved_model_dir):
        tf.io.gfile.rmtree(saved_model_dir)
    tf_rep.export_graph(saved_model_dir)

    # 3) 使用 TFLiteConverter 导出 TFLite
    print(f"[INFO] Converting SavedModel to TFLite at '{tflite_path}'...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 可根据需要添加优化选项，比如：
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # 4) 写入 .tflite 文件
    output_dir = os.path.dirname(tflite_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Successfully wrote TFLite model to '{tflite_path}'.\n")


if __name__ == "__main__":
    # 设置 ONNX 源文件目录和 TFLite 输出目录
    onnx_base_dir   = "onnx_models"
    saved_base_dir  = "saved_models"
    tflite_base_dir = "tflite_models"

    # 确保输出目录存在
    os.makedirs(tflite_base_dir, exist_ok=True)
    os.makedirs(saved_base_dir, exist_ok=True)

    # 逐个转换：
    # 1) text_encoder.onnx → saved_models/text_encoder → tflite_models/text_encoder.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "text_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "text_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "text_encoder.tflite")
    )

    # 2) visual_encoder.onnx → saved_models/visual_encoder → tflite_models/visual_encoder.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "visual_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "visual_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "visual_encoder.tflite")
    )

    # 3) post_prediction.onnx → saved_models/post_prediction → tflite_models/post_prediction.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "post_prediction.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "post_prediction"),
        tflite_path     = os.path.join(tflite_base_dir, "post_prediction.tflite")
    )

    print("[INFO] All models have been converted to TFLite successfully.")