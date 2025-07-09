import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_onnx_to_tflite(onnx_path: str, saved_model_dir: str, tflite_path: str):
    print(f"[INFO] Loading ONNX model from '{onnx_path}'...")
    onnx_model = onnx.load(onnx_path)

    print(f"[INFO] Converting ONNX to TensorFlow SavedModel at '{saved_model_dir}'...")
    tf_rep = prepare(onnx_model)        
    # 如果 saved_model_dir 已存在，删除后重建
    if os.path.isdir(saved_model_dir):
        tf.io.gfile.rmtree(saved_model_dir)
    tf_rep.export_graph(saved_model_dir)

    print(f"[INFO] Converting SavedModel to TFLite at '{tflite_path}'...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 添加TFLite兼容性设置
    try:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # 允许使用TensorFlow操作
        ]
        tflite_model = converter.convert()
        print(f"[INFO] Successfully converted with SELECT_TF_OPS support")
    except Exception as e:
        print(f"[WARN] Failed with SELECT_TF_OPS: {e}")
        print(f"[INFO] Trying with basic TFLite operations only...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_model = converter.convert()

    output_dir = os.path.dirname(tflite_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Successfully wrote TFLite model to '{tflite_path}'.\n")

if __name__ == "__main__":
    onnx_base_dir   = "onnx_models"
    saved_base_dir  = "saved_models_with_p"
    tflite_base_dir = "tflite_models_with_p"

    os.makedirs(tflite_base_dir, exist_ok=True)
    os.makedirs(saved_base_dir, exist_ok=True)

    # 1) 转换text_encoder.onnx
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "text_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "text_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "text_encoder.tflite")
    )

    # 2) 转换visual_encoder.onnx
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "visual_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "visual_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "visual_encoder.tflite")
    )

    # 3) 转换包含解码的post_prediction.onnx（与ONNX推理代码一致）
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "post_prediction.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "post_prediction_with_decode"),
        tflite_path     = os.path.join(tflite_base_dir, "post_prediction_with_decode.tflite")
    )

    print("[INFO] TFLite conversion completed with integrated decoding.")

