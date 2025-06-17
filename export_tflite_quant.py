import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf


def convert_onnx_to_tflite(onnx_path: str, saved_model_dir: str, tflite_path: str):
    # 1) Load ONNX model
    print(f"[INFO] Loading ONNX model from '{onnx_path}'...")
    onnx_model = onnx.load(onnx_path)

    # 2) Convert ONNX to TensorFlow Graph (pfrep) using onnx-tf
    print(f"[INFO] Converting ONNX to TensorFlow SavedModel at '{saved_model_dir}'...")
    tf_rep = prepare(onnx_model)  # This automatically converts NCHW to NHWC (TensorFlow default)
    # If saved_model_dir exists, remove it and recreate
    if os.path.isdir(saved_model_dir):
        tf.io.gfile.rmtree(saved_model_dir)
    tf_rep.export_graph(saved_model_dir)

    # 3) Convert SavedModel to TFLite using TFLiteConverter
    print(f"[INFO] Converting SavedModel to TFLite at '{tflite_path}'...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter._experimental_lower_tensor_list_ops = True

    tflite_model = converter.convert()

    # 4) Write the .tflite file
    output_dir = os.path.dirname(tflite_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[INFO] Successfully wrote TFLite model to '{tflite_path}'.\n")


if __name__ == "__main__":
    # Set directories for ONNX source files and TFLite output
    onnx_base_dir   = "onnx_models"
    saved_base_dir  = "saved_models"
    tflite_base_dir = "tflite_models"

    # Ensure output directories exist
    os.makedirs(tflite_base_dir, exist_ok=True)
    os.makedirs(saved_base_dir, exist_ok=True)

    # Convert each model:
    # 1) text_encoder.onnx → saved_models/text_encoder → tflite_models/text_encoder.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "text_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "text_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "text_encoder_quant_f16.tflite")
    )

    # 2) visual_encoder.onnx → saved_models/visual_encoder → tflite_models/visual_encoder.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "visual_encoder.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "visual_encoder"),
        tflite_path     = os.path.join(tflite_base_dir, "visual_encoder_quant_f16.tflite")
    )

    # 3) post_prediction.onnx → saved_models/post_prediction → tflite_models/post_prediction.tflite
    convert_onnx_to_tflite(
        onnx_path       = os.path.join(onnx_base_dir, "post_prediction.onnx"),
        saved_model_dir = os.path.join(saved_base_dir, "post_prediction"),
        tflite_path     = os.path.join(tflite_base_dir, "post_prediction_quant_f16.tflite")
    )

    print("[INFO] All models have been converted to TFLite successfully.")