#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tflite_models_analysis.py

This script analyzes TFLite models and prints their input/output information.
"""

import os
import numpy as np
import tensorflow as tf

# ------------------------------------------------------------------------------
# Load TFLite interpreters for text, visual, and post-processing models
# ------------------------------------------------------------------------------
MODEL_DIR = "tflite_models"
interpreter_text   = tf.lite.Interpreter(os.path.join(MODEL_DIR, "text_encoder.tflite"))
interpreter_visual = tf.lite.Interpreter(os.path.join(MODEL_DIR, "visual_encoder.tflite"))
interpreter_post   = tf.lite.Interpreter(os.path.join(MODEL_DIR, "post_prediction.tflite"))

interpreter_text.allocate_tensors()
interpreter_visual.allocate_tensors()
interpreter_post.allocate_tensors()

# ------------------------------------------------------------------------------
# Print model information with file export
# ------------------------------------------------------------------------------
def print_model_info(interpreter, model_name, file_handle=None):
    """Print detailed input/output information for a TFLite model."""
    def write_line(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    write_line(f"\n{'='*60}", file_handle)
    write_line(f"Model Information: {model_name}", file_handle)
    write_line(f"{'='*60}", file_handle)
    
    # Print input details
    input_details = interpreter.get_input_details()
    write_line(f"\nInput Information (Total: {len(input_details)}):", file_handle)
    write_line("-" * 40, file_handle)
    for i, detail in enumerate(input_details):
        write_line(f"Input {i+1}:", file_handle)
        write_line(f"  Name: {detail['name']}", file_handle)
        write_line(f"  Shape: {detail['shape']}", file_handle)
        write_line(f"  Data Type: {detail['dtype']}", file_handle)
        write_line(f"  Quantization Parameters: {detail['quantization_parameters']}", file_handle)
        write_line("", file_handle)
    
    # Print output details
    output_details = interpreter.get_output_details()
    write_line(f"Output Information (Total: {len(output_details)}):", file_handle)
    write_line("-" * 40, file_handle)
    for i, detail in enumerate(output_details):
        write_line(f"Output {i+1}:", file_handle)
        write_line(f"  Name: {detail['name']}", file_handle)
        write_line(f"  Shape: {detail['shape']}", file_handle)
        write_line(f"  Data Type: {detail['dtype']}", file_handle)
        write_line(f"  Quantization Parameters: {detail['quantization_parameters']}", file_handle)
        write_line("", file_handle)

# ------------------------------------------------------------------------------
# Create dummy inputs and test inference
# ------------------------------------------------------------------------------
def create_dummy_inputs(interpreter):
    """Create dummy inputs based on model input specifications."""
    input_details = interpreter.get_input_details()
    dummy_inputs = {}
    
    for detail in input_details:
        name = detail['name']
        shape = detail['shape']
        dtype = detail['dtype']
        
        # Remove 'serving_default_' prefix and ':0' suffix for key
        key = name.replace('serving_default_', '').replace(':0', '')
        
        # Create dummy data based on dtype
        if dtype == np.int64:
            dummy_inputs[key] = np.ones(shape, dtype=dtype)
        elif dtype == np.float32:
            dummy_inputs[key] = np.random.randn(*shape).astype(dtype)
        else:
            dummy_inputs[key] = np.zeros(shape, dtype=dtype)
            
    return dummy_inputs

def test_model_inference(interpreter, model_name, file_handle=None):
    """Test model inference with dummy inputs."""
    def write_line(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    write_line(f"\n{'='*60}", file_handle)
    write_line(f"Testing Model Inference: {model_name}", file_handle)
    write_line(f"{'='*60}", file_handle)
    
    try:
        # Create dummy inputs
        dummy_inputs = create_dummy_inputs(interpreter)
        write_line(f"Created Dummy Inputs:", file_handle)
        for key, value in dummy_inputs.items():
            write_line(f"  {key}: shape={value.shape}, dtype={value.dtype}", file_handle)
        
        # Set inputs
        input_details = interpreter.get_input_details()
        for detail in input_details:
            name = detail['name']
            key = name.replace('serving_default_', '').replace(':0', '')
            if key in dummy_inputs:
                interpreter.set_tensor(detail['index'], dummy_inputs[key])
        
        # Run inference
        write_line(f"\nStarting Inference...", file_handle)
        interpreter.invoke()
        write_line(f"Inference Completed!", file_handle)
        
        # Get outputs
        output_details = interpreter.get_output_details()
        write_line(f"\nOutput Results:", file_handle)
        outputs = []
        for i, detail in enumerate(output_details):
            output_data = interpreter.get_tensor(detail['index'])
            outputs.append(output_data)
            write_line(f"  Output {i+1} ({detail['name']}): shape={output_data.shape}, dtype={output_data.dtype}", file_handle)
            write_line(f"    Value Range: [{np.min(output_data):.6f}, {np.max(output_data):.6f}]", file_handle)
            write_line(f"    Mean: {np.mean(output_data):.6f}, Std: {np.std(output_data):.6f}", file_handle)
        
        write_line(f"\nModel {model_name} Test Successful!", file_handle)
        return outputs
        
    except Exception as e:
        write_line(f"\n✗ Model {model_name} Test Failed: {str(e)}", file_handle)
        return None

# ------------------------------------------------------------------------------
# Complete inference pipeline
# ------------------------------------------------------------------------------
def complete_inference_pipeline(text_prompt="dog"):
    """Run complete inference pipeline using dummy inputs."""
    print(f"\n{'='*80}")
    print(f"Complete Inference Pipeline with Text: '{text_prompt}'")
    print(f"{'='*80}")
    
    # 1. Text encoder inference
    print("\n[Step 1] Running Text Encoder...")
    dummy_text_inputs = create_dummy_inputs(interpreter_text)
    
    # Set text inputs
    input_details = interpreter_text.get_input_details()
    for detail in input_details:
        name = detail['name']
        key = name.replace('serving_default_', '').replace(':0', '')
        if key in dummy_text_inputs:
            interpreter_text.set_tensor(detail['index'], dummy_text_inputs[key])
    
    interpreter_text.invoke()
    text_outputs = []
    output_details = interpreter_text.get_output_details()
    for detail in output_details:
        text_outputs.append(interpreter_text.get_tensor(detail['index']))
    
    text_feats = text_outputs[0]  # Shape: (1, 1, 512)
    print(f"Text features shape: {text_feats.shape}")
    
    # 2. Visual encoder inference
    print("\n[Step 2] Running Visual Encoder...")
    dummy_visual_inputs = create_dummy_inputs(interpreter_visual)
    
    # Set visual inputs
    input_details = interpreter_visual.get_input_details()
    for detail in input_details:
        name = detail['name']
        key = name.replace('serving_default_', '').replace(':0', '')
        if key in dummy_visual_inputs:
            interpreter_visual.set_tensor(detail['index'], dummy_visual_inputs[key])
    
    interpreter_visual.invoke()
    visual_outputs = []
    output_details = interpreter_visual.get_output_details()
    for detail in output_details:
        visual_outputs.append(interpreter_visual.get_tensor(detail['index']))
    
    feat_2 = visual_outputs[0]  # Shape: (1, 640, 20, 20)
    feat_1 = visual_outputs[1]  # Shape: (1, 640, 40, 40)
    feat_0 = visual_outputs[2]  # Shape: (1, 320, 80, 80)
    
    print(f"Visual features shapes:")
    print(f"  feat_0: {feat_0.shape}")
    print(f"  feat_1: {feat_1.shape}")
    print(f"  feat_2: {feat_2.shape}")
    
    # 3. Post-processing inference
    print("\n[Step 3] Running Post-processing...")
    txt_masks = np.ones((1, 1), dtype=np.float32)
    
    # Set post-processing inputs
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
    
    print(f"\nPost-processing outputs:")
    for i, output in enumerate(post_outputs):
        print(f"  Output {i+1}: shape={output.shape}, range=[{np.min(output):.6f}, {np.max(output):.6f}]")
    
    print(f"\nComplete pipeline executed successfully!")
    return text_feats, visual_outputs, post_outputs

# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Create output file for model info
    output_file = "tflite_model_info.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TFLite Model Loading and Information Export\n")
        f.write("=" * 60 + "\n\n")
        
        print("TFLite Model Loading and Information Export")
        print("=" * 60)
        
        # Print information for each model
        print_model_info(interpreter_text, "Text Encoder (text_encoder.tflite)", f)
        print_model_info(interpreter_visual, "Visual Encoder (visual_encoder.tflite)", f)
        print_model_info(interpreter_post, "Post-processing Model (post_prediction.tflite)", f)
        
        # Test inference for each model
        f.write("\n" + "="*60 + "\n")
        f.write("Start Testing Model Inference\n")
        f.write("="*60 + "\n")
        
        print("\n" + "="*60)
        print("Start Testing Model Inference")
        print("="*60)
        
        test_model_inference(interpreter_text, "Text Encoder", f)
        test_model_inference(interpreter_visual, "Visual Encoder", f)
        test_model_inference(interpreter_post, "Post-processing Model", f)
    
    print(f"\nModel information exported to {output_file}")
    
    # Run complete inference pipeline with dummy data
    complete_inference_pipeline("dog")
    
    print("\n" + "="*60)
    print("All models analyzed successfully!")
    print("="*60)
