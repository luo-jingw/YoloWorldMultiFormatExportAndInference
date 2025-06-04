#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
coreml_models_analysis.py

This script analyzes CoreML models and prints their input/output information.
Note: CoreML inference is only supported on macOS, so this script only analyzes model structure.
"""

import os
import sys
import platform
import numpy as np
import warnings

# Suppress CoreML warnings about missing native libraries
warnings.filterwarnings("ignore", message=".*libcoremlpython.*")
warnings.filterwarnings("ignore", message=".*torch version.*")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
    print(f"✓ CoreMLTools version: {ct.__version__}")
except ImportError as e:
    print(f"✗ Failed to import CoreMLTools: {str(e)}")
    COREML_AVAILABLE = False

# Check environment
def check_environment():
    """Check the current environment and provide warnings/suggestions."""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check platform
    system = platform.system()
    print(f"Operating System: {system}")
    
    if system != "Darwin":
        print("⚠ Warning: CoreML models are optimized for Apple platforms (macOS/iOS)")
        print("  - Model structure analysis is supported on all platforms")
        print("  - Model inference only works on macOS 10.13+")
    
    # Check torch version if available
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        
        if torch_version < "2.1.0":
            print("⚠ Warning: CoreMLTools optimize features require PyTorch 2.1.0+")
            print("  - Basic model loading and analysis should still work")
    except ImportError:
        print("PyTorch: Not installed")
    
    # Check CoreMLTools status
    if COREML_AVAILABLE:
        print("✓ CoreMLTools: Available for model analysis")
        
        # Test basic functionality
        try:
            # Try to create a minimal spec to test functionality
            import coremltools.proto.FeatureTypes_pb2 as ft
            print("✓ CoreMLTools protobuf support: Working")
        except Exception as e:
            print(f"⚠ CoreMLTools protobuf warning: {str(e)}")
    else:
        print("✗ CoreMLTools: Not available")
        return False
    
    print("="*60)
    return True

# ------------------------------------------------------------------------------
# Load CoreML models
# ------------------------------------------------------------------------------
MODEL_DIR = "coreML_models"
TEXT_ENCODER_MODEL = os.path.join(MODEL_DIR, "text_encoder.mlpackage")
VISUAL_ENCODER_MODEL = os.path.join(MODEL_DIR, "visual_encoder.mlpackage")
POST_PREDICTION_MODEL = os.path.join(MODEL_DIR, "post_prediction.mlpackage")

# Check if models exist
models_info = [
    ("Text Encoder", TEXT_ENCODER_MODEL),
    ("Visual Encoder", VISUAL_ENCODER_MODEL),
    ("Post Prediction", POST_PREDICTION_MODEL)
]

loaded_models = {}
if COREML_AVAILABLE:
    for model_name, model_path in models_info:
        if os.path.exists(model_path):
            try:
                # Suppress warnings during model loading
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loaded_models[model_name] = ct.models.MLModel(model_path)
                print(f"✓ Loaded {model_name} from {model_path}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)}")
        else:
            print(f"⚠ Model not found: {model_path}")
else:
    print("⚠ Skipping model loading due to CoreMLTools unavailability")

# ------------------------------------------------------------------------------
# Model analysis functions
# ------------------------------------------------------------------------------
def print_model_info(model, model_name, file_handle=None):
    """Print detailed input/output information for a CoreML model."""
    def write_line(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    write_line(f"\n{'='*60}", file_handle)
    write_line(f"CoreML Model Information: {model_name}", file_handle)
    write_line(f"{'='*60}", file_handle)
    
    # Print model specification
    spec = model.get_spec()
    write_line(f"\nModel Type: {spec.WhichOneof('Type')}", file_handle)
    
    # Print input details
    if hasattr(spec.description, 'input'):
        inputs = spec.description.input
        write_line(f"\nInput Information (Total: {len(inputs)}):", file_handle)
        write_line("-" * 40, file_handle)
        for i, input_desc in enumerate(inputs):
            write_line(f"Input {i+1}:", file_handle)
            write_line(f"  Name: {input_desc.name}", file_handle)
            write_line(f"  Type: {input_desc.type.WhichOneof('Type')}", file_handle)
            
            # Get shape information based on type
            if input_desc.type.HasField('multiArrayType'):
                array_type = input_desc.type.multiArrayType
                write_line(f"  Shape: {list(array_type.shape)}", file_handle)
                write_line(f"  Data Type: {array_type.dataType}", file_handle)
            elif input_desc.type.HasField('imageType'):
                image_type = input_desc.type.imageType
                write_line(f"  Width: {image_type.width}", file_handle)
                write_line(f"  Height: {image_type.height}", file_handle)
                write_line(f"  Color Space: {image_type.colorSpace}", file_handle)
            
            write_line("", file_handle)
    
    # Print output details
    if hasattr(spec.description, 'output'):
        outputs = spec.description.output
        write_line(f"Output Information (Total: {len(outputs)}):", file_handle)
        write_line("-" * 40, file_handle)
        for i, output_desc in enumerate(outputs):
            write_line(f"Output {i+1}:", file_handle)
            write_line(f"  Name: {output_desc.name}", file_handle)
            write_line(f"  Type: {output_desc.type.WhichOneof('Type')}", file_handle)
            
            # Get shape information based on type
            if output_desc.type.HasField('multiArrayType'):
                array_type = output_desc.type.multiArrayType
                write_line(f"  Shape: {list(array_type.shape)}", file_handle)
                write_line(f"  Data Type: {array_type.dataType}", file_handle)
            
            write_line("", file_handle)

def get_model_statistics(model, model_name, file_handle=None):
    """Get additional model statistics and information."""
    def write_line(text, file_handle=None):
        print(text)
        if file_handle:
            file_handle.write(text + '\n')
    
    write_line(f"\nModel Statistics for {model_name}:", file_handle)
    write_line("-" * 40, file_handle)
    
    spec = model.get_spec()
    
    # Model size information
    try:
        model_size = os.path.getsize(models_info[list(loaded_models.keys()).index(model_name)][1])
        write_line(f"Model file size: {model_size / (1024*1024):.2f} MB", file_handle)
    except:
        write_line("Model file size: Unknown", file_handle)
    
    # Model metadata
    if hasattr(spec.description, 'metadata'):
        metadata = spec.description.metadata
        if metadata.shortDescription:
            write_line(f"Description: {metadata.shortDescription}", file_handle)
        if metadata.author:
            write_line(f"Author: {metadata.author}", file_handle)
        if metadata.license:
            write_line(f"License: {metadata.license}", file_handle)
        if metadata.userDefined:
            write_line("User defined metadata:", file_handle)
            for key, value in metadata.userDefined.items():
                write_line(f"  {key}: {value}", file_handle)
    
    # Platform compatibility
    write_line(f"Minimum deployment target: {spec.specificationVersion}", file_handle)
    
    # Model architecture info
    model_type = spec.WhichOneof('Type')
    if model_type == 'neuralNetwork':
        nn = spec.neuralNetwork
        write_line(f"Neural Network layers: {len(nn.layers)}", file_handle)
        
        # Count different layer types
        layer_types = {}
        for layer in nn.layers:
            layer_type = layer.WhichOneof('layer')
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        write_line("Layer type distribution:", file_handle)
        for layer_type, count in sorted(layer_types.items()):
            write_line(f"  {layer_type}: {count}", file_handle)

# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Check environment first
    env_ok = check_environment()
    
    if not env_ok:
        print("\nEnvironment check failed. Please install CoreMLTools:")
        print("  pip install coremltools")
        print("  # or")
        print("  conda install -c conda-forge coremltools")
        exit(1)
    
    if not loaded_models:
        print("\nNo CoreML models found or loaded.")
        print("Possible reasons:")
        print("1. Models not exported yet - run export_coreML.py first")
        print("2. CoreMLTools compatibility issues")
        print("3. Missing model files in coreML_models/ directory")
        
        if not COREML_AVAILABLE:
            print("\nCoreMLTools is not properly installed or has compatibility issues.")
            print("Try reinstalling:")
            print("  pip uninstall coremltools")
            print("  pip install coremltools")
        
        exit(1)
    
    # Create output file for model info
    output_file = "coreml_model_info.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CoreML Model Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write("Note: CoreML inference testing is only supported on macOS.\n")
        f.write("This report contains model structure analysis only.\n")
        f.write("=" * 60 + "\n\n")
        
        # Write environment info to file
        f.write("Environment Information:\n")
        f.write(f"- Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
        f.write(f"- Platform: {platform.system()}\n")
        f.write(f"- CoreMLTools: {ct.__version__ if COREML_AVAILABLE else 'Not available'}\n")
        try:
            import torch
            f.write(f"- PyTorch: {torch.__version__}\n")
        except:
            f.write("- PyTorch: Not installed\n")
        f.write("\n")
        
        print("CoreML Model Analysis Report")
        print("=" * 60)
        print("Note: CoreML inference testing is only supported on macOS.")
        print("This report contains model structure analysis only.")
        print("=" * 60)
        
        # Print information for each model
        for model_name, model in loaded_models.items():
            model_path = models_info[list(loaded_models.keys()).index(model_name)][1]
            print_model_info(model, f"{model_name} ({model_path})", f)
            get_model_statistics(model, model_name, f)
        
        # Summary information
        f.write("\n" + "="*60 + "\n")
        f.write("Summary\n")
        f.write("="*60 + "\n")
        
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        
        def write_line(text, file_handle=None):
            print(text)
            if file_handle:
                file_handle.write(text + '\n')
        
        write_line(f"Total models analyzed: {len(loaded_models)}", f)
        write_line("Available models:", f)
        for model_name in loaded_models.keys():
            write_line(f"  ✓ {model_name}", f)
        
        missing_models = []
        for model_name, model_path in models_info:
            if model_name not in loaded_models:
                missing_models.append(model_name)
        
        if missing_models:
            write_line("Missing models:", f)
            for model_name in missing_models:
                write_line(f"  ✗ {model_name}", f)
        
        write_line("\nModel deployment information:", f)
        write_line("- CoreML models can be deployed on iOS, macOS, watchOS, and tvOS", f)
        write_line("- Inference testing requires macOS 10.13 or later", f)
        write_line("- Models use .mlpackage format for better organization", f)
        write_line("- Optimized for Apple's Neural Engine when available", f)
        
        write_line("\nEnvironment notes:", f)
        write_line("- The warnings about 'libcoremlpython' are normal in non-macOS environments", f)
        write_line("- These warnings don't affect model structure analysis", f)
        write_line("- For full CoreML functionality, use macOS with proper Xcode tools", f)
    
    print(f"\n✓ Model analysis report exported to {output_file}")
    
    print("\n" + "="*60)
    print("CoreML model analysis completed successfully!")
    print("Note: Warnings about libcoremlpython are expected in non-macOS environments.")
    print("="*60)
