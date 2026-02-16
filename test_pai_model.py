"""
PAI Best Model Test Evaluation
==============================
Loads the PAI dendritic model saved during training and evaluates on test set.

CRITICAL: This script uses IDENTICAL PAI configuration as pai_yolo_training.py
to ensure load_system works correctly.

Usage:
    python test_pai_model.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

os.environ["PYTHONBREAKPOINT"] = "0"

import torch
import torch.nn as nn
from ultralytics import YOLO
import copy

# PAI Imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# ============================================================================
# CONFIGURATION - Match your training run
# ============================================================================
PAI_SAVE_NAME = "dendrite_100"  # PAI saves to ./dendrite_100/ (basename only)
DATA_YAML = "runs/data_efficiency/dendrite_100/data_subset.yaml"  # Dataset YAML from training
IMGSZ = 640

# ============================================================================
# PAI CONFIGURATION - EXACT COPY FROM pai_yolo_training.py
# ============================================================================
def setup_pai_config(use_perforated_bp: bool = False):
    """
    Configure PAI IDENTICALLY to training.
    MUST match pai_yolo_training.py exactly!
    """
    print("\n" + "=" * 60)
    print("  PAI CONFIGURATION (matching training)")
    print("=" * 60)
    
    # CRITICAL: Set Open Source GD mode (not PerforatedBP)
    if hasattr(GPA.pc, 'set_perforated_backpropagation'):
        GPA.pc.set_perforated_backpropagation(use_perforated_bp)
        if use_perforated_bp:
            print("  Perforated Backpropagation: ENABLED")
        else:
            print("  Perforated Backpropagation: DISABLED (Open Source GD)")
    
    # Disable testing mode
    if hasattr(GPA.pc, 'set_testing_dendrite_capacity'):
        GPA.pc.set_testing_dendrite_capacity(False)
    
    # Enable SafeTensors mode
    if hasattr(GPA.pc, 'set_using_safe_tensors'):
        GPA.pc.set_using_safe_tensors(True)
        print("  SafeTensors Mode: ENABLED")
    
    # Silence warnings
    if hasattr(GPA.pc, 'set_unwrapped_modules_confirmed'):
        GPA.pc.set_unwrapped_modules_confirmed(True)
    if hasattr(GPA.pc, 'set_weight_decay_accepted'):
        GPA.pc.set_weight_decay_accepted(True)
    if hasattr(GPA.pc, 'set_verbose'):
        GPA.pc.set_verbose(False)
    
    # Main feature extraction blocks - ADD DENDRITES TO THESE (same as training)
    main_blocks_to_convert = ['C3k2', 'C3k', 'C2PSA', 'Bottleneck', 'PSABlock']
    if hasattr(GPA.pc, 'append_module_names_to_convert'):
        GPA.pc.append_module_names_to_convert(main_blocks_to_convert)
        print(f"  Module Types to CONVERT: {main_blocks_to_convert}")
    
    # Normalization/activation layers - TRACK but don't add dendrites
    layers_to_track = ['BatchNorm2d', 'SiLU', 'Identity', 'Upsample', 'MaxPool2d', 'Concat']
    if hasattr(GPA.pc, 'append_module_names_to_track'):
        GPA.pc.append_module_names_to_track(layers_to_track)
    
    # BatchNorm tracking
    if hasattr(GPA.pc, 'append_module_names_to_track'):
        GPA.pc.append_module_names_to_track(["BatchNorm2d", "BatchNorm1d", "SyncBatchNorm"])
    
    # Duplicate pointer handling
    if hasattr(GPA.pc, 'set_duplicate_pointer_confirmed'):
        GPA.pc.set_duplicate_pointer_confirmed(True)
    
    print("=" * 60 + "\n")


def initialize_pai_model(model: nn.Module, save_name: str) -> nn.Module:
    """
    Initialize PAI on the model - EXACT COPY from pai_yolo_training.py
    """
    # Get basename for PAI
    save_name_basename = os.path.basename(os.path.abspath(save_name))
    
    print(f"[PAI] Initializing with save_name: '{save_name_basename}'")
    
    # CRITICAL: Auto-detect shared modules (same as training)
    id_to_names = defaultdict(list)
    
    print("[PAI] Scanning model for shared modules...")
    
    # PASS 1: Find all registered submodules (including duplicates)
    for name, module in model.named_modules(remove_duplicate=False):
        id_to_names[id(module)].append(name)
        
        # PASS 2: Check CLASS-LEVEL attributes
        module_class = type(module)
        for attr_name in dir(module_class):
            if attr_name.startswith('_'):
                continue
            try:
                attr_value = getattr(module_class, attr_name)
                if isinstance(attr_value, nn.Module):
                    full_name = f"{name}.{attr_name}" if name else attr_name
                    id_to_names[id(attr_value)].append(full_name)
            except Exception:
                pass
        
    shared_module_names = []
    for mod_id, names in id_to_names.items():
        if len(names) > 1:
            names.sort(key=len)
            duplicates = names[1:]
            formatted_duplicates = ["." + name if not name.startswith(".") else name for name in duplicates]
            shared_module_names.extend(formatted_duplicates)

    if shared_module_names:
        if hasattr(GPA.pc, 'append_module_names_to_not_save'):
            GPA.pc.append_module_names_to_not_save(shared_module_names)
            print(f"[PAI] Configured {len(shared_module_names)} shared modules to ignore")
    
    # CRITICAL: Exclude detection head layers (SAME AS TRAINING)
    detection_head_modules = [
        '.model.23', '.model.23.dfl', '.model.23.dfl.conv',
        '.model.23.cv2', '.model.23.cv2.0', '.model.23.cv2.1', '.model.23.cv2.2',
        '.model.23.cv2.0.0', '.model.23.cv2.0.1', '.model.23.cv2.0.2',
        '.model.23.cv2.1.0', '.model.23.cv2.1.1', '.model.23.cv2.1.2',
        '.model.23.cv2.2.0', '.model.23.cv2.2.1', '.model.23.cv2.2.2',
        '.model.23.cv3', '.model.23.cv3.0', '.model.23.cv3.1', '.model.23.cv3.2',
        '.model.23.cv3.0.0', '.model.23.cv3.0.1',
        '.model.23.cv3.1.0', '.model.23.cv3.1.1',
        '.model.23.cv3.2.0', '.model.23.cv3.2.1',
        '.model.23.cv3.0.0.0', '.model.23.cv3.0.0.1',
        '.model.23.cv3.0.1.0', '.model.23.cv3.0.1.1',
        '.model.23.cv3.1.0.0', '.model.23.cv3.1.0.1',
        '.model.23.cv3.1.1.0', '.model.23.cv3.1.1.1',
        '.model.23.cv3.2.0.0', '.model.23.cv3.2.0.1',
        '.model.23.cv3.2.1.0', '.model.23.cv3.2.1.1',
        '.model.9',  # SPPF
    ]
    
    if hasattr(GPA.pc, 'append_module_ids_to_track'):
        GPA.pc.append_module_ids_to_track(detection_head_modules)
        print(f"[PAI] Excluding {len(detection_head_modules)} detection head modules")
    
    # Initialize PAI
    model = UPA.initialize_pai(
        model,
        save_name=save_name_basename,
        making_graphs=False,
        maximizing_score=True
    )
    
    print("[PAI] Model initialized successfully")
    return model


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("  PAI BEST MODEL TEST EVALUATION")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check PAI save exists
    pai_folder = Path(PAI_SAVE_NAME)
    pai_file = pai_folder / "best_model.pt"
    print(f"\nPAI model path: {pai_file}")
    if not pai_file.exists():
        print(f"❌ ERROR: File not found!")
        print(f"   Looking for: {pai_file.absolute()}")
        sys.exit(1)
    print(f"✅ File exists ({pai_file.stat().st_size:,} bytes)")
    
    # Check dataset YAML
    if not Path(DATA_YAML).exists():
        print(f"❌ ERROR: Dataset YAML not found: {DATA_YAML}")
        sys.exit(1)
    
    # Step 1: Create fresh YOLO model
    print("\n[1/4] Creating fresh YOLO model...")
    yolo = YOLO('yolo11n.pt')
    model = copy.deepcopy(yolo.model)
    model.train()  # Enable training mode for PAI init
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    
    # Step 2: Configure PAI (SAME AS TRAINING!)
    print("[2/4] Configuring PAI (same as training)...")
    setup_pai_config(use_perforated_bp=False)  # Open Source GD
    
    # Step 3: Initialize PAI (SAME AS TRAINING!)
    print("[3/4] Initializing PAI...")
    model = initialize_pai_model(model, PAI_SAVE_NAME)
    model = model.to(device)
    print("   ✅ PAI initialized")
    
    # Step 4: Load PAI's best_model using load_system
    print(f"[4/4] Loading PAI best model...")
    try:
        model = UPA.load_system(model, PAI_SAVE_NAME, 'best_model', switch_call=True)
        model = model.to(device)
        print("   ✅ PAI best model loaded!")
    except Exception as e:
        print(f"   ❌ Error loading PAI model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Put model into eval mode
    model.eval()
    
    # Prevent YOLO from fusing the model (would break PAI structure)
    model.is_fused = lambda: True
    
    # Put back into YOLO wrapper
    yolo.model = model
    
    # Run test evaluation using YOLO's val
    print(f"\n" + "="*60)
    print("  RUNNING TEST EVALUATION")
    print("="*60)
    print(f"Dataset: {DATA_YAML}")
    print(f"Split: test")
    print()
    
    try:
        test_results = yolo.val(
            data=DATA_YAML,
            imgsz=IMGSZ,
            split='test',
            plots=False,
            save=False,
            verbose=True
        )
        
        # Results
        print("\n" + "="*60)
        print("  TEST RESULTS (PAI Dendritic Model)")
        print("="*60)
        print(f"  Test mAP@0.5:      {test_results.box.map50:.4f}")
        print(f"  Test mAP@0.5:0.95: {test_results.box.map:.4f}")
        print(f"  Test Precision:    {test_results.box.mp:.4f}")
        print(f"  Test Recall:       {test_results.box.mr:.4f}")
        print("="*60 + "\n")
        
        # Save results
        results_file = pai_folder / "test_evaluation_results.txt"
        with open(results_file, 'w') as f:
            f.write("PAI Dendritic Model Test Results\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {pai_file}\n")
            f.write(f"Test mAP@0.5: {test_results.box.map50:.4f}\n")
            f.write(f"Test mAP@0.5:0.95: {test_results.box.map:.4f}\n")
            f.write(f"Test Precision: {test_results.box.mp:.4f}\n")
            f.write(f"Test Recall: {test_results.box.mr:.4f}\n")
        print(f"✅ Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
