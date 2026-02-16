#!/usr/bin/env python3
"""
Test PAI Forward Functions - Quick Validation
==============================================
Tests that sigmoid, relu, and tanh forward functions work with PAI YOLO.

IMPORTANT: Forward functions are ONLY used when a DENDRITE IS ADDED.
So we must run enough epochs and set n_epochs_to_switch=1 to trigger dendrite addition.

Uses same seed (42) and data split as main experiments.
"""

import os
import sys

os.environ["PYTHONBREAKPOINT"] = "0"

import torch
import numpy as np
import random
from datetime import datetime

# ==============================================================================
# CONFIGURATION - Same as main experiments
# ==============================================================================
SEED = 42  # SAME SEED as all experiments
BATCH_SIZE = 32
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data - Use existing baseline_100 data subset (same split)
# Data - Use baseline_50 data subset (50% data) as requested
DATA_YAML = "runs/data_efficiency/baseline_50/data_subset.yaml"
FALLBACK_DATA_YAML = "VOC2007.yaml"

# Test config - Must trigger dendrite addition!
TEST_EPOCHS = 5  # Run 5 epochs: dendrite added after epoch 1, then train with it
N_EPOCHS_TO_SWITCH = 1  # Add dendrite after epoch 1


def set_seed(seed=42):
    """Set all random seeds for reproducibility - SAME as main experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] Set all random seeds to {seed}")


def test_forward_function(fwd_func_name: str):
    """
    Test a single PAI forward function.
    
    CRITICAL: Forward functions only apply when dendrites are ADDED.
    We set n_epochs_to_switch=1 to force PAI to consider adding dendrites
    after every epoch, ensuring the forward function is actually used.
    
    Args:
        fwd_func_name: 'sigmoid', 'relu', or 'tanh'
    
    Returns:
        True if successful, False if error
    """
    print(f"\n{'='*70}")
    print(f"  TESTING PAI FORWARD FUNCTION: {fwd_func_name.upper()}")
    print(f"  (n_epochs_to_switch=1 forces dendrite check every epoch)")
    print(f"{'='*70}")
    
    # Reset seed for each test (same starting point)
    set_seed(SEED)
    
    # Determine data YAML
    if os.path.exists(DATA_YAML):
        data_yaml = DATA_YAML
        print(f"[Data] Using: {DATA_YAML}")
    elif os.path.exists(FALLBACK_DATA_YAML):
        data_yaml = FALLBACK_DATA_YAML
        print(f"[Data] Using fallback: {FALLBACK_DATA_YAML}")
    else:
        print(f"[ERROR] Data YAML not found!")
        return False
    
    try:
        # Import training function
        from pai_yolo_training import train_pai_yolo
        
        # Create unique save directory for this test
        save_name = f"test_fwd_func/{fwd_func_name}_{datetime.now().strftime('%H%M%S')}"
        os.makedirs(save_name, exist_ok=True)
        
        print(f"[Config] Forward Function: {fwd_func_name}")
        print(f"[Config] Save Dir: {save_name}")
        print(f"[Config] Epochs: {TEST_EPOCHS}")
        print(f"[Config] n_epochs_to_switch: {N_EPOCHS_TO_SWITCH} (FIXED mode - adds dendrite at epoch 1)")
        print(f"[Config] Mode: FIXED (guaranteed dendrite addition)")
        print(f"[Config] Device: {DEVICE}")
        
        # Run training with this forward function
        # KEY: switch_mode='doing_fixed' GUARANTEES dendrite addition at epoch 1
        trained_model, best_val_map50, test_map50 = train_pai_yolo(
            data_yaml=data_yaml,
            epochs=TEST_EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMGSZ,
            device=DEVICE,
            lr=0.0005,  # Same as baseline sweep
            warmup_epochs=0,
            save_name=save_name,
            seed=SEED,
            # PAI params - FORCE DENDRITE ADDITION WITH FIXED MODE
            switch_mode='DOING_FIXED',  # CRITICAL: Forces dendrite at epoch n_epochs_to_switch
            n_epochs_to_switch=N_EPOCHS_TO_SWITCH,  # Add dendrite after epoch 1
            p_epochs_to_switch=1,
            max_dendrites=2,
            early_stop_patience=10,
            use_pai=True,
            use_perforated_bp=False,  # Open Source GD
            history_lookback=1,  # Not used in FIXED mode
            improvement_threshold=[0.5, 0.1, 0],  # Not used in FIXED mode
            candidate_weight_init=0.01,
            pai_forward_function=fwd_func_name,  # TEST THIS!
            data_fraction=1.0,
            scheduler_patience=3,
        )
        
        print(f"\n[SUCCESS] ✅ {fwd_func_name.upper()} forward function works!")
        print(f"  Val mAP50: {best_val_map50:.4f}")
        print(f"  Test mAP50: {test_map50:.4f}")
        return True
        
    except Exception as e:
        print(f"\n[FAILED] ❌ {fwd_func_name.upper()} forward function FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("  PAI FORWARD FUNCTION TEST")
    print("  Testing: sigmoid, relu, tanh")
    print("  Seed: 42 (same as all experiments)")
    print("="*70)
    
    forward_functions = ['sigmoid', 'relu', 'tanh']
    results = {}
    
    for fwd_func in forward_functions:
        results[fwd_func] = test_forward_function(fwd_func)
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for fwd_func, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {fwd_func.upper():10s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 ALL FORWARD FUNCTIONS WORK CORRECTLY!")
        print("   You can safely use sigmoid, relu, or tanh in the sweep.\n")
        return 0
    else:
        print("\n⚠️ SOME FORWARD FUNCTIONS FAILED!")
        print("   Check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
