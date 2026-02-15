#!/usr/bin/env python3
"""
Dendrite Parameter Sweep - Full Training from Scratch
======================================================
Trains YOLO from scratch on 100% VOC2007 data with PAI dendrites.
Uses W&B Bayesian optimization to find best dendrite configuration.

Key Features:
- Full training from scratch (no checkpoint loading)
- 100% VOC2007 data (same as baseline_100)
- Open Source GD mode (not PerforatedBP)
- 999 epochs max, early stopping patience 10
- Test evaluation at end of each run using CORRECT PAI load method
- Fixed baseline hyperparameters from sweep, only PAI params swept

FIXED BASELINE HYPERPARAMETERS (from baseline sweep winners):
- learning_rate: 0.005
- optimizer: SGD (momentum=0.937)
- warmup_epochs: 2
- scheduler_patience: 10
- scheduler_factor: 0.2
- weight_decay: 0.005

SWEPT PAI PARAMETERS:
- max_dendrites: [2, 3, 5]
- n_epochs_to_switch: [10, 15, 20]
- history_lookback: [1, 3, 5]
- improvement_threshold: [[0.01, 0.001, 0.0001, 0], [0.001, 0.0001, 0]]
- candidate_weight_init: [0.1, 0.01, 0.001]
- pai_forward_function: [sigmoid, relu, tanh]
- weight_decay: [0, 0.01] (PAI docs recommend sweeping this)

Usage:
    # Create new sweep
    python sweep_dendrite.py --create-sweep
    
    # Run sweep agent
    python sweep_dendrite.py --sweep-id <SWEEP_ID> --count 15
"""

import os
import sys
import argparse
from pathlib import Path

os.environ["PYTHONBREAKPOINT"] = "0"

import torch
import numpy as np
import random

# W&B
import wandb

# PAI Imports (only GPA needed for dendrite count tracking)
from perforatedai import globals_perforatedai as GPA


# ==============================================================================
# CONFIGURATION - FIXED BASELINE HYPERPARAMETERS
# ==============================================================================
SEED = 42
BATCH_SIZE = 32  # Fixed from baseline
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data - Use the EXACT same subset as run_experiments.py baseline_100
DATA_YAML = "runs/data_efficiency/baseline_100/data_subset.yaml"
FALLBACK_DATA_YAML = "VOC2007.yaml"

# Training - FIXED from baseline sweep WINNERS
MAX_EPOCHS = 999
# EARLY_STOP_PATIENCE is now swept: [40, 50]
FIXED_LR = 0.005              # From baseline sweep winner ✓
FIXED_OPTIMIZER = 'sgd'       # From baseline sweep winner ✓
FIXED_WEIGHT_DECAY = 0.005    # From baseline sweep winner ✓
WARMUP_EPOCHS = 2             # From baseline sweep winner ✓
SCHEDULER_PATIENCE = 10       # From baseline sweep winner ✓
SCHEDULER_FACTOR = 0.2        # From baseline sweep winner ✓

# PAI mode
USE_PERFORATED_BP = False  # Open Source GD (not PerforatedBP)

# ==============================================================================
# SWEEP PARAMETER RANGES (PAI-specific parameters to optimize)
# ==============================================================================
SWEEP_MAX_DENDRITES = [2, 3, 4, 5]
SWEEP_N_EPOCHS_TO_SWITCH = [8, 12, 15, 20, 22]
SWEEP_HISTORY_LOOKBACK = [1, 3, 5, 6]
SWEEP_IMPROVEMENT_THRESHOLD = [0, 1, 2, 3]  # Maps to threshold lists
SWEEP_CANDIDATE_WEIGHT_INIT = [0.01, 0.005, 0.001, 0.0005]
SWEEP_CANDIDATE_INIT_BY_MAIN = [True, False]  # Adaptive vs Fixed
SWEEP_PAI_FORWARD_FUNCTION = [0, 1, 2]  # 0=sigmoid, 1=relu, 2=tanh
SWEEP_EARLY_STOP_PATIENCE = [50, 60]
SWEEP_POST_DENDRITE_SCHEDULER_PATIENCE = [8, 10, 12, 15, 17]
SWEEP_DENDRITE_LR_OPTION = [0, 1]  # 0=base_lr, 1=base_lr/2


# ==============================================================================
# SWEEP CONFIGURATION
# ==============================================================================
def get_sweep_config():
    """
    Return W&B Bayesian sweep configuration.
    
    IMPORTANT: Only PAI-specific params are swept!
    Training params (LR, epochs, etc.) are FIXED to match baseline_100 sweep.
    This ensures fair comparison - only dendrite config varies.
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "val/best_mAP50",
            "goal": "maximize"
        },
        "parameters": {
            # ============ PAI-ONLY PARAMETERS (SWEPT) ============
            
            # Maximum number of dendrites to add
            "max_dendrites": {
                "values": SWEEP_MAX_DENDRITES
            },
            
            # How many epochs before switching (for PAI restructuring)
            "n_epochs_to_switch": {
                "values": SWEEP_N_EPOCHS_TO_SWITCH
            },
            
            # History lookback for improvement detection
            "history_lookback": {
                "values": SWEEP_HISTORY_LOOKBACK
            },
            
            # Improvement threshold - speed of improvement required before adding dendrite
            # Higher values = more patience (waits for smaller improvements before adding dendrite)
            # 0 = [0.01, 0.001, 0.0001, 0] - strict (quickly adds dendrites)
            # 1 = [0.001, 0.0001, 0]       - moderate
            # 2 = [0.005, 0.001, 0.0005, 0] - balanced
            # 3 = [0.0001, 0]              - lenient (slow to add dendrites)
            "improvement_threshold": {
                "values": SWEEP_IMPROVEMENT_THRESHOLD
            },
            
            # Multiplier to initialize dendrite weights
            # Lower = gentler integration, higher = faster adaptation
            "candidate_weight_init": {
                "values": SWEEP_CANDIDATE_WEIGHT_INIT
            },
            
            # Adaptive dendrite initialization (scales with main layer weights)
            # False = Fixed scale for all layers (current approach)
            # True = Adaptive scale (dendrite_init = candidate_weight_init * avg_abs(layer_weights))
            # RECOMMENDED for YOLO: Different layers (backbone/neck/head) have different weight scales
            "candidate_weight_init_by_main": {
                "values": SWEEP_CANDIDATE_INIT_BY_MAIN
            },
            
            # Forward function for dendrites
            # 0=sigmoid, 1=relu, 2=tanh
            "pai_forward_function": {
                "values": SWEEP_PAI_FORWARD_FUNCTION
            },
            
            # ============ EARLY STOPPING PATIENCE (SWEPT) ============
            "early_stop_patience": {
                "values": SWEEP_EARLY_STOP_PATIENCE
            },
            
            # ============ POST-DENDRITE SCHEDULER PATIENCE (SWEPT) ============
            # Pre-dendrite scheduler patience is FIXED at 10 (SCHEDULER_PATIENCE)
            # Post-dendrite should be >= pre-dendrite to allow recovery after dendrite addition
            # Higher values = more patience for the model to recover from dendrite disruption
            # 
            # OPTIMIZED FOR YOLO DENDRITE RECOVERY:
            # - 10: Baseline (same as pre-dendrite) - tests if patience should change
            # - 12: +2 epochs recovery (+20% more time) - gentle increase
            # - 15: +5 epochs recovery (+50% more time) - moderate increase
            # - 18: +8 epochs recovery (+80% more time) - aggressive recovery support
            # 
            # SAFETY: With n_epochs_to_switch=[8,12,15,20], these values ensure:
            # - When n_epochs=15+, patience=10-12 leaves 3-5 epoch buffer before next dendrite
            # - When n_epochs=20, patience=15-18 still leaves 2-5 epoch buffer
            # - Bayesian optimizer will learn which combinations work best
            "post_dendrite_scheduler_patience": {
                "values": SWEEP_POST_DENDRITE_SCHEDULER_PATIENCE
            },
            
            # ============ POST-DENDRITE LR (SWEPT) ============
            # Tests different learning rates after dendrite addition
            # 0 = None (use base_lr=0.005) - standard approach
            # 1 = 0.0025 (base_lr/2) - conservative, may help recovery
            "dendrite_lr_option": {
                "values": SWEEP_DENDRITE_LR_OPTION
            },
        },
    }


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def set_global_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def run_sweep_training(config=None):
    """
    Run a full training experiment with sweep config.
    """
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Ensure correct working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Determine data YAML
        if os.path.exists(DATA_YAML):
            data_yaml = DATA_YAML
        elif os.path.exists(FALLBACK_DATA_YAML):
            data_yaml = FALLBACK_DATA_YAML
            print(f"[Warning] Using fallback data YAML: {FALLBACK_DATA_YAML}")
        else:
            print(f"ERROR: Data YAML not found at {DATA_YAML} or {FALLBACK_DATA_YAML}")
            raise FileNotFoundError(f"Data YAML not found")
        
        print("\n" + "="*70)
        print("  DENDRITE SWEEP - FULL TRAINING")
        print("="*70)
        print(f"  Run ID: {run.id}")
        print(f"  Working Dir: {os.getcwd()}")
        print(f"  Data YAML: {data_yaml}")
        print(f"  Config: {dict(config)}")
        print("="*70 + "\n")
        
        # Set seed
        set_global_seed(SEED)
        
        # Import training function
        from pai_yolo_training import train_pai_yolo
        
        # ========== STEP 1: Log Configuration ==========
        print("[FIXED] Baseline hyperparameters:")
        print(f"  LR: {FIXED_LR}")
        print(f"  Optimizer: {FIXED_OPTIMIZER}")
        print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
        print(f"  Weight Decay: {FIXED_WEIGHT_DECAY} (before dendrite, then 0 after)")
        print(f"  Pre-Dendrite Scheduler Patience: {SCHEDULER_PATIENCE} (fixed)")
        print(f"  Scheduler Factor: {SCHEDULER_FACTOR}")
        print(f"  Batch Size: {BATCH_SIZE}")
        print(f"  Max Epochs: {MAX_EPOCHS}")
        
        # Map forward function
        if config.pai_forward_function == 0:
            fwd_func = 'sigmoid'
        elif config.pai_forward_function == 1:
            fwd_func = 'relu'
        else:
            fwd_func = 'tanh'
        
        # Map improvement threshold (controls how fast dendrites are added)
        if config.improvement_threshold == 0:
            thresh = [0.01, 0.001, 0.0001, 0]  # Strict: quickly adds dendrites
        elif config.improvement_threshold == 1:
            thresh = [0.001, 0.0001, 0]         # Moderate
        elif config.improvement_threshold == 2:
            thresh = [0.005, 0.001, 0.0005, 0]  # Balanced
        else:  # 3
            thresh = [0.0001, 0]                # Lenient: slow to add dendrites
        
        # Map dendrite_lr_option to actual value
        # 0 = None (use base_lr = 0.005)
        # 1 = 0.0025 (base_lr/2 - conservative recovery)
        if config.dendrite_lr_option == 0:
            dendrite_lr = None  # Will use base_lr (0.005)
            dendrite_lr_display = "None (use base_lr=0.005)"
        else:  # 1
            dendrite_lr = 0.0025  # Half of base_lr
            dendrite_lr_display = "0.0025 (base_lr/2 - conservative)"
        
        print("\n[SWEPT] PAI parameters:")
        print(f"  Max Dendrites: {config.max_dendrites}")
        print(f"  N Epochs to Switch: {config.n_epochs_to_switch}")
        print(f"  History Lookback: {config.history_lookback}")
        print(f"  Improvement Threshold: {thresh}")
        print(f"  Candidate Weight Init: {config.candidate_weight_init}")
        print(f"  Candidate Weight Init By Main: {config.candidate_weight_init_by_main} ({'Adaptive' if config.candidate_weight_init_by_main else 'Fixed'})")
        print(f"  Forward Function: {fwd_func}")
        print(f"  Early Stop Patience: {config.early_stop_patience}")
        print(f"  Post-Dendrite Scheduler Patience: {config.post_dendrite_scheduler_patience} (swept)")
        print(f"  Post-Dendrite LR: {dendrite_lr_display}")
        
        # ========== STEP 2: Run Full Training ==========
        save_name = f"sweep_dendrite_runs/{run.id}"
        os.makedirs(save_name, exist_ok=True)
        
        print(f"\n[Training] Starting full training...")
        print(f"  Data: {data_yaml}")
        print(f"  Save: {save_name}")
        
        # Run training with FIXED baseline params + SWEPT PAI params
        # NOTE: Now returns 3 values - model, val_score, test_score
        trained_model, best_val_map50, test_map50 = train_pai_yolo(
            data_yaml=data_yaml,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMGSZ,
            device=DEVICE,
            lr=FIXED_LR,  # FIXED from baseline sweep winner (0.005)
            warmup_epochs=WARMUP_EPOCHS,  # FIXED from baseline sweep winner (2)
            save_name=save_name,
            seed=SEED,
            # PAI params (SWEPT)
            n_epochs_to_switch=config.n_epochs_to_switch,
            p_epochs_to_switch=10,  # Not used in Open Source GD
            max_dendrites=config.max_dendrites,
            early_stop_patience=config.early_stop_patience,
            use_pai=True,
            use_perforated_bp=USE_PERFORATED_BP,  # Open Source GD
            history_lookback=config.history_lookback,
            improvement_threshold=thresh,
            candidate_weight_init=config.candidate_weight_init,
            candidate_weight_init_by_main=config.candidate_weight_init_by_main,  # NEW: Adaptive init
            pai_forward_function=fwd_func,
            data_fraction=1.0,  # 100% data
            scheduler_patience=SCHEDULER_PATIENCE,  # Pre-dendrite: fixed at 10
            post_dendrite_scheduler_patience=config.post_dendrite_scheduler_patience,  # SWEPT: post-dendrite
            # CRITICAL: Add missing params to match baseline and working config
            find_best_lr=True,  # From sweep best config: enables PAI auto LR testing
            # SWEPT: dendrite_lr = None (base_lr=0.005) or 0.015 (3x base_lr)
            dendrite_lr=dendrite_lr,  # From sweep: None=0.005 or 0.015
            optimizer_type=FIXED_OPTIMIZER,  # sgd from baseline sweep winner
            scheduler_factor=SCHEDULER_FACTOR,  # 0.2 from baseline sweep winner
            # Weight decay: 0.005 before dendrite, then automatically 0 after (in pai_yolo_training.py)
            weight_decay=FIXED_WEIGHT_DECAY,  # Fixed 0.005 (from baseline sweep winner)
        )
        
        # Test evaluation is NOW DONE INSIDE train_pai_yolo()
        # The returned test_map50 is from CORRECT PAI evaluation (using UPA.load_system)
        print(f"\n[Test] ✅ Test mAP50 (from training): {test_map50:.4f}")
        
        # ========== STEP 4: Get Dendrite Count ==========
        n_dendrites = 0
        try:
            if hasattr(GPA, 'pai_tracker') and hasattr(GPA.pai_tracker, 'member_vars'):
                n_dendrites = GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)
        except:
            pass
        
        # ========== STEP 5: Log Final Results ==========
        print(f"\n" + "="*70)
        print(f"  SWEEP RUN COMPLETE")
        print(f"="*70)
        print(f"  ⭐️ Test mAP50: {test_map50:.4f}")
        print(f"  --------------------------------------------------")
        print(f"  Best Val mAP50: {best_val_map50:.4f}")
        print(f"  Dendrites Added: {n_dendrites}")
        print(f"="*70)
        
        # Log to W&B
        wandb.log({
            "val/best_mAP50": best_val_map50,
            "test/mAP50": test_map50,
            "dendrites/count": n_dendrites,
        })
        
        # Summary
        wandb.summary["test/mAP50"] = test_map50
        wandb.summary["val/best_mAP50"] = best_val_map50
        wandb.summary["dendrites/count"] = n_dendrites
        wandb.summary["config/lr"] = FIXED_LR
        wandb.summary["config/optimizer"] = FIXED_OPTIMIZER
        wandb.summary["config/weight_decay"] = FIXED_WEIGHT_DECAY
        wandb.summary["config/max_dendrites"] = config.max_dendrites
        wandb.summary["config/n_epochs_to_switch"] = config.n_epochs_to_switch
        wandb.summary["config/pai_forward_function"] = fwd_func
        wandb.summary["config/candidate_weight_init"] = config.candidate_weight_init
        wandb.summary["config/candidate_weight_init_by_main"] = config.candidate_weight_init_by_main
        wandb.summary["config/post_dendrite_scheduler_patience"] = config.post_dendrite_scheduler_patience
        wandb.summary["config/early_stop_patience"] = config.early_stop_patience
        wandb.summary["config/dendrite_lr"] = dendrite_lr if dendrite_lr is not None else "base_lr"


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Dendrite Parameter Sweep')
    parser.add_argument('--create-sweep', action='store_true', 
                        help='Create new W&B sweep and print ID')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='W&B sweep ID to join')
    parser.add_argument('--count', type=int, default=15,
                        help='Number of sweep runs (default: 15)')
    parser.add_argument('--test-run', action='store_true',
                        help='Run single test without W&B sweep')
    parser.add_argument('--project', type=str, default='PAI-YOLO-Dendrite-Sweep',
                        help='W&B project name')
    args = parser.parse_args()
    
    # Verify data yaml exists
    if not os.path.exists(DATA_YAML) and not os.path.exists(FALLBACK_DATA_YAML):
        print(f"\n{'='*70}")
        print(f"  ERROR: Data YAML not found!")
        print(f"{'='*70}")
        print(f"  Primary: {DATA_YAML}")
        print(f"  Fallback: {FALLBACK_DATA_YAML}")
        print(f"\n  Please run baseline_100 experiment first:")
        print(f"    python run_experiments.py --experiments baseline_100")
        print(f"{'='*70}\n")
        sys.exit(1)
    
    print(f"\n[Config] Data YAML: {DATA_YAML if os.path.exists(DATA_YAML) else FALLBACK_DATA_YAML}")
    print(f"[Config] Max Epochs: {MAX_EPOCHS}")
    # print(f"[Config] Early Stop Patience: {EARLY_STOP_PATIENCE}") # Now swept
    print(f"[Config] Fixed LR: {FIXED_LR}")
    print(f"[Config] Fixed Optimizer: {FIXED_OPTIMIZER}")
    print(f"[Config] Mode: Open Source GD (Perforated BP: {USE_PERFORATED_BP})")
    
    wandb.login()
    
    if args.create_sweep:
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\n{'='*70}")
        print(f"  SWEEP CREATED")
        print(f"{'='*70}")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {args.project}")
        print(f"  Method: Bayesian")
        print(f"  Metric: val/best_mAP50 (maximize)")
        print(f"  Count: {args.count} experiments")
        print(f"\n  FIXED hyperparameters (from baseline sweep):")
        print(f"  - learning_rate: {FIXED_LR}")
        print(f"  - optimizer: {FIXED_OPTIMIZER}")
        print(f"  - warmup_epochs: {WARMUP_EPOCHS}")
        print(f"  - weight_decay: {FIXED_WEIGHT_DECAY} (pre-dendrite, then 0)")
        print(f"  - pre_dendrite_scheduler_patience: {SCHEDULER_PATIENCE}")
        print(f"  - scheduler_factor: {SCHEDULER_FACTOR}")
        print(f"\n  SWEPT PAI parameters:")
        print(f"  - max_dendrites: {SWEEP_MAX_DENDRITES}")
        print(f"  - n_epochs_to_switch: {SWEEP_N_EPOCHS_TO_SWITCH}")
        print(f"  - history_lookback: {SWEEP_HISTORY_LOOKBACK}")
        print(f"  - improvement_threshold: {SWEEP_IMPROVEMENT_THRESHOLD} (0=strict, 1=moderate, 2=balanced, 3=lenient)")
        print(f"  - candidate_weight_init: {SWEEP_CANDIDATE_WEIGHT_INIT}")
        print(f"  - candidate_weight_init_by_main: {SWEEP_CANDIDATE_INIT_BY_MAIN} (adaptive)")
        print(f"  - pai_forward_function: {SWEEP_PAI_FORWARD_FUNCTION} (0=sigmoid, 1=relu, 2=tanh)")
        print(f"  - early_stop_patience: {SWEEP_EARLY_STOP_PATIENCE}")
        print(f"  - post_dendrite_scheduler_patience: {SWEEP_POST_DENDRITE_SCHEDULER_PATIENCE}")
        print(f"  - dendrite_lr_option: {SWEEP_DENDRITE_LR_OPTION} (0=base_lr, 1=base_lr/2)")
        print(f"\n  To run agents:")
        print(f"  python sweep_dendrite.py --sweep-id {sweep_id} --count {args.count}")
        print(f"{'='*70}\n")
        
    elif args.sweep_id:
        print(f"\n[Sweep] Joining sweep: {args.sweep_id}")
        print(f"[Sweep] Running {args.count} experiments...")
        print(f"[Sweep] Each experiment is a separate WandB run\n")
        wandb.agent(args.sweep_id, run_sweep_training, count=args.count, project=args.project)
        
    elif args.test_run:
        print("\n[Test] Running single test with default config...")
        test_config = {
            "max_dendrites": 3,
            "n_epochs_to_switch": 15,
            "history_lookback": 3,
            "improvement_threshold": 1,  # moderate: [0.001, 0.0001, 0]
            "candidate_weight_init": 0.001,
            "candidate_weight_init_by_main": True,  # adaptive
            "pai_forward_function": 0,  # sigmoid
            "early_stop_patience": 50,
            "post_dendrite_scheduler_patience": 12,
            "dendrite_lr_option": 0,  # base_lr (0.005)
        }
        run_sweep_training(test_config)
        
    else:
        print("Usage:")
        print("  --create-sweep  : Create new W&B sweep")
        print("  --sweep-id ID   : Join existing sweep")
        print("  --test-run      : Single test run")


if __name__ == "__main__":
    main()
