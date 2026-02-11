#!/usr/bin/env python3
"""
Baseline Parameter Sweep - Full Training from Scratch
======================================================
Trains YOLO from scratch on 100% VOC2007 data WITHOUT PAI (Baseline).
Uses W&B Bayesian optimization to find best baseline configuration.

Key Features:
- Full training from scratch (no PAI)
- 100% VOC2007 data
- 500 epochs max, early stopping
- Test evaluation at end of each run
- 10 experiments with Bayesian optimization

Hyperparameters Swept:
- Learning Rate (lr)
- Learning Rate (lr)
- Optimizer (adamw, sgd, adam)
- Weight Decay (weight_decay)
- Warmup Epochs (warmup_epochs)
- Early Stop Patience (early_stop_patience)
- Scheduler Patience (scheduler_patience)
- Scheduler Factor (scheduler_factor)

Fixed Parameters:
- Batch Size: 32

Usage:
    # Create new sweep
    python sweep_baseline.py --create-sweep
    
    # Run sweep agent
    python sweep_baseline.py --sweep-id <SWEEP_ID> --count 10
"""

import os
import sys
import argparse
import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import random

# W&B
import wandb

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEED = 42
IMGSZ = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data - Use the EXACT same dataset as run_experiments.py baseline_100
DATA_YAML = "runs/data_efficiency/baseline_100/data_subset.yaml"
# Fallback if doesn't exist - use main VOC YAML
FALLBACK_DATA_YAML = "VOC2007.yaml"

# Training (not swept - these are fixed or have a different mechanism)
MAX_EPOCHS = 500
BATCH_SIZE = 32  # Fixed batch size


# ==============================================================================
# SWEEP CONFIGURATION
# ==============================================================================
def get_sweep_config():
    """
    Return W&B Bayesian sweep configuration for baseline training.
    
    These parameters match the training loop in pai_yolo_training.py.
    We're sweeping the key hyperparameters that affect baseline YOLO performance.
    
    FIXED (not swept):
    - batch_size: 32 (same as default baseline)
    """
    return {
        "method": "bayes",
        "metric": {
            "name": "val/best_mAP50",
            "goal": "maximize"
        },
        "parameters": {
            # ============ LEARNING RATE ============
            # Range based on common YOLO training values
            # Original baseline uses 0.005
            # ============ LEARNING RATE ============
            # Range based on common YOLO training values
            # Only higher LRs (0.005+) for better convergence
            "learning_rate": {
                "values": [0.02, 0.01, 0.005]
            },
            
            # ============ OPTIMIZER TYPE ============
            # Different optimizer algorithms
            "optimizer": {
                "values": ["adamw", "sgd", "adam"]
            },
            
            # ============ WEIGHT DECAY ============
            # Regularization parameter (critical for AdamW vs SGD)
            # SGD default: 0.0005, AdamW default: 0.01
            "weight_decay": {
                "values": [0.0001, 0.0005, 0.005, 0.01]
            },
            
            # ============ WARMUP EPOCHS ============
            # How many epochs of gradual LR warmup
            "warmup_epochs": {
                "values": [0, 2, 3, 5]
            },
            
            # ============ EARLY STOPPING PATIENCE ============
            # Epochs to wait without improvement before stopping
            "early_stop_patience": {
                "values": [10, 15, 20]
            },
            
            # ============ SCHEDULER PATIENCE ============
            # Epochs to wait before reducing LR (ReduceLROnPlateau)
            "scheduler_patience": {
                "values": [3, 5, 7, 10]
            },
            
            # ============ SCHEDULER FACTOR ============
            # Factor to reduce LR by when plateau detected
            "scheduler_factor": {
                "values": [0.1, 0.2, 0.5]
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
    Run a full baseline training experiment with sweep config.
    
    Each sweep run:
    1. Initializes a NEW WandB run (automatic with wandb.agent)
    2. Uses swept hyperparameters
    3. Trains baseline YOLO (no PAI)
    4. Evaluates on test set
    5. Logs all metrics to WandB
    """
    with wandb.init(config=config) as run:
        config = wandb.config
        
        # Ensure correct working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Determine data YAML to use
        if os.path.exists(DATA_YAML):
            data_yaml = DATA_YAML
        elif os.path.exists(FALLBACK_DATA_YAML):
            data_yaml = FALLBACK_DATA_YAML
            print(f"[Warning] Using fallback data YAML: {FALLBACK_DATA_YAML}")
        else:
            print(f"ERROR: No data YAML found at {DATA_YAML} or {FALLBACK_DATA_YAML}")
            print("Please run baseline_100 experiment first via run_experiments.py")
            print("Or ensure VOC2007.yaml exists in the project root")
            raise FileNotFoundError(f"Data YAML not found")
        
        print("\n" + "="*70)
        print("  BASELINE SWEEP - FULL TRAINING")
        print("="*70)
        print(f"  Run ID: {run.id}")
        print(f"  Run Name: {run.name}")
        print(f"  Working Dir: {os.getcwd()}")
        print(f"  Data YAML: {data_yaml}")
        print(f"  Config: {dict(config)}")
        print("="*70 + "\n")
        
        # Set seed (Fixed to 42)
        set_global_seed(SEED)
        
        # ========== STEP 1: Log Configuration ==========
        print("[SWEPT] Hyperparameters for this run:")
        print(f"  Learning Rate: {config.learning_rate}")
        print(f"  Optimizer: {config.optimizer}")
        print(f"  Weight Decay: {config.weight_decay}")
        print(f"  Warmup Epochs: {config.warmup_epochs}")
        print(f"  Early Stop Patience: {config.early_stop_patience}")
        print(f"  Scheduler Patience: {config.scheduler_patience}")
        print(f"  Scheduler Factor: {config.scheduler_factor}")
        print(f"[FIXED] Batch Size: {BATCH_SIZE}")
        
        # ========== STEP 2: Setup Custom Training ==========
        # We'll use a modified version of the training function
        # that accepts our swept parameters
        
        save_name = f"sweep_baseline_runs/{run.id}"
        os.makedirs(save_name, exist_ok=True)
        
        print(f"\n[Training] Starting full training...")
        print(f"  Data: {data_yaml}")
        print(f"  Save: {save_name}")
        
        # Import training function
        # NOTE: We avoid importing set_seed here to prevent conflict with local function
        from pai_yolo_training import (
            extract_yolo_model, check_gradients,
            preprocess_batch, validate, EarlyStopper
        )
        from ultralytics import YOLO
        from ultralytics.data import build_yolo_dataset, build_dataloader
        from ultralytics.data.utils import check_det_dataset
        from types import SimpleNamespace
        from tqdm import tqdm
        
        # ========== Load Model ==========
        print("[Model] Loading pretrained: yolo11n.pt")
        yolo = YOLO('yolo11n.pt')
        model = extract_yolo_model(yolo)
        model = model.to(DEVICE)
        check_gradients(model)
        
        # ========== Setup Data Loaders ==========
        print(f"[Data] Loading dataset from: {data_yaml}")
        data_dict = check_det_dataset(data_yaml)
        
        # Augmentation config matching baseline
        AUGMENTATION_CONFIG = {
            'mosaic': 1.0,
            'mixup': 0.15,
            'copy_paste': 0.3,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'fliplr': 0.5,
            'flipud': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'erasing': 0.4,
        }
        
        default_cfg = {
            'cache': None,
            'single_cls': False,
            'classes': None,
            'fraction': 1.0,
            'task': 'detect',
            'rect': False,
            'imgsz': IMGSZ,
            'project': 'runs',
            'name': 'exp',
            'close_mosaic': 0,
            'cutmix': 0.0,
            'crop_fraction': 1.0,
            'auto_augment': None,
            'mask_ratio': 4,
            'overlap_mask': True,
            'bgr': 0.0,
            'copy_paste_mode': 'flip',
        }
        
        train_cfg_dict = default_cfg.copy()
        train_cfg_dict.update(AUGMENTATION_CONFIG)
        train_cfg_dict['rect'] = False
        train_cfg = SimpleNamespace(**train_cfg_dict)
        
        train_dataset = build_yolo_dataset(
            cfg=train_cfg,
            img_path=data_dict.get('train', ''),
            batch=BATCH_SIZE,
            data=data_dict,
            mode='train'
        )
        
        train_loader = build_dataloader(
            train_dataset,
            batch=BATCH_SIZE,
            workers=8,
            shuffle=True,
            pin_memory=True
        )
        
        print(f"  Train batches: {len(train_loader)}")
        
        # ========== Setup Optimizer with SWEPT parameters ==========
        optimizer_name = config.optimizer.lower()
        print(f"[Optimizer] Using {optimizer_name.upper()} + ReduceLROnPlateau")
        
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=config.learning_rate, 
                weight_decay=config.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=config.learning_rate, 
                momentum=0.937,  # Standard YOLO momentum
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience
        )
        print(f"[Scheduler] ReduceLROnPlateau: patience={config.scheduler_patience}, factor={config.scheduler_factor}")
        
        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if DEVICE == 'cuda' else None
        if scaler:
            print("[AMP] Using CUDA mixed precision")
        
        # ========== Early Stopping ==========
        early_stopper = EarlyStopper(patience=config.early_stop_patience)
        print(f"[EarlyStopping] Patience: {config.early_stop_patience} epochs")
        
        # ========== Warmup Function ==========
        def get_warmup_factor(epoch):
            if epoch < config.warmup_epochs:
                return (epoch + 1) / config.warmup_epochs
            return 1.0
        
        print(f"[Warmup] Using {config.warmup_epochs} warmup epochs")
        
        # ========== Training Loop ==========
        best_score = float('-inf')
        best_model_state = None
        best_epoch = 0
        
        print(f"\n{'=' * 60}")
        print(f"  STARTING TRAINING: {MAX_EPOCHS} epochs max")
        print(f"{'=' * 60}\n")
        
        for epoch in range(1, MAX_EPOCHS + 1):
            # Apply warmup
            warmup_factor = get_warmup_factor(epoch - 1)
            if warmup_factor < 1.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.learning_rate * warmup_factor
            
            # Train one epoch
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                batch = preprocess_batch(batch, torch.device(DEVICE))
                optimizer.zero_grad(set_to_none=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        result = model(batch)
                        if isinstance(result, tuple):
                            loss = result[0]
                            if loss.dim() > 0:
                                loss = loss.sum()
                        else:
                            loss = result
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    result = model(batch)
                    if isinstance(result, tuple):
                        loss = result[0]
                        if loss.dim() > 0:
                            loss = loss.sum()
                    else:
                        loss = result
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
            
            pbar.close()  # Ensure progress bar closes before validation prints
            avg_loss = total_loss / max(num_batches, 1)
            
            # Validate
            yolo.model = model
            val_metrics = validate(yolo, data_yaml, IMGSZ, split='val')
            val_score = val_metrics['map50']
            
            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "val/mAP50": val_score,
                "val/mAP50-95": val_metrics['map'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "lr": optimizer.param_groups[0]['lr']
            })
            
            # Update scheduler
            if warmup_factor >= 1.0:
                scheduler.step(val_score)
            
            # Track best model
            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'best_map50': best_score,
                }, f"{save_name}/best_model.pt")
            
            # Early stopping check
            if early_stopper(val_score, epoch):
                print(f"\n[Early Stop] No improvement for {config.early_stop_patience} epochs")
                print(f"  Best mAP50: {best_score:.4f} at epoch {best_epoch}")
                break
            
            # Print epoch summary (Tabular style like pai_yolo_training.py)
            print(f"\n{'─' * 66}")
            print(f"  Epoch {epoch}/{MAX_EPOCHS} Summary")
            print(f"  {'─' * 66}")
            print(f"  {'Metric':<20} {'Value':>12} {'Best':>15}")
            print(f"  {'─' * 50}")
            print(f"  {'Train Loss':<20} {avg_loss:>12.4f} {'':>15}")
            print(f"  {'Val mAP@0.5':<20} {val_score:>12.4f} {best_score:>15.4f}")
            
            # Show star if new best
            if val_score == best_score:
                print(f"  {'':<20} {'⭐️ NEW BEST':>12} {'':>15}")
            
            print(f"  {'Learning Rate':<20} {optimizer.param_groups[0]['lr']:>12.2e} {'':>15}")
            print(f"  {'─' * 66}\n")
        
        # Load best model for test evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # ========== STEP 3: Test Evaluation ==========
        print(f"\n[Test] Running test evaluation...")
        
        test_map50 = 0.0
        try:
            yolo.model = model
            model.is_fused = lambda: True
            
            results = yolo.val(
                data=data_yaml,
                imgsz=IMGSZ,
                split='test',
                verbose=False,
                workers=0,
            )
            test_map50 = float(results.box.map50)
            print(f"  Test mAP50: {test_map50:.4f}")
            
            if hasattr(model, 'is_fused'):
                delattr(model, 'is_fused')
                
        except Exception as e:
            print(f"  ⚠️ Test evaluation failed: {e}")
            test_map50 = 0.0
        
        # ========== STEP 4: Log Final Results ==========
        print(f"\n" + "="*70)
        print(f"  SWEEP RUN COMPLETE")
        print(f"="*70)
        print(f"  Best Val mAP50: {best_score:.4f}")
        print(f"  Best Epoch: {best_epoch}")
        print(f"  Test mAP50: {test_map50:.4f}")
        print(f"  Total Epochs: {epoch}")
        print(f"="*70)
        
        # Log final metrics to W&B
        wandb.log({
            "val/best_mAP50": best_score,
            "test/mAP50": test_map50,
            "best_epoch": best_epoch,
            "total_epochs": epoch,
        })
        
        # Summary
        wandb.summary["val/best_mAP50"] = best_score
        wandb.summary["test/mAP50"] = test_map50
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["total_epochs"] = epoch
        wandb.summary["config/lr"] = config.learning_rate
        wandb.summary["config/optimizer"] = config.optimizer
        wandb.summary["config/weight_decay"] = config.weight_decay
        wandb.summary["config/batch_size"] = BATCH_SIZE
        wandb.summary["config/warmup_epochs"] = config.warmup_epochs
        wandb.summary["config/early_stop_patience"] = config.early_stop_patience
        wandb.summary["config/scheduler_patience"] = config.scheduler_patience
        wandb.summary["config/scheduler_factor"] = config.scheduler_factor


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Baseline Parameter Sweep')
    parser.add_argument('--create-sweep', action='store_true', 
                        help='Create new W&B sweep and print ID')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='W&B sweep ID to join')
    parser.add_argument('--count', type=int, default=15,
                        help='Number of sweep runs (default: 15)')
    parser.add_argument('--test-run', action='store_true',
                        help='Run single test without W&B sweep')
    parser.add_argument('--project', type=str, default='PAI-YOLO-Baseline-Sweep',
                        help='W&B project name')
    args = parser.parse_args()
    
    # Check for data YAML
    if not os.path.exists(DATA_YAML) and not os.path.exists(FALLBACK_DATA_YAML):
        print(f"\n{'='*70}")
        print(f"  WARNING: Data YAML not found!")
        print(f"{'='*70}")
        print(f"  Primary: {DATA_YAML}")
        print(f"  Fallback: {FALLBACK_DATA_YAML}")
        print(f"\n  Run baseline_100 first OR ensure VOC2007.yaml exists:")
        print(f"    python run_experiments.py --experiment baseline_100")
        print(f"{'='*70}\n")
    
    wandb.login()
    
    if args.create_sweep:
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.project)
        print(f"\n{'='*70}")
        print(f"  SWEEP CREATED")
        print(f"{'='*70}")
        print(f"  Sweep ID: {sweep_id}")
        print(f"  Project: {args.project}")
        print(f"\n  Sweep Configuration:")
        print(f"  - Method: Bayesian (bayes)")
        print(f"  - Metric: val/best_mAP50 (maximize)")
        print(f"  - Count: {args.count} experiments")
        print(f"\n  Hyperparameters being swept:")
        print(f"  - learning_rate: [0.01, 0.005, 0.002, 0.001, 0.0005]")
        print(f"  - optimizer: [adamw, sgd, adam]")
        print(f"  - weight_decay: [0.0001, 0.0005, 0.005, 0.01]")
        print(f"  - warmup_epochs: [0, 2, 3, 5]")
        print(f"  - early_stop_patience: [10, 12, 15]")
        print(f"  - scheduler_patience: [3, 5, 7, 10]")
        print(f"  - scheduler_factor: [0.1, 0.2, 0.5]")
        print(f"\n  Fixed parameters:")
        print(f"  - batch_size: 32")
        print(f"\n  To run agents:")
        print(f"  python sweep_baseline.py --sweep-id {sweep_id} --count {args.count}")
        print(f"{'='*70}\n")
        
    elif args.sweep_id:
        print(f"\n[Sweep] Joining sweep: {args.sweep_id}")
        print(f"[Sweep] Running {args.count} experiments...")
        print(f"[Sweep] Each experiment is a separate WandB run\n")
        wandb.agent(args.sweep_id, run_sweep_training, count=args.count, project=args.project)
        
    elif args.test_run:
        print("\n[Test] Running single test with default config...")
        test_config = {
            "learning_rate": 0.005,
            "optimizer": "adamw",
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "early_stop_patience": 12,
            "scheduler_patience": 5,
            "scheduler_factor": 0.1,
        }
        run_sweep_training(test_config)
        
    else:
        print("Usage:")
        print("  --create-sweep  : Create new W&B sweep")
        print("  --sweep-id ID   : Join existing sweep")
        print("  --test-run      : Single test run")
        print("\nExample workflow:")
        print("  1. python sweep_baseline.py --create-sweep")
        print("  2. python sweep_baseline.py --sweep-id <ID> --count 10")


if __name__ == "__main__":
    main()
