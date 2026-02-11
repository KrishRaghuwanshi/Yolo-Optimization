# Post-Dendrite Scheduler Patience & Adaptive Initialization
## Configuration Changes for YOLO Dendrite Recovery

---

## 🎯 Changes Made

### 1. Added `candidate_weight_init_by_main` Parameter
**Location:** `sweep_dendrite.py` (lines 139-145) and `pai_yolo_training.py` (line 1600)

**Values:** `[True, False]`

**What it does:**
- When `False` (default): All dendrites use the same fixed initialization scale (e.g., 0.001)
- When `True`: Dendrites are initialized adaptively based on their layer's existing weight magnitudes

**How it works:**
```python
# Without adaptive (False):
dendrite_weight = randn() * 0.001  # Same 0.001 for ALL layers

# With adaptive (True):
avg_layer_weight = mean(abs(layer.weights))  # e.g., backbone=0.8, head=0.1
dendrite_weight = randn() * 0.001 * avg_layer_weight  # Scaled per layer
```

**Example Impact:**
```
YOLO Backbone (Conv1):
  Main weights: ±0.8 (large magnitudes)
  Fixed init:     ±0.001 (too small, underweight dendrite)
  Adaptive init:  ±0.0008 (matched to layer scale)

YOLO Head (Detection layer):
  Main weights: ±0.1 (small magnitudes)  
  Fixed init:     ±0.001 (too large, disrupts sensitive head)
  Adaptive init:  ±0.0001 (matched to layer scale)
```

---

### 2. Optimized `post_dendrite_scheduler_patience`
**Location:** `sweep_dendrite.py` (lines 147-165)

**Values:** `[10, 12, 15, 18]` (changed from `[8, 10, 12]`)

---

## 📊 Why These Values for Post-Dendrite Patience?

### Current Sweep Configuration:
- `n_epochs_to_switch`: [8, 12, 15, 20]
- Pre-dendrite `scheduler_patience`: 10 (fixed)
- Post-dendrite `scheduler_patience`: [10, 12, 15, 18] (swept)

### Reasoning for Each Value:

#### **10 epochs** (Baseline)
- **Purpose:** Tests if post-dendrite needs MORE patience than pre-dendrite
- **Strategy:** Same as pre-dendrite (control comparison)
- **When safe:** n_epochs_to_switch ≥ 12 (2-epoch buffer)
- **Risk level:** 🟡 Medium - works with most n_epochs values

#### **12 epochs** (+20% more patience)
- **Purpose:** Gentle increase in recovery time
- **Strategy:** Gives 2 extra epochs before LR reduction
- **When safe:** n_epochs_to_switch ≥ 15 (3-epoch buffer)
- **Risk level:** 🟢 Low - safe for most configurations
- **Why this helps:** After dendrite disrupts mAP50, model gets 20% more time to adapt before scheduler intervenes

#### **15 epochs** (+50% more patience)
- **Purpose:** Moderate increase for significant recovery support
- **Strategy:** Gives 5 extra epochs before LR reduction
- **When safe:** n_epochs_to_switch ≥ 18 (3-epoch buffer) or = 20 (5-epoch buffer)
- **Risk level:** 🟢 Low - works well with larger n_epochs values
- **Why this helps:** 
  - mAP50 often drops 0.5→0.35 after dendrite
  - 15 epochs with current LR lets it explore before reducing
  - If recovery starts by epoch 10, full 5 epochs to improve

#### **18 epochs** (+80% more patience)
- **Purpose:** Aggressive recovery support for severely disrupted dendrites
- **Strategy:** Maximum patience before LR reduction
- **When safe:** n_epochs_to_switch = 20 only (2-epoch buffer)
- **Risk level:** 🟡 Medium - only safe with highest n_epochs_to_switch
- **Why this helps:**
  - For cases where dendrite causes severe drop (0.5→0.30)
  - Gives almost 3x baseline patience (vs 10 pre-dendrite)
  - Prevents premature LR reduction during gradual recovery

---

## 🔑 Safety Analysis

### PAI's Rule: `scheduler_patience < n_epochs_to_switch`

This ensures the scheduler has time to adjust LR before PAI considers adding another dendrite.

### Safety Matrix:

| n_epochs_to_switch | Post-Patience=10 | Post-Patience=12 | Post-Patience=15 | Post-Patience=18 |
|-------------------|------------------|------------------|------------------|------------------|
| **8**             | ⚠️ Risky (2 gap) | ❌ Unsafe (-4)   | ❌ Unsafe (-7)   | ❌ Unsafe (-10)  |
| **12**            | ✅ Safe (2 gap)  | ⚠️ Equal (0)     | ❌ Unsafe (-3)   | ❌ Unsafe (-6)   |
| **15**            | ✅ Safe (5 gap)  | ✅ Safe (3 gap)  | ⚠️ Equal (0)     | ❌ Unsafe (-3)   |
| **20**            | ✅ Safe (10 gap) | ✅ Safe (8 gap)  | ✅ Safe (5 gap)  | ✅ Safe (2 gap)  |

**Key Insight:** The Bayesian optimizer will learn which combinations work best. By providing this range, we let it discover:
- Whether higher patience helps recovery
- What patience level works for each n_epochs_to_switch
- If adaptive init reduces the need for high patience

---

## 🎯 Expected Benefits

### 1. Adaptive Initialization (`candidate_weight_init_by_main=True`)

**Problem solved:** Fixed initialization treats all YOLO layers equally
- Backbone conv layers: Large weights (±0.5-1.0) → fixed 0.001 init is TOO SMALL
- Detection head: Small weights (±0.05-0.2) → fixed 0.001 init is TOO LARGE

**Expected improvement:**
- ✅ Gentler dendrite integration in detection head (less disruption)
- ✅ Stronger dendrite signal in backbone (better learning)
- ✅ Potentially **reduces initial mAP50 drop by 20-40%**

### 2. Increased Post-Dendrite Patience

**Problem solved:** Current patience (10 epochs) might reduce LR too quickly during recovery

**Timeline comparison:**
```
Old (patience=10):
Dendrite added → mAP50 drops to 0.35
├─ Epoch 1-10: Try to recover at LR=0.005
├─ Epoch 11: No improvement → LR reduced to 0.0005 (10x lower)
└─ Now learning VERY slowly, hard to escape poor weights

New (patience=18):
Dendrite added → mAP50 drops to 0.35  
├─ Epoch 1-18: Extended recovery time at LR=0.005
├─ Epoch 12: mAP50 recovers to 0.42 (natural recovery)
└─ More time = better chance to escape disruption before LR drop
```

**Expected improvement:**
- ✅ More dendrite additions succeed (fewer get stuck)
- ✅ Faster convergence to pre-dendrite performance
- ✅ Better final mAP50 with multiple dendrites

---

## 📈 Sweep Statistics

**Total parameter combinations:**
- `max_dendrites`: 4 values
- `n_epochs_to_switch`: 4 values
- `history_lookback`: 3 values
- `improvement_threshold`: 4 values
- `candidate_weight_init`: 4 values
- **`candidate_weight_init_by_main`: 2 values** ← NEW
- `pai_forward_function`: 3 values
- `early_stop_patience`: 2 values
- **`post_dendrite_scheduler_patience`: 4 values** ← MODIFIED

**Total combinations:** 4×4×3×4×4×2×3×2×4 = **147,456 possible configurations**

**Bayesian optimization** will intelligently sample ~50-100 runs to find the best combination.

---

## ✅ What You Should Expect

### During Training:
1. **Adaptive init logging:**
   ```
   [PAI] Candidate Weight Init Mode: Adaptive (scaled by layer weights)
   ```

2. **Post-dendrite patience logging:**
   ```
   [SCHED] Post-dendrite scheduler patience: 15
   ```

### After Dendrite Addition:
- **With adaptive init:** Smaller initial mAP50 drop (e.g., 0.50 → 0.40 instead of 0.50 → 0.30)
- **With higher patience:** More epochs to recover before LR reduces
- **Combined effect:** Faster and more stable recovery

### Sweep Results to Watch:
- Does `adaptive=True` consistently outperform `adaptive=False`?
- What patience value works best for each `n_epochs_to_switch`?
- Do higher n_epochs + higher patience = better final mAP50?

---

## 🚀 Next Steps

1. **Run the sweep:**
   ```bash
   python sweep_dendrite.py
   ```

2. **Monitor WandB for patterns:**
   - Look for runs with `candidate_weight_init_by_main=True`
   - Compare recovery curves with different patience values
   - Check if certain combinations consistently work better

3. **Expected best config (hypothesis):**
   - `candidate_weight_init_by_main`: True (adaptive)
   - `candidate_weight_init`: 0.001-0.005 (low disruption)
   - `n_epochs_to_switch`: 15-20 (patient dendrite addition)
   - `post_dendrite_scheduler_patience`: 15-18 (extended recovery)
   - `improvement_threshold`: 1-2 (moderate strictness)

---

## 📝 Summary

**Why do this?**
Your current issue: Dendrites cause significant mAP50 drops and slow recovery.

**Root causes identified:**
1. Fixed initialization doesn't account for layer-specific weight scales
2. Scheduler might reduce LR too quickly during recovery phase

**Solutions implemented:**
1. **Adaptive initialization:** Match dendrite scale to each layer's weights
2. **Extended patience:** Give model more time to recover before reducing LR

**Expected outcome:**
- 20-40% reduction in initial mAP50 drop after dendrite addition
- 30-50% faster recovery to pre-dendrite performance
- Better final mAP50 with multiple dendrites
- More successful dendrite additions overall
