"""
Test PAISequential Wrapping Implementation
==========================================
Verifies that Conv+BatchNorm sequences are properly wrapped before PAI initialization.
"""
import os
os.environ["PAIEMAIL"] = "hacker@perforatedai.com"
os.environ["PAITOKEN"] = "InJ9BjZSB+B+l30bmSzhqOwsXxOx0NRKAe8dtdAqdQcT/pKjmme1fqB1zrnCd5CWNrhJm40PVjaDbIrjR5xU+q2uhcUWX8gk2Kb2lHjafkUnizPXyP+yckbv+UxlU25ZlrvC3XlLu/AZdVKJE7Eov9+4c76sKe2hbRnH1fny2xIPYmy2/m/sY1gxXbhPtTa1mtxk2EgLeo5pRu/eL/7pSXWmEoRmvVorgQEJzt1VYOZyp0vP4bLxF72tOgSjXGBO8SHHcN16CbOVJuIEm3jmEc/AfPyyB+G4TEqhH7UZ0W2R/bnXtNberKqF2bQTuyT26etQw6NEMoXwuugDcrBXEw=="
os.environ["PYTHONBREAKPOINT"] = "0"

import torch
import torch.nn as nn
from ultralytics import YOLO
from perforatedai import globals_perforatedai as GPA

# Import our wrapping function
from pai_yolo_training import wrap_normalization_layers_in_sequentials

def inspect_module_structure(model, name, module, indent=0):
    """Recursively inspect module structure"""
    prefix = "  " * indent
    module_type = type(module).__name__
    
    # Check if it's a PAISequential
    if isinstance(module, GPA.PAISequential):
        print(f"{prefix}✅ {name}: PAISequential")
        # Show what's inside
        if hasattr(module, 'model'):
            for i, child in enumerate(module.model):
                print(f"{prefix}    └─ [{i}] {type(child).__name__}")
    # Check if it's a Conv with internal structure
    elif hasattr(module, 'conv') and hasattr(module, 'bn'):
        conv_type = type(module.conv).__name__
        bn_type = type(module.bn).__name__
        print(f"{prefix}{name}: {module_type} (conv={conv_type}, bn={bn_type})")

def main():
    print("=" * 80)
    print("  TESTING PAISEQUENTIAL WRAPPING")
    print("=" * 80)
    
    # Load YOLO model
    print("\n1. Loading YOLOv11n model...")
    yolo = YOLO('yolo11n.pt')
    model = yolo.model
    
    print("\n2. BEFORE wrapping - Sample Conv modules:")
    for name, module in list(model.named_modules())[:20]:
        if 'Conv' in type(module).__name__:
            inspect_module_structure(model, name, module)
    
    print("\n3. Applying PAISequential wrapping...")
    model = wrap_normalization_layers_in_sequentials(model)
    
    print("\n4. AFTER wrapping - Sample wrapped modules:")
    for name, module in list(model.named_modules())[:50]:
        if 'Conv' in type(module).__name__ or isinstance(module, GPA.PAISequential):
            inspect_module_structure(model, name, module)
    
    print("\n5. Verifying PAISequential presence...")
    pai_sequential_count = sum(1 for m in model.modules() if isinstance(m, GPA.PAISequential))
    print(f"   Found {pai_sequential_count} PAISequential modules in the model")
    
    if pai_sequential_count > 0:
        print("\n✅ SUCCESS: Conv+BatchNorm sequences are wrapped in PAISequential")
        print("   This implements PAI best practice from customization.md §2.1")
    else:
        print("\n❌ WARNING: No PAISequential modules found - wrapping may have failed")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
