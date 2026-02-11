"""Minimal structure dump - no PAI needed"""
import os, sys
os.environ["PYTHONBREAKPOINT"] = "0"
from ultralytics import YOLO
import torch.nn as nn

model = YOLO('yolo11n.pt').model

print("=" * 90)
print("COMPLETE YOLO MODULE HIERARCHY")
print("=" * 90)

for name, module in model.named_modules():
    depth = name.count('.')
    indent = "  " * depth
    mtype = type(module).__name__
    
    # Check for conv/bn/act pattern
    extras = []
    if hasattr(module, 'conv') and isinstance(getattr(module, 'conv'), nn.Module):
        extras.append(f"conv={type(module.conv).__name__}")
    if hasattr(module, 'bn') and isinstance(getattr(module, 'bn'), nn.Module):
        extras.append(f"bn={type(module.bn).__name__}")
    if hasattr(module, 'act') and isinstance(getattr(module, 'act'), nn.Module):
        extras.append(f"act={type(module.act).__name__}")
    
    extra_str = f" ({', '.join(extras)})" if extras else ""
    
    # Count params
    params = sum(p.numel() for p in module.parameters(recurse=False))
    param_str = f" [{params} params]" if params > 0 else ""
    
    print(f"{indent}{name or 'ROOT'}: {mtype}{extra_str}{param_str}")

print("\n" + "=" * 90)
print("ULTRALYTICS Conv CLASS FORWARD METHOD")
print("=" * 90)

# Find the Conv class and show its forward
for name, module in model.named_modules():
    if type(module).__name__ == 'Conv':
        import inspect
        print(f"\nConv class: {type(module)}")
        print(f"Forward source:")
        try:
            src = inspect.getsource(type(module).forward)
            print(src)
        except:
            print("Could not get source")
        break
