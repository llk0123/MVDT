#!/usr/bin/env python3
"""Smoke test: verify CLIP_GRADIENTS config and optimizer build."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.solver.build import maybe_add_gradient_clipping
from dinoteacher import add_dinoteacher_config


def main():
    cfg = get_cfg()
    add_dinoteacher_config(cfg)
    cfg.merge_from_file("configs/vgg_city2fgcity.yaml")

    clip = cfg.SOLVER.CLIP_GRADIENTS
    required = ["ENABLED", "CLIP_TYPE", "CLIP_VALUE", "NORM_TYPE"]
    missing = [k for k in required if not hasattr(clip, k)]
    if missing:
        print(f"FAIL: missing CLIP_GRADIENTS fields: {missing}")
        return 1

    print("CLIP_GRADIENTS config:")
    print(f"  ENABLED    = {clip.ENABLED}")
    print(f"  CLIP_TYPE  = {clip.CLIP_TYPE}")
    print(f"  CLIP_VALUE = {clip.CLIP_VALUE}")
    print(f"  NORM_TYPE  = {clip.NORM_TYPE}")

    # Test optimizer wrapper creation (this is where CLIP_TYPE error occurred)
    dummy_params = [__import__("torch").nn.Parameter(__import__("torch").randn(2, 2))]
    opt_cls = maybe_add_gradient_clipping(cfg, __import__("torch").optim.SGD)
    optimizer = opt_cls(dummy_params, lr=0.01)
    print(f"Optimizer class: {type(optimizer).__name__}")

    # Simulate one step with gradient clipping
    loss = (dummy_params[0] ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Optimizer step with gradient clipping: OK")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
