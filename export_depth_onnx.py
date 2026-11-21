#!/usr/bin/env python3
"""
Export local HuggingFace Depth-Anything-V2-Small-hf to a standalone single-file
ONNX model (`models/depth_anything_v2_small.onnx`) for PyTorch-free portable
inference via `onnxruntime`.
"""

from __future__ import annotations

import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForDepthEstimation


class DepthWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "depth_anything_v2_small.onnx")

    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    print(f"Loading cached model: {model_id} ...")
    model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
    model.eval().cpu()

    wrapper = DepthWrapper(model).eval()
    dummy = torch.randn(1, 3, 518, 518, dtype=torch.float32)

    print(f"Exporting ONNX to: {out_path} ...")
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy,),
        out_path,
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=True,
    )
    print(f"Initial export finished in {time.time() - t0:.2f}s")

    # Ensure everything is stored in a single self-contained .onnx file (no external .data sidecar)
    data_sidecar = out_path + ".data"
    if os.path.exists(data_sidecar):
        print("Merging external tensor data into single .onnx file...")
        m = onnx.load(out_path, load_external_data=True)
        onnx.save_model(m, out_path, save_as_external_data=False)
        try:
            os.remove(data_sidecar)
        except OSError:
            pass

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Single-file ONNX size: {size_mb:.2f} MB")

    # Numerical accuracy & latency verification on CPU ONNX Runtime
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    t1 = time.time()
    ort_out = sess.run(None, {"pixel_values": dummy.numpy()})[0]
    elapsed = time.time() - t1

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    corr = float(np.corrcoef(pt_out.ravel(), ort_out.ravel())[0, 1])
    print(f"ONNX CPU inference time (518x518): {elapsed:.3f}s | out shape: {ort_out.shape}")
    print(f"PyTorch vs ONNX max_abs_diff={max_diff:.6f}, pearson_corr={corr:.6f}")


if __name__ == "__main__":
    main()
