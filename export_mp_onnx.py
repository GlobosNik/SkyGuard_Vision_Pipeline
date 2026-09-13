"""
Mixed-Precision ONNX Exporter
=========================================
Takes a trained QAT .pt model + bit_allocation.json and produces an ONNX
model with the per-layer precision baked in as DequantizeLinear ops.

Uses ORDER-BASED layer matching (k-th Conv node in ONNX = k-th Conv2d in
named_modules()), which is robust to the Conv+BatchNorm fusion that
Ultralytics applies during ONNX export. Value-based matching fails after
fusion because fused weights differ numerically from unfused PyTorch
weights; order is preserved regardless.

Usage:
    python export_mp_onnx.py \
        --model runs_qat/qat/drone_qat/weights/best.pt \
        --allocation bit_allocation.json \
        --output model_mixed_precision.onnx \
        --imgsz 640

Requirements:
    pip install ultralytics>=8.3.0 onnx numpy
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np


# ============================================================================
# Quantization math (must match the QAT fake-quant hooks exactly)
# ============================================================================

# Symmetric per-channel fake quantization
def fake_quant_per_channel(weight_np: np.ndarray, bits: int):
    """
    Identical to the QAT forward_pre_hook scheme:
        qmax  = 2^(bits-1) - 1              (127 for 8-bit, 7 for 4-bit)
        scale = per-channel |w|max / qmax   (clamped at 1e-8)
        q     = round(w / scale).clip(-qmax, qmax)
    Returns (q_int8[C,...], scale_f32[C], dequantized_f32).
    """
    qmax = 2 ** (bits - 1) - 1
    w = weight_np.astype(np.float64)
    flat = w.reshape(w.shape[0], -1)
    max_per_ch = np.abs(flat).max(axis=1)
    scale = np.maximum(max_per_ch / qmax, 1e-8).astype(np.float32)
    scale_b = scale.reshape(-1, *([1] * (w.ndim - 1))).astype(np.float64)
    q = np.clip(np.round(w / scale_b), -qmax, qmax)
    deq = (q * scale_b).astype(np.float32)
    return q.astype(np.int8), scale, deq


# ============================================================================
# ONNX graph surgery with order-based matching
# ============================================================================

def apply_mixed_precision(
    fp32_onnx_path: str,
    layer_names_in_order: list,
    bit_allocation: Dict[str, int],
    output_path: str,
    default_bits: int = 8,
) -> Dict:
    """
    Bake per-layer precision into an ONNX graph.

    Matching strategy: the k-th Conv node in the ONNX graph (in node-list
    order, which for Ultralytics exports is topological/execution order)
    corresponds to the k-th Conv2d in model.model.named_modules(). This
    survives Conv+BN fusion because fusion preserves node count and order.

    Args:
        fp32_onnx_path: FP32 ONNX file (from YOLO.export).
        layer_names_in_order: Conv2d layer names in named_modules() order.
        bit_allocation: {layer_name: bits} from bit_allocation.json.
        output_path: where to save the mixed-precision model.
        default_bits: fallback for unmatched layers.

    Returns: report dict.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(fp32_onnx_path)
    graph = model.graph

    # --- opset guard ---
    opset = next((o.version for o in model.opset_import
                  if o.domain in ('', 'ai.onnx')), None)
    if opset is not None and opset < 13:
        raise RuntimeError(
            f"ONNX opset {opset} < 13; DequantizeLinear(axis=...) requires "
            f"opset >= 13. Re-export with opset=13.")

    # --- index initializers ---
    inits = {i.name: i for i in graph.initializer}

    # --- collect Conv nodes that consume an initializer as weight ---
    conv_entries = []  # (node_index, weight_init_name)
    for idx, node in enumerate(graph.node):
        if node.op_type != 'Conv' or len(node.input) < 2:
            continue
        wname = node.input[1]
        if wname in inits:
            conv_entries.append((idx, wname))

    n_onnx_convs = len(conv_entries)
    n_ref_layers = len(layer_names_in_order)

    print(f"ONNX Conv nodes with initializer weights: {n_onnx_convs}")
    print(f"PyTorch Conv2d layers (named_modules order): {n_ref_layers}")

    if n_onnx_convs != n_ref_layers:
        print(f"WARNING: count mismatch ({n_onnx_convs} vs {n_ref_layers}).")
        print("  Proceeding with positional matching for the overlap; "
              "extra ONNX convs keep FP32.")
        if n_onnx_convs < n_ref_layers:
            layer_names_in_order = layer_names_in_order[:n_onnx_convs]

    # --- order-based mapping: k-th ONNX Conv -> k-th PyTorch layer ---
    plan = {}  # node_idx -> (layer_name, bits, weight_init_name)
    for k, (node_idx, wname) in enumerate(conv_entries):
        if k >= len(layer_names_in_order):
            break
        layer = layer_names_in_order[k]
        bits = int(bit_allocation.get(layer, default_bits))
        plan[node_idx] = (layer, bits, wname)

    matched = set(v[0] for v in plan.values())
    unmatched_alloc = [n for n in bit_allocation if n not in matched]

    # --- surgery ---
    new_nodes = []
    stats = {'16': 0, '8': 0, '4': 0, 'untouched': 0}
    for idx, node in enumerate(graph.node):
        if idx not in plan:
            new_nodes.append(node)
            continue

        layer, bits, wname = plan[idx]
        winit = inits[wname]
        w = numpy_helper.to_array(winit)

        if bits >= 16:
            stats['16'] += 1
            new_nodes.append(node)  # keep FP32
            continue

        q_int8, scale, _ = fake_quant_per_channel(w, bits)
        stats[str(bits)] = stats.get(str(bits), 0) + 1

        base = f"mpq_{layer}_w{bits}".replace('.', '_')
        q_name = f"{base}_q"
        s_name = f"{base}_scale"
        dq_name = f"{base}_dq"

        # Replace FP32 initializer with INT8 quantized + scale
        graph.initializer.remove(winit)
        graph.initializer.append(numpy_helper.from_array(q_int8, q_name))
        graph.initializer.append(numpy_helper.from_array(scale, s_name))

        # DequantizeLinear(axis=0) -> per-channel dequantized weight
        dq_node = helper.make_node(
            'DequantizeLinear',
            [q_name, s_name],
            [dq_name],
            axis=0,
            name=f"{base}_DequantizeLinear",
        )
        new_nodes.append(dq_node)

        # Rewire the Conv to consume the DQ output
        conv = onnx.NodeProto()
        conv.CopyFrom(node)
        conv.input[1] = dq_name
        new_nodes.append(conv)

    # Replace node list
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Validate
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"WARNING: onnx.checker flagged the model ({e}); saving anyway.")

    onnx.save(model, output_path)

    # --- report ---
    fp32_bytes = os.path.getsize(fp32_onnx_path)
    mp_bytes = os.path.getsize(output_path)

    # Count weight bytes by dtype for an honest size breakdown
    int8_params = 0
    fp32_params = 0
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if init.data_type == onnx.TensorProto.INT8 and init.name.startswith('mpq_') and init.name.endswith('_q'):
            int8_params += arr.size
        elif init.data_type == onnx.TensorProto.FLOAT and 'scale' not in init.name:
            # crude: count remaining float tensors as FP32 weights
            fp32_params += arr.size

    report = {
        'output_path': output_path,
        'layers_16bit_fp32': stats['16'],
        'layers_8bit_int8': stats.get('8', 0),
        'layers_4bit_int4_in_int8': stats.get('4', 0),
        'total_conv_layers': n_onnx_convs,
        'matched_layers': len(matched),
        'unmatched_allocation_layers': unmatched_alloc,
        'fp32_onnx_bytes': fp32_bytes,
        'mixed_precision_onnx_bytes': mp_bytes,
        'size_ratio': round(mp_bytes / max(fp32_bytes, 1), 4),
    }

    print(f"\n{'='*60}")
    print(f"MIXED-PRECISION ONNX EXPORT REPORT")
    print(f"{'='*60}")
    print(f"  16-bit (FP32 kept):     {report['layers_16bit_fp32']:>3} layers")
    print(f"  8-bit (INT8 QDQ):       {report['layers_8bit_int8']:>3} layers")
    print(f"  4-bit (INT4 in INT8):   {report['layers_4bit_int4_in_int8']:>3} layers")
    print(f"  Total Conv layers:      {report['total_conv_layers']:>3}")
    print(f"  Matched:                {report['matched_layers']:>3}")
    print(f"  FP32 ONNX:  {fp32_bytes/1e6:.2f} MB")
    print(f"  Mixed-precision ONNX: {mp_bytes/1e6:.2f} MB "
          f"({report['size_ratio']*100:.1f}% of FP32)")
    if unmatched_alloc:
        print(f"  WARNING: {len(unmatched_alloc)} allocation layers unmatched:")
        for n in unmatched_alloc[:5]:
            print(f"    {n}")
    print(f"{'='*60}")
    return report


# ============================================================================
# Main: load QAT model, export FP32 ONNX, apply mixed precision
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Standalone mixed-precision ONNX exporter for QAT-trained '
                    'YOLOv11n models. Uses order-based layer matching '
                    '(robust to Conv+BN fusion during Ultralytics export).')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained QAT model weights (.pt)')
    parser.add_argument('--allocation', type=str, required=True,
                        help='Path to bit_allocation.json (from Stage 2)')
    parser.add_argument('--output', type=str, default='model_mixed_precision.onnx',
                        help='Output ONNX path (default: model_mixed_precision.onnx)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--keep-fp32-onnx', action='store_true',
                        help='Keep the intermediate FP32 ONNX file')
    parser.add_argument('--fp32-onnx-path', type=str, default=None,
                        help='(Advanced) Skip Ultralytics export; use this '
                             'existing FP32 ONNX file directly')
    args = parser.parse_args()

    # --- load bit allocation ---
    with open(args.allocation) as f:
        bit_allocation = {k: int(v) for k, v in json.load(f).items()}
    print(f"Loaded bit allocation: {len(bit_allocation)} layers")
    from collections import Counter
    c = Counter(bit_allocation.values())
    for bits in sorted(c):
        print(f"  {bits}-bit: {c[bits]} layers")

    # --- get Conv2d layer names in order ---
    if args.fp32_onnx_path:
        # Advanced mode: user provides existing FP32 ONNX + we need the layer
        # names from the PyTorch model
        print(f"\nUsing existing FP32 ONNX: {args.fp32_onnx_path}")
        from ultralytics import YOLO
        import torch.nn as nn
        model = YOLO(args.model)
        layer_names = [n for n, m in model.model.named_modules()
                       if isinstance(m, nn.Conv2d)]
        fp32_path = args.fp32_onnx_path
    else:
        # Standard mode: export FP32 ONNX via Ultralytics, then collect names
        from ultralytics import YOLO
        import torch.nn as nn

        print(f"\nLoading QAT model: {args.model}")
        model = YOLO(args.model)

        # Collect Conv2d layer names BEFORE export (named_modules order)
        layer_names = [n for n, m in model.model.named_modules()
                       if isinstance(m, nn.Conv2d)]
        print(f"Found {len(layer_names)} Conv2d layers")

        # Export FP32 ONNX
        print(f"Exporting FP32 ONNX (imgsz={args.imgsz}, opset=13)...")
        fp32_path = model.export(
            format='onnx',
            imgsz=args.imgsz,
            opset=13,
            simplify=True,
            dynamic=False,
        )
        print(f"FP32 ONNX exported: {fp32_path}")

    # --- apply mixed-precision surgery ---
    print(f"\nApplying mixed-precision quantization...")
    report = apply_mixed_precision(
        fp32_path,
        layer_names,
        bit_allocation,
        args.output,
    )

    # --- cleanup ---
    if not args.keep_fp32_onnx and args.fp32_onnx_path is None:
        if os.path.exists(fp32_path) and fp32_path != args.output:
            os.remove(fp32_path)
            print(f"Removed intermediate FP32 ONNX: {fp32_path}")

    print(f"\nDone. Mixed-precision model: {args.output}")
    print(f"Validate with: "
          f"python -c \"import onnxruntime; "
          f"s=onnxruntime.InferenceSession('{args.output}'); "
          f"print('OK, inputs:', [i.name for i in s.get_inputs()])\"")

    return report


if __name__ == '__main__':
    main()
