#!/usr/bin/env python
"""Convert a nanotron Qwen2 checkpoint to HuggingFace transformers format.

Why this exists
---------------
Nanotron saves checkpoints in its own sharded layout (`model/<submodule>/.../
model_weight_pp-rank-X-of-Y_tp-rank-A-of-B.safetensors`).  This is great for
distributed training resume but not consumable by:
  * lighteval's accelerate / transformers backend
  * lm-evaluation-harness
  * vllm / sglang inference engines
  * HuggingFace Hub publishing

This script translates the layout into the HF-standard
`config.json + model.safetensors + tokenizer.json` so that downstream tools
just see a normal `Qwen2ForCausalLM` checkpoint.

Scope
-----
* Only handles **non-MoE Qwen2** with `tp=1, pp=1` (matches our training
  setup; a TP-aware path would need to gather shards across TP ranks first).
* Reads `<ckpt_dir>/config.yaml` and `<ckpt_dir>/model_config.json` for layer
  count and hidden sizes.
* Outputs into a directory with `config.json`, `tokenizer*` (copied from
  the configured tokenizer), and `model.safetensors` (single-file form).

Usage
-----
    python scripts/convert_nanotron_qwen2_to_hf.py \
        --src experiments/exp_ref_cosmopedia_v2_qwen3_seed42/ckpt/22888 \
        --dst experiments/exp_ref_cosmopedia_v2_qwen3_seed42/hf_ckpt
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load
from safetensors.torch import save_file as safe_save


# ─────────────────────────────────────────────────────────────────────────
# Helpers — locate nanotron tensors and load them as plain torch.Tensor.

def _read_one(path: Path) -> torch.Tensor:
    """nanotron stores each tensor in a single-key safetensors file
    under the key 'data'."""
    obj = safe_load(str(path))
    if "data" not in obj or len(obj) != 1:
        raise RuntimeError(f"unexpected safetensors keys in {path}: {list(obj.keys())}")
    return obj["data"]


def _ckpt_path(root: Path, *parts: str, suffix: str = "weight") -> Path:
    """Build the canonical pp/tp-tagged shard path for a given module."""
    fname = f"model_{suffix}_pp-rank-0-of-1_tp-rank-0-of-1.safetensors"
    return root.joinpath(*parts, fname)


def _norm_path(root: Path, *parts: str) -> Path:
    """Layer-norm style files are not pp/tp-tagged."""
    return root.joinpath(*parts, "model_weight.safetensors")


# ─────────────────────────────────────────────────────────────────────────
# Conversion core.

def convert(src_ckpt: Path, dst_dir: Path) -> None:
    if not src_ckpt.is_dir():
        sys.exit(f"src is not a directory: {src_ckpt}")
    model_root = src_ckpt / "model" / "model"
    if not model_root.is_dir():
        sys.exit(f"missing nanotron model dir: {model_root}")

    # --- read nanotron architecture spec ---------------------------------
    arch_json = json.loads((src_ckpt / "model_config.json").read_text())
    if not arch_json.get("is_qwen2_config"):
        sys.exit("model_config.json does not advertise is_qwen2_config=True; "
                 "this converter only handles Qwen2-family checkpoints.")
    n_layers     = arch_json["num_hidden_layers"]
    hidden       = arch_json["hidden_size"]
    n_q_heads    = arch_json["num_attention_heads"]
    n_kv_heads   = arch_json["num_key_value_heads"]
    inter        = arch_json["intermediate_size"]
    vocab        = arch_json["vocab_size"]
    head_dim     = hidden // n_q_heads
    q_size       = n_q_heads * head_dim
    kv_size      = n_kv_heads * head_dim

    print(f"[convert] arch  : Qwen2 hidden={hidden} layers={n_layers} "
          f"heads(q/kv)={n_q_heads}/{n_kv_heads} head_dim={head_dim} "
          f"inter={inter} vocab={vocab}")

    # --- iterate layers and rename tensors -------------------------------
    state: dict[str, torch.Tensor] = {}
    t0 = time.time()

    # 1. token embeddings + lm_head + final norm
    state["model.embed_tokens.weight"] = _read_one(
        _ckpt_path(model_root, "token_position_embeddings", "pp_block", "token_embedding")
    )
    state["lm_head.weight"] = _read_one(
        _ckpt_path(model_root, "lm_head", "pp_block")
    )
    state["model.norm.weight"] = _read_one(
        _norm_path(model_root, "final_layer_norm", "pp_block")
    )

    # 2. transformer blocks
    for i in range(n_layers):
        layer_root = model_root / "decoder" / str(i) / "pp_block"

        # (a) Attention QKV — nanotron packs as a single [q_size + kv + kv, hidden]
        #     matrix in the order [Q rows; K rows; V rows].  Same order for bias.
        qkv_w = _read_one(_ckpt_path(layer_root, "attn", "qkv_proj"))
        qkv_b = _read_one(_ckpt_path(layer_root, "attn", "qkv_proj", suffix="bias"))
        assert qkv_w.shape == (q_size + 2 * kv_size, hidden), \
            f"layer {i} qkv weight shape mismatch: got {tuple(qkv_w.shape)}"
        q_w, k_w, v_w = torch.split(qkv_w, [q_size, kv_size, kv_size], dim=0)
        q_b, k_b, v_b = torch.split(qkv_b, [q_size, kv_size, kv_size], dim=0)

        prefix = f"model.layers.{i}.self_attn"
        state[f"{prefix}.q_proj.weight"] = q_w.contiguous()
        state[f"{prefix}.k_proj.weight"] = k_w.contiguous()
        state[f"{prefix}.v_proj.weight"] = v_w.contiguous()
        state[f"{prefix}.q_proj.bias"]   = q_b.contiguous()
        state[f"{prefix}.k_proj.bias"]   = k_b.contiguous()
        state[f"{prefix}.v_proj.bias"]   = v_b.contiguous()

        # (b) Attention output projection
        state[f"{prefix}.o_proj.weight"] = _read_one(
            _ckpt_path(layer_root, "attn", "o_proj")
        )

        # (c) MLP — nanotron packs gate_proj and up_proj as [gate; up] of size
        #     [2 * inter, hidden].  Split into HF's separate matrices.
        gate_up = _read_one(_ckpt_path(layer_root, "mlp", "gate_up_proj"))
        assert gate_up.shape == (2 * inter, hidden), \
            f"layer {i} gate_up shape mismatch: got {tuple(gate_up.shape)}"
        gate_w, up_w = torch.split(gate_up, [inter, inter], dim=0)
        prefix = f"model.layers.{i}.mlp"
        state[f"{prefix}.gate_proj.weight"] = gate_w.contiguous()
        state[f"{prefix}.up_proj.weight"]   = up_w.contiguous()
        state[f"{prefix}.down_proj.weight"] = _read_one(
            _ckpt_path(layer_root, "mlp", "down_proj")
        )

        # (d) Layer norms (no pp/tp tag)
        state[f"model.layers.{i}.input_layernorm.weight"] = _read_one(
            _norm_path(layer_root, "input_layernorm")
        )
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = _read_one(
            _norm_path(layer_root, "post_attention_layernorm")
        )

        if (i + 1) % 8 == 0 or (i + 1) == n_layers:
            print(f"[convert] layer {i+1}/{n_layers}", flush=True)

    print(f"[convert] tensors collected in {time.time()-t0:.1f}s "
          f"(total {len(state)} tensors)")

    # --- HuggingFace config.json -----------------------------------------
    hf_config = {
        "_name_or_path": str(dst_dir.resolve()),
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "torch_dtype": "bfloat16",
        # Architecture
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_q_heads,
        "num_key_value_heads": n_kv_heads,
        "max_position_embeddings": arch_json.get("max_position_embeddings", 2048),
        "rope_theta": arch_json.get("rope_theta", 1_000_000.0),
        # Qwen2 specifics
        "rms_norm_eps": arch_json.get("rms_norm_eps", 1e-6),
        "attention_bias": True,   # nanotron `attention_bias=True` was used during training
        "tie_word_embeddings": arch_json.get("tie_word_embeddings", False),
        "vocab_size": vocab,
        # Tokenizer ids
        "bos_token_id": arch_json.get("bos_token_id", 1),
        "eos_token_id": arch_json.get("eos_token_id", 2),
        "pad_token_id": arch_json.get("pad_token_id"),
        # Activation
        "hidden_act": arch_json.get("hidden_act", "silu"),
        "use_cache": True,
        # Lighteval-compatible attention impl (FA2 by default)
        "attn_implementation": "flash_attention_2",
        # Sliding window unused
        "sliding_window": arch_json.get("sliding_window_size"),
        "use_sliding_window": False,
    }

    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "config.json").write_text(json.dumps(hf_config, indent=2))

    # --- Single-file safetensors save ------------------------------------
    out_path = dst_dir / "model.safetensors"
    print(f"[convert] writing {out_path} ({sum(t.numel() for t in state.values())/1e6:.1f}M params)")
    safe_save(state, str(out_path), metadata={"format": "pt"})

    # --- Tokenizer copy --------------------------------------------------
    nanotron_yaml = (src_ckpt / "config.yaml").read_text()
    import yaml
    cfg = yaml.safe_load(nanotron_yaml)
    tok_repo = (cfg.get("tokenizer") or {}).get("tokenizer_name_or_path")
    if not tok_repo:
        print("[convert] no tokenizer in nanotron config; you must copy one manually")
    else:
        print(f"[convert] saving tokenizer files from {tok_repo}")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tok_repo)
            tok.save_pretrained(str(dst_dir))
        except Exception as e:
            print(f"[convert] tokenizer save failed (non-fatal): {e}")

    # --- generation_config.json (small, helps inference defaults) --------
    gen_cfg = {
        "bos_token_id": hf_config["bos_token_id"],
        "eos_token_id": hf_config["eos_token_id"],
        "pad_token_id": hf_config["pad_token_id"],
        "do_sample": False,
        "max_new_tokens": 256,
    }
    (dst_dir / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))

    print(f"[convert] DONE → {dst_dir}")
    print(f"[convert] verify with: python -c \"from transformers import "
          f"AutoModelForCausalLM, AutoTokenizer; "
          f"m=AutoModelForCausalLM.from_pretrained('{dst_dir}'); "
          f"t=AutoTokenizer.from_pretrained('{dst_dir}'); "
          f"print(m.config); print(t)\"")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="nanotron ckpt dir (contains config.yaml, model/, model_config.json)")
    ap.add_argument("--dst", type=Path, required=True, help="output HF model dir")
    args = ap.parse_args()
    convert(args.src.resolve(), args.dst.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
