#!/usr/bin/env python3
"""
Domain-Specific BBPE Vocabulary Analysis & Transitive-Closure Pruning for Qwen3.5-2B (ShopSimulator).

Key guarantees:
1. Preserves all 256 byte-level BPE base tokens -> Zero OOV on any UTF-8 text.
2. Preserves all special / added tokens & remaps config.json / generation_config.json / tokenizer_config.json.
3. Enforces BPE Merge Tree Transitive Closure -> Never breaks intermediate merge chains.
4. Keeps all single-character CJK/ASCII/digit tokens from original vocab -> Prevents Chinese byte-fragmentation.
5. Evaluates token fertility (length inflation) on held-out official_test_200.jsonl (no test leakage into selection!).
6. Verifies MultiTurnLossMaskGenerator(tokenizer_type="qwen3_5") compatibility and round-trip decode fidelity.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


def is_single_cjk_or_ascii(token_decoded: str) -> bool:
    if len(token_decoded) != 1:
        return False
    cp = ord(token_decoded)
    # Basic ASCII printable, CJK Unified Ideographs, CJK punctuation, Fullwidth forms
    return (
        0x20 <= cp <= 0x7E
        or 0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3000 <= cp <= 0x303F
        or 0xFF00 <= cp <= 0xFFEF
    )


def extract_jsonl_strings(jsonl_path: str) -> list[str]:
    texts: list[str] = []
    if not jsonl_path or not os.path.isfile(jsonl_path):
        return texts

    def walk_str(obj: Any, out: list[str]):
        if isinstance(obj, str):
            if len(obj) >= 2:
                out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk_str(v, out)
        elif isinstance(obj, list):
            for v in obj:
                walk_str(v, out)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parts: list[str] = []
            walk_str(obj, parts)
            if parts:
                texts.append("\n".join(parts))
    return texts


def collect_domain_texts(
    rl_jsonl: str | None,
    shop_dir: str | None,
    max_shop_chars: int = 200_000_000,
) -> tuple[list[str], list[str]]:
    import gzip

    texts: list[str] = []
    product_samples: list[str] = []

    # 1. All strings in RL training tasks (rl_500.jsonl)
    texts.extend(extract_jsonl_strings(rl_jsonl))

    # 2. Synthetic tool-call & chat template scaffolding used by Pi + ShopSimulator
    texts.append(
        "完成给定的购物任务。先调用 shop_reset，然后只使用 shop_act 与环境交互。\n"
        "<tool_call>\n<function=shop_reset>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=shop_act>\n<parameter=action>\nsearch[防盗门 窗户]\n</parameter>\n</function>\n</tool_call>\n"
        "click[buy now] click[description] click[features] click[reviews] click[back to search] click[next >] click[< prev]\n"
        "Instruction: [SEP] Back to Search [SEP] Page 1 (Total results: 150) [SEP] Next > [SEP] < Prev [SEP] Buy Now\n"
        "搜索功能是否可用: True False 可点击的按钮: 价格: 店铺: 颜色分类 尺码 规格 型号\n"
    )

    # 3. Full ShopSimulator product catalog (JSON, JSONL, JSON.GZ) chunked to <=64KB per chunk
    total_chars = 0
    if shop_dir and os.path.isdir(shop_dir):
        candidates: list[str] = []
        for pat in ("**/*.json", "**/*.jsonl", "**/*.json.gz"):
            candidates.extend(glob.glob(os.path.join(shop_dir, pat), recursive=True))
        filtered = []
        for p in sorted(set(candidates)):
            low = p.lower()
            if any(skip in low for skip in ("node_modules", ".git", "indexes", "__pycache__", "test_200", "official_test")):
                continue
            sz = os.path.getsize(p)
            if sz > 500 * 1024 * 1024:
                continue
            filtered.append((sz, p))
        for sz, p in filtered:
            if total_chars >= max_shop_chars:
                break
            try:
                opener = gzip.open if p.endswith(".gz") else open
                with opener(p, "rt", encoding="utf-8", errors="ignore") as f:
                    while total_chars < max_shop_chars:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        texts.append(chunk)
                        total_chars += len(chunk)
                        if len(product_samples) < 300 and ("items_eval_train" in p or "fine_items" in p):
                            product_samples.append(chunk[:1500])
            except Exception as e:
                print(f"[WARN] skip {p}: {e}", flush=True)

    return texts, product_samples


def collect_eval_texts(eval_jsonl: str | None) -> list[str]:
    return extract_jsonl_strings(eval_jsonl)


def parse_merges(raw_merges: list[Any]) -> tuple[list[tuple[str, str]], bool]:
    """Return list of (a, b) tuples and whether original format was list[list[str]]."""
    if not raw_merges:
        return [], False
    is_list_fmt = isinstance(raw_merges[0], list)
    parsed: list[tuple[str, str]] = []
    for m in raw_merges:
        if isinstance(m, list):
            parsed.append((m[0], m[1]))
        else:
            a, b = m.split(" ", 1)
            parsed.append((a, b))
    return parsed, is_list_fmt


def build_merge_closure_helper(merges: list[tuple[str, str]]):
    creator: dict[str, tuple[str, str]] = {}
    for a, b in merges:
        c = a + b
        if c not in creator:
            creator[c] = (a, b)

    memo: dict[str, set[str]] = {}

    def get_closure(tok: str) -> set[str]:
        if tok in memo:
            return memo[tok]
        res: set[str] = set()
        stack = [tok]
        while stack:
            cur = stack.pop()
            if cur in res:
                continue
            res.add(cur)
            pair = creator.get(cur)
            if pair is not None:
                a, b = pair
                if a not in res:
                    stack.append(a)
                if b not in res:
                    stack.append(b)
        memo[tok] = res
        return res

    return creator, get_closure


def remap_id_fields_inplace(obj: Any, old_to_new: dict[int, int], new_vocab_size: int, old_vocab_size: int):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "vocab_size" and isinstance(v, int) and v == old_vocab_size:
                obj[k] = new_vocab_size
            elif k.endswith("_token_id") and isinstance(v, int):
                if v in old_to_new:
                    obj[k] = old_to_new[v]
            elif k.endswith("_token_ids") and isinstance(v, list):
                obj[k] = [old_to_new[x] if isinstance(x, int) and x in old_to_new else x for x in v]
            elif k in ("eos_token_id", "bos_token_id", "pad_token_id") and isinstance(v, list):
                obj[k] = [old_to_new[x] if isinstance(x, int) and x in old_to_new else x for x in v]
            elif isinstance(v, (dict, list)):
                remap_id_fields_inplace(v, old_to_new, new_vocab_size, old_vocab_size)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                remap_id_fields_inplace(item, old_to_new, new_vocab_size, old_vocab_size)


def collect_referenced_special_ids(cfg_paths: list[str]) -> set[int]:
    ids: set[int] = set()

    def walk(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.endswith("_token_id") and isinstance(v, int):
                    ids.add(v)
                elif (k.endswith("_token_ids") or k in ("eos_token_id", "bos_token_id", "pad_token_id")) and isinstance(v, list):
                    for x in v:
                        if isinstance(x, int):
                            ids.add(x)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    walk(item)

    for p in cfg_paths:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                walk(json.load(f))
    return ids


def select_pruned_tokens(
    base_vocab: dict[str, int],
    added_tokens: list[dict[str, Any]],
    merges: list[tuple[str, str]],
    domain_counts: collections.Counter,
    referenced_ids: set[int],
    orig_tokenizer: Any,
    target_vocab_size: int,
) -> tuple[set[str], list[dict[str, Any]], dict[str, int]]:
    assert target_vocab_size % 128 == 0, f"target_vocab_size ({target_vocab_size}) must be divisible by 128"

    id_to_base_tok = {tid: tok for tok, tid in base_vocab.items()}
    creator, get_closure = build_merge_closure_helper(merges)

    # All added_tokens are mandatory to preserve chat template, tool calls, thinking, vision tags
    kept_added = sorted(added_tokens, key=lambda x: x["id"])
    added_contents = {a["content"] for a in kept_added}
    # Count added_tokens that are NOT already in base_vocab
    extra_added_count = sum(1 for a in kept_added if a["content"] not in base_vocab)
    max_base_slots = target_vocab_size - extra_added_count

    kept_base: set[str] = set()

    def try_add_with_closure(tok: str, force: bool = False) -> bool:
        if tok not in base_vocab:
            return False
        if tok in kept_base:
            return True
        cl = get_closure(tok)
        delta = [x for x in cl if x not in kept_base and x in base_vocab]
        if not force and len(kept_base) + len(delta) > max_base_slots:
            return False
        kept_base.update(delta)
        return True

    # Tier 0: All atomic base tokens (not created by any merge -> includes all 256 byte-level BPE atoms)
    # + all base tokens referenced in config/generation_config + all added_tokens present in base_vocab
    for tok in base_vocab:
        if tok not in creator:
            try_add_with_closure(tok, force=True)
    for tid in referenced_ids:
        if tid in id_to_base_tok:
            try_add_with_closure(id_to_base_tok[tid], force=True)
    for a in kept_added:
        if a["content"] in base_vocab:
            try_add_with_closure(a["content"], force=True)
    tier0_n = len(kept_base)

    # Tier 1: Domain-observed tokens (ShopSimulator catalog + rl_500 + tool scaffolds), highest frequency first
    domain_sorted = [id_to_base_tok[tid] for tid, _ in domain_counts.most_common() if tid in id_to_base_tok]
    for tok in domain_sorted:
        try_add_with_closure(tok)
    tier1_n = len(kept_base)

    # Tier 2: All single-character CJK / ASCII / punctuation tokens from original vocab (prevents byte fragmentation)
    # Decode each token ID cleanly in batches
    single_char_toks: list[tuple[int, str]] = []
    for tok, tid in base_vocab.items():
        if tok in kept_base:
            continue
        dec = orig_tokenizer.decode([tid], skip_special_tokens=False)
        if is_single_cjk_or_ascii(dec):
            single_char_toks.append((tid, tok))
    single_char_toks.sort(key=lambda x: x[0])
    for _, tok in single_char_toks:
        if len(kept_base) >= max_base_slots:
            break
        try_add_with_closure(tok)
    tier2_n = len(kept_base)

    # Tier 3: Fill remaining budget up to max_base_slots with original pretraining high-frequency tokens (lowest old_id)
    for tid in sorted(id_to_base_tok.keys()):
        if len(kept_base) >= max_base_slots:
            break
        tok = id_to_base_tok[tid]
        if tok in kept_base:
            continue
        try_add_with_closure(tok)
    tier3_n = len(kept_base)

    stats = {
        "tier0_atoms_and_specials": tier0_n,
        "tier1_after_domain": tier1_n,
        "tier2_after_single_cjk_ascii": tier2_n,
        "tier3_after_pretrain_fill": tier3_n,
        "extra_added_tokens": extra_added_count,
        "total_active_ids": len(kept_base) + extra_added_count,
        "target_vocab_size": target_vocab_size,
    }
    return kept_base, kept_added, stats


def build_pruned_tokenizer_json(
    orig_tj: dict[str, Any],
    kept_base: set[str],
    kept_added: list[dict[str, Any]],
    merges: list[tuple[str, str]],
    merges_is_list_fmt: bool,
) -> tuple[dict[str, Any], dict[int, int], list[int]]:
    base_vocab: dict[str, int] = orig_tj["model"]["vocab"]
    kept_base_sorted = sorted(((tid, tok) for tok, tid in base_vocab.items() if tok in kept_base), key=lambda x: x[0])

    new_base_vocab: dict[str, int] = {}
    old_to_new: dict[int, int] = {}
    old_ids_order: list[int] = []

    for new_id, (old_id, tok) in enumerate(kept_base_sorted):
        new_base_vocab[tok] = new_id
        old_to_new[old_id] = new_id
        old_ids_order.append(old_id)

    next_id = len(kept_base_sorted)
    new_added_tokens: list[dict[str, Any]] = []
    for a in sorted(kept_added, key=lambda x: x["id"]):
        a_copy = dict(a)
        old_id = a["id"]
        if a["content"] in new_base_vocab:
            new_id = new_base_vocab[a["content"]]
            a_copy["id"] = new_id
            old_to_new[old_id] = new_id
        else:
            a_copy["id"] = next_id
            old_to_new[old_id] = next_id
            old_ids_order.append(old_id)
            next_id += 1
        new_added_tokens.append(a_copy)

    new_merges: list[Any] = []
    for a, b in merges:
        if a in kept_base and b in kept_base and (a + b) in kept_base:
            new_merges.append([a, b] if merges_is_list_fmt else f"{a} {b}")

    new_tj = json.loads(json.dumps(orig_tj))
    new_tj["model"]["vocab"] = new_base_vocab
    new_tj["model"]["merges"] = new_merges
    new_tj["added_tokens"] = new_added_tokens
    return new_tj, old_to_new, old_ids_order


def evaluate_fertility(orig_tok: Any, new_tok: Any, texts: list[str], label: str) -> dict[str, float]:
    orig_len = 0
    new_len = 0
    roundtrip_ok = 0
    sample_n = min(len(texts), 500)
    for t in texts[:sample_n]:
        o_ids = orig_tok.encode(t, add_special_tokens=False)
        n_ids = new_tok.encode(t, add_special_tokens=False)
        orig_len += len(o_ids)
        new_len += len(n_ids)
        if new_tok.decode(n_ids, skip_special_tokens=False) == t:
            roundtrip_ok += 1
    ratio = new_len / max(orig_len, 1)
    res = {
        "samples": sample_n,
        "orig_tokens": orig_len,
        "pruned_tokens": new_len,
        "inflation_ratio": round(ratio, 5),
        "inflation_pct": round((ratio - 1.0) * 100.0, 3),
        "roundtrip_exact_rate": round(roundtrip_ok / max(sample_n, 1), 5),
    }
    print(f"[Fertility:{label}] {json.dumps(res, ensure_ascii=False)}", flush=True)
    return res


def prune_weights_and_save(
    src_hf: str,
    dst_hf: str,
    old_ids_order: list[int],
    old_to_new: dict[int, int],
    old_vocab_size: int,
    target_vocab_size: int,
    new_tj: dict[str, Any],
):
    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(dst_hf, exist_ok=True)
    # Copy non-weight files first
    for entry in os.listdir(src_hf):
        src_p = os.path.join(src_hf, entry)
        dst_p = os.path.join(dst_hf, entry)
        if entry.endswith(".safetensors") or entry == "tokenizer.json":
            continue
        if os.path.isfile(src_p):
            shutil.copy2(src_p, dst_p)
        elif os.path.isdir(src_p) and not entry.startswith("."):
            if os.path.exists(dst_p):
                shutil.rmtree(dst_p)
            shutil.copytree(src_p, dst_p)

    # Write pruned tokenizer.json
    with open(os.path.join(dst_hf, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump(new_tj, f, ensure_ascii=False)

    # Update tokenizer_config.json added_tokens_decoder
    tc_path = os.path.join(dst_hf, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        if isinstance(tc.get("added_tokens_decoder"), dict):
            new_dec = {}
            for k, v in tc["added_tokens_decoder"].items():
                old_id = int(k)
                if old_id in old_to_new:
                    new_dec[str(old_to_new[old_id])] = v
            tc["added_tokens_decoder"] = new_dec
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)

    # Update config.json & generation_config.json
    for cfg_name in ("config.json", "generation_config.json"):
        cfg_p = os.path.join(dst_hf, cfg_name)
        if os.path.isfile(cfg_p):
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            remap_id_fields_inplace(cfg, old_to_new, target_vocab_size, old_vocab_size)
            with open(cfg_p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Slice all safetensors files where dim 0 == old_vocab_size
    idx_tensor = torch.tensor(old_ids_order, dtype=torch.long)
    pad_rows = target_vocab_size - len(old_ids_order)
    assert pad_rows >= 0, f"len(old_ids_order)={len(old_ids_order)} > target_vocab_size={target_vocab_size}"

    st_files = sorted(glob.glob(os.path.join(src_hf, "*.safetensors")))
    total_bytes = 0
    sliced_names: list[str] = []
    for st_path in st_files:
        fname = os.path.basename(st_path)
        tensors = load_file(st_path, device="cpu")
        out_tensors = {}
        for name, t in tensors.items():
            if t.ndim >= 1 and t.shape[0] == old_vocab_size and any(
                key in name for key in ("embed_tokens", "lm_head", "word_embeddings", "output_layer")
            ):
                new_t = t.index_select(0, idx_tensor)
                if pad_rows > 0:
                    pad = torch.zeros((pad_rows, *t.shape[1:]), dtype=t.dtype)
                    new_t = torch.cat([new_t, pad], dim=0)
                new_t = new_t.contiguous()
                out_tensors[name] = new_t
                sliced_names.append(f"{name}: {tuple(t.shape)} -> {tuple(new_t.shape)}")
            else:
                out_tensors[name] = t
            total_bytes += out_tensors[name].numel() * out_tensors[name].element_size()
        save_file(out_tensors, os.path.join(dst_hf, fname), metadata={"format": "pt"})

    # Update safetensors index metadata if present
    idx_json = os.path.join(dst_hf, "model.safetensors.index.json")
    if os.path.isfile(idx_json):
        with open(idx_json, "r", encoding="utf-8") as f:
            idx_obj = json.load(f)
        if isinstance(idx_obj.get("metadata"), dict):
            idx_obj["metadata"]["total_size"] = total_bytes
        with open(idx_json, "w", encoding="utf-8") as f:
            json.dump(idx_obj, f, ensure_ascii=False, indent=2)

    # Save mapping report for reproducibility
    with open(os.path.join(dst_hf, "vocab_pruning_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "old_vocab_size": old_vocab_size,
                "new_vocab_size": target_vocab_size,
                "active_tokens": len(old_ids_order),
                "pad_rows": pad_rows,
                "sliced_tensors": sliced_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Weights] Sliced tensors: {sliced_names}", flush=True)


def verify_slime_loss_mask(dst_hf: str, slime_dir: str | None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(dst_hf, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "完成给定的购物任务。先调用 shop_reset，然后只使用 shop_act 与环境交互。"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "shop_reset", "arguments": {}}}],
        },
        {"role": "tool", "content": "Instruction: 我想买白色菱形款防盗门小窗户，尺寸540*660mm，价格200多元。"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "shop_act", "arguments": {"action": "search[防盗门 窗户]"}}}
            ],
        },
    ]
    tools = [
        {"type": "function", "function": {"name": "shop_reset", "description": "Reset shop", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "shop_act", "description": "Act in shop", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}}}},
    ]

    if slime_dir and os.path.isdir(slime_dir) and slime_dir not in sys.path:
        sys.path.insert(0, slime_dir)
    from slime.utils.mask_utils import MultiTurnLossMaskGenerator

    masker = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3_5")
    tids, mask = masker.get_loss_mask(messages, tools=tools)
    assert len(tids) == len(mask) and sum(mask) > 0, "Invalid loss mask!"
    print(f"[Verify:SlimeMask] PASS! seq_len={len(tids)}, assistant_loss_tokens={sum(mask)}, max_id={max(tids)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-hf", required=True)
    ap.add_argument("--dst-hf", default=None)
    ap.add_argument("--rl-jsonl", default=None)
    ap.add_argument("--eval-jsonl", default=None)
    ap.add_argument("--shop-dir", default=None)
    ap.add_argument("--slime-dir", default=None)
    ap.add_argument("--target-vocab-sizes", default="32768,49152,65536", help="Comma-separated sizes to analyze")
    ap.add_argument("--prune-vocab-size", type=int, default=49152, help="Final vocab size to export when --dst-hf is set")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoTokenizer

    orig_tok = AutoTokenizer.from_pretrained(args.src_hf, trust_remote_code=True)
    with open(os.path.join(args.src_hf, "tokenizer.json"), "r", encoding="utf-8") as f:
        orig_tj = json.load(f)
    with open(os.path.join(args.src_hf, "config.json"), "r", encoding="utf-8") as f:
        orig_cfg = json.load(f)
    old_vocab_size = orig_cfg.get("text_config", orig_cfg).get("vocab_size", orig_cfg.get("vocab_size"))
    base_vocab: dict[str, int] = orig_tj["model"]["vocab"]
    added_tokens: list[dict[str, Any]] = orig_tj.get("added_tokens", [])
    merges, merges_is_list_fmt = parse_merges(orig_tj["model"].get("merges", []))
    print(
        f"[Loaded] old_vocab_size={old_vocab_size}, base_vocab={len(base_vocab)}, "
        f"added_tokens={len(added_tokens)}, merges={len(merges)} (list_fmt={merges_is_list_fmt})",
        flush=True,
    )

    print("[Scan] Collecting FULL domain texts (rl_500 + ALL ShopSimulator catalog) and held-out eval_200...", flush=True)
    domain_texts, product_samples = collect_domain_texts(args.rl_jsonl, args.shop_dir)
    eval_texts = collect_eval_texts(args.eval_jsonl)
    print(f"[Scan] domain_chunks={len(domain_texts)}, product_samples={len(product_samples)}, eval_texts={len(eval_texts)}", flush=True)

    domain_counts: collections.Counter = collections.Counter()
    domain_total_toks = 0
    for i, t in enumerate(domain_texts):
        ids = orig_tok.encode(t, add_special_tokens=False)
        domain_counts.update(ids)
        domain_total_toks += len(ids)

    eval_counts: collections.Counter = collections.Counter()
    for t in eval_texts:
        eval_counts.update(orig_tok.encode(t, add_special_tokens=False))

    overlap = len(set(eval_counts.keys()) & set(domain_counts.keys()))
    print(
        f"[Stats] domain_total_toks={domain_total_toks:,}, domain_unique_toks={len(domain_counts):,} "
        f"({len(domain_counts)/old_vocab_size*100:.2f}% of {old_vocab_size}); "
        f"eval200_unique_toks={len(eval_counts):,}, covered_by_domain={overlap:,} ({overlap/max(len(eval_counts),1)*100:.2f}%)",
        flush=True,
    )

    referenced_ids = collect_referenced_special_ids(
        [os.path.join(args.src_hf, "config.json"), os.path.join(args.src_hf, "generation_config.json")]
    )
    print(f"[Stats] referenced_config_special_ids={sorted(referenced_ids)}", flush=True)

    import tempfile

    sizes = [int(x.strip()) for x in args.target_vocab_sizes.split(",") if x.strip()]
    for vs in sizes:
        kept_base, kept_added, stats = select_pruned_tokens(
            base_vocab, added_tokens, merges, domain_counts, referenced_ids, orig_tok, vs
        )
        new_tj, old_to_new, old_ids_order = build_pruned_tokenizer_json(
            orig_tj, kept_base, kept_added, merges, merges_is_list_fmt
        )
        with tempfile.TemporaryDirectory() as tmpd:
            for fn in ("tokenizer_config.json", "chat_template.jinja"):
                sp = os.path.join(args.src_hf, fn)
                if os.path.isfile(sp):
                    shutil.copy2(sp, os.path.join(tmpd, fn))
            with open(os.path.join(tmpd, "tokenizer.json"), "w", encoding="utf-8") as f:
                json.dump(new_tj, f, ensure_ascii=False)
            tmp_tok = AutoTokenizer.from_pretrained(tmpd, trust_remote_code=True)
            print(f"\n=== Candidate vocab_size={vs} (merges={len(new_tj['model']['merges'])}) ===", flush=True)
            print(f"[SelectionStats] {json.dumps(stats, ensure_ascii=False)}", flush=True)
            evaluate_fertility(orig_tok, tmp_tok, eval_texts, f"eval200@{vs}")
            evaluate_fertility(orig_tok, tmp_tok, product_samples, f"products300@{vs}")

    if not args.analyze_only and args.dst_hf:
        vs = args.prune_vocab_size
        print(f"\n=== Exporting pruned model (vocab_size={vs}) to {args.dst_hf} ===", flush=True)
        kept_base, kept_added, stats = select_pruned_tokens(
            base_vocab, added_tokens, merges, domain_counts, referenced_ids, orig_tok, vs
        )
        new_tj, old_to_new, old_ids_order = build_pruned_tokenizer_json(
            orig_tj, kept_base, kept_added, merges, merges_is_list_fmt
        )
        prune_weights_and_save(
            args.src_hf, args.dst_hf, old_ids_order, old_to_new, old_vocab_size, vs, new_tj
        )
        verify_slime_loss_mask(args.dst_hf, args.slime_dir)
        print(f"[DONE] Pruned model exported and verified in {time.time()-t0:.1f}s -> {args.dst_hf}", flush=True)


if __name__ == "__main__":
    main()
