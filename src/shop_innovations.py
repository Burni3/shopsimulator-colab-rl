"""Custom RL & Privileged-Context OPSD (On-Policy Self-Distillation) plugins for ShopSimulator.

Kept strictly outside the upstream `pi-slime-shopsimulator` repository and loaded via PYTHONPATH.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from typing import Any

import aiohttp
import torch

from examples.ShopSimulator import generate as base_gen
from examples.ShopSimulator.common import SUB_SCORES
from slime.utils.types import Sample

logger = logging.getLogger(__name__)
_OPSD_SEM = None


def _get_opsd_sem():
    import asyncio
    global _OPSD_SEM
    if _OPSD_SEM is None:
        _OPSD_SEM = asyncio.Semaphore(int(os.environ.get("OPSD_CONCURRENCY", "1")))
    return _OPSD_SEM


def _compute_composite_reward(sample: Sample) -> float:
    """Blend r_loose with r_hard and strict-success bonus to close the loose-vs-hard gap."""
    mode = os.environ.get("SHOP_REWARD_MODE", "loose").strip().lower()
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    r_loose = float(sample.reward or 0.0)
    if mode != "composite":
        return r_loose
    if not meta.get("env_done"):
        return 0.0
    detail = meta.get("reward_detail") or {}
    if not detail:
        return r_loose
    values = {
        name: float(detail.get(name, 1.0 if name == "r_option" else 0.0))
        for name in SUB_SCORES
    }
    r_hard = 1.0
    for v in values.values():
        r_hard *= v
    r_success = 1.0 if all(v == 1.0 for v in values.values()) else 0.0
    return 0.5 * r_loose + 0.5 * r_hard + 0.2 * r_success


def shaped_reward_post_process(args, samples: list[Sample]):
    """Group-normalized rewards with optional loose+hard composite shaping."""
    expected = int(args.n_samples_per_prompt)
    raw_rewards = [_compute_composite_reward(sample) for sample in samples]
    grouped: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, sample in enumerate(samples):
        if sample.group_index is None:
            raise RuntimeError("ShopSimulator sample is missing group_index")
        candidate_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        if candidate_id is None:
            raise RuntimeError("ShopSimulator sample is missing candidate identity")
        grouped[int(sample.group_index)][int(candidate_id)].append(position)

    normalized = [0.0] * len(samples)
    use_std = bool(getattr(args, "grpo_std_normalization", True))
    nonzero_variance_groups = 0
    for group_index, candidates in grouped.items():
        if len(candidates) != expected:
            raise RuntimeError(
                f"ShopSimulator group {group_index} has {len(candidates)} candidates, expected {expected}"
            )
        group_positions = [pos for positions in candidates.values() for pos in positions]
        removed = [
            samples[pos].remove_sample or samples[pos].status == Sample.Status.ABORTED
            for pos in group_positions
        ]
        if any(removed):
            if not all(samples[pos].remove_sample for pos in group_positions):
                raise RuntimeError(f"ShopSimulator group {group_index} was only partially filtered")
            continue

        candidate_rewards = []
        for candidate_id, positions in candidates.items():
            values = [raw_rewards[pos] for pos in positions]
            if not all(math.isfinite(v) for v in values) or any(abs(v - values[0]) > 1e-6 for v in values[1:]):
                raise RuntimeError(f"inconsistent reward across candidate {candidate_id} fan-out fragments")
            candidate_rewards.append((candidate_id, values[0], positions))
        if max(v for _, v, _ in candidate_rewards) != min(v for _, v, _ in candidate_rewards):
            nonzero_variance_groups += 1
        mean = sum(v for _, v, _ in candidate_rewards) / expected
        centered = [v - mean for _, v, _ in candidate_rewards]
        if use_std:
            std = math.sqrt(sum(v * v for v in centered) / max(expected - 1, 1))
            centered = [v / (std + 1e-6) for v in centered]
        for (_, _, positions), adv in zip(candidate_rewards, centered, strict=True):
            for pos in positions:
                normalized[pos] = adv

    require_signal = os.environ.get("SHOP_REQUIRE_NONZERO_VARIANCE_PER_ROLLOUT", "0") not in {"0", "false", "False"}
    if require_signal and not getattr(args, "debug_rollout_only", False) and nonzero_variance_groups == 0:
        raise RuntimeError("zero-signal ShopSimulator rollout: no complete candidate group has reward variance")
    return raw_rewards, normalized


def _build_privileged_hint(meta: dict[str, Any]) -> str:
    """Build concise privileged outcome-grounded diagnostic context for OPSD self-teacher."""
    done = bool(meta.get("env_done"))
    detail = meta.get("reward_detail") or {}
    if not done or not detail:
        reason = meta.get("termination_reason") or "未完成下单"
        return (
            f"[事后特权诊断] 当前轨迹未能成功完成下单（终止原因：{reason}）。"
            "请以高标准审查每一步工具调用：优先精准搜索、核对全部商品属性/规格选项/价格上限后及时完成购买，"
            "避免无效重复操作或过早放弃。\n"
        )
    r_type = float(detail.get("r_type", 0.0))
    r_att = float(detail.get("r_att", 0.0))
    r_option = float(detail.get("r_option", 1.0))
    r_price = float(detail.get("r_price", 0.0))
    right_prod = meta.get("purchase_asin") is not None and meta.get("purchase_asin") == meta.get("goal_asin")
    issues = []
    if r_type < 1.0:
        issues.append("商品品类不匹配(r_type<1)")
    if r_att < 1.0:
        issues.append(f"商品属性未完全满足(r_att={r_att:.2f})")
    if r_option < 1.0:
        issues.append("规格/颜色/尺码选项选错(r_option<1)")
    if r_price < 1.0:
        issues.append("价格超出预算上限(r_price<1)")
    if not issues:
        return (
            "[事后特权诊断] 本次购物决策完全正确（品类、属性、规格选项、价格约束全部 1.0 达标"
            + ("且精准命中目标商品" if right_prod else "")
            + "）。请确认并强化此类规范高效的检索、核验与下单决策。\n"
        )
    issue_text = "；".join(issues)
    return (
        f"[事后特权诊断] 本次购买存在以下硬约束失分：{issue_text}。"
        "请严格按照目标要求审查每一步搜索词、商品筛选、规格选项点选和价格核验，"
        "对导致约束违背的草率或错误决策给予低置信度。\n"
    )


def _zero_teacher_logprobs(sample: Sample) -> torch.Tensor:
    resp_len = max(int(sample.response_length or 0), 1)
    if sample.rollout_log_probs is not None and len(sample.rollout_log_probs) == resp_len:
        return torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
    return torch.zeros(resp_len, dtype=torch.float32)


async def _fetch_self_teacher_logprobs(
    sglang_url: str,
    input_ids: list[int],
    response_length: int,
    temperature: float,
) -> torch.Tensor | None:
    """Query the live colocate SGLang engine (prefill-only) for privileged self-teacher logprobs."""
    if response_length <= 0 or len(input_ids) <= response_length:
        return None
    start_len = max(0, len(input_ids) - response_length - 1)
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": start_len,
    }
    timeout = aiohttp.ClientTimeout(total=60)
    async with _get_opsd_sem():
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as client:
            async with client.post(f"{sglang_url}/generate", json=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning("OPSD self-teacher request failed (%s): %s", resp.status, text[:200])
                    return None
                data = await resp.json(content_type=None)
    meta_info = data.get("meta_info") if isinstance(data, dict) else None
    raw_lp = meta_info.get("input_token_logprobs") if isinstance(meta_info, dict) else None
    if not raw_lp:
        return None
    # SGLang returns (logprob, token_id, ...) starting at start_len; the first token at index 0 has None logprob.
    vals = [float(item[0]) for item in raw_lp if item and item[0] is not None]
    if len(vals) < response_length:
        return None
    tensor = torch.tensor(vals[-response_length:], dtype=torch.float32)
    if not torch.all(torch.isfinite(tensor)):
        return None
    return tensor


async def generate_opd_external(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run Pi+ShopSimulator rollout, then query a larger external SGLang teacher (e.g. Qwen3.5-4B) for OPD logprobs."""
    samples = await base_gen.generate(args, base_sample, sampling_params, evaluation=evaluation)
    if evaluation or not getattr(args, "use_opd", False):
        return samples

    teacher_url = os.environ.get("OPD_TEACHER_URL", "http://127.0.0.1:13141")
    temperature = float(getattr(args, "rollout_temperature", 1.0) or 1.0)
    for sample in samples:
        if sample.remove_sample or sample.status == Sample.Status.ABORTED or not sample.tokens:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            continue
        resp_len = int(sample.response_length or 0)
        if resp_len <= 0 or len(sample.tokens) <= resp_len:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            continue
        try:
            t_lp = await _fetch_self_teacher_logprobs(
                sglang_url=teacher_url,
                input_ids=list(sample.tokens),
                response_length=resp_len,
                temperature=temperature,
            )
        except Exception as exc:
            logger.warning("External OPD teacher exception: %s", exc)
            t_lp = None
        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        if t_lp is not None:
            sample.teacher_log_probs = t_lp
            meta["opd_external_applied"] = True
        else:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            meta["opd_external_applied"] = False
        sample.metadata = meta
    return samples


async def generate_opsd(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Run real Pi+ShopSimulator rollout, then compute Privileged-Context OPSD teacher_log_probs."""
    samples = await base_gen.generate(args, base_sample, sampling_params, evaluation=evaluation)
    if evaluation or not getattr(args, "use_opd", False):
        return samples

    state = base_gen.AdapterService(args)
    tokenizer = state.adapter.tokenizer
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    max_ctx = int(getattr(args, "rollout_max_context_len", 16384) or 16384)
    temperature = float(getattr(args, "rollout_temperature", 1.0) or 1.0)

    for sample in samples:
        if sample.remove_sample or sample.status == Sample.Status.ABORTED or not sample.tokens:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            continue

        resp_len = int(sample.response_length or 0)
        if resp_len <= 0 or len(sample.tokens) <= resp_len:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            continue

        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        hint_text = "\n" + _build_privileged_hint(meta)
        hint_ids = tokenizer.encode(hint_text, add_special_tokens=False)

        prompt_ids = list(sample.tokens[:-resp_len])
        response_ids = list(sample.tokens[-resp_len:])

        # Inject privileged diagnostic cleanly at the end of the system message (before first <|im_end|>),
        # preserving valid Qwen3.5 chat template structure.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id in prompt_ids:
            insert_pos = prompt_ids.index(im_end_id)
            priv_prompt_ids = prompt_ids[:insert_pos] + hint_ids + prompt_ids[insert_pos:]
        else:
            priv_prompt_ids = hint_ids + prompt_ids

        if len(priv_prompt_ids) + len(response_ids) <= max_ctx:
            priv_input_ids = priv_prompt_ids + response_ids
            try:
                t_lp = await _fetch_self_teacher_logprobs(
                    sglang_url=sglang_url,
                    input_ids=priv_input_ids,
                    response_length=resp_len,
                    temperature=temperature,
                )
            except Exception as exc:
                logger.warning("OPSD self-teacher exception: %s", exc)
                t_lp = None
        else:
            t_lp = None

        if t_lp is not None:
            sample.teacher_log_probs = t_lp
            meta["opsd_privileged_applied"] = True
        else:
            sample.teacher_log_probs = _zero_teacher_logprobs(sample)
            meta["opsd_privileged_applied"] = False
        sample.metadata = meta

    return samples
