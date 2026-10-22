#!/usr/bin/env python3
"""Original trajectory-to-view preprocessing.

Public dataset: https://huggingface.co/datasets/McGill-NLP/agent-reward-bench
Download on the experiment server with huggingface_hub.snapshot_download:
  repo_id="McGill-NLP/agent-reward-bench", repo_type="dataset",
  revision="b6d17e646009d6cb63d5dd7be78807b680693f61",
  local_dir="./data/agent-reward-bench"

Use --source-records for the existing labeled trajectory index and
--dataset-root for the downloaded dataset. Generate the five final views
with --variant availability_clean, availability_no_reasoning,
availability_no_terminal_dom, availability_no_history, and
availability_reasoning_support_tagged, respectively. The public dataset
terms apply; raw data, screenshots, and model weights are not bundled here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DIRECT_SYSTEM_PROMPT = """You are an expert evaluator of web-browsing agent executions.
Decide whether the agent actually completed the user's requested Web task using only the visible execution record. Judge the achieved outcome, not the agent's intention or confidence.

Evaluation principles:
- Information-seeking tasks succeed only when the requested answer is provided and supported by the visible Web evidence.
- Navigation tasks succeed only when the requested destination or state is visibly reached.
- Modification or transaction tasks succeed only when the requested change is visibly completed or confirmed.
- Action errors, loops, wrong entities, incomplete scope, and unsupported claims are evidence of failure.
- Missing evidence is not proof of success. You must still make a binary decision.

Return exactly three lines:
SUCCESS or FAILURE
CONFIDENCE: a number from 0.00 to 1.00
RATIONALE: one concise sentence grounded in the supplied record"""


DIRECT_REVERSE_SYSTEM_PROMPT = DIRECT_SYSTEM_PROMPT.replace(
    "SUCCESS or FAILURE", "FAILURE or SUCCESS"
)


AVAILABILITY_SYSTEM_PROMPT = """You are an expert evaluator of web-browsing agent executions.
Decide whether the agent actually completed the user's requested Web task using only the supplied execution record. Judge the achieved outcome, not the agent's intention or confidence.

Evaluation principles:
- Information-seeking tasks succeed only when the requested answer is provided and supported by the available Web evidence.
- Navigation tasks succeed only when the requested destination or state is reached in the available evidence.
- Modification or transaction tasks succeed only when the requested change is completed or confirmed in the available evidence.
- Action errors, loops, wrong entities, incomplete scope, and unsupported claims are evidence of failure only when they are actually observed.
- A channel marked UNAVAILABLE was not observed. Its absence is neither success evidence nor failure evidence. Infer the outcome from the remaining channels.
- You must still make a binary decision.

Return exactly three lines:
SUCCESS or FAILURE
CONFIDENCE: a number from 0.00 to 1.00
RATIONALE: one concise sentence grounded in the supplied record"""


REASONING_CONTEXT_SYSTEM_PROMPT = """You are an expert evaluator of web-browsing agent executions.
Decide whether the agent actually completed the user's requested Web task using only the supplied execution record. Judge the achieved outcome, not the agent's intention or confidence.

Evidence semantics:
- URLs, browser actions, action errors, and visible accessibility/DOM states are observable execution evidence.
- Agent reasoning is retained as unverified context. It may contain useful calculations, constraint analysis, and descriptions that help interpret the execution.
- A claim in agent reasoning is not by itself evidence that a page state, side effect, transaction, or task completion occurred. Such claims require support from observable execution evidence.
- A channel marked UNAVAILABLE was not observed. Its absence is neither success evidence nor failure evidence.
- You must still make a binary decision.

Return exactly three lines:
SUCCESS or FAILURE
CONFIDENCE: a number from 0.00 to 1.00
RATIONALE: one concise sentence grounded in the supplied record"""


ANCHORED_REASONING_CONTEXT_SYSTEM_PROMPT = """You are an expert evaluator of web-browsing agent executions.
Decide whether the agent actually completed the user's requested Web task using only the supplied execution record. Judge the achieved outcome, not the agent's intention or confidence.

Evidence semantics:
- URLs, browser actions, action errors, and visible accessibility/DOM states are observable execution evidence.
- Agent reasoning is unverified context. Only nonredundant snippets aligned with newly observed Web-state anchors are retained.
- Alignment makes a reasoning snippet relevant, but does not make its claim evidence. Page states, side effects, transactions, and task completion still require observable support.
- A channel marked UNAVAILABLE was not observed. Its absence is neither success evidence nor failure evidence.
- You must still make a binary decision.

Return exactly three lines:
SUCCESS or FAILURE
CONFIDENCE: a number from 0.00 to 1.00
RATIONALE: one concise sentence grounded in the supplied record"""


SUPPORT_TAGGED_REASONING_CONTEXT_SYSTEM_PROMPT = """You are an expert evaluator of web-browsing agent executions.
Decide whether the agent actually completed the user's requested Web task using only the supplied execution record. Judge the achieved outcome, not the agent's intention or confidence.

Evidence semantics:
- URLs, browser actions, action errors, and visible accessibility/DOM states are observable execution evidence.
- Agent reasoning is retained as unverified context. Each reasoning step has an observable-anchor support count measured against its current Web state, executed action, and immediately following Web state.
- The support count measures alignment and relevance, not truth. Even fully aligned reasoning does not prove that a side effect, transaction, or task completion occurred.
- A missing support count means the available channels cannot measure alignment; it is not evidence of failure.
- A channel marked UNAVAILABLE was not observed. Its absence is neither success evidence nor failure evidence.
- You must still make a binary decision.

Return exactly three lines:
SUCCESS or FAILURE
CONFIDENCE: a number from 0.00 to 1.00
RATIONALE: one concise sentence grounded in the supplied record"""


AVAILABILITY_VARIANTS = {
    "availability_clean": frozenset(),
    "availability_no_reasoning": frozenset({"reasoning_history"}),
    "availability_no_terminal_dom": frozenset({"terminal_dom"}),
    "availability_no_history": frozenset(
        {"url_history", "action_history", "reasoning_history", "error_history"}
    ),
    "availability_reasoning_context": frozenset(),
    "availability_reasoning_anchored": frozenset(),
    "availability_reasoning_support_tagged": frozenset(),
}


_ACTION_TAG_RE = re.compile(r"<action>.*?</action>", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_ANCHOR_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%$-]{2,}")
_ANCHOR_STOPWORDS = frozenset(
    {
        "action",
        "agent",
        "button",
        "click",
        "clicked",
        "current",
        "fill",
        "filled",
        "find",
        "goal",
        "link",
        "next",
        "page",
        "reasoning",
        "result",
        "results",
        "search",
        "step",
        "task",
        "textbox",
        "that",
        "this",
        "using",
        "will",
        "with",
    }
)


AUTHORITY_DELTA_SYSTEM_PROMPT = """You are a binary outcome sensor for a recorded Web-agent execution.
Decide whether the user's exact Web task was actually completed using only the supplied observable evidence.

Evaluate the evidence in temporal and causal order:
1. A terminal answer, receipt, created object, persisted post-state, or exact final rendered result is the strongest evidence.
2. Action-bound changes in visible Web state can establish what an action actually changed.
3. Clicks, fills, navigation, agent claims, and action errors describe attempts; they are not outcomes by themselves.

Apply these Web-specific semantics:
- A later exact post-state may establish success despite an earlier failed attempt or stale error.
- An explicit final contradiction, wrong entity/value, incomplete scope, stale constraint, or missing requested output means failure.
- Information-seeking tasks require the requested answer to be delivered and visibly grounded; merely locating a source is insufficient.
- Mutation or transaction tasks require visible confirmation of the requested committed state; a prepared form or clicked control is insufficient.
- Rendered-view tasks require the requested controls and result state together, without incompatible retained constraints.
- Do not infer success from the agent's confidence, intention, or summary.

Internally check target identity, requested operation, values and constraints, scope completeness, and terminal outcome.
Return exactly one token as the first response token: SUCCESS or FAILURE."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        help="Optional benchmark filter when source-records is the unified trajectory index.",
    )
    parser.add_argument(
        "--variant",
        choices=(
            "direct_exact",
            "direct_order_reverse",
            "authority_delta",
            *AVAILABILITY_VARIANTS,
        ),
        default="direct_exact",
    )
    return parser.parse_args()


def compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def excerpt(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    head = max(0, limit // 2 - 14)
    tail = max(0, limit - head - 29)
    return text[:head] + "\n...[VISIBLE TEXT OMITTED]...\n" + text[-tail:]


def visible_state(step: dict[str, Any], limit: int) -> str:
    return excerpt(step.get("axtree_pruned") or step.get("axtree") or "", limit)


def axtree_lines(step: dict[str, Any]) -> list[str]:
    value = step.get("axtree_pruned") or step.get("axtree") or ""
    return [" ".join(line.split()) for line in str(value).splitlines() if line.strip()]


def anchor_tokens(value: Any) -> set[str]:
    """Extract concrete lexical anchors without interpreting their truth."""
    tokens = {
        match.group(0).casefold()
        for match in _ANCHOR_TOKEN_RE.finditer(str(value or ""))
    }
    return {
        token
        for token in tokens
        if token not in _ANCHOR_STOPWORDS
        and (any(character.isdigit() for character in token) or len(token) >= 4)
    }


def anchored_reasoning_by_step(
    steps: list[dict[str, Any]],
    *,
    limit: int = 180,
) -> dict[int, str]:
    """Project reasoning onto novel anchors in the observed Web-state sequence."""
    previous_state_tokens: set[str] = set()
    covered_anchors: set[str] = set()
    selected: dict[int, str] = {}
    for index, step in enumerate(steps):
        state_tokens = anchor_tokens(
            step.get("axtree_pruned") or step.get("axtree") or ""
        )
        newly_observed = state_tokens - previous_state_tokens
        error_tokens = anchor_tokens(step.get("last_action_error"))
        available_anchors = (newly_observed | error_tokens) - covered_anchors

        reasoning = _ACTION_TAG_RE.sub(" ", str(step.get("reasoning") or ""))
        retained_sentences: list[str] = []
        retained_anchors: set[str] = set()
        for sentence in _SENTENCE_SPLIT_RE.split(reasoning):
            sentence = " ".join(sentence.split())
            if not sentence:
                continue
            support = anchor_tokens(sentence) & available_anchors
            if not support:
                continue
            retained_sentences.append(sentence)
            retained_anchors.update(support)

        if retained_sentences:
            selected[index] = compact(" ".join(retained_sentences), limit)
            covered_anchors.update(retained_anchors)
        previous_state_tokens = state_tokens
    return selected


def support_tagged_reasoning_by_step(
    steps: list[dict[str, Any]],
    *,
    limit: int = 240,
) -> dict[int, tuple[str, int | None, int]]:
    """Attach label-free transition support while retaining reasoning content."""
    tagged: dict[int, tuple[str, int | None, int]] = {}
    for index, step in enumerate(steps):
        reasoning = compact(
            _ACTION_TAG_RE.sub(" ", str(step.get("reasoning") or "")), limit
        )
        if not reasoning:
            continue
        reasoning_tokens = anchor_tokens(reasoning)
        following = steps[index + 1] if index + 1 < len(steps) else {}
        observable_tokens = anchor_tokens(
            " ".join(
                str(value or "")
                for value in (
                    step.get("url"),
                    step.get("axtree_pruned") or step.get("axtree"),
                    step.get("action"),
                    step.get("last_action_error"),
                    following.get("url"),
                    following.get("axtree_pruned") or following.get("axtree"),
                    following.get("last_action_error"),
                )
            )
        )
        matched = (
            len(reasoning_tokens & observable_tokens)
            if reasoning_tokens and observable_tokens
            else None
        )
        tagged[index] = (reasoning, matched, len(reasoning_tokens))
    return tagged


def line_delta(previous: list[str], current: list[str]) -> tuple[list[str], list[str]]:
    previous_set = set(previous)
    current_set = set(current)
    return (
        [line for line in current if line not in previous_set],
        [line for line in previous if line not in current_set],
    )


def availability_block(
    masked_channels: frozenset[str],
    *,
    reasoning_context_only: bool = False,
    reasoning_anchored_only: bool = False,
    reasoning_support_tagged: bool = False,
) -> str:
    channels = (
        "terminal_dom",
        "url_history",
        "action_history",
        "reasoning_history",
        "error_history",
    )
    states = []
    for channel in channels:
        if channel in masked_channels:
            state = "UNAVAILABLE"
        elif channel == "reasoning_history" and reasoning_anchored_only:
            state = "AVAILABLE_ANCHORED_CONTEXT_ONLY"
        elif channel == "reasoning_history" and reasoning_support_tagged:
            state = "AVAILABLE_SUPPORT_TAGGED_CONTEXT"
        elif channel == "reasoning_history" and reasoning_context_only:
            state = "AVAILABLE_CONTEXT_ONLY"
        else:
            state = "AVAILABLE"
        states.append(f"{channel}: {state}")
    return "\n".join(["EVIDENCE AVAILABILITY", *states])


def build_user_prompt(
    raw: dict[str, Any],
    *,
    include_state_deltas: bool = False,
    masked_channels: frozenset[str] | None = None,
    reasoning_context_only: bool = False,
    reasoning_anchored_only: bool = False,
    reasoning_support_tagged: bool = False,
) -> str:
    availability_enabled = masked_channels is not None
    masked_channels = masked_channels or frozenset()
    steps = raw.get("steps") or []
    final = steps[-1] if steps else {}
    terminal_dom = (
        "[EVIDENCE_UNAVAILABLE: terminal_dom]"
        if "terminal_dom" in masked_channels
        else visible_state(final, 8_000)
    )
    final_block = "\n".join(
        [
            "FINAL OBSERVABLE STATE",
            f"URL: {compact(final.get('url'), 600)}",
            f"Action: {compact(final.get('action'), 700)}",
            f"Action error: {compact(final.get('last_action_error'), 900)}",
            "Visible accessibility/DOM text:",
            terminal_dom,
        ]
    )
    records: list[str] = []
    previous_lines: list[str] = []
    anchored_reasoning = (
        anchored_reasoning_by_step(steps) if reasoning_anchored_only else {}
    )
    tagged_reasoning = (
        support_tagged_reasoning_by_step(steps)
        if reasoning_support_tagged
        else {}
    )
    history_available = not {
        "url_history",
        "action_history",
        "reasoning_history",
        "error_history",
    }.issubset(masked_channels)
    for step_index, step in enumerate(steps if history_available else []):
        fields = [
            f"STEP {step.get('num', '?')}",
        ]
        if "url_history" not in masked_channels:
            fields.append(f"URL: {compact(step.get('url'), 240)}")
        if reasoning_anchored_only:
            reasoning = anchored_reasoning.get(step_index, "")
        elif reasoning_support_tagged:
            reasoning = tagged_reasoning.get(step_index, ("", None, 0))[0]
        else:
            reasoning = compact(step.get("reasoning"), 240)
        action = compact(step.get("action"), 280)
        error = compact(step.get("last_action_error"), 240)
        if reasoning and "reasoning_history" not in masked_channels:
            if reasoning_anchored_only:
                label = "Agent reasoning (observable-state anchored context)"
            elif reasoning_support_tagged:
                _, matched, total = tagged_reasoning[step_index]
                support = "unavailable" if matched is None else f"{matched}/{total}"
                label = (
                    "Agent reasoning (unverified context; "
                    f"observable-anchor support {support})"
                )
            elif reasoning_context_only:
                label = "Agent reasoning (unverified context)"
            else:
                label = "Reasoning"
            fields.append(f"{label}: {reasoning}")
        if action and "action_history" not in masked_channels:
            fields.append(f"Action: {action}")
        if error and "error_history" not in masked_channels:
            fields.append(f"Action error: {error}")
        if include_state_deltas:
            current_lines = axtree_lines(step)
            added, removed = line_delta(previous_lines, current_lines)
            if added:
                fields.append("Visible text added after this step:\n" + excerpt("\n".join(added), 700))
            if removed:
                fields.append(
                    "Visible text no longer present after this step:\n"
                    + excerpt("\n".join(removed), 350)
                )
            previous_lines = current_lines
        records.append("\n".join(fields))
    history_block = (
        "\n\n---\n\n".join(records)
        if records
        else "[EVIDENCE_UNAVAILABLE: chronological_history]"
    )
    prefix = ""
    if availability_enabled:
        prefix = availability_block(
            masked_channels,
            reasoning_context_only=reasoning_context_only,
            reasoning_anchored_only=reasoning_anchored_only,
            reasoning_support_tagged=reasoning_support_tagged,
        ) + "\n\n"
    return (
        prefix
        + "USER TASK\n"
        + str(raw.get("goal") or "")
        + "\n\n"
        + final_block
        + "\n\nCHRONOLOGICAL EXECUTION RECORD\n"
        + history_block
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_record_id(row: dict[str, Any]) -> str:
    value = row.get("record_id")
    if value:
        return str(value)
    required = ("benchmark", "task_id", "agent")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise KeyError(f"Cannot construct record_id; missing {missing}")
    return f"{row['benchmark']}::{row['task_id']}::{row['agent']}"


GROUP_ID_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    # VisualWebArena publishes resized variants of the same underlying task.
    # They must remain in one split to prevent template-level leakage.
    "visualwebarena": ((r"^visualwebarena\.resized\.", "visualwebarena."),),
}


def source_group_id(row: dict[str, Any]) -> str:
    """Return a leakage-safe semantic task group for grouped evaluation."""
    group_id = str(row.get("group_id") or row["task_id"])
    for pattern, replacement in GROUP_ID_ALIASES.get(
        str(row.get("benchmark")), ()
    ):
        group_id = re.sub(pattern, replacement, group_id)
    return group_id


def source_labels(row: dict[str, Any]) -> dict[str, Any]:
    def normalize(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        raise ValueError(
            f"Record {source_record_id(row)} has non-binary success label: {value!r}"
        )

    labels = row.get("labels")
    if isinstance(labels, dict) and labels.get("success") is not None:
        return {**labels, "success": normalize(labels["success"])}
    success = row.get("trajectory_success_bool")
    if success is None:
        majority = str(row.get("trajectory_success_majority") or "").strip().lower()
        if majority == "successful":
            success = True
        elif majority == "unsuccessful":
            success = False
    if success is None:
        raise ValueError(f"Record {source_record_id(row)} has no expert success label")
    return {"success": normalize(success)}


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.source_records)
    rows = [row for row in rows if row.get("valid", True) is True]
    if args.benchmark:
        rows = [
            row
            for row in rows
            if row.get("benchmark") == args.benchmark
        ]
    rows.sort(key=lambda row: source_record_id(row))
    record_ids = [source_record_id(row) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Source records contain duplicate record IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    prompt_chars: list[int] = []
    label_counts: Counter[int] = Counter()
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, 1):
            source = args.dataset_root / row["relative_path"]
            raw = json.loads(source.read_text(encoding="utf-8"))
            include_state_deltas = args.variant == "authority_delta"
            masked_channels = AVAILABILITY_VARIANTS.get(args.variant)
            reasoning_anchored_only = (
                args.variant == "availability_reasoning_anchored"
            )
            reasoning_support_tagged = (
                args.variant == "availability_reasoning_support_tagged"
            )
            reasoning_context_only = args.variant in {
                "availability_reasoning_context",
                "availability_reasoning_anchored",
                "availability_reasoning_support_tagged",
            }
            if reasoning_support_tagged:
                system_prompt = SUPPORT_TAGGED_REASONING_CONTEXT_SYSTEM_PROMPT
            elif reasoning_anchored_only:
                system_prompt = ANCHORED_REASONING_CONTEXT_SYSTEM_PROMPT
            elif reasoning_context_only:
                system_prompt = REASONING_CONTEXT_SYSTEM_PROMPT
            elif masked_channels is not None:
                system_prompt = AVAILABILITY_SYSTEM_PROMPT
            elif include_state_deltas:
                system_prompt = AUTHORITY_DELTA_SYSTEM_PROMPT
            elif args.variant == "direct_order_reverse":
                system_prompt = DIRECT_REVERSE_SYSTEM_PROMPT
            else:
                system_prompt = DIRECT_SYSTEM_PROMPT
            user_prompt = build_user_prompt(
                raw,
                include_state_deltas=include_state_deltas,
                masked_channels=masked_channels,
                reasoning_context_only=reasoning_context_only,
                reasoning_anchored_only=reasoning_anchored_only,
                reasoning_support_tagged=reasoning_support_tagged,
            )
            labels = source_labels(row)
            label_counts[int(labels["success"])] += 1
            item = {
                "record_id": source_record_id(row),
                "relative_path": row["relative_path"],
                "benchmark": row["benchmark"],
                "task_id": row["task_id"],
                "group_id": source_group_id(row),
                "agent": row["agent"],
                "labels": labels,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "prompt_variant": args.variant,
                "evidence_view": (
                    args.variant.removeprefix("availability_")
                    if masked_channels is not None
                    else None
                ),
            }
            line = json.dumps(item, ensure_ascii=False, sort_keys=True)
            digest.update((line + "\n").encode("utf-8"))
            output.write(line + "\n")
            prompt_chars.append(len(system_prompt) + len(user_prompt))
            if index % 50 == 0 or index == len(rows):
                print(json.dumps({"prepared": index, "total": len(rows)}), flush=True)
    summary = {
        "records": len(rows),
        "benchmark": args.benchmark,
        "groups": len({source_group_id(row) for row in rows}),
        "label_distribution": {
            "failure": label_counts[0],
            "success": label_counts[1],
        },
        "prompt_variant": args.variant,
        "sha256": digest.hexdigest(),
        "prompt_chars_mean": sum(prompt_chars) / len(prompt_chars) if prompt_chars else 0,
        "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
