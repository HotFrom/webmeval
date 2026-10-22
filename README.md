# WebMEval

Minimal source code for **WebMEval: Multi-View Latent Evaluation of Web-Agent Task Completion**.

WebMEval constructs complete, evidence-omission, and support-tagged trajectory views. A shared linear readout is fitted on four training views. At inference, normalized margins from the complete and support-tagged views are combined with a weight and threshold selected from training task-group out-of-fold predictions.

## Files

| File | Purpose |
| --- | --- |
| `core.py` | Shared Ridge/LDA readouts, feature validation, grouped folds, calibration, fusion, and metrics. |
| `run.py` | Original fixed-split and task-group evaluation runners, with machine-specific paths replaced by command-line arguments. |
| `prepare_data.py` | Original trajectory serialization and evidence-view construction. |

This is a three-file source release. `run.py` operates on existing frozen hidden-state features; it does not launch a language-model server or repeat encoder inference. Raw trajectories, model weights, feature caches, split files, and archived reference outputs are not included in this repository.

## Environment

Use a Linux experiment server with Python 3.12. The archived numerical environment used:

```bash
python -m pip install numpy==2.3.5 scipy==1.18.0 scikit-learn==1.9.0 huggingface_hub
```

The paper's representations were extracted from Qwen3.8-27B-FP8 at language layer 64, with a 32,768-token limit, thinking disabled, and zero generated tokens. Encoder deployment is outside this minimal source release.

## Dataset Download

The four benchmarks are provided through [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench): WebArena, WorkArena, VisualWebArena, and AssistantBench.

Download the public textual trajectories and annotations on the experiment server:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="McGill-NLP/agent-reward-bench",
    repo_type="dataset",
    revision="b6d17e646009d6cb63d5dd7be78807b680693f61",
    local_dir="./data/agent-reward-bench",
    allow_patterns=["README.md", "data/*", "cleaned/**"],
)
```

The dataset's terms of use and third-party data conditions apply. The source dataset and its authors retain their respective rights.

## Data Processing

`prepare_data.py` takes an existing labeled trajectory-index JSONL and the downloaded dataset root. Each index record supplies `benchmark`, `task_id`, `agent`, `relative_path`, and a binary success label through `labels.success` or the supported expert-label fields. `record_id` and `group_id` may be supplied explicitly; otherwise the original preprocessing functions derive them.

Generate the five views by repeating the preprocessing command:

```bash
mkdir -p ./prepared

for view in clean no_reasoning no_terminal_dom no_history reasoning_support_tagged; do
  python prepare_data.py \
    --source-records /path/to/records.webarena385.jsonl \
    --dataset-root ./data/agent-reward-bench \
    --benchmark webarena \
    --variant "availability_${view}" \
    --output "./prepared/records.webarena385.availability_${view}.jsonl"
done
```

| View | Modification |
| --- | --- |
| `clean` | Complete evidence, including agent reasoning. |
| `no_reasoning` | Omit agent reasoning. |
| `no_terminal_dom` | Mask terminal page text. The field uses recorded AXTree text. |
| `no_history` | Omit chronological interaction history while retaining the terminal block. |
| `reasoning_support_tagged` | Retain cleaned reasoning with lexical support counts from observed transitions. |

The first four views train the shared head. `clean` and `reasoning_support_tagged` are the two deployment views. Support counts measure lexical overlap, not whether a claim is true.

## Run Evaluation

Provide the original frozen-artifact root with prepared records, feature files, task partitions, and reference evaluation files. The expected layout is:

```text
frozen/
  datasets/prepared/
    records.<dataset-stem>.availability_<view>.jsonl
    split.webarena385.train200_test185.json
  results/<result-name>/
    features/<view>.final64.npz
    evaluation/summary.json
    evaluation/predictions.jsonl
```

For VisualWebArena, the reference evaluation files are under `evaluation/full300/`. Feature artifacts must retain the original provenance metadata, record ordering, and prompt identities checked by `core.py`.

Run one dataset at a time:

```bash
python run.py --dataset webarena --frozen-root /path/to/frozen --output-root ./results
python run.py --dataset workarena --frozen-root /path/to/frozen --output-root ./results
python run.py --dataset visualwebarena --frozen-root /path/to/frozen --output-root ./results
python run.py --dataset assistantbench --frozen-root /path/to/frozen --output-root ./results
```

WebArena uses the fixed 200-training/185-test task-disjoint split. The other datasets retain their original five task-group folds: WorkArena has 472 trajectories, VisualWebArena 300, and AssistantBench 132. VisualWebArena resized and original task identifiers share a canonical group. The exact artifact stems and result-directory names are defined in `DATASETS` in `run.py`.

Each dataset's output directory must not already exist. Use a new `--output-root` for another run.

### Final Method Output

The final WebMEval result is:

```python
import json

with open("results/webarena/RESULT.json") as f:
    results = json.load(f)

webmeval = results["frozen_complete_reference"]["weighted_mean_calibrated"]
print(webmeval)
```

The archived runner also retains score-level comparison controls. In particular, the `authority_*` entries are controls, not the final weighted-fusion method. Per-record scores and decisions are saved in `predictions.jsonl`; heads and calibration records are saved alongside the result.

These cached-feature runs do not measure model evaluation latency.
