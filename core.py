"""Final shared-head and calibrated weighted-fusion implementation.

The functions below are copied unchanged from the archived experiment code.
Former local modules are consolidated here; no prediction formula is changed.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                            f1_score, log_loss, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

EPSILON = 1e-8
SCHEMA = "webjev-feature-provenance-v1"



class RidgeProbe:
    """Closed-form dual ridge readout, matching the core AnyJEV L2 mechanism."""

    def __init__(self, n_classes: int, ridge_lambda: float):
        self.n_classes = n_classes
        self.ridge_lambda = ridge_lambda
        self.classes_ = np.arange(n_classes)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RidgeProbe":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1e-6
        x = (x - self.mean) / self.scale
        targets = np.zeros((len(y), self.n_classes), dtype=np.float64)
        targets[np.arange(len(y)), y] = 1.0
        x_mean = x.mean(axis=0)
        target_mean = targets.mean(axis=0)
        x_centered = x - x_mean
        target_centered = targets - target_mean
        # np.matmul on the macOS Accelerate build can leave spurious floating-point
        # status flags for these tall matrices. einsum computes the same Gram matrix
        # without changing the closed-form ridge solution.
        system = np.einsum("nd,md->nm", x_centered, x_centered, optimize=True)
        system.flat[:: len(system) + 1] += self.ridge_lambda
        solved = np.linalg.solve(system, target_centered)
        self.weights = np.einsum("nd,nk->dk", x_centered, solved, optimize=True)
        self.bias = target_mean - np.einsum(
            "d,dk->k", x_mean, self.weights, optimize=True
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        scores = np.einsum("nd,dk->nk", x, self.weights, optimize=True) + self.bias
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

class LDAProbe:
    """Shrunk Fisher discriminant solved through the sample-space system."""

    def __init__(self, n_classes: int, shrinkage: float):
        self.n_classes = n_classes
        self.shrinkage = shrinkage
        self.classes_ = np.arange(n_classes)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LDAProbe":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0) + 1e-6
        x = (x - self.mean) / self.scale
        n, width = x.shape
        global_mean = x.mean(axis=0)
        means = np.stack(
            [x[y == label].mean(axis=0) if np.any(y == label) else global_mean
             for label in range(self.n_classes)]
        )
        residual = x - means[y]
        tau = float(np.sum(residual * residual) / max(1, n - 1) / width)
        tau = max(tau, EPSILON)
        alpha = float(np.clip(self.shrinkage, 1e-3, 1.0))
        mean_matrix = means.T
        if alpha >= 1.0:
            self.weights = mean_matrix / (alpha * tau)
        else:
            coefficient = (1.0 - alpha) / max(1, n - 1)
            gram = np.einsum("nd,md->nm", residual, residual, optimize=True)
            system = (alpha * tau) * np.eye(n) + coefficient * gram
            rhs = np.einsum("nd,dk->nk", residual, mean_matrix, optimize=True)
            solved = np.linalg.solve(system, rhs)
            correction = np.einsum(
                "nd,nk->dk", residual, solved, optimize=True
            )
            self.weights = (
                mean_matrix - coefficient * correction
            ) / (alpha * tau)
        priors = np.asarray(
            [max(1, np.sum(y == label)) for label in range(self.n_classes)],
            dtype=float,
        )
        quadratic = np.einsum(
            "kd,dk->k", means, self.weights, optimize=True
        )
        self.bias = -0.5 * quadratic + np.log(priors / priors.sum())
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        scores = np.einsum("nd,dk->nk", x, self.weights, optimize=True) + self.bias
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def digest_ids(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(str(row["record_id"]) for row in rows) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def fit_head(
    kind: str,
    regularization: float,
    features: np.ndarray,
    labels: np.ndarray,
) -> RidgeProbe | LDAProbe:
    if kind == "ridge":
        return RidgeProbe(2, regularization).fit(features, labels)
    if kind == "lda":
        return LDAProbe(2, regularization).fit(features, labels)
    raise ValueError(f"Unknown head kind: {kind}")

def make_group_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros(len(labels)), labels, groups))
    assignments: dict[str, int] = {}
    for fold_index, (_, valid) in enumerate(result):
        for group in np.unique(groups[valid]):
            key = str(group)
            if key in assignments:
                raise RuntimeError(f"Group {key} appears in multiple validation folds")
            assignments[key] = fold_index
    if set(assignments) != {str(value) for value in np.unique(groups)}:
        raise RuntimeError("Group folds do not cover every task group")
    return result

def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def prompt_digest(rows: Iterable[dict[str, Any]]) -> str:
    payload = [
        {
            "record_id": str(row["record_id"]),
            "prompt_variant": row.get("prompt_variant"),
            "messages": row["messages"],
        }
        for row in rows
    ]
    return sha256_bytes(stable_json(payload).encode("utf-8"))

def parse_provenance(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("Unsupported or missing WebJEV feature provenance")
    expected = value.get("digest")
    unsigned = dict(value)
    unsigned.pop("digest", None)
    measured = sha256_bytes(stable_json(unsigned).encode("utf-8"))
    if expected != measured:
        raise ValueError("Feature provenance digest mismatch")
    return value

def validate_service_attestation_artifact(
    raw: str, provenance: dict[str, Any]
) -> dict[str, Any]:
    value = json.loads(raw)
    if value.get("schema") != "webjev-sglang-service-attestation-v1":
        raise ValueError("Unsupported embedded SGLang service attestation")
    expected = provenance["config"].get("service_run_attestation_digest")
    measured = sha256_bytes(stable_json(value).encode("utf-8"))
    if expected != measured:
        raise ValueError("Embedded SGLang service attestation digest mismatch")
    runtime = value.get("runtime") or {}
    if runtime.get("service_nonce") != value.get("service_nonce"):
        raise ValueError("Embedded SGLang service/runtime nonce mismatch")
    if int(runtime.get("actual_end_layer", -1)) != int(
        provenance["config"].get("hidden_layer", -2)
    ):
        raise ValueError("Embedded SGLang runtime layer mismatch")
    return value

def validate_provenance(
    value: dict[str, Any],
    *,
    expected_variant: str | None = None,
    expected_records: Path | None = None,
    expected_rows: list[dict[str, Any]] | None = None,
) -> None:
    if expected_variant is not None and value["prompt_variant"] != expected_variant:
        raise ValueError(
            f"Feature prompt variant mismatch: {value['prompt_variant']} != {expected_variant}"
        )
    if expected_records is not None and value["records_sha256"] != sha256_file(expected_records):
        raise ValueError("Feature artifact was extracted from different prepared records")
    if expected_rows is not None and value["prompt_sha256"] != prompt_digest(expected_rows):
        raise ValueError("Feature artifact was extracted from different prompt messages")

def validate_compatible(
    values: dict[str, dict[str, Any]],
    *,
    allowed_config_differences: frozenset[str] = frozenset(),
) -> None:
    if not values:
        raise ValueError("No feature provenance supplied")
    fields = (
        "model_identity_sha256",
        "tokenizer_identity_sha256",
        "extractor",
    )
    reference_name, reference = next(iter(values.items()))
    for name, value in values.items():
        for field in fields:
            if value[field] != reference[field]:
                raise ValueError(
                    f"Feature provenance {field} differs between {reference_name} and {name}"
                )
        ignored_config = allowed_config_differences | frozenset(
            {"service_run_attestation_digest"}
        )
        reference_config = {
            key: item
            for key, item in reference["config"].items()
            if key not in ignored_config
        }
        value_config = {
            key: item
            for key, item in value["config"].items()
            if key not in ignored_config
        }
        if value_config != reference_config:
            raise ValueError(
                f"Feature extraction config differs between {reference_name} and {name}"
            )

def expected_variant_for_view(name: str) -> str:
    return name if name.startswith("availability_") else f"availability_{name}"

def records_path_for_view(
    anchor_path: Path,
    anchor_view: str,
    target_view: str,
    *,
    variants_by_view: dict[str, str] | None = None,
) -> Path:
    variants = variants_by_view or {}
    anchor_variant = variants.get(anchor_view, expected_variant_for_view(anchor_view))
    target_variant = variants.get(target_view, expected_variant_for_view(target_view))
    if anchor_variant not in anchor_path.name:
        raise ValueError(
            f"Records filename must contain its prompt variant {anchor_variant}: "
            f"{anchor_path}"
        )
    target = anchor_path.with_name(
        anchor_path.name.replace(anchor_variant, target_variant, 1)
    )
    if not target.is_file():
        raise ValueError(f"Missing prepared records for {target_view}: {target}")
    return target

def load_feature_view(
    path: Path,
    expected_ids: list[str],
    expected_digest: str,
    *,
    expected_variant: str | None = None,
    expected_records_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Expected feature IDs contain duplicates")
    with np.load(path, allow_pickle=False) as stored:
        if "provenance_json" not in stored:
            raise ValueError(f"Feature artifact lacks provenance: {path}")
        provenance = parse_provenance(str(stored["provenance_json"].item()))
        service_attestation = None
        if provenance["extractor"] == "sglang_final_hidden_plus_lm_head":
            if "service_attestation_json" not in stored:
                raise ValueError(f"Feature artifact lacks service attestation: {path}")
            service_attestation = validate_service_attestation_artifact(
                str(stored["service_attestation_json"].item()), provenance
            )
        ids = stored["record_ids"].astype(str).tolist()
        digest = str(stored["record_digest"].item())
        features = np.asarray(stored["features"], dtype=np.float32)
        truncated = np.asarray(stored["truncated"], dtype=bool)
    if ids != expected_ids:
        raise ValueError(f"Feature record order mismatch: {path}")
    if digest != expected_digest:
        raise ValueError(f"Feature record digest mismatch: {path}")
    if np.any(truncated):
        raise ValueError(f"Missing-evidence evaluation requires zero truncation: {path}")
    if features.ndim != 2 or features.shape[0] != len(expected_ids):
        raise ValueError(f"Feature matrix shape mismatch: {path}: {features.shape}")
    if truncated.shape != (len(expected_ids),):
        raise ValueError(f"Truncation vector shape mismatch: {path}: {truncated.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features: {path}")
    expected_rows = (
        read_jsonl(expected_records_path) if expected_records_path is not None else None
    )
    validate_provenance(
        provenance,
        expected_variant=expected_variant,
        expected_records=expected_records_path,
        expected_rows=expected_rows,
    )
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else None
    )
    return features, {
        "extraction": summary,
        "provenance": provenance,
        "service_attestation": service_attestation,
    }

def load_feature_views(
    paths: dict[str, Path],
    expected_ids: list[str],
    expected_digest: str,
    *,
    records_path: Path,
    records_view: str = "clean",
    records_by_view: dict[str, Path] | None = None,
    variants_by_view: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    features: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    explicit_records = records_by_view or {}
    variants = variants_by_view or {}
    unknown_records = set(explicit_records) - set(paths)
    unknown_variants = set(variants) - set(paths)
    if unknown_records or unknown_variants:
        raise ValueError(
            f"Unknown view overrides: records={sorted(unknown_records)}, "
            f"variants={sorted(unknown_variants)}"
        )
    for name, path in paths.items():
        expected_records_path = explicit_records.get(name) or records_path_for_view(
            records_path,
            records_view,
            name,
            variants_by_view=variants,
        )
        features[name], metadata[name] = load_feature_view(
            path,
            expected_ids,
            expected_digest,
            expected_variant=variants.get(name, expected_variant_for_view(name)),
            expected_records_path=expected_records_path,
        )
    validate_compatible(
        {name: item["provenance"] for name, item in metadata.items()}
    )
    return features, metadata

def hashed_mask_indices(record_ids: list[str], missing_views: list[str]) -> np.ndarray:
    if not missing_views:
        raise ValueError("At least one missing view is required")
    selected = []
    for record_id in record_ids:
        digest = hashlib.sha256(record_id.encode("utf-8")).digest()
        selected.append(int.from_bytes(digest[:8], "big") % len(missing_views))
    return np.asarray(selected, dtype=int)

def make_augmented_training(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    record_ids: list[str],
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    view_names = list(features)
    if strategy == "all_views":
        return (
            np.concatenate([features[name][train] for name in view_names], axis=0),
            np.concatenate([labels[train] for _ in view_names], axis=0),
            {name: int(len(train)) for name in view_names},
        )
    if strategy != "clean_plus_hashed_mask":
        raise ValueError(f"Unknown training strategy: {strategy}")
    missing_views = [name for name in view_names if name != "clean"]
    selected = hashed_mask_indices([record_ids[index] for index in train], missing_views)
    masked = np.stack(
        [features[missing_views[choice]][index] for index, choice in zip(train, selected)]
    )
    counts = Counter(missing_views[choice] for choice in selected)
    return (
        np.concatenate([features["clean"][train], masked], axis=0),
        np.concatenate([labels[train], labels[train]], axis=0),
        {"clean": int(len(train)), **{name: int(counts[name]) for name in missing_views}},
    )

def fit_robust_model(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train: np.ndarray,
    kind: str,
    regularization: float,
) -> Any:
    record_ids = [str(index) for index in range(len(labels))]
    augmented_x, augmented_y, _ = make_augmented_training(
        features, labels, train, record_ids, "all_views"
    )
    return fit_head(kind, regularization, augmented_x, augmented_y)

def decision_margin(model: Any, features: np.ndarray) -> np.ndarray:
    """Read the binary pre-softmax margin from a fitted Ridge/LDA head."""

    values = (np.asarray(features, dtype=np.float64) - model.mean) / model.scale
    scores = np.einsum("nd,dk->nk", values, model.weights, optimize=True) + model.bias
    return scores[:, 1] - scores[:, 0]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def metrics(labels, prediction, score=None):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    y, p = np.asarray(labels, dtype=int), np.asarray(prediction, dtype=int)
    assert y.shape == p.shape and y.ndim == 1
    assert set(np.unique(y)) == {0, 1} and set(np.unique(p)).issubset({0, 1})
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    result = {"count": len(y), "coverage": 1.0, "correct": tp + tn,
              "accuracy": float(accuracy_score(y, p)),
              "balanced_accuracy": float(balanced_accuracy_score(y, p)),
              "macro_f1": float(f1_score(y, p, average="macro", labels=[0, 1], zero_division=0)),
              "FS": fp, "FF": fn, "TP": tp, "TN": tn}
    if score is not None:
        result["roc_auc"] = float(roc_auc_score(y, score))
    return result

def threshold_candidates(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not len(values):
        raise ValueError("Finite nonempty one-dimensional calibration scores required")
    unique = np.unique(values)
    return np.unique(np.r_[np.nextafter(unique[0], -np.inf),
                           unique[:-1] / 2 + unique[1:] / 2,
                           np.nextafter(unique[-1], np.inf), 0.0])

def select_threshold(values, labels):
    values, y = np.asarray(values, dtype=float), np.asarray(labels, dtype=int)
    if y.shape != values.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Aligned binary calibration labels required")
    candidates = threshold_candidates(values)
    p = values[None, :] >= candidates[:, None]
    tp, tn = (p & (y == 1)).sum(axis=1), (~p & (y == 0)).sum(axis=1)
    fp, fn = (p & (y == 0)).sum(axis=1), (~p & (y == 1)).sum(axis=1)
    bacc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
    index = max(range(len(candidates)), key=lambda i:
                (float(bacc[i]), float(f1[i]), -abs(float(candidates[i])), float(candidates[i])))
    return {"threshold": float(candidates[index]), "oof_balanced_accuracy": float(bacc[index]),
            "oof_macro_f1": float(f1[index])}

def normalize(values, fit_indices):
    values = np.asarray(values, dtype=float)
    scale = max(1e-8, float(np.median(np.abs(values[fit_indices]))))
    return values / scale, scale

def base_scores(reference, support, reference_scale, support_scale):
    e, s = np.asarray(reference, dtype=float), np.asarray(support, dtype=float)
    en, sn = e / reference_scale, s / support_scale
    mean = (en + sn) / 2
    return {"single": en, "raw_mean": (e + s) / 2,
            "scaled_mean": mean, "and": np.minimum(en, sn),
            "authority": np.minimum(en, mean)}

def fit_policies(oof, labels, alphas):
    policies = {}
    for panel, scores in oof.items():
        selected = {name: select_threshold(scores[name], labels)
                    for name in ("single", "scaled_mean", "authority")}
        weighted = []
        for alpha in alphas:
            values = float(alpha) * scores["reference_normalized"] + (1 - float(alpha)) * scores["support_normalized"]
            weighted.append({"alpha": float(alpha), **select_threshold(values, labels)})
        best = max(weighted, key=lambda item: (item["oof_balanced_accuracy"], item["oof_macro_f1"],
                   -abs(item["alpha"] - 0.5), -abs(item["threshold"]), item["threshold"]))
        selected["weighted_mean"] = best
        selected["weighted_candidates"] = weighted
        policies[panel] = selected
    return policies

def apply_policies(scores, policy):
    weighted = policy["weighted_mean"]
    blend = weighted["alpha"] * scores["reference_normalized"] + (1 - weighted["alpha"]) * scores["support_normalized"]
    return {"single_zero": scores["single"],
            "single_calibrated": scores["single"] - policy["single"]["threshold"],
            "raw_mean_zero": scores["raw_mean"], "scaled_mean_zero": scores["scaled_mean"],
            "scaled_mean_calibrated": scores["scaled_mean"] - policy["scaled_mean"]["threshold"],
            "weighted_mean_calibrated": blend - weighted["threshold"],
            "and_zero": scores["and"], "authority_zero": scores["authority"],
            "authority_calibrated": scores["authority"] - policy["authority"]["threshold"]}

def paired(y, ours, other, groups, reps, seed):
    y, ours, other = np.asarray(y), np.asarray(ours), np.asarray(other)
    wins = int(((ours == y) & (other != y)).sum())
    losses = int(((ours != y) & (other == y)).sum())
    n = wins + losses
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    unique = sorted(set(groups))
    counts = []
    for group in unique:
        mask = groups == group
        def confusion(pred):
            return [int(((y == a) & (pred == b) & mask).sum()) for a, b in ((1, 1), (0, 0), (0, 1), (1, 0))]
        counts.append([confusion(ours), confusion(other)])
    counts = np.asarray(counts)
    sampled = np.random.default_rng(seed).integers(0, len(unique), (reps, len(unique)))
    aggregated = counts[sampled].sum(axis=1)
    tp, tn, fp, fn = [aggregated[:, :, i] for i in range(4)]
    valid = ((tp + fn) > 0).all(axis=1) & ((tn + fp) > 0).all(axis=1)
    bacc = (tp[valid] / (tp[valid] + fn[valid]) + tn[valid] / (tn[valid] + fp[valid])) / 2
    delta = bacc[:, 0] - bacc[:, 1]
    return {"authority_only_correct": wins, "control_only_correct": losses,
            "mcnemar_exact_p": p, "bootstrap_valid_reps": int(valid.sum()),
            "bacc_delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist()}
