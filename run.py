#!/usr/bin/env python3
"""Final archived feature-to-decision runners (no encoder rerun).

Usage on a Linux server:
  python run.py --dataset webarena --frozen-root /path/to/frozen --output-root ./results
  python run.py --dataset workarena --frozen-root /path/to/frozen --output-root ./results

Dependencies: numpy==2.3.5, scipy==1.18.0, scikit-learn==1.9.0.
The existing frozen data/features/reference split files are runtime inputs,
not bundled into this three-file source-only package. The primary result is
RESULT.json -> frozen_complete_reference -> weighted_mean_calibrated.
Other score-level controls remain exactly as in the original experiments.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sklearn
from typing import Any
from core import (apply_policies, base_scores, fit_policies, metrics, normalize,
                  paired, sha256, decision_margin, fit_robust_model,
                  load_feature_views, digest_ids, make_group_folds)
ROOT = Path(__file__).resolve().parent

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def read(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for name in ("mean", "scale", "weights", "bias"):
        digest.update(np.ascontiguousarray(getattr(model, name), dtype=np.float64).tobytes())
    return digest.hexdigest()

def resolve_split(
    rows: list[dict[str, Any]], split: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    record_ids = [str(row["record_id"]) for row in rows]
    id_to_index = {record_id: index for index, record_id in enumerate(record_ids)}
    if len(id_to_index) != len(record_ids):
        raise ValueError("Duplicate record IDs")
    train_ids = [str(value) for value in split["train_record_ids"]]
    test_ids = [str(value) for value in split["test_record_ids"]]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate train record IDs")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate test record IDs")
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test record IDs overlap")
    if set(train_ids) | set(test_ids) != set(record_ids):
        raise ValueError("Split and records do not match exactly")
    train = np.asarray([id_to_index[value] for value in train_ids], dtype=int)
    test = np.asarray([id_to_index[value] for value in test_ids], dtype=int)
    groups = np.asarray([str(row["group_id"]) for row in rows])
    if set(groups[train]) & set(groups[test]):
        raise ValueError("Task-group leakage")
    return train, test

COMMON_CONFIG = {'training_views': ['clean', 'no_reasoning', 'no_terminal_dom', 'no_history'], 'score_views': ['clean', 'no_reasoning', 'reasoning_support_tagged'], 'reference_panels': {'frozen_complete_reference': 'clean', 'strict_reasoning_free_reference': 'no_reasoning'}, 'calibration_folds': 5, 'seed': 20261002, 'weighted_fusion_alphas': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], 'rules': ['single_zero', 'single_calibrated', 'raw_mean_zero', 'scaled_mean_zero', 'scaled_mean_calibrated', 'weighted_mean_calibrated', 'and_zero', 'authority_zero', 'authority_calibrated'], 'bootstrap_reps': 2000, 'threshold_selection': 'training-group OOF balanced accuracy, then Macro-F1, then nearest zero, then higher threshold'}

DATASETS = {'webarena': {'benchmark': 'webarena', 'train_count': 200, 'test_count': 185, 'head_kind': 'lda', 'regularization': 0.9}, 'workarena': {'benchmark': 'workarena', 'record_stem': 'workarena472', 'result_name': 'workarena472_current_oof_uncalibrated', 'record_count': 472, 'group_count': 118, 'outer_folds': 5, 'evaluation_subset': ''}, 'visualwebarena': {'benchmark': 'visualwebarena', 'record_stem': 'visualwebarena300', 'result_name': 'visualwebarena300_current_oof_uncalibrated', 'record_count': 300, 'group_count': 100, 'outer_folds': 5, 'evaluation_subset': 'full300'}, 'assistantbench': {'benchmark': 'assistantbench', 'record_stem': 'assistantbench132', 'result_name': 'assistantbench132_current_oof_uncalibrated', 'record_count': 132, 'group_count': 33, 'outer_folds': 5, 'evaluation_subset': ''}}

def run_fixed(cfg, output_root):
    assert sys.platform == 'linux', 'Run only on the server'
    frozen = Path(cfg['frozen_root'])
    out = output_root / cfg['benchmark']
    out.mkdir(exist_ok=False)
    save(out / 'CONFIG.json', cfg)
    save(out / 'STATUS.json', {'status': 'running', 'stage': 'loading_frozen_features'})
    records = frozen / 'datasets/prepared/records.webarena385.availability_clean.jsonl'
    split_path = records.with_name('split.webarena385.train200_test185.json')
    reference_path = frozen / 'results/webarena385_train200_test185_current/evaluation/predictions.jsonl'
    original_summary_path = reference_path.with_name('summary.json')
    rows = read(records)
    ids = [r['record_id'] for r in rows]
    y = np.asarray([r['labels']['success'] for r in rows], dtype=int)
    groups = np.asarray([r['group_id'] for r in rows])
    (train, test) = resolve_split(rows, json.loads(split_path.read_text()))
    assert len(train) == cfg['train_count'] and len(test) == cfg['test_count']
    assert not set(groups[train]) & set(groups[test])
    needed = cfg['training_views'] + ['reasoning_support_tagged']
    paths = {name: frozen / f'results/webarena385_train200_test185_current/features/{name}.final64.npz' for name in needed}
    (features, provenance) = load_feature_views(paths, ids, digest_ids(rows), records_path=records)
    prepared = {name: records.with_name(f'records.webarena385.availability_{name}.jsonl') for name in needed}
    sources = [ROOT / 'run.py', ROOT / 'core.py', ROOT / 'prepare_data.py', records, reference_path, *paths.values(), *prepared.values()]
    hashes = {str(p): sha256(p) for p in set(sources)}
    save(out / 'SOURCE_SHA256.json', hashes)
    prompt_audit = {}
    for name in ('clean', 'no_reasoning', 'reasoning_support_tagged'):
        items = read(prepared[name])
        assert [r['record_id'] for r in items] == ids
        count = sum(('Reasoning:' in r['messages'][1]['content'] for r in items))
        prompt_audit[name] = {'count': len(items), 'literal_reasoning_lines_present': count, 'sha256': hashes[str(prepared[name])]}
    assert prompt_audit['clean']['literal_reasoning_lines_present'] == 385
    assert prompt_audit['no_reasoning']['literal_reasoning_lines_present'] == 0
    save(out / 'VIEW_AUDIT.json', prompt_audit)

    def fit(indices):
        return fit_robust_model({name: features[name] for name in cfg['training_views']}, y, indices, cfg['head_kind'], cfg['regularization'])

    def margins(model):
        return {name: decision_margin(model, features[name]) for name in cfg['score_views']}
    oof = {panel: {name: np.full(len(train), np.nan) for name in ('single', 'raw_mean', 'scaled_mean', 'and', 'authority', 'reference_normalized', 'support_normalized')} for panel in cfg['reference_panels']}
    fold_log = []
    for (fold, (fit_local, valid_local)) in enumerate(make_group_folds(y[train], groups[train], cfg['calibration_folds'], cfg['seed'])):
        (fit_indices, valid_indices) = (train[fit_local], train[valid_local])
        assert not set(groups[fit_indices]) & set(groups[valid_indices])
        model = fit(fit_indices)
        values = margins(model)
        scales = {name: normalize(value, fit_indices)[1] for (name, value) in values.items()}
        for (panel, reference) in cfg['reference_panels'].items():
            scores = base_scores(values[reference], values['reasoning_support_tagged'], scales[reference], scales['reasoning_support_tagged'])
            scores['reference_normalized'] = values[reference] / scales[reference]
            scores['support_normalized'] = values['reasoning_support_tagged'] / scales['reasoning_support_tagged']
            for (name, array) in scores.items():
                oof[panel][name][valid_local] = array[valid_indices]
        fold_log.append({'fold': fold, 'train': fit_indices.tolist(), 'validation': valid_indices.tolist(), 'head_fingerprint': fingerprint(model), 'scales': scales})
        print(json.dumps({'calibration_fold_completed': fold, 'train': len(fit_indices), 'validation': len(valid_indices)}), flush=True)
    assert all((np.isfinite(v).all() for panel in oof.values() for v in panel.values()))
    policies = fit_policies(oof, y[train], cfg['weighted_fusion_alphas'])
    save(out / 'CALIBRATION.json', {'train_indices': train.tolist(), 'folds': fold_log, 'policies': policies, 'selection_rule': cfg['threshold_selection']})
    model = fit(train)
    head_id = fingerprint(model)
    np.savez_compressed(out / 'shared_head.npz', mean=model.mean, scale=model.scale, weights=model.weights, bias=model.bias)
    values = margins(model)
    scales = {name: normalize(value, train)[1] for (name, value) in values.items()}
    np.savez_compressed(out / 'margins.npz', record_ids=np.asarray(ids), train=train, test=test, labels=y, groups=groups, **values)
    np.savez_compressed(out / 'training_oof_scores.npz', labels=y[train], record_ids=np.asarray(ids)[train], **{f'{p}__{k}': v for (p, scores) in oof.items() for (k, v) in scores.items()})
    save(out / 'HEAD.json', {'fingerprint': head_id, 'head_file_sha256': sha256(out / 'shared_head.npz'), 'scales': scales, 'head_kind': cfg['head_kind'], 'regularization': cfg['regularization'], 'training_views': cfg['training_views'], 'final_head_count': 1, 'training_records': len(train), 'calibration_head_count': len(fold_log)})
    rules = {}
    for (panel, reference) in cfg['reference_panels'].items():
        scores = base_scores(values[reference], values['reasoning_support_tagged'], scales[reference], scales['reasoning_support_tagged'])
        scores['reference_normalized'] = values[reference] / scales[reference]
        scores['support_normalized'] = values['reasoning_support_tagged'] / scales['reasoning_support_tagged']
        rules[panel] = {name: value[test] for (name, value) in apply_policies(scores, policies[panel]).items()}
        assert list(rules[panel]) == cfg['rules']
    frozen_predictions = {r['record_id']: r for r in read(reference_path)}
    baseline = rules['frozen_complete_reference']['authority_zero']
    assert all((int(baseline[i] >= 0) == frozen_predictions[ids[j]]['predictions']['authority_capped'] for (i, j) in enumerate(test))), 'Do not substitute frozen predictions if reconstruction fails'
    original = json.loads(original_summary_path.read_text())
    assert np.isclose(scales['clean'], original['margin_scales']['clean'], rtol=1e-10)
    assert np.isclose(scales['reasoning_support_tagged'], original['margin_scales']['reasoning_context'], rtol=1e-10)
    save(out / 'DECISION_LOCK.json', {'head_fingerprint': head_id, 'calibration_sha256': sha256(out / 'CALIBRATION.json'), 'config_sha256': sha256(out / 'CONFIG.json'), 'source_hashes_sha256': sha256(out / 'SOURCE_SHA256.json'), 'test_rules_fixed_before_metric_reporting': True, 'frozen_predictions_reproduced': len(test)})
    result = {}
    for (panel, scores) in rules.items():
        reference_prediction = (scores['authority_zero'] >= 0).astype(int)
        result[panel] = {}
        for (name, value) in scores.items():
            p = (value >= 0).astype(int)
            result[panel][name] = {**metrics(y[test], p, value), 'head_fingerprint': head_id, 'paired_authority_vs_control': paired(y[test], reference_prediction, p, groups[test], cfg['bootstrap_reps'], cfg['seed'])}
    save(out / 'RESULT.json', result)
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for (i, j) in enumerate(test):
            item = {'record_id': ids[j], 'group_id': str(groups[j]), 'label': int(y[j]), 'head_fingerprint': head_id, 'margins': {p: {k: float(v[i]) for (k, v) in scores.items()} for (p, scores) in rules.items()}, 'predictions': {p: {k: int(v[i] >= 0) for (k, v) in scores.items()} for (p, scores) in rules.items()}}
            handle.write(json.dumps(item) + '\n')
    save(out / 'RUN.json', {'finished_utc': datetime.now(timezone.utc).isoformat(), 'model_calls': 0, 'frozen_inputs_modified': False, 'train_count': len(train), 'test_count': len(test), 'feature_provenance': provenance})
    assert all((sha256(Path(name)) == value for (name, value) in hashes.items()))
    save(out / 'STATUS.json', {'status': 'finished_pending_verification'})
    print(json.dumps(result, indent=2), flush=True)

def run_grouped(cfg, output_root):
    assert sys.platform == 'linux', 'Run only on the server'
    frozen = Path(cfg['frozen_root'])
    out = output_root / cfg['benchmark']
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'CONFIG.json', cfg)
    status_path = out / 'STATUS.json'
    save(status_path, {'status': 'running', 'stage': 'loading_features'})
    records = frozen / f"datasets/prepared/records.{cfg['record_stem']}.availability_clean.jsonl"
    reference_dir = frozen / 'results' / cfg['result_name'] / 'evaluation'
    if cfg['evaluation_subset']:
        reference_dir = reference_dir / cfg['evaluation_subset']
    reference_path = reference_dir / 'predictions.jsonl'
    summary_path = reference_dir / 'summary.json'
    rows = read(records)
    ids = [r['record_id'] for r in rows]
    y = np.asarray([r['labels']['success'] for r in rows], dtype=int)
    groups = np.asarray([r['group_id'] for r in rows])
    assert len(ids) == len(set(ids)) == cfg['record_count']
    assert len(set(groups)) == cfg['group_count']
    reference = {r['record_id']: r for r in read(reference_path)}
    assert set(reference) == set(ids)
    assert all((reference[k]['truth_success'] == int(y[i]) and reference[k]['group_id'] == groups[i] for (i, k) in enumerate(ids)))
    assignment = np.asarray([reference[k]['outer_fold'] for k in ids], dtype=int)
    assert set(assignment) == set(range(cfg['outer_folds']))
    assert all((len(set(assignment[groups == g])) == 1 for g in set(groups)))
    canonical_groups = np.asarray([g.replace('visualwebarena.resized.', 'visualwebarena.') for g in groups])
    assert all((len(set(assignment[canonical_groups == g])) == 1 for g in set(canonical_groups)))
    original = json.loads(summary_path.read_text())
    original_folds = {r['fold']: r for r in original['folds']}
    needed = cfg['training_views'] + ['reasoning_support_tagged']
    paths = {name: frozen / 'results' / cfg['result_name'] / 'features' / f'{name}.final64.npz' for name in needed}
    (features, provenance) = load_feature_views(paths, ids, digest_ids(rows), records_path=records)
    prepared = {name: records.with_name(f"records.{cfg['record_stem']}.availability_{name}.jsonl") for name in needed}
    sources = [ROOT / 'run.py', ROOT / 'core.py', ROOT / 'prepare_data.py', records, reference_path, *paths.values(), *prepared.values()]
    hashes = {str(p): sha256(p) for p in set(sources)}
    save(out / 'SOURCE_SHA256.json', hashes)
    save(out / 'CONFIG.json', cfg)
    prompt_audit = {}
    for name in cfg['score_views']:
        items = read(prepared[name])
        assert [r['record_id'] for r in items] == ids
        prompt_audit[name] = {'records': len(items), 'literal_reasoning_lines_present': sum(('Reasoning:' in r['messages'][1]['content'] for r in items))}
    assert prompt_audit['no_reasoning']['literal_reasoning_lines_present'] == 0
    save(out / 'VIEW_AUDIT.json', prompt_audit)
    save(out / 'ORIGINAL_PROTOCOL.json', {'record_ids': ids, 'groups': groups.tolist(), 'outer_assignment': assignment.tolist(), 'folds': [{'fold': k, 'selected_head': v['selected_head'], 'train_records': v['train_records'], 'test_records': v['test_records']} for (k, v) in original_folds.items()]})
    rules = {p: {r: np.full(len(rows), np.nan) for r in cfg['rules']} for p in cfg['reference_panels']}
    (fold_logs, heads) = ([], [])
    head_ids = np.empty(len(rows), dtype='U64')
    for outer in range(cfg['outer_folds']):
        (train, test) = (np.flatnonzero(assignment != outer), np.flatnonzero(assignment == outer))
        assert not set(groups[train]) & set(groups[test])
        old = original_folds[outer]
        assert (len(train), len(test)) == (old['train_records'], old['test_records'])
        (kind, regularization) = (old['selected_head']['head_kind'], old['selected_head']['regularization'])

        def fit(indices):
            return fit_robust_model({k: features[k] for k in cfg['training_views']}, y, indices, kind, regularization)

        def margins(model):
            return {k: decision_margin(model, features[k]) for k in cfg['score_views']}
        folder = out / f'fold{outer}'
        folder.mkdir()
        oof = {panel: {name: np.full(len(train), np.nan) for name in ('single', 'raw_mean', 'scaled_mean', 'and', 'authority', 'reference_normalized', 'support_normalized')} for panel in cfg['reference_panels']}
        inner_log = []
        for (inner, (a, b)) in enumerate(make_group_folds(y[train], groups[train], cfg['calibration_folds'], cfg['seed'] + outer + 1)):
            (fit_indices, valid_indices) = (train[a], train[b])
            assert not set(groups[fit_indices]) & set(groups[valid_indices])
            auxiliary = fit(fit_indices)
            values = margins(auxiliary)
            scales = {name: normalize(value, fit_indices)[1] for (name, value) in values.items()}
            for (panel, ref) in cfg['reference_panels'].items():
                scores = base_scores(values[ref], values['reasoning_support_tagged'], scales[ref], scales['reasoning_support_tagged'])
                scores['reference_normalized'] = values[ref] / scales[ref]
                scores['support_normalized'] = values['reasoning_support_tagged'] / scales['reasoning_support_tagged']
                for (name, array) in scores.items():
                    oof[panel][name][b] = array[valid_indices]
            inner_log.append({'fold': inner, 'train': fit_indices.tolist(), 'validation': valid_indices.tolist(), 'head_fingerprint': fingerprint(auxiliary), 'scales': scales})
        assert all((np.isfinite(v).all() for panel in oof.values() for v in panel.values()))
        policies = fit_policies(oof, y[train], cfg['weighted_fusion_alphas'])
        save(folder / 'CALIBRATION.json', {'train_indices': train.tolist(), 'folds': inner_log, 'policies': policies, 'selection_rule': cfg['threshold_selection']})
        np.savez_compressed(folder / 'training_oof_scores.npz', labels=y[train], record_ids=np.asarray(ids)[train], **{f'{p}__{k}': v for (p, scores) in oof.items() for (k, v) in scores.items()})
        model = fit(train)
        head_id = fingerprint(model)
        head_ids[test] = head_id
        np.savez_compressed(folder / 'shared_head.npz', mean=model.mean, scale=model.scale, weights=model.weights, bias=model.bias)
        values = margins(model)
        scales = {name: normalize(value, train)[1] for (name, value) in values.items()}
        np.savez_compressed(folder / 'margins.npz', train=train, test=test, **values)
        for (name, old_key) in (('clean', 'clean'), ('reasoning_support_tagged', 'reasoning_context')):
            assert np.allclose(values[name][test], [reference[ids[j]]['margins'][old_key] for j in test], rtol=1e-09, atol=1e-08)
        assert np.isclose(scales['clean'], old['clean_margin_scale'], rtol=1e-09)
        assert np.isclose(scales['reasoning_support_tagged'], old['reasoning_context_margin_scale'], rtol=1e-09)
        for (panel, ref) in cfg['reference_panels'].items():
            scores = base_scores(values[ref], values['reasoning_support_tagged'], scales[ref], scales['reasoning_support_tagged'])
            scores['reference_normalized'] = values[ref] / scales[ref]
            scores['support_normalized'] = values['reasoning_support_tagged'] / scales['reasoning_support_tagged']
            readouts = apply_policies(scores, policies[panel])
            assert list(readouts) == cfg['rules']
            for (name, value) in readouts.items():
                rules[panel][name][test] = value[test]
        reconstructed = rules['frozen_complete_reference']['authority_zero'][test]
        assert np.allclose(reconstructed, [reference[ids[j]]['margins']['authority_capped'] for j in test], rtol=1e-09, atol=1e-08)
        assert all((int(reconstructed[i] >= 0) == reference[ids[j]]['predictions']['authority_capped'] for (i, j) in enumerate(test)))
        head_info = {'outer_fold': outer, 'fingerprint': head_id, 'file_sha256': sha256(folder / 'shared_head.npz'), 'scales': scales, 'head_kind': kind, 'regularization': regularization, 'training_views': cfg['training_views'], 'train': train.tolist(), 'test': test.tolist()}
        save(folder / 'HEAD.json', head_info)
        heads.append(head_info)
        fold_logs.append({'fold': outer, 'train_records': len(train), 'test_records': len(test), 'head_fingerprint': head_id, 'frozen_predictions_reproduced': len(test)})
        print(json.dumps({'completed_outer_fold': outer + 1, **fold_logs[-1]}), flush=True)
        save(status_path, {'status': 'running', 'completed_outer_folds': outer + 1})
    assert all((np.isfinite(v).all() for p in rules.values() for v in p.values()))
    save(out / 'DECISION_LOCK.json', {'source_hashes_sha256': sha256(out / 'SOURCE_SHA256.json'), 'protocol_sha256': sha256(out / 'ORIGINAL_PROTOCOL.json'), 'test_rules_fixed_before_metric_reporting': True, 'heads': [{'outer_fold': h['outer_fold'], 'fingerprint': h['fingerprint'], 'calibration_sha256': sha256(out / f"fold{h['outer_fold']}/CALIBRATION.json")} for h in heads]})
    result = {}
    for (panel, scores) in rules.items():
        primary = (scores['authority_zero'] >= 0).astype(int)
        result[panel] = {}
        for (name, value) in scores.items():
            prediction = (value >= 0).astype(int)
            result[panel][name] = {**metrics(y, prediction, value), 'shared_head_per_fold': True, 'paired_authority_vs_control': paired(y, primary, prediction, groups, cfg['bootstrap_reps'], cfg['seed'])}
    old_primary = original['readout_metrics']['authority_capped']
    assert abs(result['frozen_complete_reference']['authority_zero']['balanced_accuracy'] - old_primary['balanced_accuracy']) < 1e-12
    save(out / 'RESULT.json', result)
    with (out / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for (i, record_id) in enumerate(ids):
            item = {'record_id': record_id, 'group_id': str(groups[i]), 'label': int(y[i]), 'outer_fold': int(assignment[i]), 'head_fingerprint': str(head_ids[i]), 'margins': {p: {k: float(v[i]) for (k, v) in scores.items()} for (p, scores) in rules.items()}, 'predictions': {p: {k: int(v[i] >= 0) for (k, v) in scores.items()} for (p, scores) in rules.items()}}
            handle.write(json.dumps(item) + '\n')
    save(out / 'RUN.json', {'finished_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'model_calls': 0, 'record_count': len(rows), 'groups': len(set(groups)), 'outer_folds': len(heads), 'one_shared_head_per_outer_fold': True, 'separate_heads_per_control': False, 'frozen_predictions_reproduced': len(rows), 'folds': fold_logs, 'feature_provenance': provenance})
    assert all((sha256(Path(name)) == value for (name, value) in hashes.items()))
    save(status_path, {'status': 'finished_pending_verification'})
    print(json.dumps(result, indent=2), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    cfg = {**COMMON_CONFIG, **DATASETS[args.dataset], "frozen_root": str(args.frozen_root)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (run_fixed if args.dataset == "webarena" else run_grouped)(cfg, args.output_root)

if __name__ == "__main__":
    main()
