#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.root':
        import uproot
        with uproot.open(path) as f:
            treenames = [k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree']
            tree = f[treenames[0]]
            return tree.arrays(library='ak')
    elif ext == '.parquet':
        return ak.from_parquet(path)
    else:
        raise RuntimeError(f'Unsupported file type: {ext}')


def discover_columns(fields, score_prefix='score_'):
    raw_score_cols = sorted([c for c in fields if c.startswith(score_prefix)])
    raw_label_cols = sorted([c for c in fields if c.startswith('label_')])
    def score_base(name):
        base = name[len(score_prefix):]
        if base.startswith('label_'):
            base = base[len('label_'):]
        return base
    def label_base(name):
        return name[len('label_'):]
    score_base_to_name = {score_base(c): c for c in raw_score_cols}
    label_base_to_name = {label_base(c): c for c in raw_label_cols}
    bases = sorted(set(score_base_to_name.keys()) & set(label_base_to_name.keys()))
    if not bases:
        raise RuntimeError('Could not align score_*/label_* columns.')
    score_cols = [score_base_to_name[b] for b in bases]
    label_cols = [label_base_to_name[b] for b in bases]
    return bases, score_cols, label_cols


def build_multiclass_targets_and_scores(tables, bases, score_cols, label_cols):
    table = ak.concatenate(tables, axis=0) if len(tables) > 1 else tables[0]
    label_matrix = np.vstack([ak.to_numpy(table[c]).astype(int) for c in label_cols]).T
    y_true = label_matrix.argmax(axis=1)
    y_score = np.vstack([ak.to_numpy(table[c]).astype(float) for c in score_cols]).T
    return y_true, y_score, table


def compute_discriminant_score(score_signal, score_background):
    """
    Compute the discriminant score d = score(S) / (score(S) + score(B))
    This is the correct way to handle multi-class outputs for binary classification.
    """
    # Avoid division by zero
    denominator = score_signal + score_background
    with np.errstate(divide='ignore', invalid='ignore'):
        discriminant = np.where(denominator > 0, score_signal / denominator, 0.0)
    return discriminant


def compute_optimal_f1_score(y_true, y_score):
    """
    Compute the optimal F1 score by finding the best threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    f1_scores = np.nan_to_num(f1_scores)
    # Return the maximum F1 score
    return np.max(f1_scores) if len(f1_scores) > 0 else 0.0


def summarize_group(name, patterns, score_prefix='score_', verbose=False):
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(files)
    if verbose:
        print(f'[INFO] Group {name}: {len(files)} files')
        for fp in files[:10]:
            print(f'  - {fp}')
    if not files:
        raise RuntimeError(f'No files matched for group {name}')
    sample = load_table(files[0])
    bases, score_cols, label_cols = discover_columns(sample.fields, score_prefix=score_prefix)
    tables = [load_table(fp) for fp in files]
    y_true, y_score, table = build_multiclass_targets_and_scores(tables, bases, score_cols, label_cols)

    # Standard multi-class metrics (unchanged)
    macro_ovo = roc_auc_score(y_true, y_score, multi_class='ovo', average='macro')
    y_true_onehot = np.eye(len(bases), dtype=int)[y_true]
    ap_macro_ovr = average_precision_score(y_true_onehot, y_score, average='macro')

    # CORRECTED: Macro PR AUC and F1 using discriminant score (signals vs QCD)
    qcd_candidates = [b for b in bases if b.lower() == 'qcd'] or [b for b in bases if 'qcd' in b.lower()]
    qcd_base = qcd_candidates[0]
    qcd_idx = bases.index(qcd_base)
    
    if verbose:
        print(f'[INFO] Using discriminant d = score(S) / (score(S) + score(B)) for {name}')
        print(f'[INFO] QCD base: {qcd_base}, Signal bases: {[b for b in bases if b != qcd_base]}')
    
    pr_aucs = []
    f1_max_list = []
    
    for b in bases:
        if b == qcd_base:
            continue
        
        # Get labels
        y_sig = ak.to_numpy(table[f'label_{b}']).astype(bool)
        y_qcd = ak.to_numpy(table[f'label_{qcd_base}']).astype(bool)
        
        # Filter to only signal and background events
        mask = (y_sig | y_qcd)
        y_bin = y_sig[mask].astype(int)
        
        # Get scores for signal and background
        sig_idx = bases.index(b)
        score_signal = ak.to_numpy(table[score_cols[sig_idx]]).astype(float)[mask]
        score_qcd = ak.to_numpy(table[score_cols[qcd_idx]]).astype(float)[mask]
        
        # CORRECTED: Use discriminant score instead of raw signal score
        discriminant = compute_discriminant_score(score_signal, score_qcd)
        
        # Calculate PR AUC using discriminant
        prec, rec, _ = precision_recall_curve(y_bin, discriminant)
        pr_auc = np.trapz(prec[::-1], rec[::-1]) if len(rec) > 1 else float('nan')
        pr_aucs.append(pr_auc)
        
        # Calculate optimal F1 using discriminant
        f1_max = compute_optimal_f1_score(y_bin, discriminant)
        f1_max_list.append(f1_max)
        
        if verbose:
            print(f'    Signal {b}: PR AUC={pr_auc:.4f}, F1 max={f1_max:.4f}')
    
    macro_pr_auc_signal_qcd = float(np.nanmean(pr_aucs)) if pr_aucs else float('nan')
    macro_f1_signal_qcd = float(np.nanmean(f1_max_list)) if f1_max_list else float('nan')
    
    if verbose:
        print(f'    Macro PR AUC (signals vs QCD): {macro_pr_auc_signal_qcd:.4f}')
        print(f'    Macro F1 (signals vs QCD): {macro_f1_signal_qcd:.4f}')

    return {
        'name': name,
        'bases': bases,
        'num_samples': int(y_true.shape[0]),
        'roc_auc_macro_ovo': float(macro_ovo),
        'avg_precision_macro_ovr': float(ap_macro_ovr),
        'pr_auc_macro_signal_qcd': float(macro_pr_auc_signal_qcd),
        'f1_macro_signal_qcd': float(macro_f1_signal_qcd),
    }


def main():
    parser = argparse.ArgumentParser(description='Comparación por dataset size (1M,2M,10M) para ParT y PN - JetClass full usando discriminante normalizado')
    parser.add_argument('--part_1m', type=str, default='predictions/JetClass/Pythia/full/ParT/1M/*_pred_*.root')
    parser.add_argument('--part_2m', type=str, default='predictions/JetClass/Pythia/full/ParT/2M/*_pred_*.root')
    parser.add_argument('--part_10m', type=str, default='predictions/JetClass/Pythia/full/ParT/10M/*_pred_*.root')
    parser.add_argument('--pn_1m', type=str, default='predictions/JetClass/Pythia/full/PN/1M/*_pred_*.root')
    parser.add_argument('--pn_2m', type=str, default='predictions/JetClass/Pythia/full/PN/2M/*_pred_*.root')
    parser.add_argument('--pn_10m', type=str, default='predictions/JetClass/Pythia/full/PN/10M/*_pred_*.root')
    parser.add_argument('--score-prefix', type=str, default='score_')
    parser.add_argument('--output-dir', type=str, default='graficos_comparativos_v2/JetClass/dataset_size_comparison')
    parser.add_argument('--title', type=str, default='JetClass full - Dataset size comparison (Discriminante Normalizado)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print('[INFO] Usando discriminante normalizado d = score(S) / (score(S) + score(B)) para métricas signals vs QCD')

    # Summaries
    res = {
        'ParT': {
            '1M': summarize_group('ParT_full_1M', [args.part_1m], score_prefix=args.score_prefix, verbose=args.verbose),
            '2M': summarize_group('ParT_full_2M', [args.part_2m], score_prefix=args.score_prefix, verbose=args.verbose),
            '10M': summarize_group('ParT_full_10M', [args.part_10m], score_prefix=args.score_prefix, verbose=args.verbose),
        },
        'PN': {
            '1M': summarize_group('PN_full_1M', [args.pn_1m], score_prefix=args.score_prefix, verbose=args.verbose),
            '2M': summarize_group('PN_full_2M', [args.pn_2m], score_prefix=args.score_prefix, verbose=args.verbose),
            '10M': summarize_group('PN_full_10M', [args.pn_10m], score_prefix=args.score_prefix, verbose=args.verbose),
        }
    }

    # CSV table
    csv_path = os.path.join(args.output_dir, 'macro_summary_by_size_discriminant.csv')
    with open(csv_path, 'w') as f:
        f.write('model,size,num_samples,roc_auc_macro_ovo,avg_precision_macro_ovr,pr_auc_macro_signal_qcd,f1_macro_signal_qcd\n')
        for model in ['ParT', 'PN']:
            for size in ['1M', '2M', '10M']:
                r = res[model][size]
                f.write(f"{model},{size},{r['num_samples']},{r['roc_auc_macro_ovo']:.6f},{r['avg_precision_macro_ovr']:.6f},{r['pr_auc_macro_signal_qcd']:.6f},{r['f1_macro_signal_qcd']:.6f}\n")

    # Plots per metric
    sizes = ['1M', '2M', '10M']
    x = np.arange(len(sizes))
    width = 0.35

    def bar_plot(metric_key, ylabel, filename):
        plt.figure(figsize=(8, 5))
        vals_part = [res['ParT'][s][metric_key] for s in sizes]
        vals_pn = [res['PN'][s][metric_key] for s in sizes]
        bars1 = plt.bar(x - width/2, vals_part, width, label='ParT')
        bars2 = plt.bar(x + width/2, vals_pn, width, label='PN')
        
        # Add value labels on bars
        for bars, vals in [(bars1, vals_part), (bars2, vals_pn)]:
            for bar, val in zip(bars, vals):
                plt.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.xticks(x, sizes)
        plt.ylim(0.0, 1.0)
        plt.ylabel(ylabel)
        plt.xlabel('Dataset Size')
        title_text = args.title
        if 'signal_qcd' in metric_key:
            title_text += '\n(usando discriminante normalizado)'
        plt.title(title_text)
        plt.legend()
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(args.output_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()
        if args.verbose:
            print(f'[OK] Saved: {out_path}')

    bar_plot('roc_auc_macro_ovo', 'ROC AUC (macro OVO)', 'roc_auc_macro_ovo_by_size.png')
    bar_plot('avg_precision_macro_ovr', 'Average Precision (macro OVR)', 'avg_precision_macro_ovr_by_size.png')
    bar_plot('pr_auc_macro_signal_qcd', 'PR AUC (signals vs QCD, macro)', 'pr_auc_macro_signal_qcd_by_size.png')
    bar_plot('f1_macro_signal_qcd', 'F1 (signals vs QCD, macro)', 'f1_macro_signal_qcd_by_size.png')

    print('[OK] Saved:', csv_path)
    print('[OK] Evaluación completada usando discriminante normalizado para métricas signals vs QCD')


if __name__ == '__main__':
    main()