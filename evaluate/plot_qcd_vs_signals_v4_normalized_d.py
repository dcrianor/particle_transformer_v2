#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def discover_columns(array_fields, score_prefix):
    score_columns = [c for c in array_fields if c.startswith(score_prefix)]
    label_columns = []
    # Labels may be present as separate boolean columns matching score names without prefix
    for sc in score_columns:
        base = sc[len(score_prefix):]
        if base in array_fields:
            label_columns.append(base)
    # Fallback: also consider any field that starts with 'label_' if present
    label_like = [c for c in array_fields if c.startswith('label_')]
    for c in label_like:
        if c not in label_columns:
            label_columns.append(c)
    return sorted(score_columns), sorted(set(label_columns))


def infer_qcd_name(label_columns, qcd_name_hint):
    # Prefer explicit hint
    if qcd_name_hint:
        # Allow variants like 'QCD' or 'label_QCD'
        if qcd_name_hint in label_columns:
            return qcd_name_hint
        prefixed = f'label_{qcd_name_hint}'
        if prefixed in label_columns:
            return prefixed
        # Try case-insensitive match
        for c in label_columns:
            if qcd_name_hint.lower() in c.lower():
                return c
    # Heuristic: pick any column containing 'qcd'
    for c in label_columns:
        if 'qcd' in c.lower():
            return c
    raise RuntimeError('Cannot infer QCD label column. Please provide --qcd-name explicitly.')


def normalize_label_name(label_name):
    # Strip common prefixes like 'label_'
    return label_name.replace('label_', '')


def load_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.root':
        import uproot
        with uproot.open(path) as f:
            treenames = [k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree']
            if not treenames:
                raise RuntimeError(f'No TTree found in {path}')
            if len(treenames) > 1:
                # If multiple, default to first; allow override later if needed
                treename = treenames[0]
            else:
                treename = treenames[0]
            tree = f[treename]
            table = tree.arrays(library='ak')
    elif ext == '.parquet':
        table = ak.from_parquet(path)
    else:
        raise RuntimeError(f'Unsupported file extension: {ext}')
    return table


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
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    f1_scores = np.nan_to_num(f1_scores)
    # Return the maximum F1 score
    return np.max(f1_scores) if len(f1_scores) > 0 else 0.0


def compute_curves_for_signal_vs_qcd(y_signal_bool, y_qcd_bool, score_signal, score_background, verbose=False):
    """
    Compute ROC curves using the correct discriminant score for multi-class models.
    
    Steps:
    1. Filter to only include signal and background events
    2. Compute discriminant d = score(S) / (score(S) + score(B))
    3. Use standard ROC curve computation on this discriminant
    4. Compute F1 score using the discriminant
    """
    # Step 5 from the guide: Create filtered dataset with only S and B events
    valid_mask = (y_signal_bool | y_qcd_bool)
    y_true = y_signal_bool[valid_mask]  # 1 for signal, 0 for background
    score_s = score_signal[valid_mask]
    score_b = score_background[valid_mask]
    
    if verbose:
        n_signal = int(y_true.sum())
        n_background = int((~y_true).sum())
        print(f'    Filtered dataset: {n_signal} signal, {n_background} background events')
    
    # Step 4 from the guide: Calculate the discriminant
    discriminant = compute_discriminant_score(score_s, score_b)
    
    # Convert to numpy arrays for sklearn
    y_true_np = np.asarray(y_true, dtype=np.int32)
    discriminant_np = np.asarray(discriminant, dtype=np.float32)
    
    # Step 6 from the guide: Generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true_np, discriminant_np)
    roc_auc = auc(fpr, tpr)
    
    # NEW: Calculate optimal F1 score
    f1_optimal = compute_optimal_f1_score(y_true_np, discriminant_np)
    
    if verbose:
        print(f'    Discriminant range: [{discriminant_np.min():.4f}, {discriminant_np.max():.4f}]')
        print(f'    ROC AUC: {roc_auc:.4f}')
        print(f'    Optimal F1 Score: {f1_optimal:.4f}')
    
    return fpr, tpr, thresholds, roc_auc, discriminant_np, f1_optimal


def plot_roc(ax, tpr, fpr, label):
    ax.plot(fpr, tpr, lw=2, label=label)


def plot_rejection(ax, tpr, fpr, label):
    # rejection = 1 / FPR, avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rejection = np.where(fpr > 0, 1.0 / fpr, np.inf)
    ax.plot(tpr, rejection, lw=2, label=label)


def rejection_at_eff(tpr, fpr, target_eff=0.5):
    """
    Step 7 from the guide: Determine background rejection at given signal efficiency.
    """
    # Find closest index where TPR >= target
    idx = np.searchsorted(tpr, target_eff, side='left')
    if idx >= len(fpr):
        return 0.0
    return (1.0 / fpr[idx]) if fpr[idx] > 0 else np.inf


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar correctamente modelos multi-clase para clasificación binaria señal vs QCD usando el discriminante normalizado.'
    )
    parser.add_argument('--inputs', nargs='+', default=None, 
                       help='Rutas a archivos de predicciones (.root o .parquet). Soporta globs. (Modo simple, un solo grupo)')
    parser.add_argument('--group', action='append', default=None, 
                       help='Definir grupo como nombre=glob. Se puede repetir. (Modo comparativo JetClass)')
    parser.add_argument('--output-dir', type=str, default='graficos_comparativos_v2/JetClass', 
                       help='Directorio raíz de salida de gráficos')
    parser.add_argument('--qcd-name', type=str, default='QCD', 
                       help='Nombre (o substring) de la columna de labels para QCD')
    parser.add_argument('--score-prefix', type=str, default='score_', 
                       help='Prefijo de columnas de score en el archivo de predicciones')
    parser.add_argument('--title', type=str, default=None, 
                       help='Título opcional para los gráficos')
    parser.add_argument('--verbose', action='store_true', 
                       help='Imprimir información detallada de archivos y señales')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Solo listar archivos y señales detectadas, sin generar gráficos')
    args = parser.parse_args()

    # Build groups
    groups = []  # list of (name, file_list)
    if args.group:
        for spec in args.group:
            if '=' not in spec:
                raise SystemExit('Use --group nombre=glob')
            name, pattern = spec.split('=', 1)
            files = glob.glob(pattern)
            if args.verbose:
                print(f'[INFO] Grupo {name}: patrón={pattern}, archivos={len(files)}')
                for fp in files[:10]:
                    print(f'       - {fp}')
                if len(files) > 10:
                    print('       ...')
            if not files:
                print(f'[WARN] Grupo {name} no tiene archivos para patrón: {pattern}', file=sys.stderr)
            groups.append((name, sorted(files)))
    elif args.inputs:
        files = []
        for p in args.inputs:
            files.extend(glob.glob(p))
        if not files:
            print('No input files found for given patterns.', file=sys.stderr)
            sys.exit(1)
        groups.append(('default', sorted(files)))
    else:
        print('Debe especificar --group o --inputs', file=sys.stderr)
        sys.exit(1)

    # Load first group's first file to discover columns
    if not groups or not groups[0][1]:
        print('No hay archivos para procesar.', file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f'[INFO] Cargando archivo de ejemplo para descubrir columnas: {groups[0][1][0]}')
    
    sample_table = load_table(groups[0][1][0])
    fields = sample_table.fields
    score_cols, label_cols = discover_columns(fields, args.score_prefix)
    
    if not score_cols or not label_cols:
        raise RuntimeError('No se detectaron columnas de score/label en el primer archivo.')
    
    if args.verbose:
        print(f'[INFO] Columnas de score detectadas: {score_cols}')
        print(f'[INFO] Columnas de label detectadas: {label_cols}')
    
    qcd_label_col = infer_qcd_name(label_cols, args.qcd_name)
    qcd_base = normalize_label_name(qcd_label_col)
    
    # Mapeos base
    base_to_score = {}
    for sc in score_cols:
        base = sc[len(args.score_prefix):]
        base = normalize_label_name(base)
        base_to_score[base] = sc
    
    base_to_label = {normalize_label_name(lc): lc for lc in label_cols}
    candidate_bases = sorted(set(base_to_label.keys()) & set(base_to_score.keys()))
    signal_bases = [b for b in candidate_bases if b.lower() != qcd_base.lower()]
    
    if args.verbose:
        print(f'[INFO] QCD base: {qcd_base}')
        print(f'[INFO] Señales detectadas: {signal_bases}')
        print(f'[INFO] Usando discriminante d = score(S) / (score(S) + score(B)) para evaluación correcta')

    # Verificar que tenemos score para QCD
    if qcd_base not in base_to_score:
        raise RuntimeError(f'No se encontró columna de score para QCD ({qcd_base}). Columnas disponibles: {list(base_to_score.keys())}')

    # Prepara salida
    ensure_dir(args.output_dir)
    if args.verbose:
        print(f'[INFO] Directorio de salida: {args.output_dir}')

    # Acumular curvas por señal y grupo
    curves = {sig: {} for sig in signal_bases}  # curves[signal][group] = (fpr,tpr,roc_auc,prec,rec,pr_auc,f1_opt)
    summaries = {sig: [] for sig in signal_bases}

    for group_name, file_list in groups:
        if not file_list:
            continue
        
        # Concatenar todas las tablas del grupo
        if args.verbose:
            print(f'[INFO] Procesando grupo {group_name} con {len(file_list)} archivos')
        
        tables = [load_table(fp) for fp in file_list]
        table = ak.concatenate(tables, axis=0) if len(tables) > 1 else tables[0]

        for base in signal_bases:
            if args.verbose:
                print(f'  [INFO] Procesando señal {base} vs {qcd_base}')
            
            label_col = base_to_label[base]
            score_col = base_to_score[base]
            qcd_score_col = base_to_score[qcd_base]
            
            y_signal = ak.to_numpy(table[label_col]).astype(bool)
            y_qcd = ak.to_numpy(table[base_to_label[qcd_base]]).astype(bool)
            scores_signal = ak.to_numpy(table[score_col]).astype(np.float32)
            scores_qcd = ak.to_numpy(table[qcd_score_col]).astype(np.float32)
            
            if args.verbose:
                n_sig = int(y_signal.sum())
                n_bkg = int(y_qcd.sum())
                print(f'    Total events: {len(y_signal)}, Signal: {n_sig}, Background: {n_bkg}')

            if args.dry_run:
                # No calcular curvas, solo inspección
                curves[base][group_name] = (
                    np.array([0, 1]),  # fpr
                    np.array([0, 1]),  # tpr
                    0.0,               # roc_auc
                    np.array([1, 0]),  # prec (dummy)
                    np.array([0, 1]),  # rec (dummy)
                    0.0,               # pr_auc
                    0.0                # f1_opt (dummy)
                )
                continue

            # CORRECTED: Usar la función corregida que calcula el discriminante Y el F1 score
            fpr, tpr, thr, roc_auc, discriminant, f1_optimal = compute_curves_for_signal_vs_qcd(
                y_signal, y_qcd, scores_signal, scores_qcd, verbose=args.verbose
            )
            
            # Para PR curve, también usamos el discriminant filtrado
            valid_mask = (y_signal | y_qcd)
            y_true_filtered = y_signal[valid_mask].astype(int)
            discriminant_filtered = discriminant
            
            prec, rec, _ = precision_recall_curve(y_true_filtered, discriminant_filtered)
            pr_auc = auc(rec, prec) if len(rec) > 1 else float('nan')
            
            # CORRECTED: Now storing F1 score correctly
            curves[base][group_name] = (fpr, tpr, roc_auc, prec, rec, pr_auc, f1_optimal)
            
            # Step 7: Calculate background rejection at specific efficiencies
            rej50 = rejection_at_eff(tpr, fpr, target_eff=0.5)
            rej99 = rejection_at_eff(tpr, fpr, target_eff=0.99)
            summaries[base].append((group_name, roc_auc, rej50, rej99, f1_optimal))
            
            if args.verbose:
                print(f'    ROC AUC: {roc_auc:.4f}')
                print(f'    PR AUC: {pr_auc:.4f}')
                print(f'    F1 Score (optimal): {f1_optimal:.4f}')
                print(f'    Rejection@50%: {rej50:.2f}')
                print(f'    Rejection@99%: {rej99:.2f}')

    if args.dry_run:
        print('[OK] Dry-run completado. Configuración detectada:')
        for base in signal_bases:
            print(f'  - Señal: {base}')
        print(f'  - Background: {qcd_base}')
        print(f'  - Método: Discriminante normalizado d = score(S) / (score(S) + score(B))')
        return

    # Graficar comparativos por señal
    for base in signal_bases:
        sig_dir = os.path.join(args.output_dir, base)
        ensure_dir(sig_dir)

        # ROC comparativo
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        for group_name, curve_data in curves[base].items():
            fpr, tpr, aucv = curve_data[0], curve_data[1], curve_data[2]
            ax_roc.plot(fpr, tpr, lw=2, label=f'{group_name} (AUC={aucv:.3f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        title = f'ROC: {base} vs {qcd_base} (Discriminante Normalizado)'
        if args.title:
            title += f' - {args.title}'
        ax_roc.set_title(title)
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        out_roc = os.path.join(sig_dir, f'roc_compare_{base}.png')
        fig_roc.savefig(out_roc, dpi=220)
        plt.close(fig_roc)

        # Rejection comparativo
        fig_rej, ax_rej = plt.subplots(figsize=(8, 6))
        for group_name, curve_data in curves[base].items():
            fpr, tpr = curve_data[0], curve_data[1]
            with np.errstate(divide='ignore', invalid='ignore'):
                rejection = np.where(fpr > 0, 1.0 / fpr, np.inf)
            ax_rej.plot(tpr, rejection, lw=2, label=f'{group_name}')
        ax_rej.set_yscale('log')
        ax_rej.set_xlabel('True Positive Rate (Signal Efficiency)')
        ax_rej.set_ylabel('1/FPR (Background Rejection)')
        title = f'Background Rejection: {base} vs {qcd_base}'
        if args.title:
            title += f' - {args.title}'
        ax_rej.set_title(title)
        ax_rej.legend(loc='lower left')
        ax_rej.grid(True, alpha=0.3)
        fig_rej.tight_layout()
        out_rej = os.path.join(sig_dir, f'rejection_compare_{base}.png')
        fig_rej.savefig(out_rej, dpi=220)
        plt.close(fig_rej)

        # CSV resumen - CORRECTED to include F1 score
        lines = ['group,auc,rejection@50%,rejection@99%,f1_score']
        for row in summaries[base]:
            lines.append('{},{:.6f},{:.6g},{:.6g},{:.6f}'.format(*row))
        out_csv = os.path.join(sig_dir, f'summary_{base}.csv')
        with open(out_csv, 'w') as f:
            f.write('\n'.join(lines) + '\n')
            
        if args.verbose:
            print(f'  [OK] Guardado: {out_roc}')
            print(f'  [OK] Guardado: {out_rej}')
            print(f'  [OK] Guardado: {out_csv}')

    # Gráficas conjuntas por grupo: todas las señales en una sola
    for group_name, file_list in groups:
        if not file_list:
            continue
            
        # ROC all signals
        fig_rall, ax_rall = plt.subplots(figsize=(10, 8))
        for base in signal_bases:
            if group_name in curves[base]:
                curve_data = curves[base][group_name]
                fpr, tpr, aucv = curve_data[0], curve_data[1], curve_data[2]
                ax_rall.plot(fpr, tpr, lw=2, label=f'{base} (AUC={aucv:.3f})')
        ax_rall.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax_rall.set_xlabel('False Positive Rate')
        ax_rall.set_ylabel('True Positive Rate')
        title = f'ROC Curves - {group_name} (Discriminante Normalizado)'
        if args.title:
            title += f' - {args.title}'
        ax_rall.set_title(title)
        ax_rall.legend(loc='lower right')
        ax_rall.grid(True, alpha=0.3)
        fig_rall.tight_layout()
        out_rall = os.path.join(args.output_dir, f'roc_all_signals_{group_name}.png')
        fig_rall.savefig(out_rall, dpi=240)
        plt.close(fig_rall)

        # PR all signals
        fig_prall, ax_prall = plt.subplots(figsize=(10, 8))
        for base in signal_bases:
            if group_name in curves[base]:
                curve_data = curves[base][group_name]
                prec, rec, prauc = curve_data[3], curve_data[4], curve_data[5]
                ax_prall.plot(rec, prec, lw=2, label=f'{base} (PR AUC={prauc:.3f})')
        ax_prall.set_xlabel('Recall (True Positive Rate)')
        ax_prall.set_ylabel('Precision')
        title = f'Precision-Recall Curves - {group_name}'
        if args.title:
            title += f' - {args.title}'
        ax_prall.set_title(title)
        ax_prall.legend(loc='lower left')
        ax_prall.grid(True, alpha=0.3)
        fig_prall.tight_layout()
        out_prall = os.path.join(args.output_dir, f'pr_all_signals_{group_name}.png')
        fig_prall.savefig(out_prall, dpi=240)
        plt.close(fig_prall)

        # Rejection all signals
        fig_rejall, ax_rejall = plt.subplots(figsize=(10, 8))
        for base in signal_bases:
            if group_name in curves[base]:
                curve_data = curves[base][group_name]
                fpr, tpr = curve_data[0], curve_data[1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rejection = np.where(fpr > 0, 1.0 / fpr, np.inf)
                ax_rejall.plot(tpr, rejection, lw=2, label=f'{base}')
        ax_rejall.set_yscale('log')
        ax_rejall.set_xlabel('True Positive Rate (Signal Efficiency)')
        ax_rejall.set_ylabel('1/FPR (Background Rejection)')
        title = f'Background Rejection - {group_name}'
        if args.title:
            title += f' - {args.title}'
        ax_rejall.set_title(title)
        ax_rejall.legend(loc='lower left')
        ax_rejall.grid(True, alpha=0.3)
        fig_rejall.tight_layout()
        out_rejall = os.path.join(args.output_dir, f'rejection_all_signals_{group_name}.png')
        fig_rejall.savefig(out_rejall, dpi=240)
        plt.close(fig_rejall)

        if args.verbose:
            print(f'[OK] Gráficas conjuntas para {group_name} guardadas')

    # CORRECTED: Crear CSV completo con todas las métricas incluyendo F1 score
    complete_summary = []
    for base in signal_bases:
        for group_name, file_list in groups:
            if group_name in curves[base] and file_list:
                curve_data = curves[base][group_name]
                if len(curve_data) >= 7:  # Make sure we have F1 score
                    fpr, tpr, aucv, prec, rec, prauc, f1_opt = curve_data[:7]
                else:  # Fallback for old format
                    fpr, tpr, aucv, prec, rec, prauc = curve_data[:6]
                    f1_opt = 0.0
                    
                rej50 = rejection_at_eff(tpr, fpr, target_eff=0.5)
                rej99 = rejection_at_eff(tpr, fpr, target_eff=0.99)
                complete_summary.append([
                    base,           # signal_class
                    group_name,     # group
                    aucv,          # roc_auc
                    prauc,         # pr_auc
                    f1_opt,        # f1_score (NOW CORRECTLY CALCULATED)
                    rej50,         # rejection@50%
                    rej99          # rejection@99%
                ])
    
    # Guardar CSV completo
    complete_csv_path = os.path.join(args.output_dir, 'complete_metrics_summary.csv')
    lines = ['signal_class,group,roc_auc,pr_auc,f1_score,rejection@50%,rejection@99%']
    for row in complete_summary:
        lines.append('{},{},{:.6f},{:.6f},{:.6f},{:.6g},{:.6g}'.format(*row))
    
    with open(complete_csv_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    
    if args.verbose:
        print(f'[OK] CSV completo guardado en: {complete_csv_path}')

    print(f'[OK] Evaluación completada usando discriminante normalizado d = score(S) / (score(S) + score(B))')
    print(f'[OK] Salidas guardadas en: {args.output_dir}')


if __name__ == '__main__':
    main()