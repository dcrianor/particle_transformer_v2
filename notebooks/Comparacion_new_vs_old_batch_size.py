import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import glob
from sklearn.metrics import roc_curve, auc, precision_recall_curve

RESULTS_DIR = 'notebooks/resultados/comp_batch_size'
os.makedirs(RESULTS_DIR, exist_ok=True)

JET_CLASSES = [
    "HToBB", "HToCC", "HToGG", "HToWW4Q", "HToWW2Q1L",
    "TTBar", "TTBarLep", "WToQQ", "ZToQQ", "ZJetsToNuNu"
]

def load_all_class_predictions(base_dir, feature_set, model, dataset_size):
    pattern = f"{base_dir}/JetClass/Pythia/{feature_set}/{model}/{dataset_size}/[0-9]*-[0-9]*_pred_*.root"
    all_files = glob.glob(pattern)
    if not all_files:
        raise ValueError(f"No se encontraron archivos para {feature_set} en {pattern}")
    print(f"Encontrados {len(all_files)} archivos para {feature_set} en {dataset_size}:")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")
    dfs = []
    for file_path in all_files:
        with uproot.open(file_path) as f:
            tree_name = f.keys()[0]
            df = f[tree_name].arrays(library="pd")
            df['source_file'] = os.path.basename(file_path)
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

def plot_macro_avg_roc_comparison(dfs, nombres, save_path=None, titulo='Comparación ROC macro-promedio'):
    plt.figure(figsize=(10, 8))
    for df, nombre in zip(dfs, nombres):
        score_cols = [c for c in df.columns if c.startswith('score_')]
        y_true = df['_label_'].values
        scores = df[score_cols].values
        n_classes = scores.shape[1]
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i), scores[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, lw=2, label=f'{nombre} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(titulo)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_macro_avg_pr_comparison(dfs, nombres, save_path=None, titulo='Comparación PR macro-promedio'):
    plt.figure(figsize=(10, 8))
    for df, nombre in zip(dfs, nombres):
        score_cols = [c for c in df.columns if c.startswith('score_')]
        y_true = df['_label_'].values
        scores = df[score_cols].values
        n_classes = scores.shape[1]
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve((y_true == i), scores[:, i])
            precision = np.flip(precision)
            recall = np.flip(recall)
            mean_precision += np.interp(all_recall, recall, precision)
        mean_precision /= n_classes
        pr_auc = auc(all_recall, mean_precision)
        plt.plot(all_recall, mean_precision, lw=2, label=f'{nombre} (AUC={pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_auc_bars_by_class(dfs, nombres, save_path, jet_classes):
    aucs = {nombre: [] for nombre in nombres}
    for df, nombre in zip(dfs, nombres):
        score_cols = [c for c in df.columns if c.startswith('score_')]
        y_true = df['_label_'].values
        scores = df[score_cols].values
        for i, jet in enumerate(jet_classes):
            fpr, tpr, _ = roc_curve((y_true == i), scores[:, i])
            aucs[nombre].append(auc(fpr, tpr))
    x = np.arange(len(jet_classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    for idx, nombre in enumerate(nombres):
        ax.bar(x + idx*width - width/2, aucs[nombre], width, label=nombre)
    ax.set_ylabel('AUC ROC')
    ax.set_title('Comparación de AUC ROC por clase')
    ax.set_xticks(x)
    ax.set_xticklabels(jet_classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    # Guardar también en CSV
    df_auc = pd.DataFrame(aucs, index=jet_classes)
    df_auc.to_csv(save_path.replace('.png', '.csv'))

def main():
    base_dir = 'predictions'
    feature_set = 'full'
    model = 'PN'

    # 2M
    preds_2M_new = load_all_class_predictions(base_dir, feature_set, model, '2M/new')
    preds_2M_old = load_all_class_predictions(base_dir, feature_set, model, '2M/old')
    print(f'2M new: {len(preds_2M_new)} ejemplos')
    print(f'2M old: {len(preds_2M_old)} ejemplos')

    # 10M
    preds_10M_new = load_all_class_predictions(base_dir, feature_set, model, '10M/new')
    preds_10M_old = load_all_class_predictions(base_dir, feature_set, model, '10M/old')
    print(f'10M new: {len(preds_10M_new)} ejemplos')
    print(f'10M old: {len(preds_10M_old)} ejemplos')

    # Comparación para 2M
    plot_macro_avg_roc_comparison([preds_2M_new, preds_2M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'roc_macro_2M.png'),
        titulo='Curva ROC macro-promedio (2M)')
    plot_macro_avg_pr_comparison([preds_2M_new, preds_2M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'pr_macro_2M.png'),
        titulo='Curva PR macro-promedio (2M)')
    plot_auc_bars_by_class([preds_2M_new, preds_2M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'auc_roc_por_clase_2M.png'),
        jet_classes=JET_CLASSES)

    # Comparación para 10M
    plot_macro_avg_roc_comparison([preds_10M_new, preds_10M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'roc_macro_10M.png'),
        titulo='Curva ROC macro-promedio (10M)')
    plot_macro_avg_pr_comparison([preds_10M_new, preds_10M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'pr_macro_10M.png'),
        titulo='Curva PR macro-promedio (10M)')
    plot_auc_bars_by_class([preds_10M_new, preds_10M_old], ['new', 'old'],
        save_path=os.path.join(RESULTS_DIR, 'auc_roc_por_clase_10M.png'),
        jet_classes=JET_CLASSES)

if __name__ == "__main__":
    main() 