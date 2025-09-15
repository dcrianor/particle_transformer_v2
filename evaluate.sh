#!/bin/bash

# Script para monitorear el progreso de los entrenamientos
# Uso: ./evaluate.sh [log_file]

echo "Iniciando evaluación de modelos..."
echo "=================================="

# Comparación de modelos para JetClass 10M full
echo "Ejecutando comparación ParT vs PN para 10M full..."
python plot_qcd_vs_signals_v4_normalized_d.py \
  --group "ParT_full_10M=predictions/JetClass/Pythia/full/ParT/10M/*_pred_*.root" \
  --group "PN_full_10M=predictions/JetClass/Pythia/full/PN/10M/*_pred_*.root" \
  --qcd-name label_QCD \
  --title "JetClass 10M full" \
  --output-dir graficos_comparativos_v5/JetClass/model_comparison

echo "Ejecutando comparación macro OVO para ParT vs PN para 10M full..."

python compare_macro_ovo_model_v2.py \
  --part_10m "predictions/JetClass/Pythia/full/ParT/10M/*_pred_*.root" \
  --pn_10m "predictions/JetClass/Pythia/full/PN/10M/*_pred_*.root" \
  --output-dir "graficos_comparativos_v5/JetClass/model_comparison/macro" \
  --title "Comparación ParT vs PN - JetClass" \
  --verbose

# Comparación de features para ParT
echo "Ejecutando comparación ParT: kin vs kinpid vs full..."
python plot_qcd_vs_signals_v4_normalized_d.py \
  --group "ParT_kin=predictions/JetClass/Pythia/kin/ParT/10M/*_pred_*.root" \
  --group "ParT_kinpid=predictions/JetClass/Pythia/kinpid/ParT/10M/*_pred_*.root" \
  --group "ParT_full=predictions/JetClass/Pythia/full/ParT/10M/*_pred_*.root" \
  --qcd-name label_QCD \
  --title "ParT 10M: kin vs kinpid vs full" \
  --output-dir graficos_comparativos_v5/JetClass/features_subset_comparison/ParT

# Comparación de features para PN
echo "Ejecutando comparación PN: kin vs kinpid vs full..."
python plot_qcd_vs_signals_v4_normalized_d.py \
  --group "PN_kin=predictions/JetClass/Pythia/kin/PN/10M/*_pred_*.root" \
  --group "PN_kinpid=predictions/JetClass/Pythia/kinpid/PN/10M/*_pred_*.root" \
  --group "PN_full=predictions/JetClass/Pythia/full/PN/10M/*_pred_*.root" \
  --qcd-name label_QCD \
  --title "PN 10M: kin vs kinpid vs full" \
  --output-dir graficos_comparativos_v5/JetClass/features_subset_comparison/PN

echo "Ejecutando comparación macro OVO para kin vs kinpid vs full..."
python compare_macro_ovo_features_v2.py \
  --title "JetClass 10M - Features subsets comparison" \
  --output-dir graficos_comparativos_v5/JetClass/features_subset_comparison

# Comparación de tamaños de dataset para PN
echo "Ejecutando comparación PN: 1M vs 2M vs 10M..."
python plot_qcd_vs_signals_v4_normalized_d.py \
  --group "PN_1M=predictions/JetClass/Pythia/full/PN/1M/*_pred_*.root" \
  --group "PN_2M=predictions/JetClass/Pythia/full/PN/2M/*_pred_*.root" \
  --group "PN_10M=predictions/JetClass/Pythia/full/PN/10M/*_pred_*.root" \
  --qcd-name label_QCD \
  --title "PN full: 1M vs 2M vs 10M" \
  --output-dir graficos_comparativos_v5/JetClass/dataset_size_comparison/PN

# Comparación de tamaños de dataset para ParT
echo "Ejecutando comparación ParT: 1M vs 2M vs 10M..."
python plot_qcd_vs_signals_v4_normalized_d.py \
  --group "ParT_1M=predictions/JetClass/Pythia/full/ParT/1M/*_pred_*.root" \
  --group "ParT_2M=predictions/JetClass/Pythia/full/ParT/2M/*_pred_*.root" \
  --group "ParT_10M=predictions/JetClass/Pythia/full/ParT/10M/*_pred_*.root" \
  --qcd-name label_QCD \
  --title "JetClass full: 1M vs 2M vs 10M" \
  --output-dir graficos_comparativos_v5/JetClass/dataset_size_comparison/ParT

echo "Ejecutando comparación macro dataset sizes: 1M vs 2M vs 10M..."
python compare_macro_ovo_dataset_sizes_v2.py \
    --title "JetClass full - Dataset size comparison" \
    --output-dir graficos_comparativos_v5/JetClass/dataset_size_comparison

echo "=================================="
echo "Evaluación completada!"