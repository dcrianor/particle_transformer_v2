#!/bin/bash

# set -x  # Comentado para reducir verbosidad

source env.sh

echo "🚀 Iniciando entrenamiento con args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi


# "kin", "kinpid", "full"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "❌ Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# "1M", "2M", "10M", "100M"
DATASET_SIZE=$3
[[ -z ${DATASET_SIZE} ]] && DATASET_SIZE="1M"

if ! [[ "${DATASET_SIZE}" =~ ^(1M|2M|10M|100M)$ ]]; then
    echo "❌ Invalid dataset size ${DATASET_SIZE}!"
    exit 1
fi

echo "📊 Configuración: Modelo=$model, Features=$FEATURE_TYPE, Dataset=$DATASET_SIZE"

# Calculate validation and test sizes based on training size
# Validation: 5% of training size, Test: 20% of training size
case ${DATASET_SIZE} in
    "1M")
        DATASET_VAL_SIZE="50k"
        DATASET_TEST_SIZE="200k"
        ;;
    "2M")
        DATASET_VAL_SIZE="100k"
        DATASET_TEST_SIZE="400k"
        ;;
    "10M")
        DATASET_VAL_SIZE="500k"
        DATASET_TEST_SIZE="2M"
        ;;
    "100M")
        DATASET_VAL_SIZE="5M"
        DATASET_TEST_SIZE="20M"
        ;;
esac


epochs=125  # Número de épocas 

# Convertir DATASET_SIZE a número de eventos
case "${DATASET_SIZE}" in
    "1M")   NUM_EVENTS=1000000 ;;
    "2M")   NUM_EVENTS=2000000 ;;
    "10M")  NUM_EVENTS=10000000 ;;
    "100M") NUM_EVENTS=100000000 ;;
    *) 
        echo "❌ DATASET_SIZE desconocido: ${DATASET_SIZE}"
        exit 1
        ;;
esac

# Cálculo de samples_per_epoch y samples_per_epoch_val
samples_per_epoch=$(( ($NUM_EVENTS / 10000) * 1024 / $NGPUS ))   
samples_per_epoch_val=$(( ($NUM_EVENTS / 10000) * 128 ))      

dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 128 --start-lr 5e-4" ## acá le cambié el tamaño del batch y el learning rate
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "❌ Invalid model $model!"
    exit 1
fi

# currently only Pythia
SAMPLE_TYPE=Pythia

echo "🔧 Ejecutando comando de entrenamiento..."

$CMD \
    --data-train \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/train_${DATASET_SIZE}/ZJetsToNuNu_*.root" \
    --data-val "${DATADIR}/${SAMPLE_TYPE}/val_${DATASET_VAL_SIZE}/*.root" \
    --data-test \
    "HToBB:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/HToBB_*.root" \
    "HToCC:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/HToCC_*.root" \
    "HToGG:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/HToGG_*.root" \
    "HToWW2Q1L:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/HToWW2Q1L_*.root" \
    "HToWW4Q:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/HToWW4Q_*.root" \
    "TTBar:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/TTBar_*.root" \
    "TTBarLep:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/TTBarLep_*.root" \
    "WToQQ:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/WToQQ_*.root" \
    "ZToQQ:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/ZToQQ_*.root" \
    "ZJetsToNuNu:${DATADIR}/${SAMPLE_TYPE}/test_${DATASET_TEST_SIZE}/ZJetsToNuNu_*.root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/${DATASET_SIZE}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/${DATASET_SIZE}/{auto}${suffix}.log --predict-output predictions/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/${DATASET_SIZE}/{auto}${suffix}_pred.root \
    --tensorboard JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/${DATASET_SIZE}/{auto}${suffix} \
    "${@:4}"

echo "✅ Entrenamiento completado para $model $FEATURE_TYPE $DATASET_SIZE"
