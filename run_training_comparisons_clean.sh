#!/bin/bash

# Script seguro para ejecutar comparaciones de entrenamiento
# Versión limpia - ignora errores menores de sensors

# Configuración
BATCH_SIZE=128
SCRIPT_NAME="./train_JetClass_fix.sh"
LOG_FILE="training_log_$(date +%Y%m%d_%H%M%S).txt"
TEMP_LOG_FILE="temperature_log_$(date +%Y%m%d_%H%M%S).txt"

# Arrays con las configuraciones
MODELS=("ParT" "PN")
FEATURES=("full")
DATASET_SIZES=("1M")

# Configuración de temperatura (en Celsius)
MAX_GPU_TEMP=85
MAX_CPU_TEMP=90
COOLDOWN_TEMP=70

# Función para monitorear temperatura (versión limpia)
monitor_temperature() {
    # Obtener temperatura GPU (ignorar errores)
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 2>/dev/null || echo "0")
    
    # Obtener temperatura CPU (ignorar errores de sensors)
    local cpu_temp="0"
    if command -v sensors &> /dev/null; then
        # Extraer temperatura CPU correctamente
        cpu_temp=$(sensors 2>/dev/null | grep "Core 0" | awk '{print $3}' | sed 's/+//' | sed 's/°C//' | cut -d'.' -f1 2>/dev/null || echo "0")
    fi
    
    # Verificar que las temperaturas sean números válidos
    if ! [[ "$gpu_temp" =~ ^[0-9]+$ ]]; then
        gpu_temp="0"
    fi
    if ! [[ "$cpu_temp" =~ ^[0-9]+$ ]]; then
        cpu_temp="0"
    fi
    
    # Solo loggear si las temperaturas son válidas
    if [ "$gpu_temp" -gt 0 ] || [ "$cpu_temp" -gt 0 ]; then
        echo "$(date): GPU: ${gpu_temp}°C, CPU: ${cpu_temp}°C" >> "$TEMP_LOG_FILE"
    fi
    
    # Verificar si la temperatura es demasiado alta
    if [ "$gpu_temp" -gt "$MAX_GPU_TEMP" ] || [ "$cpu_temp" -gt "$MAX_CPU_TEMP" ]; then
        echo "⚠️  ALERTA: Temperatura alta - GPU: ${gpu_temp}°C, CPU: ${cpu_temp}°C"
        echo "⏸️  Pausando por 30 minutos para enfriamiento..."
        sleep 1800  # 30 minutos
        return 1
    fi
    
    return 0
}

# Función para esperar hasta que la temperatura baje
wait_for_cooldown() {
    echo "🔄 Esperando a que la temperatura baje a ${COOLDOWN_TEMP}°C..."
    
    while true; do
        # Obtener temperatura GPU
        local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 2>/dev/null || echo "0")
        
        # Obtener temperatura CPU (ignorar errores)
        local cpu_temp="0"
        if command -v sensors &> /dev/null; then
            cpu_temp=$(sensors 2>/dev/null | grep "Core 0" | awk '{print $3}' | sed 's/+//' | sed 's/°C//' | cut -d'.' -f1 2>/dev/null || echo "0")
        fi
        
        # Verificar que las temperaturas sean números válidos
        if ! [[ "$gpu_temp" =~ ^[0-9]+$ ]]; then
            gpu_temp="0"
        fi
        if ! [[ "$cpu_temp" =~ ^[0-9]+$ ]]; then
            cpu_temp="0"
        fi
        
        if [ "$gpu_temp" -lt "$COOLDOWN_TEMP" ] && [ "$cpu_temp" -lt "$COOLDOWN_TEMP" ]; then
            echo "✅ Temperatura normalizada - GPU: ${gpu_temp}°C, CPU: ${cpu_temp}°C"
            break
        fi
        
        echo "⏳ GPU: ${gpu_temp}°C, CPU: ${cpu_temp}°C - Esperando 5 minutos..."
        sleep 300  # 5 minutos
    done
}

# Función para ejecutar un entrenamiento
run_training() {
    local model=$1
    local features=$2
    local dataset_size=$3
    local run_number=$4
    local total_runs=$5
    
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Ejecutando: $model $features $dataset_size" | tee -a "$LOG_FILE"
    echo "Progreso: $run_number/$total_runs" | tee -a "$LOG_FILE"
    echo "Comando: DDP_NGPUS=1 $SCRIPT_NAME $model $features $dataset_size --batch-size $BATCH_SIZE" | tee -a "$LOG_FILE"
    echo "Hora de inicio: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # Monitorear temperatura antes de empezar (ignorar errores)
    monitor_temperature 2>/dev/null || true
    
    # Ejecutar el comando con filtrado de salida
    echo "🚀 Iniciando entrenamiento..."
    
    # Redirigir toda la salida al log completo, pero mostrar solo líneas importantes
    # Ignorar errores de sensors en la salida
    DDP_NGPUS=1 $SCRIPT_NAME $model $features $dataset_size --batch-size $BATCH_SIZE 2>&1 | \
        tee -a "$LOG_FILE" | \
        grep -E "(🚀|📊|🔧|Epoch|✅|❌|Error|Exception|Traceback)" | \
        grep -v "temp1_max_alarm" | \
        grep -v "Can't read" || true
    
    # Verificar si la ejecución fue exitosa
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Éxito: $model $features $dataset_size" | tee -a "$LOG_FILE"
    else
        echo "❌ Error: $model $features $dataset_size" | tee -a "$LOG_FILE"
    fi
    
    echo "Hora de finalización: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Función para mostrar el progreso
show_progress() {
    local current=$1
    local total=$2
    local model=$3
    local features=$4
    local dataset_size=$5
    
    echo "Progreso: $current/$total - $model $features $dataset_size"
}

# Función para calcular tiempo estimado
estimate_time() {
    local remaining=$1
    local hours_per_run=9
    local total_hours=$((remaining * hours_per_run))
    local days=$((total_hours / 24))
    local hours=$((total_hours % 24))
    
    echo "Tiempo estimado restante: ${days} días y ${hours} horas"
}

# Configurar logs
echo "=== Inicio de entrenamiento: $(date) ===" > "$LOG_FILE"
echo "=== Log de temperatura: $(date) ===" > "$TEMP_LOG_FILE"

# Contar total de ejecuciones
total_runs=0
for model in "${MODELS[@]}"; do
    for features in "${FEATURES[@]}"; do
        for dataset_size in "${DATASET_SIZES[@]}"; do
            ((total_runs++))
        done
    done
done

echo "🚀 Iniciando ejecución de $total_runs entrenamientos..." | tee -a "$LOG_FILE"
echo "📊 Tiempo estimado total: $((total_runs * 9)) horas ($((total_runs * 9 / 24)) días)" | tee -a "$LOG_FILE"
echo "🌡️  Monitoreando temperatura - Máx GPU: ${MAX_GPU_TEMP}°C, Máx CPU: ${MAX_CPU_TEMP}°C" | tee -a "$LOG_FILE"
echo "⏸️  Pausa entre experimentos: 30 minutos" | tee -a "$LOG_FILE"
echo "🧹 Modo limpio: ignorando errores menores de sensors" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Ejecutar todas las combinaciones
current_run=0

for model in "${MODELS[@]}"; do
    echo "🔄 Procesando modelo: $model" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    for features in "${FEATURES[@]}"; do
        for dataset_size in "${DATASET_SIZES[@]}"; do
            ((current_run++))
            show_progress $current_run $total_runs $model $features $dataset_size
            run_training $model $features $dataset_size $current_run $total_runs
            
            # Pausa entre experimentos (excepto el último)
            if [ $current_run -lt $total_runs ]; then
                remaining=$((total_runs - current_run))
                echo "⏸️  Pausa de 30 minutos entre experimentos..." | tee -a "$LOG_FILE"
                echo "$(estimate_time $remaining)" | tee -a "$LOG_FILE"
                echo "Hora de reanudación: $(date -d '+30 minutes')" | tee -a "$LOG_FILE"
                echo "" | tee -a "$LOG_FILE"
                
                # Monitorear temperatura durante la pausa (ignorar errores)
                for i in {1..6}; do
                    sleep 300  # 5 minutos
                    monitor_temperature 2>/dev/null || true
                done
            fi
        done
    done
    
    echo "✅ Completado modelo: $model" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "🎉 ¡Todas las ejecuciones completadas!" | tee -a "$LOG_FILE"
echo "📋 Resumen: $total_runs entrenamientos ejecutados" | tee -a "$LOG_FILE"
echo "📁 Logs guardados en: $LOG_FILE y $TEMP_LOG_FILE" | tee -a "$LOG_FILE" 