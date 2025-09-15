#!/bin/bash

# Script para monitorear el progreso de los entrenamientos
# Uso: ./monitor_training.sh [log_file]

# Encontrar el archivo de log más reciente
if [ -z "$1" ]; then
    LOG_FILE=$(ls training_log_*.txt 2>/dev/null | tail -1 2>/dev/null)
    if [ -z "$LOG_FILE" ]; then
        echo "❌ No se encontraron archivos de log de entrenamiento"
        echo "Archivos de log disponibles:"
        ls training_log_*.txt 2>/dev/null || echo "No hay archivos de log"
        exit 1
    fi
else
    LOG_FILE="$1"
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ No se encontró archivo de log: $LOG_FILE"
    echo "Archivos de log disponibles:"
    ls training_log_*.txt 2>/dev/null || echo "No hay archivos de log"
    exit 1
fi

echo "📊 Monitoreando entrenamiento desde: $LOG_FILE"
echo "Presiona Ctrl+C para salir"
echo "=" * 60

# Función para mostrar el progreso
show_progress() {
    # Configuración real del script de comparaciones
    local total_runs=6  # 2 modelos × 3 features × 1 dataset = 6
    local completed_runs=$(grep -c "✅ Éxito\|❌ Error" "$LOG_FILE" 2>/dev/null || echo "0")
    local current_run=$(grep "Ejecutando:" "$LOG_FILE" | tail -1 | grep -o "Progreso: [0-9]*/[0-9]*" | cut -d' ' -f2 | cut -d'/' -f1 2>/dev/null || echo "0")
    
    # Si no hay current_run, calcular basado en completed_runs
    if [ "$current_run" -eq 0 ] 2>/dev/null && [ "$completed_runs" -gt 0 ] 2>/dev/null; then
        current_run=$((completed_runs + 1))
    elif [ "$current_run" -eq 0 ] 2>/dev/null; then
        current_run=1
    fi
    
    echo "📈 Progreso: $completed_runs/$total_runs completados"
    echo "🔄 Ejecutando: Experimento $current_run/$total_runs"
    
    # Mostrar experimentos esperados
    echo ""
    echo "📋 Experimentos configurados:"
    echo "   • PN kin 10M"
    echo "   • PN kinpid 10M" 
    echo "   • PN full 10M"
    echo "   • ParT kin 10M"
    echo "   • ParT kinpid 10M"
    echo "   • ParT full 10M"
    
    # Mostrar últimos 5 experimentos ejecutados
    echo ""
    echo "📋 Últimos experimentos ejecutados:"
    grep -E "(Ejecutando:|✅ Éxito|❌ Error)" "$LOG_FILE" | tail -5 | while read line; do
        if [[ $line == *"Ejecutando:"* ]]; then
            echo "🔄 $line"
        elif [[ $line == *"✅ Éxito"* ]]; then
            echo "✅ $line"
        elif [[ $line == *"❌ Error"* ]]; then
            echo "❌ $line"
        fi
    done
    
    # Mostrar temperatura actual en tiempo real
    echo ""
    echo "🌡️  Temperatura actual:"
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 2>/dev/null || echo "N/A")
    local cpu_temp=$(sensors 2>/dev/null | grep "Core 0" | awk '{print $3}' | sed 's/+//' | sed 's/°C//' | cut -d'.' -f1 2>/dev/null || echo "N/A")
    echo "   GPU: ${gpu_temp}°C, CPU: ${cpu_temp}°C"
    
    # Mostrar última temperatura del log también
    local temp_file=$(ls temperature_log_*.txt 2>/dev/null | tail -1)
    if [ -f "$temp_file" ]; then
        local last_temp=$(tail -1 "$temp_file" 2>/dev/null)
        if [ ! -z "$last_temp" ]; then
            echo "📝 Última temperatura registrada: $last_temp"
        fi
    fi
}

# Monitoreo en tiempo real
while true; do
    clear
    show_progress
    echo ""
    echo "⏳ Actualizando en 30 segundos... (Ctrl+C para salir)"
    sleep 30
done 