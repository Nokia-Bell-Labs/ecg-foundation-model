#!/bin/bash

# Models to test
models=("ked" "ecgfounder" "stmem" "moment" "moirai")

# Configuration
DEVICES=0
BATCH_SIZE=64
LEARNING_RATE=1e-3
LINEAR_PROB=false
SEED=42

# Log file for tracking runs
log_file="run_all_models_$(date +%Y%m%d_%H%M%S).log"
echo "Starting experiment runs at $(date)" | tee "$log_file"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local exp=$3
    local task_type=$4
    local task_override=$5
    
    echo "Running: Model=$model, Dataset=$dataset, Type=$task_type ${task_override:+Task=$task_override}"
    echo "$(date): Starting $model with $dataset ($exp) task.type=$task_type ${task_override:+exp.task=$task_override}" >> "$log_file"
    
    # Build the command
    cmd="python script/downstream.py model=\"$model\" dataset=\"$dataset\" exp=\"$exp\" task.type=\"$task_type\" exp.devices=$DEVICES exp.batch_size=$BATCH_SIZE exp.learning_rate=$LEARNING_RATE model.model_params.linear_prob=$LINEAR_PROB seed=$SEED"
    if [ -n "$task_override" ]; then
        cmd="$cmd exp.task=\"$task_override\""
    fi
    
    # Run the command
    eval "$cmd" 2>&1 | tee -a "logs/${model}_${dataset}_${task_override:-${task_type}}_$(date +%Y%m%d_%H%M%S).log"
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "$(date): SUCCESS - $model with $dataset" >> "$log_file"
        echo "✓ Completed successfully"
    else
        echo "$(date): FAILED - $model with $dataset - Exit code: $exit_code" >> "$log_file"
        echo "✗ Failed (exit code: $exit_code)"
    fi
    
    echo "----------------------------------------"
    sleep 2
}

# Run experiments for each model
for MODEL in "${models[@]}"; do
    echo ""
    echo "========================================"
    echo "=== STARTING EXPERIMENTS FOR MODEL: $MODEL ==="
    echo "========================================"
    echo ""
    
    echo "=== CLASSIFICATION TASKS ==="
    
    # Standard classification tasks
    run_experiment "$MODEL" "music_data" "music_downstream" "classification"
    run_experiment "$MODEL" "chapman_data" "chapman_downstream" "classification"
    run_experiment "$MODEL" "mimiciv_data" "mimiciv_downstream" "classification"
    
    # PTB-XL classification tasks
    ptbxl_tasks=("diagnostic" "subdiagnostic" "superdiagnostic" "form" "rhythm")
    for task in "${ptbxl_tasks[@]}"; do
        run_experiment "$MODEL" "ptbxl_data" "ptbxl_downstream" "classification" "$task"
    done
    
    # Icentia classification tasks
    icentia_tasks=("beat" "rhythm")
    for task in "${icentia_tasks[@]}"; do
        run_experiment "$MODEL" "icentia_data" "icentia_downstream" "classification" "$task"
    done
    
    # MCMED classification tasks
    mcmed_class_tasks=("ed-dispo" "dc-dispo" "acuity")
    for task in "${mcmed_class_tasks[@]}"; do
        run_experiment "$MODEL" "mcmed_data" "mcmed_downstream" "classification" "$task"
    done
    
    echo ""
    echo "=== REGRESSION TASKS ==="
    
    # Regression tasks
    run_experiment "$MODEL" "mimiciv_data" "mimiciv_downstream" "regression"
    
    mcmed_reg_tasks=("sbp" "dbp")
    for task in "${mcmed_reg_tasks[@]}"; do
        run_experiment "$MODEL" "mcmed_data" "mcmed_downstream" "regression" "$task"
    done
    
    aurorabp_tasks=("sbp" "dbp")
    for task in "${aurorabp_tasks[@]}"; do
        run_experiment "$MODEL" "aurorabp_data" "aurorabp_downstream" "regression" "$task"
    done
    
    echo ""
    echo "=== COMPLETED ALL EXPERIMENTS FOR MODEL: $MODEL ==="
    echo ""
done

