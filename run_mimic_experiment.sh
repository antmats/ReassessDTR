#!/bin/bash
#SBATCH -A NAISS2024-5-480 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 2-00:00:00

ReassessDTR_env="/mimer/NOBACKUP/groups/inpole/ppdev/rl_env.sif"
ppdev_env="/mimer/NOBACKUP/groups/inpole/ppdev/ppdev_env.sif"

project="mlhc2025"  # wandb project name

reward_setting="Outcome"
max_seq_len=6

num_runs=10

bc_algo="discrete-imitation-rnn"
bc_num_epochs=50
bc_metric="PatientWiseF1"

rl_algos=("dqn-obs_cat" "discrete-bcq-obs_cat" "discrete-cql-obs_cat")
rl_num_epochs=50
rl_metric="WIS_truncated"

export WANDB_API_KEY="WANDB_API_KEY"
export APPTAINERENV_WANDB_PROJECT="$project"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 experiment_dir"
    exit 1
fi
experiment_dir="$1"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "This script must be run as a Slurm array job. For example:"
    echo "  sbatch --output=\"\${experiment_dir}/logs/%x_%A_%a.out\" --job-name=\"rl_mimic\" --array=1-10 run_mimic_experiment.sh \"\${experiment_dir}\""
    exit 1
fi

cd ~
rsync -r ReassessDTR "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ReassessDTR"

rsync -av "$ReassessDTR_env" ./env.sif

bind="--bind ${TMPDIR}/ReassessDTR:/mnt/ReassessDTR"

env="MIMIC3Sepsis${reward_setting}Env"
seed=$((SLURM_ARRAY_TASK_ID - 1))
trial=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

# Prepare data.
num_seeds=$(find "$experiment_dir" -maxdepth 1 -type d -name 'trial_*' | wc -l)
apptainer exec --bind "${HOME}/ppdev:/mnt/ppdev" "$ppdev_env" python "${HOME}/ppdev/scripts/save_split_indices_to_file.py" \
    --config_path "${experiment_dir}/default_config.yml" \
    --output_path ./DTRGym/MIMIC3SepsisEnv \
    --num_seeds "$num_seeds"
apptainer exec $bind --nv env.sif python ./DTRGym/MIMIC3SepsisEnv/run_preprocess.py \
    --seed "$seed" \
    --reward_option "$reward_setting" \
    --max_len "$max_seq_len" \
    --use_predefined_splits
echo "Data preparation completed."

# Copy extracted actions to the experiment directory.
rsync -av "./DTRGym/${env}/offline_data/action.csv" "${experiment_dir}/trial_${trial}/action.csv"

# Fit behavior policy model.
bc_sweep_id=$(apptainer exec $bind --nv env.sif python ./experiment/run_sepsis.py \
    --env "$env" \
    --algo_name "$bc_algo" \
    --nonlinear \
    --train_buffer all_ope_train \
    --val_buffer all_val \
    --test_buffer_keyword all_val \
    --role sweep \
    --epoch "$bc_num_epochs" \
    --OPE_methods "$bc_metric" \
    --OPE_metric "$bc_metric" \
    --logdir "${PWD}" \
    --n_runs "$num_runs" \
    --project "$project" | tail -n 1)
echo "Behavior policy model training completed."

# Obtain the best behavior policy model.

results_dir=$(find . -maxdepth 1 -type d -name "${env}*" | head -n 1)
if [ -z "$results_dir" ]; then
    echo "No results directory was found."
    exit 1
fi

echo "Copying results..."
./cleanup_checkpoints.sh "$results_dir"
rsync -r "$results_dir" "${experiment_dir}/trial_${trial}"

best_score=-1
best_trial_dir=""
for trial_dir in ${results_dir}/${bc_algo}/*/; do
    summary_file="${trial_dir}run_summary.json"
    if [ -f "$summary_file" ]; then
        score=$(apptainer exec env.sif python -c "
import json
with open('$summary_file') as f:
    data = json.load(f)
print(data.get('all_val-${bc_metric}', -1))
")
        if (( $(echo "$score > $best_score" | bc -l) )); then
            best_score="$score"
            best_trial_dir="$trial_dir"
        fi
    else
        echo "Summary file not found: ${summary_file}."
    fi
done

if [ -n "$best_trial_dir" ]; then
    echo "Best trial found: ${best_trial_dir} with score ${best_score}."
else
    echo "Failed to obtain the best behavior policy model."
    exit 1
fi

# Calibrate the behavior policy model.
# (Luo et al. suggest that calibrating the behavior policy model increases variance in OPE.)
#apptainer exec $bind --nv env.sif python ./experiment/run_calibration.py \
#    --project "$project" \
#    --env_name "$env" \
#    --behavioural_model_path "${best_trial_dir}/policy.pth" \
#    --model_save_path "${best_trial_dir}/calibrated_model.pt" \
#    --plot_save_dir "$best_trial_dir"

# Learn RL policies.
for rl_algo in "${rl_algos[@]}"; do
    echo "Running RL algorithm: ${rl_algo}."
    apptainer exec $bind --nv env.sif python ./experiment/run_sepsis.py \
        --project "$project" \
        --env "$env" \
        --algo_name "$rl_algo" \
        --linear \
        --role sweep \
        --train_buffer all_train \
        --val_buffer all_val \
        --test_buffer_keyword test \
        --epoch "$rl_num_epochs" \
        --behavioural_sweep_id "$bc_sweep_id" \
        --behavioural_model_path "${best_trial_dir}/policy.pth" \
        --OPE_methods "$rl_metric" \
        --OPE_metric "$rl_metric" \
        --n_runs "$num_runs" \
        --logdir "${PWD}"
        #--calibrate_behavioural \
        #--calibrated_model_path "${best_trial_dir}/calibrated_model.pt"

    echo "Copying results..."
    ./cleanup_checkpoints.sh "$results_dir"
    rsync -r "$results_dir" "${experiment_dir}/trial_${trial}"
done

echo "Experiment completed successfully."
