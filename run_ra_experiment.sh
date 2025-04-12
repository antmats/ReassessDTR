#!/bin/bash
#SBATCH -A NAISS2024-5-480 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 1-00:00:00

image="/mimer/NOBACKUP/groups/inpole/rapomo/rl_env.sif"

project="mlhc2025"  # wandb project name

num_runs=10

bc_algo="discrete-imitation"
bc_num_epochs=50
bc_metric="PatientWiseF1"

rl_algos=("dqn" "discrete-bcq" "discrete-cql")
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
    echo "  sbatch --output=\"\${experiment_dir}/logs/%x_%A_%a.out\" --job-name=\"rl_ra\" --array=1-10 run_ra_experiment.sh \"\${experiment_dir}\""
    exit 1
fi

cd ~
rsync -r ReassessDTR "$TMPDIR" --exclude="*_env"
cd "${TMPDIR}/ReassessDTR"

rsync -av "$image" ./env.sif

bind="--bind ${TMPDIR}/ReassessDTR:/mnt/ReassessDTR"

env="RAEnv"
seed=$((SLURM_ARRAY_TASK_ID - 1))
trial=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

# Prepare data.

rapomo_env="/mimer/NOBACKUP/groups/inpole/rapomo/rapomo_env.sif"
apptainer exec --bind "${HOME}/rapomo:/mnt/rapomo" "$rapomo_env" python -c "
import os
import joblib
import pandas as pd
from rapomo.utils import load_config
from rapomo.data import get_data_handler, get_dataset
from rapomo.data import utils as utils

config_path = os.path.join('$experiment_dir', f'trial_{$SLURM_ARRAY_TASK_ID:03d}', 'config.yml')
config = load_config(config_path)
assert config['experiment']['data_handler'] == '4'

data_handler = get_data_handler(config)
data_train, data_val, data_test = data_handler.split_data()
data_train = utils.filter_cohort(data_train, c_stage='stage')

preprocessor = data_handler.get_preprocessor('minmax', 'onehot')

for split, data in zip(['train', 'val', 'test'], [data_train, data_val, data_test]):
    dataset = get_dataset(data, config)
    if hasattr(preprocessor, 'n_features_in_'):
        inputs = preprocessor.transform(dataset.inputs)
    else:
        inputs = preprocessor.fit_transform(dataset.inputs)
    inputs = pd.DataFrame(
        inputs,
        columns=preprocessor.get_feature_names_out(),
    )
    dataset.inputs = inputs
    output_file_path = os.path.join('./DTRGym/${env}/', f'{split}.pkl')
    dataset.save_data(output_file_path)
"

apptainer exec $bind --nv env.sif python ./DTRGym/RAEnv/run_preprocess.py
echo "Data preparation completed."

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
