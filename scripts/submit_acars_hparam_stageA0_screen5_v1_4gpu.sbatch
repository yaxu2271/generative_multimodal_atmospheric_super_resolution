#!/bin/bash
#SBATCH -A rmaulik
#SBATCH -p a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=acars_hpA0
#SBATCH --output=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260620__acars_hparam_stageA0_screen5_v1_4gpu/logs/acars_hpA0_g%a_%j.out
#SBATCH --error=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260620__acars_hparam_stageA0_screen5_v1_4gpu/logs/acars_hpA0_g%a_%j.err
#SBATCH --array=0-3

set -euo pipefail

GROUP_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
N_GROUPS=4

REPO=/depot/rmaulik/data/yangxu/repos/aircraft_13var_obs_protocol
RUN_ROOT=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260620__acars_hparam_stageA0_screen5_v1_4gpu
PYTHON=/home/xu2279/.conda/envs/igra311/bin/python
SCREEN_TIMESTEPS=0,8,18,28,38

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/src/igra_gen:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "ACARS Stage A0 screen5 hyperparameter ablation"
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "group=${GROUP_ID}/${N_GROUPS}"
echo "timesteps=${SCREEN_TIMESTEPS}"
echo "repo=${REPO}"
echo "run_root=${RUN_ROOT}"
echo "python=${PYTHON}"
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short

LAMBDAS=(0.01 0.025 0.05 0.1 0.2)
EXPERIMENTS=(
  acars_around5_v1_simple_sparse
  acars_around5_v2_cell_balanced_sparse
  acars_around5_v4_equal_cell_mean
  acars_around25_v1_simple_sparse
  acars_around25_v2_cell_balanced_sparse
  acars_around25_v4_equal_cell_mean
)

combo_idx=0
for lambda_aircraft in "${LAMBDAS[@]}"; do
  lambda_tag="${lambda_aircraft//./p}"
  for experiment in "${EXPERIMENTS[@]}"; do
    if (( combo_idx % N_GROUPS == GROUP_ID )); then
      combo_name="${experiment}_lambda${lambda_tag}"
      combo_root="${RUN_ROOT}/runs/${combo_name}"
      echo
      echo "=== combo_idx=${combo_idx} combo=${combo_name} ==="
      echo "experiment=${experiment}"
      echo "lambda_aircraft=${lambda_aircraft}"
      echo "output=${combo_root}"
      mkdir -p "${combo_root}"
      "${PYTHON}" -m igra_gen.run_aircraft_13var_persistent \
        --output_root "${combo_root}" \
        --experiments "${experiment}" \
        --timesteps "${SCREEN_TIMESTEPS}" \
        --ens 16 \
        --seed 17 \
        --num_steps 50 \
        --likelihood_mode multimodal \
        --std_igra 5e-4 \
        --gamma_igra 2e-6 \
        --lambda_igra 1.0 \
        --std_aircraft 5e-4 \
        --gamma_aircraft 2e-6 \
        --lambda_aircraft "${lambda_aircraft}"
    fi
    combo_idx=$((combo_idx + 1))
  done
done

echo
echo "Completed group ${GROUP_ID} at $(date -Is)"
