#!/bin/bash
#SBATCH -A rmaulik
#SBATCH -p a100-40gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --job-name=acars_stgB
#SBATCH --output=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260621__acars_stageB_std_gamma_first20_v1_3gpu/logs/acars_stgB_g%a_%j.out
#SBATCH --error=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260621__acars_stageB_std_gamma_first20_v1_3gpu/logs/acars_stgB_g%a_%j.err
#SBATCH --array=0-2

set -euo pipefail

GROUP_ID="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
N_GROUPS=3

REPO=/depot/rmaulik/data/yangxu/repos/aircraft_13var_obs_protocol
RUN_ROOT=/depot/rmaulik/data/yangxu/runs/aircraft_madis_abo_13var/20260621__acars_stageB_std_gamma_first20_v1_3gpu
PYTHON=/home/xu2279/.conda/envs/igra311/bin/python

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/samples" "${RUN_ROOT}/manifests"

cd "${REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/src/igra_gen:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "ACARS Stage B first20 std x gamma ablation"
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "group=${GROUP_ID}/${N_GROUPS}"
echo "repo=${REPO}"
echo "run_root=${RUN_ROOT}"
echo "python=${PYTHON}"
git rev-parse --show-toplevel
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short

LAMBDAS=(0.025 0.05)
STDS=(2.5e-4 5e-4 1e-3)
GAMMAS=(1e-6 2e-6 5e-6)
EXPERIMENTS=(
  acars_around25_v1_simple_sparse
  acars_around25_v4_equal_cell_mean
)

tag_val() {
  echo "$1" | sed 's/\./p/g; s/-/m/g'
}

combo_idx=0
for experiment in "${EXPERIMENTS[@]}"; do
  for lambda_aircraft in "${LAMBDAS[@]}"; do
    for std_aircraft in "${STDS[@]}"; do
      for gamma_aircraft in "${GAMMAS[@]}"; do
        lambda_tag="$(tag_val "${lambda_aircraft}")"
        std_tag="$(tag_val "${std_aircraft}")"
        gamma_tag="$(tag_val "${gamma_aircraft}")"
        if (( combo_idx % N_GROUPS == GROUP_ID )); then
          combo_name="${experiment}_lambda${lambda_tag}_std${std_tag}_gamma${gamma_tag}"
          combo_root="${RUN_ROOT}/runs/${combo_name}"
          echo
          echo "=== combo_idx=${combo_idx} combo=${combo_name} ==="
          echo "experiment=${experiment}"
          echo "lambda_aircraft=${lambda_aircraft}"
          echo "std_aircraft=${std_aircraft}"
          echo "gamma_aircraft=${gamma_aircraft}"
          echo "output=${combo_root}"
          mkdir -p "${combo_root}"
          "${PYTHON}" -m igra_gen.run_aircraft_13var_persistent \
            --output_root "${combo_root}" \
            --experiments "${experiment}" \
            --timesteps first20_12h \
            --ens 16 \
            --seed 17 \
            --num_steps 50 \
            --likelihood_mode multimodal \
            --std_igra 5e-4 \
            --gamma_igra 2e-6 \
            --lambda_igra 1.0 \
            --std_aircraft "${std_aircraft}" \
            --gamma_aircraft "${gamma_aircraft}" \
            --lambda_aircraft "${lambda_aircraft}"
        fi
        combo_idx=$((combo_idx + 1))
      done
    done
  done
done

echo
echo "Completed group ${GROUP_ID} at $(date -Is)"
