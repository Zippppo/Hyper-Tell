#!/usr/bin/env bash
# Run VoxTell inference on every case under niigz_data/ for all 25 organ
# prompts that the BDMAP ground-truth set covers.
#
# Usage:
#     bash tests/run_all_cases.sh [GPU_ID]
#
# Optional env overrides:
#     INPUT_DIR   (default: niigz_data)
#     OUTPUT_DIR  (default: outputs)
#     MODEL_DIR   (default: models-weight/voxtell/voxtell_v1.1)
#     CONDA_ENV   (default: voxtell)
set -euo pipefail

GPU_ID="${1:-1}"
INPUT_DIR="${INPUT_DIR:-niigz_data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
MODEL_DIR="${MODEL_DIR:-models-weight/voxtell/voxtell_v1.1}"
CONDA_ENV="${CONDA_ENV:-voxtell}"

# Auto-activate the voxtell conda env if voxtell-predict isn't on PATH.
if ! command -v voxtell-predict >/dev/null 2>&1; then
    for hook in \
        "${CONDA_PREFIX:-}/etc/profile.d/conda.sh" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [[ -f "${hook}" ]]; then
            # shellcheck disable=SC1090
            source "${hook}"
            break
        fi
    done
    conda activate "${CONDA_ENV}"
fi

if ! command -v voxtell-predict >/dev/null 2>&1; then
    echo "voxtell-predict not found even after activating '${CONDA_ENV}'." >&2
    echo "Activate the env manually: conda activate ${CONDA_ENV}" >&2
    exit 1
fi

PROMPTS=(
    "liver" "spleen" "right kidney" "left kidney" "pancreas"
    "gallbladder" "stomach" "aorta" "postcava" "esophagus"
    "duodenum" "colon" "intestine" "rectum" "bladder" "prostate"
    "left adrenal gland" "right adrenal gland" "celiac trunk"
    "hepatic vessel" "portal vein and splenic vein"
    "left lung" "right lung" "left femur" "right femur"
)

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
cases=("${INPUT_DIR}"/*/)
if [[ ${#cases[@]} -eq 0 ]]; then
    echo "No cases found under ${INPUT_DIR}/" >&2
    exit 1
fi

for case_dir in "${cases[@]}"; do
    case_id="$(basename "${case_dir}")"
    image="${case_dir}ct.nii.gz"
    out="${OUTPUT_DIR}/${case_id}_test"

    if [[ ! -f "${image}" ]]; then
        echo "[skip] ${case_id}: ${image} not found" >&2
        continue
    fi

    echo "==> ${case_id}  ->  ${out}"
    voxtell-predict \
        -i "${image}" \
        -o "${out}" \
        -m "${MODEL_DIR}" \
        -p "${PROMPTS[@]}" \
        --device cuda \
        --gpu "${GPU_ID}" \
        --verbose
done
