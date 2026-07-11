#!/usr/bin/env bash
set -euo pipefail
cd /mnt/infini-data/test/quan_space/codespace/aidd_0604/code_0602_opo
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
/mnt/infini-data/test/quan_space/envs/aidd/bin/python scripts/make_coreflow_paper_figures.py  --output outputs/paper_figures/coreflow_20260624  --device cuda:0
