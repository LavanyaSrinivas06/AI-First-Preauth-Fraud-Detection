# Thesis Quality Workbench

This folder contains the thesis-quality improvements:
1) Unit + functional tests (+ coverage target)
2) Latency + stress testing
3) Optimization pass notes/results
4) Drift detection + retraining trigger (PSI)
5) Evaluation protocol + plots
6) Drift robustness + adversarial simulations
7) Cost-sensitive analysis + threshold optimization
8) Final benchmark table + executive summary inputs
9) Reproducible environment + one-click run script


# Thesis Quality Runner

This folder contains "examiner-facing" scripts that run evaluation, drift checks, and system benchmarks
using the project's production scripts under `scripts/`.

## One click run (API must be running)
bash thesis_quality/reproducibility/run_all.sh

## Outputs
- thesis_quality/evaluation/results/*
- thesis_quality/drift/results/*
- thesis_quality/benchmarking/latency/results/*
- thesis_quality/benchmarking/load/results/*
