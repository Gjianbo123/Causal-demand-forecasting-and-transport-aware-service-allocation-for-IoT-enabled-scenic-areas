# Causal demand forecasting and transport-aware service allocation for IoT-enabled scenic areas

Research source code for forecasting synthetic scenic-area demand and allocating service resources under travel, inventory, and queue constraints. The repository contains the current forecasting and transport-aware rollout implementations, experimental comparison and audit scripts, and an independent, small CPU demonstration.

**This is a source-only release.** Historical experiment archives, trained weights, generated datasets, and manuscript outputs are not included. The small demonstration below is runnable from the supplied configuration; it does not reproduce the manuscript's final experiments or their reported results.

## Method and scope

The current method, identified as **v4** in the source, combines a seasonal demand anchor with learned graph residual forecasts and Graph WaveNet forecasts. A transport-aware receding-horizon rollout heuristic uses these predictions to evaluate resource transfers while accounting for travel time, resources already in transit, temporary loss of service during relocation, queue costs, and movement costs. The rollout is a heuristic and provides no global optimality guarantee.

Here, **causal** means that forecasting and decision inputs respect observation time and avoid access to future realized demand. It does not mean causal identification, treatment-effect estimation, or a causal discovery model. Historical templates and preprocessing are fitted on training data.

Synthetic inflow is exogenous; service queues respond to allocation. The supplied simulator and experiments concern fully synthetic data, rather than a validated deployment at a real scenic area. The original E-STGNN / ScenicFlow-Sim implementation is not included, and the new implementations should not be described as a recovery of that missing code.

## Repository contents

| Path | Purpose |
| --- | --- |
| `iotexp/forecast_improvement.py` | Seasonal anchor and learned graph residual predictor |
| `iotexp/improvement_runtime.py` | Forecast runtime and combined-predictor integration |
| `iotexp/transport_rollout.py` | Transport-aware marginal rollout scheduler |
| `iotexp/control_supplement.py` | Flow MPC, dispatch execution, and feasibility checks |
| `iotexp/scenic.py`, `iotexp/scenic_forecast.py` | Synthetic scenic-area environment and forecasting support |
| `iotexp/forecast_supplement.py` | Forecast comparison models and staged evaluation utilities |
| `iotexp/engine.py`, `iotexp/fusion.py` | PPO training/evaluation and forecast-fusion comparisons |
| `iotexp/telemetry.py` | Observation timestamps, missingness, noise, and delay handling |
| `configs/` | Runnable demo configuration and additional research configurations |
| `scripts/` | Training, evaluation, audits, plotting, dashboard, and manuscript utilities |
| `tests/` | Unit tests for temporal access, forecasting, resource constraints, and statistics |
| `adapters/` | Explicit placeholder for connecting an original external project |
| `validation/` | Historical demo validation notes, logs, and output-checking utility |

`README_ZH.md` and `START_HERE.txt` preserve the earlier Chinese demonstration instructions. They describe an earlier stage of the project. Their links to `docs/INTEGRATION_ZH.md` and `docs/PROTOCOL_ZH.md` refer to files absent from this source release.

## Run the independent CPU demonstration

Use Python 3.12 for the demonstration dependency set in `requirements.txt`. Run commands from the repository root. No real-world dataset, API key, or paid service is needed for this demonstration.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m iotexp run --config configs/demo_quick.json --out runs/my_demo
.\.venv\Scripts\python.exe validation/check_outputs.py runs/my_demo
```

### Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m iotexp run --config configs/demo_quick.json --out runs/my_demo
.venv/bin/python validation/check_outputs.py runs/my_demo
```

The demo uses a separate six-node synthetic queue environment, a ridge-regression forecaster, and five PPO input-fusion variants (F0-F4). Its configuration uses two training seeds and 192 environment interactions per policy. These deliberately small budgets check the workflow; they do not establish research performance or reproduce the v4 rollout method.

The `run` command trains, evaluates, and produces summaries and plots. It refuses to overwrite an existing output directory; choose a new `--out` path for another run. Typical outputs include `manifest.json`, checkpoints, episode metrics, raw traces, paired comparisons, and robustness plots. Demo results are marked as such. With only two training seeds, formal significance claims are not supported.

The following additional commands are available after installing dependencies:

```bash
python -m iotexp matrix
python -m iotexp summarize --out runs/my_demo
```

Use the virtual environment's Python executable, or activate that environment, when running these commands.

## Current research pipeline and required archives

The v4 research workflow is represented by these entry points:

| Script | Role |
| --- | --- |
| `scripts/run_forecast_improvement.py` | Develop and select the residual predictor |
| `scripts/select_forecast_combination.py` | Select the forecast combination using development data |
| `scripts/run_improvement_final.py` | Freeze inputs and run final forecasting/control evaluation |
| `scripts/analyze_improvement_v4.py` | Aggregate final experimental records |
| `scripts/audit_improvement_v4.py` | Check recorded results and provenance |
| `scripts/time_improvement_v4.py` | Measure the local computation pipeline |
| `scripts/plot_forecast_consistency.py`, `scripts/plot_control_evidence.py`, `scripts/plot_development_computation.py` | Render additional experiment figures from archived records |

These scripts are not a self-contained, one-command reproduction package. They reference prior-stage artifacts under `runs/scenic_rebuild_v2/`, `runs/supplement_v3/`, and `runs/improvement_v4/`, including simulator configuration, protocol and freeze records, training statistics, selected checkpoints, and prediction arrays. **Those directories are absent from this release.** Plotting and publication scripts also expect generated files under `output/`, which are absent. Some training scripts specify CUDA, and some document-rendering utilities depend on local tools and paths.

To evaluate the archived v4 setup, first obtain the matching research artifacts and restore their expected relative paths. Preserve source and asset hashes, follow the recorded development/freeze protocol, and do not substitute newly generated files while describing results as the original archived run. Creating a new complete experiment requires supplying and documenting a new protocol and the missing upstream artifacts. This release does not claim that its source alone reproduces any previously reported table or figure.

`requirements-rebuilt.txt` records a separate scientific environment used during the reconstructed study (Python 3.11.5 and PyTorch 2.7.1). Do not install both pinned requirements files into the same environment as a single combined specification. The optional manuscript/PDF utilities additionally use packages such as `python-docx`, `Pillow`, `lxml`, `latex2mathml`, `pdfplumber`, and `pypdf`; browser rendering and LaTeX compilation require their own tools. These optional publication utilities are not needed for the CPU demo.

## Interpretation and provenance

- Keep nominal and increased-demand conditions separate, and compare controllers on the same demand realizations and evaluation subset.
- Report movement, backlog, overload, and computation costs alongside waiting-time improvements. Lower waiting does not imply improvement in every metric.
- Use independent worlds or training seeds according to the recorded aggregation protocol; repeated decisions within an episode are not independent replications.
- Synthetic demonstrations, interface mock-ups, and archived screenshots do not establish real-world effectiveness.
- Validation files in this repository are historical records. A fresh test execution is needed to assess the current checkout and local environment.

No numerical superiority claim is made by this source-only release.
