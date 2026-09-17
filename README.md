# Homeostatic inhibitory plasticity

A research implementation of homeostatic inhibitory plasticity, based on
*Homeostatic Inhibitory Plasticity in Daleian ANNs*. It compares three ways of
regulating neural activity on MNIST and evaluates the resulting representations
with a linear classifier.

| Model | Activity regulation |
|---|---|
| `rec` | Recurrent inhibition: excitatory activity drives an inhibitory neuron |
| `ff` | Feedforward inhibition: the input drives an inhibitory neuron |
| `thresh` | Adaptive excitatory thresholds, without an inhibitory population |

Baselines include initial sparse representations, raw pixels and a frozen MLP
representation. [SPEC.md](SPEC.md) defines the equations, metrics, selection rules
and their implementation references.

This repository is a publication snapshot of the reduced implementation of
the [homeostaticNN research project](https://github.com/TsujiKeiDayo/homeostaticNN).
Its earlier development history is retained separately. Saved notebook
outputs use repository-relative paths or `[local path]` placeholders;
this display-only cleanup preserves the code, numerical results and figures.

## 1. Set up the environment

The verified environment uses **Windows 11 x64, Python 3.13.1 and an NVIDIA GPU**.
The current machine has a GeForce RTX 4070 Laptop GPU. The pinned packages include
PyTorch **2.13.0+cu126** and torchvision **0.28.0+cu126**, built for CUDA 12.6.
Other operating systems, CPU execution and equivalence across GPUs have not been
validated for this setup.

Run these PowerShell commands from the repository root. Install Python 3.13.1
first; the version check should report that version. If this checkout already
has the verified `.venv`, skip creation and package installation.

```powershell
py -3.13 --version
py -3.13 -m venv .venv
& './.venv/Scripts/python.exe' -m pip install -r requirements-lock.txt
& './.venv/Scripts/python.exe' -m pip install --no-deps --no-build-isolation -e .
```

Check the environment before running an experiment:

```powershell
& './.venv/Scripts/python.exe' -m pip check
& './.venv/Scripts/python.exe' -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available` must be `True` for the configured `DEVICE = "cuda"`. The machine
needs an NVIDIA driver compatible with these PyTorch wheels.

[requirements-lock.txt](requirements-lock.txt) records the verified package
versions; [pyproject.toml](pyproject.toml) declares the package and its broader
dependency requirements. The lock includes notebook execution, UMAP and test
tools. JupyterLab is not included; the notebook workflow below uses VS Code.

For conda, [environment.yml](environment.yml) uses the same Python version and
pip lock:

```powershell
conda env create -f environment.yml
conda activate hnn2-reduced
python -m pip check
```

The conda recipe has not been recreated and tested independently of `.venv`.
With conda, replace the explicit `.venv` executable in subsequent commands with
the activated environment's `python` and select that environment in the editor.

## 2. Choose a notebook

Open this repository in VS Code with the Python and Jupyter extensions. Select
`.venv/Scripts/python.exe` as the notebook kernel. The repository's VS Code
settings also point to this interpreter.

The notebooks contain saved cell outputs, including **39 figures** from the
verified run. You can inspect those outputs without rerunning an experiment.
Rerunning cells requires the environment and input files described below. The
existing Git attributes name an optional `nbstripout` filter; configuring that
filter in another checkout strips outputs when notebooks are committed.

| Notebook | Purpose | Inputs needed to rerun |
|---|---|---|
| [00_mnist_overview.ipynb](notebooks/00_mnist_overview.ipynb) | Split sizes, class balance, example images and pixel distributions | Prepared MNIST NPZ |
| [run_experiment.ipynb](notebooks/run_experiment.ipynb) | HP search, training, baselines and saved status | Prepared MNIST NPZ; saved HP selection when resuming training |
| [01_selection.ipynb](notebooks/01_selection.ipynb) | Learning-rate selection, target selection and responses across targets | Complete experiment results and HP selection |
| [02_adaptation.ipynb](notebooks/02_adaptation.ipynb) | Activity, residuals, parameter trajectories and fixed-point diagnostics | Complete results; saved additional analysis for the additional chapter |
| [03_distributions.ipynb](notebooks/03_distributions.ipynb) | Activity distributions, S scores and bin diagnostics | Complete experiment results and HP selection |
| [04_classifier.ipynb](notebooks/04_classifier.ipynb) | Accuracy, paired seed differences and classifier learning curves | Complete experiment results and HP selection |
| [05_geometry.ipynb](notebooks/05_geometry.ipynb) | Silhouette scores, endpoint checks and UMAP panels | Complete results; saved UMAP coordinates for the additional chapter |

Each notebook runs independently from top to bottom. After changing inputs or
experiment settings, save the notebook, restart its kernel and run all cells.
For a visual adjustment, rerun the nearby display-settings and plotting cells.

## 3. Prepare the data

`data/` and `outputs/` are local artifacts excluded from Git. A fresh clone
contains notebook outputs but does not contain the prepared MNIST input or the
saved experiment arrays needed to reproduce the figures.

The default input is `data/mnist_seed-0.npz`. Reuse it if available. To create it,
review `DATASET_PARAMS`, `DATASET_OUTPUT` and `RAW_CACHE` in
[experiment_settings.py](experiment_settings.py), then run:

```powershell
& './.venv/Scripts/python.exe' scripts/make_dataset.py
```

This explicit preparation step may download MNIST and refuses to overwrite an
existing dataset. Other entry points read the prepared input. For a different
recipe, choose a new dataset output path and update the experiment's `DATASET`.

## 4. Run an experiment

Start with [run_experiment.ipynb](notebooks/run_experiment.ipynb). Its Settings cell
loads [experiment_settings.py](experiment_settings.py); notebook overrides do not
rewrite that shared file.

1. Set `DATASET` to the prepared input and choose a new `EXPERIMENT_ROOT`.
2. Review the model, target, learning-rate and seed population, together with
   the training and saving settings.
3. Enable the required stages with `RUN_HP`, `RUN_TRAINING` and `RUN_BASELINES`.
   All three default to `False`; enable all three for a complete experiment.
4. Choose an `EXECUTION_NAME`, save, restart the kernel and execute from the top.
   With computation enabled, settings are recorded under
   `results/notebook_runs/<EXECUTION_NAME>/`.

To train from saved HP results, set `SELECTION_SOURCE`, leave `RUN_HP=False` and
enable the required later stages. The saved selection supplies the model, target
and seed population. To inspect an existing experiment, point `EXPERIMENT_ROOT`
at it and leave all three switches off.

Current shared settings use 484 excitatory units, 200 settling updates,
10 adaptation epochs and a 15-epoch HP sweep. Each of the three models has
9 targets, 16 candidate learning rates and 3 seeds. HP selection scores epoch
index 7 and checks the next 2 epochs. `BATCH_RUNS=100` controls concurrent GPU
conditions; after an out-of-memory error, the workflow retries unfinished
conditions at widths 50 and 25. Review the population before a full run.

Completed results are reused only when recorded conditions match. Unfinished
conditions restart from the beginning. Use a new experiment location when
changing scientific conditions or saving options, and a new `EXECUTION_NAME`
when changing notebook execution settings.

### Script entry points

Scripts use `experiment_settings.py` directly. For one condition:

```powershell
& './.venv/Scripts/python.exe' scripts/run_single.py
```

| Task | Script | Main settings |
|---|---|---|
| One condition | [run_single.py](scripts/run_single.py) | `SINGLE_SPEC`, `CONFIG`, `ANALYSIS`, `SINGLE_OUTPUT` |
| HP through training and baselines | [run_pipeline.py](scripts/run_pipeline.py) | Sweep population, `CONFIG`, `RULE`, `MLP`, `DEVICE`, `BATCH_RUNS` |
| HP search and selection | [run_sweep.py](scripts/run_sweep.py) | `SWEEP_SPECS`, `SWEEP_CONFIG`, `RULE` |
| Training from saved HP | [run_experiments.py](scripts/run_experiments.py) | `SELECTION_OUTPUT`, `CONFIG` |
| Initial sparse, raw and MLP baselines | [run_baselines.py](scripts/run_baselines.py) | `MODELS`, `SEEDS`, `BASELINE_REPRESENTATIONS`, `MLP`, `CONFIG` |
| Retrain a classifier | [run_readout.py](scripts/run_readout.py) | `READOUT_SOURCE`, `READOUT_LR`, `READOUT_EPOCHS`, `READOUT_NAME` |
| Analyze saved runs | [run_analysis.py](scripts/run_analysis.py) | `ANALYSIS_SOURCES`, `ANALYSIS`, `UMAP_PARAMS`, recomputation switches |

For separate stages, run `run_sweep.py`, then `run_experiments.py`, then
`run_baselines.py`. Training from a saved selection does not launch another HP
search. Select saved run paths from the training or baseline `last_run.csv`;
example paths in the shared settings may use a different eta from your selection.

## 5. Inspect and save figures

Notebooks `01` through `05` read
[figure_settings.py](notebooks/figure_settings.py). Set `RESULTS` to the
experiment's `results` directory and `OUTPUT` to the figure-export root.
For an external HP selection, also set `selection_directory` in the loading cell
to that selection's location.

`DISPLAY_TARGETS=None` selects one target per model by the seed-median final
validation loss of the classifier trained on adapted features. To override the
display, use a dictionary such as `{'rec': .17, 'ff': .55, 'thresh': 1.55}`.
This leaves HP eta selection unchanged. Candidate evidence and effective display
targets are recorded separately.

Figures appear in cells by default; export switches are off. To export:

- For a chapter, set `SAVE_SECTION=True` and a new `SECTION_EXPORT_NAME`.
- For a whole notebook, set `SAVE_NUMBERS=True`, choose a new `EXPORT_NAME` and
  execute its final export cell.
- Save the notebook before exporting so its recorded source matches the settings
  used. Existing destinations are rejected. PNG defaults to on; PDF is off.

Additional chapters in `02` and `05` read saved numeric snapshots by default.
To create them, explicitly enable `RUN_ADDITIONAL_INFERENCE` or `RUN_UMAP`, choose
a new `ADDITIONAL_PATH`, save and execute, then return the switch to `False`.
These steps replay inference or fit UMAP without updating the homeostatic model.
Missing snapshots leave the corresponding additional figures unavailable.
`SAVE_ADDITIONAL_FIGURES` with a new `ADDITIONAL_EXPORT_NAME` exports only those
figures.

## 6. Settings and saved files

[experiment_settings.py](experiment_settings.py) holds experiment choices;
[config.py](src/hnn2/config.py) defines scientific defaults. The current shared
name is `synthetic_long_v1`, but the input is **MNIST**. Choose a descriptive new
`EXPERIMENT_NAME` for a new script-based experiment. The experiment notebook
defaults to the shared name with `_notebook_v1` appended.

```text
outputs/<experiment_name>/
├─ results/
│  ├─ hp/candidates/
│  ├─ hp/selection/
│  ├─ plasticity/
│  ├─ baselines/
│  ├─ single/
│  ├─ readout_variants/<readout_name>/
│  └─ notebook_runs/<execution_name>/
└─ analysis/notebook_figures_v2/
   ├─ <notebook_owner>/<export_name>/
   │  ├─ tables/figures/
   │  └─ figures/
   └─ additional/<analysis_name>/
```

Each operation creates its own directories. Scientific results keep conditions,
arrays and tables together. `COMPLETE` marks a finished result; failures are
recorded in `FAILED.json`. The notebook execution record's own `COMPLETE` marker
only confirms that its settings were recorded.

Eight shared `SAVE_*` switches control features, full-test features, encoder
weights, activity, summaries, classifier histories, PNG and PDF. All are on
except PDF. Disabling numeric saves preserves learning and evaluation but may
remove inputs required by result notebooks. Final evaluation and HP selection
evidence remain available.

Results include effective conditions, input identity and source copies. The
copies document configuration and are not standalone distributions. Provenance
contains paths as well as checksums, so moving results to another machine may
require more than changing `RESULTS`.

## 7. Read and extend the implementation

| Area | Start here |
|---|---|
| One experiment from input to evaluation | [single.py](src/hnn2/single.py) |
| State equations and plasticity | [dynamics.py](src/hnn2/model/dynamics.py), [plasticity.py](src/hnn2/model/plasticity.py), [SPEC.md](SPEC.md) |
| Adaptation and batched GPU runs | [adapt.py](src/hnn2/adapt.py), [hp/engine.py](src/hnn2/hp/engine.py) |
| HP selection and analysis targets | [hp/select.py](src/hnn2/hp/select.py), [figure_tables.py](src/hnn2/figure_tables.py) |
| Metrics | [metrics](src/hnn2/metrics) |
| Saving and reuse | [result_io.py](src/hnn2/result_io.py), [postprocess.py](src/hnn2/postprocess.py) |
| Figure aggregation and layout | The result notebook cells |

## 8. Verification

For a small set of checks without dataset downloads or experiment training:

```powershell
& './.venv/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider tests/unit/test_config.py tests/unit/test_metrics.py tests/unit/test_hp_select.py tests/unit/test_hp_validation.py
```

The full suite contains training tests. Some notebook regression tests use local
archived reference code under `history/`, which is excluded from Git.

On 2026-09-17, all **7 notebooks / 153 code cells** ran successfully in fresh
kernels with local inputs and saved results. Their **39 figures** were embedded
in cell outputs. The experiment notebook's three computation switches were off;
additional inference and UMAP fitting were also off. This verifies execution
with saved results, not fresh full-scale training or GUI interaction.

With the local reference exports and additional numeric snapshots available:

```powershell
& './.venv/Scripts/python.exe' -B scripts/check_saved_figures.py
```

This compares generated tables and figure coverage with saved reference exports,
blocks training and additional inference, and removes its temporary exports.
It requires local artifacts absent from a fresh clone.
