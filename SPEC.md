# Scientific specification

This repository implements the model family studied in *Homeostatic Inhibitory
Plasticity in Daleian ANNs*. This specification connects the equations to the
implementation and records deliberate differences from earlier versions. See
[README.md](README.md) for environment setup, execution and saved artifacts.

The equations are inherited from the earlier `clean` implementation. Historical
choices appear in Section 5; active experiment settings are in
[experiment_settings.py](experiment_settings.py), with defaults in
[config.py](src/hnn2/config.py).

## 1. Dynamics: a discrete-time settling map

Let `x` be the input, `v_E` and `v_I` the excitatory and inhibitory membrane
potentials, `o_E` the excitatory output, and `theta` the excitatory threshold.
Leak factors are `dec_X = exp(-1 / tau_X)`, with defaults `tau_E = 2` and
`tau_I = 20`.

| No. | Equation | Models |
|---|---|---|
| (1) | v_E(t) = dec_E · v_E(t−1) + W_EIn · x − W_EI · v_I(t−1) | `rec`, `ff` |
| (2) | v_E(t) = dec_E · v_E(t−1) + W_EIn · x | `thresh` |
| (3) | o_E(t) = max(v_E(t) − theta, 0) | All |
| (4) | v_I(t) = dec_I · v_I(t−1) + W_IE · o_E(t−1) | `rec` |
| (5) | v_I(t) = dec_I · v_I(t−1) + W_IIn · x | `ff` |

Both membrane-potential updates read the previous state. The excitatory output
then uses the newly updated `v_E`. Changing this order changes the model.

The default settling duration is **200 state updates** (V1). The terminal state
serves as the approximate steady response for adaptation and representation
extraction. A fixed update budget does not guarantee convergence for every input
and parameter setting.

For a constant net drive `I`, the leaky map has fixed point `I / (1 - dec)`.
Steady input gains are approximately 2.541 for the E pathway and 20.50 for the
I pathway, a ratio of approximately 8.07. This amplification is intentionally
retained. Analytical fixed points check the iterative calculation; they do not
replace it. The recurrent fixed-point check uses the settled excitatory output.

## 2. Homeostatic adaptation and selection

### Adaptation rules

Apply one update after settling each batch, with default batch size `B = 64`.
The target activity is `rho` (`targ` in code), and `[z]+ = max(z, 0)`.

| No. | Equation | Models |
|---|---|---|
| (6) | delta_W_EI = (o_E − rho) · s_pre^T / B; W_EI ← [W_EI + eta · delta_W_EI]+ | `rec`, `ff`; s_pre = v_I |
| (7) | delta_theta = mean_B(o_E − rho); theta ← [theta + eta · delta_theta]+ | `thresh` |
| (8) | eta(e) = eta_0 · d^max(0, e − s) | All; zero-based epoch e |

Here `d = eta_decay = 0.95` and `s = eta_decay_start_epoch = 3` by default.
The learning rate is unchanged through epoch index 3, then decays. The default
adaptation duration is 10 epochs, separate from the 200 settling updates within
each batch.

The rules preserve the earlier study's differences from Vogels et al. (2011):
the presynaptic factor is inhibitory **membrane potential**, updates occur once
per batch at the settling endpoint, and the update is averaged over the batch.
The presynaptic signal is an explicit argument of `model.plasticity.update`.
Homeostatic adaptation uses neither class labels nor loss differentiation.

For one excitatory unit, the expected unclamped weight update satisfies

```text
E[delta_w_i] ∝ (mean_b(o_ib) − rho) · mean_b(v_Ib)
               + cov_b(o_ib, v_Ib).
```

When mean inhibitory potential is positive, an interior zero-update condition
constrains the `v_I`-weighted mean activity to the target, rather than necessarily
the unweighted mean. For the threshold rule, the corresponding interior condition
is unweighted mean activity equal to the target. Nonnegative clamping can
introduce boundary conditions.

The `rate_wmean` monitor records the weighted mean. These conditions concern
training batches, so `train_probe` also records `rate_mean_train` and
`rate_wmean_train` alongside validation monitoring. Validation/test residuals are
distinct from the training fixed-point condition. Historical rankings and
residual thresholds are not automatic pass/fail criteria.

### Learning-rate selection: rule v4

HP search evaluates explicit `RunSpec` candidates using `workflows.run_sweep`.
For each `(model, targ)` and candidate `eta`, form the seed-mean score:

```text
J(e) = mean_seed(l_mean(e) + lambda · rate_var(e)),  lambda = 1.
score = J(at_epoch).
fluctuation = max(J(at_epoch + 1), ..., J(at_epoch + N)) / score − 1.
```

A candidate is stable when `fluctuation <= stability_tol`. Select the stable
candidate with the smallest score, breaking exact ties by smaller eta. If none
is stable, select the unfiltered minimum and record `argmin_unstable`. A zero
score followed by a zero plateau has zero fluctuation; a zero score followed by
a positive value has infinite fluctuation. `stability_tol=None` disables the
filter. Required epochs and matching seed coverage are checked before scoring.

The forward window detects transient minima as activity crosses the target. A
backward relative-change window was rejected because a near-zero objective can
make small absolute changes appear arbitrarily large.

The rule's defaults and current experiment overrides differ:

| Parameter | `SelectionRule()` default | Current `experiment_settings.py` |
|---|---:|---:|
| `at_epoch` (zero-based) | 9 | 7 |
| `stability_epochs` | 3 | 2 |
| `stability_tol` | 0.5 | 0.5 |
| `var_weight` | 1.0 | 1.0 |

The current experiment trains for 10 epochs and sweeps for 15. Its score is
measured after the eighth epoch, with stability checked at epoch indices 8 and 9.
Training duration is recorded separately as `training_epochs`. A sweep must
cover both training duration and the full forward window; its length is not
inferred solely from `at_epoch`.

### Target selection for analysis

For each model, compare the **seed-median final validation loss** of the linear
classifier trained on adapted features. Candidates must use the same seed set.
Among targets within absolute difference `1e-6` of the minimum (`rtol=0`), choose
the smaller target. Use the loss at the final classifier step, not the best loss
in its history or the training loss.

This active rule, introduced on 2026-09-16, supersedes target selection by the
adaptation-monitor composite. Eta selection still uses rule v4. Missing final
losses are rejected rather than inferred. `DISPLAY_TARGETS` can explicitly
override the displayed target per model; candidate evidence and the effective
choice are recorded separately.

## 3. Initialization and monitoring

The current configuration uses 484 excitatory units. The `rec` and `ff`
initializers support one inhibitory unit; `thresh` has no inhibitory population.
Let `mu` be the row mean of the input-to-E weight matrix.

| No. | Equation | Notes |
|---|---|---|
| (9) | W_EIn ~ Exp(1) / sqrt(n_in); mu = row_mean(W_EIn); W_IE = mu^T / sqrt(n_E); W_EI = mu / sqrt(n_I); W_IIn ~ Exp(1) / sqrt(n_in); theta_0 = theta_init · 1 | Generate only weights required by the model. `mu` uses the original float64 draws before conversion to float32. |
| (10) | L = (mean_i(r_i) − rho)^2 + lambda · Var_i(r_i); lambda = 1 gives L = mean_i((r_i − rho)^2) | r_i is unit i's sample-mean activity. Population variance makes this a per-unit MSE. |

The current `theta_init` is **0.145**, used in the MNIST experiment and config
defaults since 2026-09-16. The earlier value was 0.175; older fixtures retain
their original settings and expected values.

Equation (10) is a monitoring and selection objective evaluated on validation
data every epoch. The adaptation rule does not read its variance weight. The
earlier Phase 1/dissertation objective used `lambda = 0.3`.

For warm start, present the first 10 training samples sequentially in batches of
one. Replicate the resulting terminal state as the initial state for a batch.
The source is always the training split, including when encoding evaluation
data. Adaptation builds this state at each epoch's start and reuses it across
that epoch's batches; probes rebuild it with the post-epoch parameters.
Validation probes are measured each epoch; test analysis probes are taken after
the first and final adaptation epochs.

## 4. Equations, implementation and verification

Implementation paths are relative to `src/hnn2/`; test paths are relative to
`tests/`, unless written with an explicit repository-root prefix. Some historical
comparison tests require local reference artifacts.

| Element | Implementation | Verification |
|---|---|---|
| Equations (1)–(5), update order and fixed points | `model/dynamics.py: step, analytic_fixed_point` | `dynamics/test_fixed_point.py` |
| 200-update settling loop | `simulate.py: settle` | `dynamics/test_fixed_point.py`, including a manual one-step comparison |
| Equations (6)–(7), direction and clamping | `model/plasticity.py: update` | `dynamics/test_properties.py`, including input immutability |
| Equation (8), learning-rate schedule | `model/plasticity.py: eta_schedule` | `dynamics/test_properties.py` |
| Equation (9), initialization and streams | `model/init.py: build_params` | `unit/test_rng.py` |
| Equation (10), weighted means and training probes | `adapt.py: monitor_stats, monitor_loss, train_probe` | `dynamics/`, `integration/test_reduced_workflows.py` |
| Run-axis batched execution | `hp/engine.py: run_adaptation_batch`, `model/params.py: stack, select` | `dynamics/test_batched_core.py`, `dynamics/test_engine_equivalence.py`: golden 2-D behavior, sequential equivalence at 1e-5 and prefix behavior |
| HP sweep and selection | `workflows.py`, `hp/rules.py`, `hp/select.py` | `integration/test_reduced_workflows.py`, `unit/test_hp_select.py`, `unit/test_hp_validation.py` |
| Descriptive power-law fit for selected eta | `hp/scaling.py: fit_power_law` | `unit/test_scaling.py`; no research-ranking pass/fail criterion |
| Analysis-target selection | `figure_tables.py: analysis_target_tables` | `unit/test_figure_revision.py` |
| Warm start | `simulate.py: warm_start_state` | `dynamics/test_properties.py` |
| MNIST preparation, stratification and paired resolutions | `data.py` | `unit/test_data.py`: fixed raw pools, saved-array round trips and historical comparisons |
| Entropy concentration scores and bins | `metrics/entropy.py`, `metrics/bins.py` | `unit/test_metrics.py`, `unit/test_regressions.py` |
| Cosine silhouette | `metrics/silhouette.py` | `unit/test_regressions.py` |
| Linear classifier: Adam, learning rate 3e-3, 15 epochs | `readout.py`, `encoders.py` | `dynamics/test_golden_p4.py`, `integration/test_reduced_workflows.py` |
| Saved-result tables and figures | `notebooks/01` through `05`, `figure_tables.py`, `figure_plots.py`, `scripts/run_analysis.py` | Notebook tests in `tests/unit/`; `scripts/check_saved_figures.py` with local reference exports |

### Metric conventions

For histogram probabilities `p_k`, Shannon entropy is
`H = -sum_k(p_k * log2(p_k))`, omitting zero-probability terms. For `K` bins,
`Hhat = H / log2(K)` and `S = 1 - Hhat`; higher S means greater concentration.
`S_prime = S * (1 - zero_bin_frac)`. The lowest bin represents near-silent
activity, not necessarily exact zeros. For a single bin, the implementation uses
a normalizing denominator of 1.

Inherited population bins span `[0, 6]` in steps of `0.05` (120 bins);
dataset/lifetime bins span `[0, 3]` in steps of `0.05` (60 bins). The notebooks
label as **dataset** the same definition stored under **lifetime** keys.
Alternative diagnostic bins do not replace the main definitions.

A histogram with no counted mass has undefined `S` and `S_prime` (`NaN`), rather
than concentration 1. Undefined silhouette returns `None`. Out-of-range mass
must raise an error or be explicitly allowed and recorded. A `BinSpec` whose
range is not divisible by its width is rejected, because its generated edges
would not reach the declared upper bound.

## 5. Scientific choices inherited from earlier implementations

The original implementation preceded `clean`, from which this reduced repository
inherits its equations. Identifiers V1–V13 remain useful when reading source
comments. Historical settings below do not override the active rules above.

| ID | Choice and rationale |
|---|---|
| V1 | Define settling as 200 state updates; the original loop performed 199. The earlier comparison recorded no effect on layer 2. |
| V2 | Use named random streams for each weight. Do not generate the unused 484-by-484 `w_EE`; numerical compatibility with the original seed sequence was deliberately relinquished. |
| V3 | Warm-start from training samples for every split. The original implementation used the split being evaluated. Analysis and classifier features now share the training-source protocol. |
| V4 | Reject or explicitly record histogram mass outside the declared range. The original implementation silently discarded it. |
| V5 | Record target-selection evidence and boundary flags instead of relying on manual constants. Earlier multi-criterion argmax tables are diagnostic artifacts; final validation loss now determines analysis targets (Section 2). |
| V6 | Use validation adaptation probes every epoch and test analysis probes after the first and final epochs. The original implementation probed test data every epoch; historical first/final analysis positions are retained. |
| V7 | Permit targets at or below `theta_init` with an informational warning. The original hard constraint excluded that region. |
| V8 | Fit UMAP only when explicitly requested, with recorded parameters and seed. Treat it as a qualitative visualization, separately from cosine silhouette. |
| V9 | Use `readout/init` and `readout/shuffle` for classifiers; the MLP initializer uses `mlp/readout_init`. `single.run_representation` and `postprocess.readout_only` share the classifier assignment. |
| V10 | Set the monitoring variance weight to 1, making the objective per-unit MSE. This coefficient does not enter the homeostatic update itself. |
| V11 | Select eta per `(model, targ)` with the fixed 16-point grid from 1e-6 to 30 and rule v4. Original v4 defaults scored epoch 9 with a three-epoch forward window and 50% rise tolerance; current overrides appear in Section 2. |
| V12 | Preserve single-run and run-axis batched computation. GPU rounding can vary with batch width; record it with the conditions. |
| V13 | Include model, target, eta and seed in readable condition directories. Compare conditions before reuse and reject mismatched completed outputs. |

The following historical observations explain choices and limits; they are not
new experimental results or acceptance thresholds:

- The 2026-09-05 fixed-point investigation reported unweighted `rec`/`ff` means
  below target in the original `rho > theta_init` region, and `ff` means above
  target below `theta_init`. The covariance term in Section 2 explains why the
  weighted and unweighted conditions differ.
- Rule v4 defaults were fixed on 2026-09-05. `sensitivity.csv` records sensitivity
  and `divergence_vs_legacy.csv` compares against the dissertation objective
  (`lambda=0.3`, epoch 9, no stability filter). The decay factor, decay-start epoch
  and 10-epoch adaptation budget were retained from that study.
- A backward relative-variation filter was rejected after the 2026-09-04
  diagnosis described in Section 2. Earlier rule v3 used a fixed-point residual
  gate that validation generalization gaps could make unreachable. It was
  archived in the predecessor project under tag `hp-v3-2026-09-02`; it is not an
  execution entry point here.
- In a migration comparison, `ff` activity differed by up to `4.33e-5` across
  batch widths in both old and reduced implementations. Agreement between
  implementations was checked at the same width. This does not establish
  bitwise equivalence across devices or execution widths.

## 6. Recording, reproducibility and limits

- Inputs are explicit NPZ files. Downloading and preprocessing occur only during
  explicit dataset creation. Both resolutions use the same sampled MNIST images;
  arrays and the preparation recipe are saved together.
- `seed_index` indexes `PRIMES` before named streams are derived. Indices 0–8
  preserve the prior `clean` assignments; unused reserved streams were removed.
- Results record conditions, input identity, NPZ arrays and CSV tables.
  `COMPLETE` is written last; exceptions before completion produce `FAILED.json`.
- HP curves and the selection rule are separate artifacts. Selection can be
  repeated from saved curves; training records selected and used eta with the
  selection source.
- Missing save flags or classifier-stream declarations in older results are
  rejected rather than reconstructed. A missing schema version alone is a
  separate compatibility question and does not relax required saved fields.
- Tests cover equations, fixed points, update order, initialization independence
  and representative numerical comparisons. Their scope depends on the selected
  tests and available reference artifacts.
- Historical numerical differences are retained. A later saved-activity replay
  comparison had 12 unequal cases out of 18, with maximum absolute difference
  `1.0967254638671875e-05`; its cause was not investigated. Final features matched
  exactly for all 9 checked runs. These observations do not justify relaxing
  tolerances or relabeling unequal comparisons as equal.
- README verification establishes notebook execution with saved inputs, not
  fresh full-scale training, scientific agreement for every condition, or
  CPU/other-GPU equivalence.

Local migration and experiment history provides additional audit records, but
`history/` and `docs/` are excluded from Git. This specification and the README
describe the current implementation without requiring those files.
