"""実験設定はここで編集する。importだけでは計算・保存・downloadしない。

3d-2で指定されたMNIST条件。科学既定値は hnn2.config.ExperimentConfig
にあり、探索候補・学習予算はここへ明示する。GPUのみで実行し、メモリ不足なら
同時計算数を100→50→25へ下げて未完了条件を再試行する。条件を変えたら
新しい EXPERIMENT_NAME を使う。
"""
from dataclasses import replace
from pathlib import Path

from hnn2.config import ExperimentConfig, RunSpec
from hnn2.data import DatasetParams
from hnn2.hp.rules import SelectionRule
from hnn2.single import AnalysisSettings, MlpSettings
from hnn2.result_io import SaveOptions
from hnn2.umap_embed import UmapParams

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = Path(__file__).resolve()

# 入力・出力・実行環境。準備済みMNISTを再利用する。
MNIST_DATASET = ROOT / "data/mnist_seed-0.npz"
DATASET = MNIST_DATASET
OUTPUT_ROOT = ROOT / "outputs"
EXPERIMENT_NAME = "synthetic_long_v1"  # ユーザー指定名。入力はMNIST。
EXPERIMENT_ROOT = OUTPUT_ROOT / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_ROOT / "results"
DEVICE = "cuda"
TORCH_THREADS = 1  # PyTorchのCPU演算内の並列数。GPU条件数とは別。
BATCH_RUNS = 100  # GPUで同時に計算する条件数。メモリ不足なら50、25で再試行。

# モデル・条件・seed_index（用途別乱数の素数seed列へのインデックス）。
MODELS = ("rec", "ff", "thresh")
SEEDS = (0, 1, 2)
SINGLE_SPEC = RunSpec(model_code="rec", targ=.35, eta=.001, seed_index=0)

# 共通のモデル条件。未指定項目は科学既定値を使う。
BASE_CONFIG = ExperimentConfig(theta_init=0.145)  # ユーザー指定。E=484、settle=200、warm start=10、batch=64。

# 本実験・Baselineの後段classifier。n_epochsはplasticity learningの回数。
CONFIG = replace(BASE_CONFIG, n_epochs=10, readout_epochs=15, readout_eval_every=1)
MLP = MlpSettings(epochs=15, batch_size=64)  # MLP表現の学習。後段classifierと別。
BASELINE_REPRESENTATIONS = ("raw", "mlp_frozen")

# HP探索と既存のη選択規則。評価位置は0始まり。本学習の長さとは独立。
# 本学習対象とseedは保存済みHP条件から引き継ぎ、各model・targの選択ηを使う。
SWEEP_CONFIG = replace(BASE_CONFIG, n_epochs=15)
SWEEP_TARGETS = (.17, .20, .27, .35, .55, .90, 1.15, 1.40, 1.55)
SWEEP_ETAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
              1e-2, 3e-2, 1e-1, 3e-1, 1., 3., 10., 30.)
SWEEP_SPECS = [RunSpec(model, targ, eta, seed) for model in MODELS
               for targ in SWEEP_TARGETS for eta in SWEEP_ETAS for seed in SEEDS]
RULE = SelectionRule(at_epoch=7, stability_epochs=2)
# at_epoch=7は8 epoch終了時。安定性窓は続く2 epoch、上昇許容50%。
# GPU実行・中心化OFF・100→50→25の再試行をユーザー承認済み。

# 解析定義。
ANALYSIS = AnalysisSettings()
UMAP_PARAMS = UmapParams(n_neighbors=3)

# 保存だけを制御する。Falseでも学習・評価の計算は同じ。全入口で共通。
SAVE_FEATURES = True  # train・val・testの特徴とラベル。
SAVE_FULL_TEST_FEATURES = True  # full testの特徴とラベル。SAVE_FEATURESとは独立。
SAVE_ENCODER_WEIGHTS = True  # 該当する学習前後の重みとBaselineのencoder。
SAVE_ACTIVITY = True  # 保存対象epochの活動配列とラベル。
SAVE_SUMMARIES = True  # 軌跡・活動要約。
SAVE_READOUT_HISTORY = True  # 後段classifierのstep履歴。最終評価は常に保存。
SAVE_FIGURE_PNG = True
SAVE_FIGURE_PDF = False  # PDFが必要なときだけTrueにする。

# 上のスイッチを数値保存APIへまとめて渡す。変更は上の8項目で行う。
SAVING = SaveOptions(save_features=SAVE_FEATURES, save_full_test=SAVE_FULL_TEST_FEATURES,
                     save_encoder_weights=SAVE_ENCODER_WEIGHTS, save_activity=SAVE_ACTIVITY,
                     save_summaries=SAVE_SUMMARIES, save_readout_history=SAVE_READOUT_HISTORY)

# 各入口の出力先。EXPERIMENT_NAMEを変えると、実験全体の保存先が変わる。
# 条件フォルダ内のNPZ・CSV・設定・完了記録は、従来どおり一緒に置く。
SINGLE_OUTPUT = RESULTS_ROOT / "single/rec_target-0.35_seed-0"
EXPERIMENTS_OUTPUT = RESULTS_ROOT / "plasticity"
BASELINES_OUTPUT = RESULTS_ROOT / "baselines"
SWEEP_OUTPUT = RESULTS_ROOT / "hp/candidates"
SELECTION_OUTPUT = RESULTS_ROOT / "hp/selection"

# 保存済み結果の再利用。対象パスは results/plasticity/last_run.csv から選ぶ。
# 以下は従来の例。HPで選ばれたηによって実際のパスは異なる。上流は自動実行しない。
READOUT_SOURCE = EXPERIMENTS_OUTPUT / "rec/target-0.35_eta-0.001/seed-0"
READOUT_LR = .02
READOUT_EPOCHS = 3
READOUT_NAME = "lr-0.02_v1"  # 再学習条件・対象を変える場合は別名にする。
READOUT_OUTPUT = RESULTS_ROOT / "readout_variants" / READOUT_NAME
ANALYSIS_SOURCES = [EXPERIMENTS_OUTPUT / f"rec/target-0.35_eta-0.001/seed-{seed}"
                    for seed in SEEDS]
ANALYSIS_NAME = "comparison_v1"  # 解析条件や図の体裁を変える場合は別名にする。
ANALYSIS_OUTPUT = EXPERIMENT_ROOT / "analysis" / ANALYSIS_NAME
RECOMPUTE_METRICS = False
COMPUTE_UMAP = False

# データ準備はmake_dataset.pyから明示実行。既存入力を上書きしない。
RAW_CACHE = ROOT / "data/raw"
DATASET_OUTPUT = MNIST_DATASET
DATASET_PARAMS = DatasetParams.from_config(BASE_CONFIG)
