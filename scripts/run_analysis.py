"""Plot saved monitor curves; recompute metrics and UMAP only when enabled."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from hnn2.postprocess import compare_tables, analyse_saved, embed_saved
from hnn2.plots import save_figure
from hnn2.result_io import write_result

from experiment_settings import (
    ANALYSIS_SOURCES as SOURCES, ANALYSIS_OUTPUT as OUTPUT, RECOMPUTE_METRICS,
    COMPUTE_UMAP, ANALYSIS, UMAP_PARAMS, SETTINGS_PATH, SAVE_FIGURE_PNG, SAVE_FIGURE_PDF,
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    table = compare_tables(SOURCES, "monitor")
    write_result(
        OUTPUT / "tables/monitor",
        {"representation": "saved_plot", "source": [str(p) for p in SOURCES],
         "save_figure_png": SAVE_FIGURE_PNG, "save_figure_pdf": SAVE_FIGURE_PDF},
        tables={"plot_data": table}, script_path=__file__, settings_path=SETTINGS_PATH,
    )

    if SAVE_FIGURE_PNG or SAVE_FIGURE_PDF:
        fig, ax = plt.subplots()
        try:
            for seed, rows in table.groupby("seed_index"):
                ax.plot(rows.epoch, rows.rate_mean, label=f"seed {seed}")
            ax.set(xlabel="Epoch (zero based)", ylabel="Mean activity")
            ax.legend()
            if SAVE_FIGURE_PNG:
                save_figure(fig, OUTPUT / "figures/rate_mean.png")
            if SAVE_FIGURE_PDF:
                save_figure(fig, OUTPUT / "figures/rate_mean.pdf")
        finally:
            plt.close(fig)

    # Optional analysis uses saved arrays from the first source.
    if RECOMPUTE_METRICS:
        analyse_saved(
            SOURCES[0], OUTPUT / "tables/metrics", analysis=ANALYSIS,
            script_path=__file__, settings_path=SETTINGS_PATH,
        )
    if COMPUTE_UMAP:
        embed_saved(
            SOURCES[0], OUTPUT / "arrays/umap", UMAP_PARAMS,
            script_path=__file__, settings_path=SETTINGS_PATH,
        )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
