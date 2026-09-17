"""Independent notebook execution with fixed saved arrays; no scientific runs."""
import hashlib
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_fixtures import fixed_experiment
from hnn2.figure_tables import load_figure_runs
from notebook_cells import SCORE_ARMS, SCORE_TABLES, frozen_reference

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"rec": .35, "ff": .55, "thresh": .35}
CHAPTERS = {
    "01_selection": (3, "hp_curves hp_landscape hp_selected hp_used_eta target_evidence target_statistics target_candidate_evidence target_diagnostic_candidates analysis_targets conditions coverage"),
    "02_adaptation": (8, "monitor"),  # Fixture has two seeds, not three.
    "03_distributions": (5, "activity_metrics activity_histograms activity_statistics endpoint_differences endpoint_difference_statistics entropy overall_entropy overall_entropy_statistics overall_S_median_pivot selected_entropy_scores selected_entropy_statistics"),
    "04_classifier": (4, "summary selected_summary accuracy_statistics paired_accuracy accuracy_differences history validation_points validation_trajectory threshold_detail threshold_summary"),
    "05_geometry": (1, "silhouette silhouette_statistics endpoint_anchors"),
}

GUARD = '''
import hnn2.single as single
import hnn2.workflows as workflows
import hnn2.mlp as mlp
import hnn2.umap_embed as umap
def forbidden(*args, **kwargs):
    raise AssertionError("No training, inference, UMAP, or whole-chapter Python wrappers")
for module, names in (
    (single, ("run_adaptation", "fit_readout", "sparse_features")),
    (workflows, ("run_adaptation_batch", "run_representation")),
    (mlp, ("train_mlp_e2e",)), (umap, ("embed",)),
):
    for name in names:
        setattr(module, name, forbidden)
'''


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = fixed_experiment(tmp_path_factory.mktemp("v2_notebooks"))
    runs, expected = load_figure_runs(root)
    return root, frozen_reference().make_figure_tables(runs, expected, root / "hp/selection", TARGETS)


def test_owned_tables_cover_v2_once():
    names = [name for _, tables in CHAPTERS.values() for name in tables.split()]
    assert len(names) == len(set(names)) == 36


@pytest.mark.parametrize("source,figure", [("activity", "G3_activity_histograms"), ("features", "G3_endpoint_features")])
@pytest.mark.parametrize("colors", [{}, {"ep00": "purple", "ep02": "green", "initial": "purple", "trained": "green"}])
def test_histogram_stage_colors(saved, source, figure, colors):
    """Use the actual notebook cells; check every panel and both broken-axis halves."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from hnn2.figure_plots import _histogram_axes
    from hnn2.plots import MODEL_COLORS
    results, tables = saved
    notebook = nbformat.read(ROOT / "notebooks/03_distributions.ipynb", as_version=4)
    namespace = dict(plt=plt, np=__import__("numpy"), tables=tables,
                     targets=TARGETS, models=list(TARGETS), figures={},
                     chosen_paths=set(tables["selected_summary"].run),
                     MODEL_COLORS=MODEL_COLORS, _histogram_axes=_histogram_axes,
                     SHOW_FIGURES=False)
    for tag in (f"style-{figure}", f"plot-{figure}"):
        cell = next(c for c in notebook.cells if tag in c.metadata.get("tags", []))
        if tag.startswith("plot-"):
            namespace["CURVE_COLORS"] = colors
        exec(compile(cell.source, f"03_distributions:{cell.id}", "exec"), namespace)
    figure_object = namespace["figures"][figure]
    panels = []
    for axis in figure_object.axes:
        patches = axis.patches
        assert len(patches) == 2
        assert patches[0].get_edgecolor() != patches[1].get_edgecolor()
        for patch in patches:
            stage = patch.get_label().split(";")[0]
            if stage in colors:
                assert patch.get_edgecolor() == to_rgba(colors[stage])
        panels.append(tuple((p.get_label(), p.get_edgecolor()) for p in patches))
    # Color identity remains the same whenever a stage reappears in another panel.
    by_stage = {}
    for panel in panels:
        for label, color in panel:
            stage = label.split(";")[0]
            assert by_stage.setdefault(stage, color) == color
    plt.close(figure_object)


@pytest.mark.parametrize("stem", CHAPTERS)
def test_independent_notebook_matches_saved_tables(saved, tmp_path, stem):
    results, reference = saved
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    before = path.read_bytes()
    source_hashes = hashes(results)
    notebook = nbformat.read(path, as_version=4)
    first_section = True
    for cell in notebook.cells:
        assert cell.source.isascii(), "Notebook text, comments, and labels must be English"
        if cell.cell_type == "code":
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, f"{stem}:{cell.id}", "exec")
        else:
            assert not any(line.startswith("    ") for line in cell.source.splitlines()), "Prose must render as Markdown, not an indented code block"
        if "section-export" in cell.metadata.get("tags", []) and first_section:
            cell.source = cell.source.replace("SAVE_SECTION = False", "SAVE_SECTION = True")
            first_section = False
        if "parameters" in cell.metadata.get("tags", []):
            # Keep every real default parameter; override only fixture inputs/outputs.
            cell.source += f'''\nRESULTS = Path({str(results)!r})
OUTPUT = Path({str(tmp_path)!r})
ADDITIONAL_PATH = OUTPUT / "additional" / "unused"
DISPLAY_TARGETS = {TARGETS!r}
SAVE_NUMBERS = True
SHOW_FIGURES = False
'''
    notebook.cells.insert(2, nbformat.v4.new_code_cell(GUARD))
    notebook.cells.append(nbformat.v4.new_code_cell('''
# Export one chapter as PDF, independently of the all-figure PNG snapshot.
SAVE_PNG, SAVE_PDF = False, True
first_figure = next(iter(figures))
export_section("single_pdf", [first_figure], OWNED_TABLES)
try:
    export_section(EXPORT_NAME, list(figures), OWNED_TABLES)
except FileExistsError:
    pass
else:
    raise AssertionError("An existing output must not be overwritten")
assert not plt.get_fignums(), "Displayed figures must not accumulate open managers"
'''))
    NotebookClient(notebook, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute(env={**os.environ, "MPLBACKEND": "Agg"})
    owner = stem.split("_", 1)[1]
    destination = tmp_path / owner / "all_v1"
    numeric = destination / "tables/figures"
    count, table_names = CHAPTERS[stem]
    assert {p.stem for p in numeric.glob("*.csv")} == set(table_names.split())
    assert len(list((destination / "figures").glob("*.png"))) == count
    expected_figures = {
        tag.removeprefix('plot-') for cell in notebook.cells
        for tag in cell.metadata.get('tags', []) if tag.startswith('plot-G')
    }
    if stem == '02_adaptation':
        expected_figures |= {
            f'G2_parameter_{model}_seed-{seed}'
            for model in TARGETS for seed in (0, 1)
        }
    assert {p.stem for p in (destination / "figures").glob("*.png")} == expected_figures
    assert not list(destination.rglob("*.pdf"))  # Default PDF remains OFF.
    for name in table_names.split():
        # CSV round-trip is the public contract; empty strings become NaN on both sides.
        from io import StringIO
        expected = reference[name]
        if name in SCORE_TABLES:
            expected = expected[expected.arm.isin(SCORE_ARMS)].reset_index(drop=True)
        wanted = pd.read_csv(StringIO(expected.to_csv(index=False)), float_precision="round_trip")
        actual = pd.read_csv(numeric / f"{name}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(actual, wanted, check_exact=True, obj=name)
    metadata = json.loads((numeric / "conditions.json").read_text(encoding="utf-8"))
    assert metadata["display_targets"] == TARGETS
    assert len(metadata["source"]) == 22
    assert (numeric / "COMPLETE").is_file()
    assert (numeric / "experiment.ipynb").read_bytes() == before
    figure_sources = json.loads((destination / "figures/figure_sources.json").read_text(encoding="utf-8"))
    assert figure_sources["numeric_source"] == str(numeric)
    assert "notebook_settings.json" in figure_sources["numeric_source_sha256"]
    single = tmp_path / owner / "single_pdf"
    assert len(list(single.rglob("*.pdf"))) == 1
    assert not list(single.rglob("*.png"))
    chapter = next(p for p in (tmp_path / owner).iterdir() if p.name not in ("all_v1", "single_pdf"))
    assert (chapter / "tables/figures/COMPLETE").is_file()
    assert list(chapter.rglob("*.png"))
    assert source_hashes == hashes(results)
    assert path.read_bytes() == before
