"""Explicit extra chapters and saved-number redraw; never train a model."""
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_fixtures import fixed_experiment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("stem,switch", [("02_adaptation", "RUN_ADDITIONAL_INFERENCE"), ("05_geometry", "RUN_UMAP")])
def test_compute_save_load_redraw_without_recomputation(tmp_path, stem, switch):
    results = fixed_experiment(tmp_path / "fixture")
    output = tmp_path / "output"
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    source = nbformat.read(path, 4)
    # The extra chapters depend on shared input checks, not other drawing chapters.
    notebook = nbformat.v4.new_notebook(cells=[c for c in source.cells
        if c.id.endswith(("-01", "-02", "-03", "-05")) or c.id.startswith("extra-")])
    params = next(c for c in notebook.cells if "parameters" in c.metadata.get("tags", []))
    params.source += f'''
RESULTS = Path({str(results)!r})
OUTPUT = Path({str(output)!r})
ADDITIONAL_PATH = OUTPUT / "additional"
DISPLAY_TARGETS = {{"rec": .35, "ff": .55, "thresh": .35}}
SHOW_FIGURES = False
{switch} = True
'''
    guard = nbformat.v4.new_code_cell('''
import hnn2.adapt as adapt
import hnn2.model.plasticity as plasticity
import hnn2.single as single
import hnn2.umap_embed as umap
def forbidden(*args, **kwargs):
    raise AssertionError("No training allowed")
adapt.update = plasticity.update = single.fit_readout = forbidden
def fixed_embedding(features, params, *, random_state):
    assert params.n_neighbors == 3
    return np.column_stack([np.arange(len(features)), np.asarray(features).mean(axis=1)]).astype(np.float32)
umap.embed = fixed_embedding  # Only the optional UMAP fit is replaced.
''')
    notebook.cells.insert(1, guard)
    for cell in notebook.cells:
        if "additional-export" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace("SAVE_ADDITIONAL_FIGURES = False", "SAVE_ADDITIONAL_FIGURES = True")
    notebook.cells.append(nbformat.v4.new_code_cell('''
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ADDITIONAL_PATH.iterdir() if p.is_file()}
import hnn2.saved_inference as inference
inference.training_order_activity = inference.fixed_point_detail = forbidden
umap.embed = forbidden
'''))
    for tag in ("additional-load", "additional-style", "additional-plot"):
        notebook.cells.append(nbformat.v4.new_code_cell(next(c.source for c in source.cells if tag in c.metadata.get("tags", []))))
    notebook.cells.append(nbformat.v4.new_code_cell('''
assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ADDITIONAL_PATH.iterdir() if p.is_file()}
SAVE_PNG, SAVE_PDF = False, True
export_additional_figures(OUTPUT / "pdf_redraw")
try:
    export_additional_figures(OUTPUT / "pdf_redraw")
except FileExistsError:
    pass
else:
    raise AssertionError("Refuse existing exports")
assert not plt.get_fignums()
'''))
    # Tampered source numbers must fail before loading/redrawing.
    load = next(c.source for c in source.cells if "additional-load" in c.metadata.get("tags", []))
    notebook.cells.append(nbformat.v4.new_code_cell('''
numeric_file = next(ADDITIONAL_PATH.glob("*.csv"))
numeric_file.write_bytes(numeric_file.read_bytes()+b"\\n")
try:
    exec(''' + repr(load) + ''')
except ValueError as exc:
    assert "changed" in str(exc)
else:
    raise AssertionError("Tampered numbers accepted")
'''))
    NotebookClient(notebook, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute(env={**os.environ, "MPLBACKEND": "Agg"})
    numeric = output / "additional"
    assert (numeric / "experiment.ipynb").read_bytes() == path.read_bytes()
    if stem.startswith("02"):
        assert len(pd.read_csv(numeric / "fixed_point_summary.csv")) == 6
        assert len(pd.read_csv(numeric / "inference_checks.csv")) == 18
        assert len(list(numeric.glob("states_*.npz"))) == 6
        figure_count = 6
    else:
        coordinates = pd.read_csv(numeric / "umap_coordinates.csv")
        assert coordinates.groupby("panel_group").run.nunique().to_dict() == {"reference": 5, "trained": 6}
        for _, frame in coordinates.groupby("run"):
            assert frame.sample_index.tolist() == list(range(len(frame)))
        figure_count = 2
    assert len(list(output.rglob("*.png"))) == figure_count
    assert len(list(output.rglob("*.pdf"))) == figure_count
