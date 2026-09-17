"""Test the actual saved-figure cells; archived code is only an expected-value oracle."""
import importlib.util
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
STEMS = ('01_selection', '02_adaptation', '03_distributions', '04_classifier', '05_geometry')
TABLE_CELLS = {
    '01_selection': (7, 11, 19, 20, 21, 22, 23, 31),
    '02_adaptation': (9,),
    '03_distributions': (7, 11, 23, 28, 29, 30),
    '04_classifier': (7, 11, 19, 27, 28, 39, 40),
    '05_geometry': (7, 11, 12, 20),
}
SCORE_ARMS = (
    'rec_initial', 'rec_trained', 'ff_initial', 'ff_trained',
    'thresh_initial', 'thresh_trained',
)
SCORE_TABLES = (
    'activity_metrics', 'activity_histograms', 'activity_statistics', 'entropy',
    'overall_entropy', 'overall_entropy_statistics',
    'selected_entropy_scores', 'selected_entropy_statistics',
)


def frozen_reference():
    path = ROOT / 'history/figures_v1/snapshot/src/hnn2/figure_tables.py'
    spec = importlib.util.spec_from_file_location('hnn2._frozen_tables', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cells(stem):
    return {c.id: c for c in nbformat.read(ROOT/'notebooks'/f'{stem}.ipynb', 4).cells}


def run_cell(stem, number, namespace):
    cell = cells(stem)[f'{stem}-{number:02d}']
    exec(compile(cell.source, f'{stem}:{cell.id}', 'exec'), namespace)


def table_results(runs, expected, selection_directory, targets=None):
    result = {}
    for stem in STEMS:
        ns = {}
        run_cell(stem, 1, ns)
        ns.update(runs=runs, expected=expected, selection_directory=selection_directory,
                  DISPLAY_TARGETS=targets, RESULTS=selection_directory.parent.parent,
                  OUTPUT=selection_directory.parent, display=lambda *a: None, print=lambda *a: None)
        run_cell(stem, 3, ns)
        ns['require_complete'](expected)
        # Inject the in-memory fixture before the unchanged input/target checks.
        source = cells(stem)[f'{stem}-05'].source
        exec(source[source.index('parts = []'):], ns)
        for number in TABLE_CELLS[stem]:
            run_cell(stem, number, ns)
        result.update({name: ns['tables'][name] for name in ns['OWNED_TABLES']})
    return result


def draw_figures(tables, runs, targets):
    result = {}
    for stem in STEMS:
        ns = {}
        run_cell(stem, 1, ns)
        models = [m for m in ('rec', 'ff', 'thresh') if m in targets]
        arms = [f'{m}_{stage}' for m in models for stage in ('initial', 'trained')]
        if stem != '03_distributions':
            arms += ['raw', 'mlp_frozen']
        selected = ns['selected_runs'](runs, targets)
        ns.update(tables=tables, runs=runs, targets=targets, models=models, arms=arms,
                  chosen=tables['selected_summary'], summary=tables['summary'],
                  arm_labels=[ns['_arm_label'](a, targets) for a in arms],
                  selected=selected, chosen_paths=set(selected), figures={}, SHOW_FIGURES=False)
        for cell in cells(stem).values():
            if any(tag.startswith(('style-', 'plot-')) for tag in cell.metadata.get('tags', [])):
                exec(compile(cell.source, f'{stem}:{cell.id}', 'exec'), ns)
        result.update(ns['figures'])
    return result


def threshold_results(runs, expected, targets):
    ns = {}
    run_cell('04_classifier', 1, ns)
    ns.update(runs=runs, expected=expected, targets=targets, tables={})
    for number in (39, 40):
        run_cell('04_classifier', number, ns)
    return ns['tables']['threshold_detail'], ns['tables']['threshold_summary']


def paired_results(summary):
    ns = {}
    run_cell('04_classifier', 1, ns)
    ns.update(tables={'summary': summary}, summary=summary)
    run_cell('04_classifier', 19, ns)
    return ns['tables']['paired_accuracy'], ns['tables']['accuracy_differences']


def statistics(frame, groups, columns):
    ns = {}
    run_cell('04_classifier', 1, ns)
    run_cell('04_classifier', 7, ns)
    return ns['describe_values'](frame, groups, columns)
