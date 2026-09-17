"""Standard-experiment notebook controls, using saved fixtures and stubbed learning."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from figure_fixtures import BASE, CONFIG, FROZEN, fixed_experiment
from hnn2.config import experiment_config_from_json
from hnn2.hp.rules import SelectionRule
from hnn2.single import MlpSettings
from hnn2.workflows import read_selection

NOTEBOOK = ROOT / 'notebooks/run_experiment.ipynb'


def run_cell(name, namespace):
    cell = next(c for c in nbformat.read(NOTEBOOK, 4).cells if c.id == name)
    exec(compile(cell.source, f'{NOTEBOOK.name}:{name}', 'exec'), namespace)


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    import hnn2.workflows as workflows
    import hnn2.single as single
    import hnn2.mlp as mlp

    def forbidden(*args, **kwargs):
        pytest.fail('Notebook verification must not train')

    for module, names in (
        (workflows, ('run_adaptation_batch', 'run_representation')),
        (single, ('run_adaptation', 'fit_readout', 'sparse_features')),
        (mlp, ('train_mlp_e2e',)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    return fixed_experiment(tmp_path_factory.mktemp('standard_reference'))


def setup_namespace(tmp_path, **changes):
    ns = {}
    run_cell('imports', ns)
    run_cell('settings', ns)
    curves = pd.read_csv(BASE / 'curves.csv')
    ns.update(
        DATASET=BASE / 'dataset.npz', EXPERIMENT_ROOT=tmp_path / 'new_experiment',
        CONFIG=CONFIG, SWEEP_CONFIG=replace(CONFIG, n_epochs=int(curves.epoch.max()) + 1),
        RULE=SelectionRule(**FROZEN['hp_rule']), MLP=MlpSettings(**FROZEN['mlp']),
        MODELS=('rec', 'ff', 'thresh'), TARGETS=(.35, .55), SEEDS=(0, 1),
        ETAS=tuple(sorted(curves.eta.unique())), DEVICE='cpu',
        display=lambda *args: None, print=lambda *args: None,
    )
    ns.update(changes)
    return ns


def copy_stage(source, destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    planned = pd.read_csv(destination / 'expected_runs.csv')
    planned['path'] = planned.path.map(lambda p: str(destination / Path(p).relative_to(source)))
    planned.to_csv(destination / 'expected_runs.csv', index=False)
    return planned.assign(state='complete', reason='')


@pytest.mark.parametrize('failure', [None, 'hp', 'training', 'baselines'])
def test_full_stage_order_saved_overrides_and_failure_stop(saved, tmp_path, failure):
    ns = setup_namespace(tmp_path, RUN_HP=True, RUN_TRAINING=True, RUN_BASELINES=True)
    ns['CONFIG'] = replace(CONFIG, readout_lr=.0123)
    run_cell('prepare', ns)
    run_cell('record', ns)
    calls = []

    def check(name, kwargs):
        calls.append(name)
        assert kwargs['script_path'] == NOTEBOOK
        assert kwargs['settings_path'] == ns['shared'].SETTINGS_PATH
        assert kwargs['device'] == 'cpu'
        if failure == name:
            raise RuntimeError(f'Injected {name} failure')

    def hp(dataset, specs, config, rule, candidates, destination, **kwargs):
        check('hp', kwargs)
        assert dataset is ns['dataset'] and specs == ns['SWEEP_SPECS']
        assert kwargs['training_config'] is ns['CONFIG']
        shutil.copytree(saved / 'hp/selection', destination)
        return read_selection(destination)[0]

    def training(source, dataset, config, destination, **kwargs):
        check('training', kwargs)
        assert source == ns['HP_SOURCE'] and dataset is ns['dataset']
        assert config.readout_lr == .0123
        assert kwargs['saving'] is ns['SAVING']
        return copy_stage(saved / 'plasticity', destination)

    def baselines(dataset, models, seeds, config, destination, **kwargs):
        check('baselines', kwargs)
        assert set(models) == {'rec', 'ff', 'thresh'} and seeds == [0, 1]
        assert dataset is ns['dataset'] and config.readout_lr == .0123
        assert kwargs['mlp'] == ns['MLP']
        return copy_stage(saved / 'baselines', destination)

    ns.update(run_sweep_selection=hp, run_selected_experiments=training, run_baselines=baselines)
    if failure:
        with pytest.raises(RuntimeError, match=f'Injected {failure} failure'):
            for stage in ('hp', 'training', 'baselines'):
                run_cell(stage, ns)
    else:
        for stage in ('hp', 'training', 'baselines', 'results'):
            run_cell(stage, ns)
        assert len(ns['summary']) == 22 and ns['status'].state.eq('complete').all()
    order = ['hp', 'training', 'baselines']
    assert calls == (order[:order.index(failure) + 1] if failure else order)
    recorded = json.loads((ns['EXECUTION_PATH'] / 'effective_settings.json').read_text(encoding='utf-8'))
    assert recorded['config']['readout_lr'] == .0123
    assert recorded['baseline_representations'] == ['raw', 'mlp_frozen']
    assert (ns['EXECUTION_PATH'] / 'experiment.ipynb').read_bytes() == NOTEBOOK.read_bytes()
    assert (ns['BASELINES_OUTPUT'] / 'expected_runs.csv').is_file()


@pytest.mark.parametrize('stage', ['training', 'baselines'])
def test_partial_run_uses_saved_selection_population_without_hp(saved, tmp_path, stage):
    ns = setup_namespace(
        tmp_path, SELECTION_SOURCE=saved / 'hp/selection', MODELS=('rec',), SEEDS=(99,),
        **{'RUN_TRAINING': stage == 'training', 'RUN_BASELINES': stage == 'baselines'},
    )
    before = hashes(saved)
    calls = []
    run_cell('prepare', ns)
    run_cell('record', ns)

    def forbidden(*args, **kwargs):
        pytest.fail('A disabled stage was called')

    def requested(*args, **kwargs):
        calls.append(stage)
        if stage == 'training':
            assert args[0] == saved / 'hp/selection'
        else:
            assert set(args[1]) == {'rec', 'ff', 'thresh'} and args[2] == [0, 1]
        return pd.DataFrame()

    ns.update(run_sweep_selection=forbidden, run_selected_experiments=forbidden, run_baselines=forbidden)
    ns['run_selected_experiments' if stage == 'training' else 'run_baselines'] = requested
    for cell in ('hp', 'training', 'baselines'):
        run_cell(cell, ns)
    assert calls == [stage] and hashes(saved) == before
    recorded = json.loads((ns['EXECUTION_PATH'] / 'effective_settings.json').read_text(encoding='utf-8'))
    assert recorded['seeds'] == [99] and recorded['active_seeds'] == [0, 1]
    assert recorded['selection']['path'] == str((saved / 'hp/selection').resolve())


def test_execution_record_reuse_and_changed_settings_rejected(tmp_path):
    ns = setup_namespace(tmp_path, RUN_BASELINES=True)
    run_cell('prepare', ns)
    run_cell('record', ns)
    before = hashes(ns['RESULTS'])
    run_cell('record', ns)
    assert hashes(ns['RESULTS']) == before
    ns['CONFIG'] = replace(CONFIG, readout_lr=.009)
    with pytest.raises(ValueError, match='Execution settings changed'):
        run_cell('record', ns)
    assert hashes(ns['RESULTS']) == before


@pytest.mark.parametrize('change,error', [
    ({'RUN_TRAINING': True}, 'complete HP selection'),
    ({'RUN_HP': True, 'SELECTION_SOURCE': Path('elsewhere')}, 'SELECTION_SOURCE=None'),
    ({'EXECUTION_NAME': '../outside'}, 'single folder name'),
    ({'RUN_HP': 1}, 'switches'),
])
def test_invalid_setup_stops_before_creating_results(tmp_path, change, error):
    ns = setup_namespace(tmp_path, **change)
    with pytest.raises(ValueError, match=error):
        run_cell('prepare', ns)
    assert not ns['EXPERIMENT_ROOT'].exists()


def test_missing_conditions_remain_visible(saved, tmp_path):
    destination = tmp_path / 'experiment/results'
    copy_stage(saved / 'plasticity', destination / 'plasticity')
    copy_stage(saved / 'baselines', destination / 'baselines')
    ns = setup_namespace(tmp_path, EXPERIMENT_ROOT=destination.parent)
    run_cell('prepare', ns)
    planned = pd.read_csv(destination / 'plasticity/expected_runs.csv')
    path = Path(planned.path.iloc[0])
    # Simulate an interrupted copy without modifying the reference fixture.
    (path / 'COMPLETE').unlink()
    run_cell('results', ns)
    assert len(ns['status']) == 22 and len(ns['summary']) == 21
    assert ns['status'].state.eq('incomplete').sum() == 1


@pytest.mark.parametrize('existing', [False, True])
def test_fresh_kernel_preview_never_trains_or_writes(saved, tmp_path, existing):
    notebook = nbformat.read(NOTEBOOK, 4)
    for cell in notebook.cells:
        assert cell.source.isascii()
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
    target = saved.parent if existing else tmp_path / 'empty'
    selection_conditions = json.loads((saved / 'hp/selection/conditions.json').read_text(encoding='utf-8'))
    settings = next(c for c in notebook.cells if c.id == 'settings')
    settings.source += f'''
DATASET = Path({str(BASE / 'dataset.npz')!r})
EXPERIMENT_ROOT = Path({str(target)!r})
CONFIG = experiment_config_from_json({json.dumps(asdict(CONFIG))!r})
SWEEP_CONFIG = experiment_config_from_json({json.dumps(selection_conditions['config'])!r})
'''
    notebook.cells.insert(2, nbformat.v4.new_code_cell('''
def forbidden(*args, **kwargs):
    raise AssertionError('A preview must not compute or write')
run_sweep_selection = run_selected_experiments = run_baselines = forbidden
write_result = save_expected_runs = forbidden
'''))
    before = hashes(saved)
    NotebookClient(notebook, timeout=90, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute(
                       env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert hashes(saved) == before
    if not existing:
        assert not target.exists()
