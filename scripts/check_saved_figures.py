"""Check current notebooks against saved numbers, without training or extra inference."""

from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'notebooks')]
import figure_settings as settings


def main():
    # Fingerprint the reference tree before executing notebooks in a temporary output.
    reference = settings.OUTPUT.resolve()
    protected = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in reference.rglob('*') if p.is_file()
    }
    temporary_root = (ROOT / 'tmp').resolve()
    temporary_root.mkdir(exist_ok=True)
    records = []
    score_arms = {
        f'{model}_{stage}' for model in ('rec', 'ff', 'thresh')
        for stage in ('initial', 'trained')
    }

    with tempfile.TemporaryDirectory(prefix='saved-figures-', dir=temporary_root) as temporary:
        destination = Path(temporary).resolve()
        # TemporaryDirectory removes only the freshly created child on exit.
        assert destination.is_relative_to(temporary_root) and destination != temporary_root
        for path in sorted((ROOT / 'notebooks').glob('0[1-5]_*.ipynb')):
            nb = nbformat.read(path, 4)
            owner = path.stem.split('_', 1)[1]

            # Redirect exports and make accidental training or inference fail immediately.
            for cell in nb.cells:
                if 'parameters' in cell.metadata.get('tags', []):
                    cell.source += f'''
OUTPUT = Path({str(destination)!r})
ADDITIONAL_PATH = Path({str(reference/'additional'/f'{owner}_v1')!r})
SAVE_NUMBERS = True
SAVE_PNG, SAVE_PDF = True, False
SHOW_FIGURES = False
EXPORT_NAME = "check"
'''
            nb.cells.insert(2, nbformat.v4.new_code_cell('''
import hnn2.single as single
import hnn2.adapt as adapt
import hnn2.model.plasticity as plasticity
import hnn2.saved_inference as inference
import hnn2.simulate as simulate
import hnn2.umap_embed as umap
def forbidden(*args, **kwargs):
    raise AssertionError("This check must only read saved numbers")
single.run_adaptation = single.fit_readout = single.sparse_features = forbidden
adapt.update = plasticity.update = simulate.encode = forbidden
inference.training_order_activity = inference.fixed_point_detail = umap.embed = forbidden
'''))

            print('Checking', path.name, flush=True)
            NotebookClient(
                nb, timeout=240, kernel_name='python3',
                resources={'metadata': {'path': str(ROOT)}},
            ).execute(env={**os.environ, 'MPLBACKEND': 'Agg'})

            # Preserve saved values; distribution comparisons now use six sparse-model arms.
            generated = destination / owner / 'check'
            original = reference / owner / 'all_v1'
            numeric = generated / 'tables/figures'
            prior = original / 'tables/figures'
            assert {p.name for p in numeric.glob('*.csv')} == {p.name for p in prior.glob('*.csv')}
            rows = 0
            for csv in numeric.glob('*.csv'):
                actual = pd.read_csv(csv, float_precision='round_trip')
                expected = pd.read_csv(prior / csv.name, float_precision='round_trip')
                if owner == 'distributions' and 'arm' in expected.columns:
                    expected = expected[expected.arm.isin(score_arms)].reset_index(drop=True)
                    assert set(actual.arm) == score_arms, csv.name
                pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=csv.name)
                rows += len(actual)
            metadata = json.loads((numeric / 'conditions.json').read_text(encoding='utf-8'))
            old = json.loads((prior / 'conditions.json').read_text(encoding='utf-8'))
            for key in (
                'source', 'selection', 'population_sources', 'display_targets', 'analysis_targets'
            ):
                assert metadata[key] == old[key], key

            # Compare image coverage; numeric equality is checked through the tables above.
            images = list(generated.rglob('*.png'))
            assert {p.relative_to(generated) for p in images} == {
                p.relative_to(original) for p in original.rglob('*.png')
            }
            assert not list(generated.rglob('*.pdf'))
            records.append({
                'notebook': path.name, 'tables': len(list(numeric.glob('*.csv'))),
                'rows': rows, 'figures': len(images),
            })

    # Verify reference files after the temporary exports have been removed.
    assert all(
        p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
        for p, digest in protected.items()
    )
    report = {
        'status': 'passed', 'notebooks': records, 'reference': str(reference),
        'tables': sum(r['tables'] for r in records), 'rows': sum(r['rows'] for r in records),
        'figures': sum(r['figures'] for r in records), 'unchanged_current_files': len(protected),
        'training': False, 'inference': False, 'umap_fit': False, 'html': False,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
