import os
import sys
from pathlib import Path

# Ensure project root is on the Python path for direct test execution
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Create a temporary symlink so llm_integration can load prompt files
symlink_path = Path('/lofn')
if not symlink_path.exists():
    symlink_path.symlink_to(project_root / 'lofn', target_is_directory=True)

import lofn.llm_integration as li


def test_process_facets_creativity_spectrum(monkeypatch):
    captured = {}

    def fake_run_llm_chain(chains, name, args, max_retries, model, debug, expected_schema=None):
        captured['args'] = args
        return {'facets': []}

    monkeypatch.setattr(li, 'run_llm_chain', fake_run_llm_chain)

    cs = {'transformative': 10, 'inventive': 20, 'literal': 70}
    li.process_facets({}, 'text', 'concept', 'medium', 1, creativity_spectrum=cs)

    assert captured['args']['creativity_spectrum_transformative'] == 10
    assert captured['args']['creativity_spectrum_inventive'] == 20
    assert captured['args']['creativity_spectrum_literal'] == 70
