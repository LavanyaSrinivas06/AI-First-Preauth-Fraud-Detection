import os
import sys


def test_requirements_exists():
    assert os.path.exists('requirements.txt'), 'requirements.txt is required for reproducibility'


def test_reproduce_script_exists():
    assert os.path.exists('scripts/reproduce.py'), 'scripts/reproduce.py should exist'


def test_progress_file_exists():
    # check that a progress file was created by our earlier steps
    p1 = '.github/appmod/code-migration/20260121194111/progress.md'
    p2 = '.github/appmod/code-migration/20260121193913/progress.md'
    assert os.path.exists(p1) or os.path.exists(p2), 'Expected at least one progress.md file in .github/appmod/code-migration'
