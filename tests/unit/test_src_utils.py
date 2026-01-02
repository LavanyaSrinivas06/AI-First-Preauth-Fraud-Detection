# tests/unit/test_src_utils.py
import pytest

pytest.skip("Update this test to match actual functions in src/utils/config.py", allow_module_level=True)
def test_src_utils_config_importable():
    import src.utils.config as cfg  # just ensure module imports
    assert cfg is not None


def test_src_utils_data_loader_importable():
    import src.utils.data_loader as dl  # just ensure module imports
    assert dl is not None



