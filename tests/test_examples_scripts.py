import glob
import runpy
import pytest

EXAMPLES_DIR = "examples_scripts"


@pytest.mark.examples
@pytest.mark.parametrize("script", glob.glob(f"{EXAMPLES_DIR}/*.py"))
def test_example_scripts(script):
    """Run each tangled example script to ensure it runs without error."""
    runpy.run_path(script)
