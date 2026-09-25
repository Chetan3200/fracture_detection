"""Offline regressions for guards that must survive optimized Python."""
import ast
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PREPARE = ROOT / "data_prep" / "prepare_data.py"
MINER = ROOT / "mine_hard_negatives.py"


class ValidationGuardTests(unittest.TestCase):
    def run_python(self, *args, **kwargs):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [sys.executable, "-B", *args],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            **kwargs,
        )

    def test_target_scripts_have_no_assert_nodes(self):
        for script in (PREPARE, MINER):
            tree = ast.parse(script.read_text(), filename=str(script))
            self.assertEqual(
                [node for node in ast.walk(tree) if isinstance(node, ast.Assert)],
                [],
                script.name,
            )

    def test_require_rejects_false_under_optimized_python(self):
        for module_name, script in (("prepare_data", PREPARE), ("miner", MINER)):
            code = (
                "import importlib.util; "
                f"spec = importlib.util.spec_from_file_location({module_name!r}, {str(script)!r}); "
                "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "module.require(False, 'guard survived -O')"
            )
            result = self.run_python("-O", "-c", code)
            self.assertNotEqual(result.returncode, 0, module_name)
            self.assertIn("ValueError: guard survived -O", result.stderr)

    def test_miner_help_is_offline(self):
        result = self.run_python(str(MINER), "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Mine training hard negatives", result.stdout)

    def test_miner_refuses_existing_output_before_ml_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "existing-output"
            output.mkdir()
            result = self.run_python(
                str(MINER), "--data", str(Path(temporary) / "data"), "--output", str(output)
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)
        self.assertIn("existing results are never overwritten", result.stderr)
        self.assertNotIn("No module named", result.stderr)


if __name__ == "__main__":
    unittest.main()
