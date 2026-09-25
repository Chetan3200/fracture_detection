"""Refactor-only checks: source parity and complete helper recovery. CPU/offline."""
from pathlib import Path
import ast
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_medgemma as app
from test_medgemma_hf_backup import fixture, API, Add


def node_text(node):
    return ast.dump(node, include_attributes=False)


class Tests(unittest.TestCase):
    def test_training_and_generation_constructor_calls_unchanged(self):
        original = ast.parse((ROOT / 'archive/train_medgemma_original.py').read_text())
        current = ast.parse((ROOT / 'train_medgemma.py').read_text())
        names = {'BitsAndBytesConfig', 'AutoModelForImageTextToText.from_pretrained',
                 'prepare_model_for_kbit_training', 'get_peft_model', 'LoraConfig',
                 'AutoProcessor.from_pretrained', 'Gemma3ImageProcessor.from_pretrained',
                 'TrainingArguments', 'Trainer', 'trainer.train', 'trainer.save_model',
                 'inference_model.generate', 'processor', 'processor.tokenizer',
                 'processor.apply_chat_template', 'masked_labels', 'set_seed'}
        def calls(tree):
            return sorted(node_text(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                          and ast.unparse(n.func) in names)
        self.assertEqual(calls(original), calls(current))

    def test_collator_and_gpu_callbacks_unchanged(self):
        before = ast.parse((ROOT / 'archive/train_medgemma_original.py').read_text())
        after = ast.parse((ROOT / 'train_medgemma.py').read_text())
        for name in ('RecordDataset', 'image_for', 'collate', 'CheckAndLog',
                     'generation_check', 'primary_call', 'barrier'):
            a = next(n for n in ast.walk(before) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            b = next(n for n in ast.walk(after) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            self.assertEqual(node_text(a), node_text(b), name)

    def test_new_archive_includes_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, checkpoint = fixture(tmp)
            helper = b'# mocked helper\n'
            (output / 'medgemma_support.py').write_bytes(helper)
            cfg_path = output / 'run_config.json'
            cfg = json.loads(cfg_path.read_text())
            cfg['helper_sha256'] = hashlib.sha256(helper).hexdigest()
            cfg_path.write_text(json.dumps(cfg))
            api = API()
            receipt = app.HFCheckpointBackup(api, Add, 'test/backup', output, 2).upload(checkpoint)
            with zipfile.ZipFile(io.BytesIO(api.blobs[receipt['archive_path']])) as archive:
                self.assertEqual(archive.read('medgemma_support.py'), helper)

    def test_missing_or_changed_helper_blocks_new_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, checkpoint = fixture(tmp)
            cfg_path = output / 'run_config.json'
            cfg = json.loads(cfg_path.read_text())
            cfg['helper_sha256'] = hashlib.sha256(b'expected').hexdigest()
            cfg_path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'helper'):
                app.backup_payload(output, checkpoint, 2)
            (output / 'medgemma_support.py').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'helper'):
                app.backup_payload(output, checkpoint, 2)

    def test_legacy_resume_helper_is_also_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, checkpoint = fixture(tmp)
            (output / 'medgemma_support.py').write_text('# legacy resume helper')
            files, _, _ = app.backup_payload(output, checkpoint, 2)
            self.assertIn(output / 'medgemma_support.py', files)

    def test_training_signature_unchanged(self):
        original = ast.parse((ROOT / 'archive/train_medgemma_original.py').read_text())
        current = ast.parse((ROOT / 'train_medgemma.py').read_text())
        def signature(tree):
            return next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'signature' for target in node.targets))
        self.assertEqual(node_text(signature(original)), node_text(signature(current)))

    def test_smoke_gate_covers_trainer_helper_and_shared_config(self):
        import run_medgemma as launcher
        self.assertEqual(launcher.current_hashes(), {
            'script_sha256': app.file_digest(ROOT / 'train_medgemma.py'),
            'helper_sha256': app.file_digest(ROOT / 'medgemma_support.py'),
            'config_sha256': app.file_digest(ROOT.parent / 'config.py'),
        })


if __name__ == '__main__':
    unittest.main(verbosity=2)
