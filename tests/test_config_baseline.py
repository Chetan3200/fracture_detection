"""Offline first-pass regressions. GPU, YAML decoder and Hub operations are mocked.

Run from the repo root:
  python -B -m unittest discover -s tests -p test_config_baseline.py -v
No dataset download, training, GPU calls, package installation or HF writes occur.
"""
from collections import defaultdict
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import train_baseline as baseline


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("FRACTURE_TEST_TMP"))
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent_environment = dict(os.environ)
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def args(self, *extra):
        return ["--project-root", str(self.root), *extra]

    def dataset(self):
        data = self.root / "data/grazpedwri_yolo"
        data.mkdir(parents=True)
        for split in ("train", "val"):
            (data / "images" / split).mkdir(parents=True)
            (data / "labels" / split).mkdir(parents=True)
        manifest = b"offline fixture,not a real clinical dataset\n"
        self.fixture_sha = hashlib.sha256(manifest).hexdigest()
        (self.root / "split_manifest.csv").write_bytes(manifest)
        (data / "split_manifest.csv").write_bytes(manifest)
        (data / "dataset_info.json").write_text(json.dumps({"manifest_sha256": self.fixture_sha}))
        (data / "data.yaml").write_text(json.dumps({"path": "/old/server/data", "train": "images/train",
                                                  "val": "images/val", "test": "images/test", "names": ["fracture"]}))
        (self.root / "yolo26s.pt").write_bytes(b"fixture COCO checkpoint; never loaded")
        self.yaml = SimpleNamespace(safe_load=json.loads, safe_dump=lambda d, **kw: json.dumps(d))
        return data

    def metadata_context(self, stack):
        stack.enter_context(patch.object(config, "MANIFEST_SHA256", self.fixture_sha))
        stack.enter_context(patch.dict(sys.modules, {"yaml": self.yaml}))


class ConfigTests(Fixture):
    def test_defaults_are_root_relative_without_creating_directories(self):
        cfg = config.load_config(self.root)
        self.assertEqual(cfg.yolo_data_dir, self.root / "data/grazpedwri_yolo")
        self.assertEqual(cfg.runs_dir, self.root / "runs")
        self.assertEqual(cfg.manifest_path, self.root / "split_manifest.csv")
        self.assertEqual(cfg.candidates_path, self.root / "hard_negatives_review/ranked_negatives.csv")
        self.assertFalse(cfg.data_dir.exists())
        self.assertFalse(cfg.runs_dir.exists())
        self.assertIsNone(cfg.hf_home)

    def test_env_quotes_comments_export_and_paths(self):
        (self.root / ".env").write_text('# comment\nexport DATA_DIR="datasets with spaces" # note\nRUNS_DIR=outputs\nHF_REPO_ID=owner/repo\nEMPTY=\n')
        cfg = config.load_config(self.root)
        self.assertEqual(cfg.data_dir, self.root / "datasets with spaces")
        self.assertEqual(cfg.runs_dir, self.root / "outputs")
        self.assertEqual(cfg.hf_repo_id, "owner/repo")
        self.assertNotIn("HF_REPO_ID", os.environ)

    def test_process_environment_overrides_dotenv(self):
        (self.root / ".env").write_text('DATA_DIR=dotenv-data\nHF_REPO_ID=owner/dotenv\n')
        os.environ.update(DATA_DIR="process-data", HF_REPO_ID="owner/process")
        cfg = config.load_config(self.root)
        self.assertEqual(cfg.data_dir, self.root / "process-data")
        self.assertEqual(cfg.hf_repo_id, "owner/process")

    def test_activation_preserves_environment_and_resolves_sdk_paths(self):
        (self.root / ".env").write_text('HF_HOME="cache/hf home"\nKAGGLEHUB_CACHE=cache/kaggle\nHF_REPO_ID=owner/dotenv\n')
        os.environ["HF_REPO_ID"] = "owner/process"
        cfg = config.load_config(self.root)
        cfg.activate_environment()
        self.assertEqual(os.environ["HF_REPO_ID"], "owner/process")
        self.assertEqual(os.environ["HF_HOME"], str(self.root / "cache/hf home"))
        self.assertEqual(os.environ["KAGGLEHUB_CACHE"], str(self.root / "cache/kaggle"))

    def test_no_default_hf_home_changes_existing_login_location(self):
        cfg = config.load_config(self.root)
        cfg.activate_environment()
        self.assertNotIn("HF_HOME", os.environ)

    def test_blank_optional_sdk_paths_do_not_replace_defaults(self):
        (self.root / '.env').write_text('HF_HOME=\nKAGGLEHUB_CACHE=\n')
        cfg = config.load_config(self.root)
        cfg.activate_environment()
        self.assertNotIn('HF_HOME', os.environ)
        self.assertNotIn('KAGGLEHUB_CACHE', os.environ)

    def test_default_project_root_does_not_follow_cwd(self):
        before = Path.cwd()
        try:
            os.chdir(self.root)
            self.assertEqual(config.load_config().project_root, ROOT)
        finally:
            os.chdir(before)

    def test_explicit_missing_env_file_rejected(self):
        with self.assertRaises(FileNotFoundError):
            config.load_config(self.root, "missing.env")

    def test_env_syntax_errors_do_not_echo_values(self):
        secret = "not-a-real-secret-do-not-print"
        for content in [f'TOKEN="{secret}', f'TOKEN={secret} unexpected', f'bad-key={secret}']:
            (self.root / ".env").write_text(content)
            with self.assertRaises(ValueError) as error:
                config.load_config(self.root)
            self.assertNotIn(secret, str(error.exception))

    def test_public_configuration_excludes_credentials(self):
        secret = "not-a-real-secret-do-not-print"
        (self.root / ".env").write_text(f'HF_TOKEN={secret}\n')
        cfg = config.load_config(self.root)
        self.assertNotIn(secret, repr(cfg))
        self.assertNotIn(secret, json.dumps(cfg.public_dict()))
        with redirect_stdout(io.StringIO()) as out:
            baseline.main(self.args("--print-config"))
        self.assertNotIn(secret, out.getvalue())
        self.assertNotIn("HF_TOKEN", os.environ)

    def test_hash_contract_and_matched_batches(self):
        self.assertEqual(config.MANIFEST_SHA256, "1719f37f442512c3c4fcab8350bedc0f27ff2dcebc082761299912b897747034")
        self.assertEqual(config.CANDIDATE_CSV_SHA256, "5af4772bf275a4e84a815670d2988fb0408119ea05e0b2ffe0cb3251fdf1e66d")
        self.assertEqual(config.HARD_NEGATIVE_POOL_SHA256, "2db84c52f39debc56ad6d6bdae9428ef8d6f2589e7e208781bcaa5f6ddf1a719")
        self.assertEqual(config.YOLO_BATCH_SIZES, {640: 35, 960: 14})


class BaselineArgumentsTests(Fixture):
    def test_direct100_defaults_and_resolution_batches(self):
        for size, batch in [(640, 35), (960, 14)]:
            args, cfg = baseline.resolve_arguments(self.args("--imgsz", str(size), "--print-config"))
            self.assertEqual((args.epochs, args.seed, args.batch, args.device, args.backup_every), (100, 43, batch, 0, 10))
            self.assertEqual(args.dataset, self.root / "data/grazpedwri_yolo")

    def test_cli_overrides_environment_and_dotenv(self):
        (self.root / ".env").write_text('HF_REPO_ID=owner/dotenv\nRUNS_DIR=env-runs\n')
        os.environ["HF_REPO_ID"] = "owner/process"
        args, cfg = baseline.resolve_arguments(self.args("--repo-id", "owner/cli", "--dataset", "other-data",
                    "--manifest", "manifest.csv", "--output-root", "cli-runs", "--batch", "12", "--seed", "99"))
        self.assertEqual(args.repo_id, "owner/cli")
        self.assertEqual(args.dataset, self.root / "other-data")
        self.assertEqual(args.manifest, self.root / "manifest.csv")
        self.assertEqual(args.output_root, self.root / "cli-runs")
        self.assertEqual((args.batch, args.seed), (12, 99))

    def test_outputs_cannot_be_created_inside_dataset(self):
        with self.assertRaisesRegex(ValueError, 'outside the prepared dataset'):
            baseline.resolve_arguments(self.args('--print-config', '--output-root', 'data/grazpedwri_yolo/generated'))

    def test_bad_numbers_rejected(self):
        for flags in [("--epochs", "0"), ("--batch", "0"), ("--seed", "-1"), ("--device", "-1"), ("--backup-every", "0")]:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                baseline.resolve_arguments(self.args("--print-config", *flags))

    def test_backup_boundary_and_destination_required(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            baseline.resolve_arguments(self.args("--epochs", "99", "--repo-id", "owner/repo"))
        with self.assertRaisesRegex(ValueError, "HF_REPO_ID"):
            baseline.resolve_arguments(self.args())
        baseline.resolve_arguments(self.args("--no-hf-backup", "--epochs", "99"))
        with self.assertRaisesRegex(ValueError, "owner/repository"):
            baseline.resolve_arguments(self.args("--repo-id", "https://huggingface.co/owner/repo"))

    def test_training_recipe_is_explicit_and_validation_only(self):
        args, _ = baseline.resolve_arguments(self.args("--print-config"))
        s = baseline.training_settings(args)
        self.assertEqual((s['optimizer'], s['lr0'], s['lrf'], s['momentum'], s['weight_decay'], s['nbs']),
                         ('SGD', .01, .01, .937, .0005, 64))
        self.assertEqual((s['patience'], s['split'], s['cos_lr'], s['close_mosaic'], s['rect']), (0, 'val', False, 10, False))
        self.assertTrue(s['amp']); self.assertTrue(s['deterministic']); self.assertIsNone(s['nms'])
        self.assertNotIn('resume', s)

    def test_cli_help_and_print_work_without_gpu_dependencies(self):
        for flags in [["--help"], self.args("--imgsz", "960", "--print-config")]:
            result = subprocess.run([sys.executable, "-B", str(ROOT / 'train_baseline.py'), *flags],
                                    cwd=self.root, capture_output=True, text=True, env=self.parent_environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            if "--help" not in flags:
                value = json.loads(result.stdout)
                self.assertEqual(value['training']['batch'], 14)
                self.assertFalse((self.root / 'runs').exists())


class DatasetTests(Fixture):
    def test_rebase_is_run_local_and_reads_no_test_directory(self):
        data = self.dataset(); before = (data / 'data.yaml').read_bytes()
        with ExitStack() as stack:
            self.metadata_context(stack)
            args, _ = baseline.resolve_arguments(self.args('--no-hf-backup'))
            info, yaml = baseline.validate_dataset(args)
        self.assertEqual(yaml['path'], str(data))
        self.assertEqual((data / 'data.yaml').read_bytes(), before)
        self.assertFalse((data / 'images/test').exists())

    def test_manifest_mismatch_fails(self):
        self.dataset(); (self.root / 'split_manifest.csv').write_bytes(b'wrong')
        with ExitStack() as stack:
            self.metadata_context(stack)
            args, _ = baseline.resolve_arguments(self.args('--no-hf-backup'))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                baseline.validate_dataset(args)

    def test_wrong_split_and_missing_directories_fail(self):
        data = self.dataset()
        with ExitStack() as stack:
            self.metadata_context(stack)
            args, _ = baseline.resolve_arguments(self.args('--no-hf-backup'))
            body = json.loads((data / 'data.yaml').read_text()); body['train'] = 'images/test'
            (data / 'data.yaml').write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, 'Unexpected train'):
                baseline.validate_dataset(args)
            body['train'] = 'images/train'; (data / 'data.yaml').write_text(json.dumps(body))
            (data / 'labels/val').rmdir()
            with self.assertRaisesRegex(ValueError, 'directory missing'):
                baseline.validate_dataset(args)


class MockedTrainingTests(Fixture):
    def runtime(self, stack, *, public=False, fail_uploads=False, batch_change=False, skip_final=False, early_end=False):
        data = self.dataset(); self.metadata_context(stack)
        state = SimpleNamespace(calls=[], commits=[], private=not public, data=data)
        torch = SimpleNamespace(__version__=config.EXPECTED_TORCH, version=SimpleNamespace(cuda='12.8'),
            cuda=SimpleNamespace(is_available=lambda:True, device_count=lambda:4, set_device=lambda d:None,
                                 synchronize=lambda d:None, get_device_name=lambda d:'Mock GPU'),
            ones=lambda *a, **k:SimpleNamespace(add_=lambda n:None))
        class Hub:
            def whoami(self): state.calls.append('whoami'); return {'name':'fixture'}
            def create_repo(self, **kw): state.calls.append(('create_repo', kw))
            def repo_info(self, **kw): return SimpleNamespace(private=state.private)
            def create_commit(self, **kw):
                state.commits.append(kw)
                if fail_uploads: raise RuntimeError('fixture upload failure')
                return SimpleNamespace(commit_url='https://example.invalid/fixture-commit')
        class Detector:
            def __init__(self, weights):
                state.calls.append(('load_model', weights)); self.ckpt_path=self_root/'yolo26s.pt'; self.callbacks=defaultdict(list)
            def add_callback(self, name, fn): self.callbacks[name].append(fn)
            def train(self, **kw):
                state.calls.append(('train', kw))
                run=Path(kw['project'])/kw['name']; (run/'weights').mkdir()
                (run/'args.yaml').write_text(json.dumps(kw))
                self.trainer=SimpleNamespace(save_dir=run, start_epoch=0, epochs=kw['epochs'], batch_size=kw['batch'], epoch=0)
                if batch_change: self.trainer.batch_size-=1
                for fn in self.callbacks['on_train_start']: fn(self.trainer)
                for epoch in range(1 if early_end else kw['epochs']):
                    self.trainer.epoch=epoch
                    for fn in self.callbacks['on_train_epoch_start']: fn(self.trainer)
                    (run/'weights/last.pt').write_bytes(f'fixture last {epoch}'.encode())
                    (run/'weights/best.pt').write_bytes(f'fixture best {epoch}'.encode())
                    (run/'results.csv').write_text('epoch,fitness\n'+'\n'.join(f'{e},0' for e in range(1,epoch+2)))
                    if not (skip_final and epoch==kw['epochs']-1):
                        for fn in self.callbacks['on_model_save']: fn(self.trainer)
        self_root=self.root
        stack.enter_context(patch.dict(sys.modules, {'torch':torch, 'ultralytics':SimpleNamespace(__version__=config.EXPECTED_ULTRALYTICS, YOLO=Detector),
            'huggingface_hub':SimpleNamespace(__version__='0.36.2', HfApi=Hub, CommitOperationAdd=lambda **kw:SimpleNamespace(**kw))}))
        stack.enter_context(patch.object(baseline.time, 'sleep', return_value=None))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return state

    def test_local_only_one_direct100_call_and_no_hf_calls(self):
        with ExitStack() as stack:
            state=self.runtime(stack)
            baseline.main(self.args('--imgsz','960','--no-hf-backup'))
        calls=[c[1] for c in state.calls if isinstance(c,tuple) and c[0]=='train']
        self.assertEqual(len(calls),1)
        kw=calls[0]; self.assertEqual((kw['epochs'],kw['batch'],kw['imgsz'],kw['seed']),(100,14,960,43))
        self.assertNotIn('resume',kw); self.assertEqual(state.commits,[]); self.assertNotIn('whoami',state.calls)
        run=Path(kw['project'])/kw['name']
        saved=json.loads((run/'resolved_config.json').read_text())
        self.assertFalse(saved['hf_backup']['enabled'])
        self.assertEqual(json.loads((run/'data.yaml').read_text())['path'],str(state.data))
        self.assertEqual(json.loads((state.data/'data.yaml').read_text())['path'],'/old/server/data')
        self.assertEqual(set(saved['source_sha256']),{'train_baseline.py','config.py','data_prep/prepare_data.py'})
        for name,h in saved['source_sha256'].items(): self.assertEqual(config.sha256(run/'source'/name),h)
        self.assertFalse((run/'source/.env').exists())

    def test_hf_initialization_and_every_ten_epochs(self):
        with ExitStack() as stack:
            state=self.runtime(stack)
            baseline.main(self.args('--repo-id','owner/repo'))
        self.assertEqual(len(state.commits),11)
        epochs=[]
        for commit in state.commits[1:]:
            ops=commit['operations']; remote=[op.path_in_repo for op in ops]
            info=next(op for op in ops if op.path_in_repo.endswith('/checkpoint_info.json'))
            body=json.loads(Path(info.path_or_fileobj).read_text()); epochs.append(body['completed_epoch'])
            self.assertTrue(any(p.endswith('/best.pt') for p in remote)); self.assertTrue(any(p.endswith('/last.pt') for p in remote))
            self.assertTrue(any(p.endswith('/resolved_config.json') for p in remote))
            self.assertTrue(any(p.endswith('/source/config.py') for p in remote))
            self.assertFalse(any('/.env' in p for p in remote))
            self.assertEqual(body['total_epochs'],100)
        self.assertEqual(epochs,list(range(10,101,10)))
        created=next(c[1] for c in state.calls if isinstance(c,tuple) and c[0]=='create_repo')
        self.assertTrue(created['private'])

    def test_public_repository_rejected_before_commits_and_training(self):
        with ExitStack() as stack:
            state=self.runtime(stack,public=True)
            with self.assertRaisesRegex(ValueError,'public'):
                baseline.main(self.args('--repo-id','owner/repo'))
        self.assertEqual(state.commits,[])
        self.assertFalse(any(isinstance(c,tuple) and c[0]=='train' for c in state.calls))

    def test_upload_failure_stops_before_training_and_rebuilds_operations(self):
        before=Path.cwd()
        with ExitStack() as stack:
            state=self.runtime(stack,fail_uploads=True)
            with self.assertRaisesRegex(RuntimeError,'HF backup failed'):
                baseline.main(self.args('--repo-id','owner/repo'))
        self.assertEqual(len(state.commits),3)
        self.assertIsNot(state.commits[0]['operations'][0],state.commits[1]['operations'][0])
        self.assertFalse(any(isinstance(c,tuple) and c[0]=='train' for c in state.calls))
        self.assertEqual(Path.cwd(),before)
        self.assertEqual(len(list((self.root/'runs').glob('*/resolved_config.json'))),1)

    def test_silent_batch_change_is_rejected(self):
        with ExitStack() as stack:
            self.runtime(stack,batch_change=True)
            with self.assertRaisesRegex(ValueError,'batch changed'):
                baseline.main(self.args('--no-hf-backup'))

    def test_missing_final_hf_backup_is_rejected(self):
        with ExitStack() as stack:
            self.runtime(stack,skip_final=True)
            with self.assertRaisesRegex(ValueError,'Final HF backup'):
                baseline.main(self.args('--repo-id','owner/repo'))

    def test_early_local_only_return_is_not_reported_as_complete(self):
        with ExitStack() as stack:
            self.runtime(stack, early_end=True)
            with self.assertRaisesRegex(ValueError, 'before all requested epochs'):
                baseline.main(self.args('--no-hf-backup'))

    def test_all_new_python_sources_compile(self):
        for p in [ROOT/'config.py',ROOT/'train_baseline.py',Path(__file__)]:
            compile(p.read_text(),str(p),'exec')


if __name__=='__main__':
    unittest.main(verbosity=2)
