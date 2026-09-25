"""Standard-library tests. All GPU/model/decoder operations are mocked."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace, ModuleType
import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = HERE.parent
EVALUATION_HELPERS = REPOSITORY_ROOT / 'evaluation_scripts'
BASELINE_HELPER_SHA256 = {
    'checkpoints.py': 'bdec58d060134bb6f7ccd6390b99e91b8543fce98a70885e9b28c406b96a6492',
    'medgemma_model.py': '6be2a28cbcdba4e3d903e6f713f985c0f950498fddda6256b1b2436479703702',
}
sys.path[:0] = [str(HERE), str(REPOSITORY_ROOT)]
import inference_common as ic
import infer_yolo26 as iy
import infer_medgemma as im


class HelperConsolidationTests(unittest.TestCase):
    def test_entrypoints_use_canonical_evaluation_helpers(self):
        helpers = {
            'checkpoints.py': iy.checkpoints,
            'medgemma_model.py': im.medgemma_model,
        }
        for name, module in helpers.items():
            with self.subTest(name=name):
                canonical = EVALUATION_HELPERS / name
                self.assertEqual(Path(module.__file__).resolve(), canonical.resolve())
                self.assertEqual(Path(module.__file__).read_bytes(), canonical.read_bytes())
                self.assertEqual(ic.sha256(module.__file__), BASELINE_HELPER_SHA256[name])
                self.assertFalse((HERE / name).exists())


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('INFERENCE_TEST_TMP'))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input'
        self.source.mkdir()
        self.image = self.source / 'a.png'
        self.image.write_bytes(b'fake image bytes; no decoder involved')
        self.other = self.source / 'b.jpg'
        self.other.write_bytes(b'other fake image bytes')
        self.output = self.root / 'output'
        self.checkpoint = self.root / 'best.pt'
        self.checkpoint.write_bytes(b'not a real checkpoint; resolver is mocked')

    def report(self):
        return {'status': 'complete', 'split': 'val', 'protocol': {'selected_on': 'val',
                'confidence_cutoff': .5, 'identity': {'model': 'yolo26', 'checkpoint': {'weights_sha256': ic.sha256(self.checkpoint)},
                'score_floor': .001, 'max_detections': 300, 'confidence_comparison': 'score > cutoff',
                'inference': {'collection': {'imgsz': 960}, 'prediction_batch': 8},
                'versions': {'torch': 'fake-version'}, 'source_sha256': {'checkpoints.py': ic.sha256(iy.checkpoints.__file__)}}}}

    def protocol_file(self, report=None):
        p = self.root / 'evaluation.json'
        p.write_text(json.dumps(report or self.report()))
        return p

    def prediction(self, scores=(.8,)):
        return {'pred_xyxy': [[10,20,50,60] for _ in scores], 'scores': list(scores),
                'status': 'valid_nonempty' if scores else 'valid_empty'}


class InputTests(Fixture):
    def test_file_and_directory(self):
        files,root,out=ic.plan_inputs(self.image,False,self.output,'yolo26')
        self.assertEqual(files,[self.image]);self.assertEqual(root,self.source);self.assertFalse(out.exists())
        files,_,_=ic.plan_inputs(self.source,False,self.output,'yolo26')
        self.assertEqual(files,[self.image,self.other])

    def test_recursive_and_extension_filter(self):
        sub=self.source/'nested';sub.mkdir();third=sub/'c.PNG';third.write_bytes(b'c')
        (self.source/'ignored.txt').write_text('not an image')
        self.assertEqual(len(ic.plan_inputs(self.source,False,self.output,'yolo26')[0]),2)
        self.assertEqual(len(ic.plan_inputs(self.source,True,self.output,'yolo26')[0]),3)

    def test_reject_existing_output(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError,'already exists'):ic.plan_inputs(self.source,False,self.output,'yolo26')

    def test_reject_output_inside_source(self):
        with self.assertRaisesRegex(ValueError,'outside the source'):ic.plan_inputs(self.source,False,self.source/'out','yolo26')

    def test_reject_dicom_and_missing_input(self):
        dicom=self.source/'x.dcm';dicom.write_bytes(b'x')
        with self.assertRaisesRegex(ValueError,'DICOM'):ic.plan_inputs(dicom,False,self.output,'yolo26')
        with self.assertRaisesRegex(ValueError,'does not exist'):ic.plan_inputs(self.source/'missing.png',False,self.output,'yolo26')

    def test_reject_empty_folder(self):
        empty=self.root/'empty';empty.mkdir()
        with self.assertRaisesRegex(ValueError,'No supported images'):ic.plan_inputs(empty,False,self.output,'yolo26')

    def test_reject_escaping_symlink(self):
        elsewhere=self.root/'elsewhere.png';elsewhere.write_bytes(b'x')
        (self.source/'link.png').symlink_to(elsewhere)
        with self.assertRaisesRegex(ValueError,'escapes'):ic.plan_inputs(self.source,False,self.output,'yolo26')


class FilteringTests(Fixture):
    def select(self, scores, cutoff=.5):
        return ic.select_detections([[10,20,50,60] for _ in scores],scores,cutoff,100,120)

    def test_strict_cutoff_and_floor(self):
        selected,count,_=self.select([.001,.5,.5001,.8])
        self.assertEqual(count,3);self.assertEqual(len(selected),2)
        self.assertGreater(selected[0]['score'],selected[1]['score'])

    def test_float32_score_and_scalar_comparison(self):
        selected,_,_=self.select([.5+1e-10],.5)
        self.assertEqual(selected,[])
        selected,_,_=self.select([ic.float32(.37)],.37)
        self.assertEqual(selected,[])
        selected,count,_=self.select([ic.float32(.001)],.001)
        self.assertEqual((selected,count),([],0))

    def test_stable_ties(self):
        boxes=[[1,1,10,10],[2,2,11,11],[3,3,12,12]]
        selected,_,_=ic.select_detections(boxes,[.8,.8,.8],.5,100,120)
        self.assertEqual([x['xyxy'][0] for x in selected],[1,2,3])

    def test_cap_and_empty(self):
        selected,count,limited=self.select([.8]*301)
        self.assertEqual((len(selected),count,limited),(300,300,True))
        self.assertEqual(self.select([]),([],0,False))

    def test_reject_bad_scores_and_boxes(self):
        for score in [float('nan'),float('inf'),-.1,1.1,True]:
            with self.subTest(score=score),self.assertRaises(ValueError):self.select([score])
        for box in [[50,20,10,60],[-1,0,30,40],[10,20,101,60],[10,20,float('nan'),60],[1,2,3]]:
            with self.subTest(box=box),self.assertRaises(ValueError):ic.select_detections([box],[.8],.5,100,120)

    def test_invalid_output_distinct_from_empty(self):
        failure={'status':'format_failure','pred_xyxy':[],'scores':[],'generated_text':'bad','truncated':True}
        bad=ic.make_record(self.image,Path('a.png'),failure,.5,100,120,'medgemma')
        empty=ic.make_record(self.image,Path('a.png'),self.prediction([]),.5,100,120,'medgemma')
        self.assertEqual(bad['status'],'invalid_model_output');self.assertEqual(empty['status'],'no_boxes_above_cutoff')
        self.assertEqual(bad['score_type'],'coordinate_token_likelihood_proxy')
        self.assertEqual(bad['generated_text'],'bad');self.assertTrue(bad['truncated'])

    def test_native_clipped_degenerate_boxes_do_not_abort(self):
        selected,count,_=ic.select_detections([[0,20,0,60]],[.002],.5,100,120)
        self.assertEqual((selected,count),([],1))
        selected,count,_=ic.select_detections([[0,20,0,60]],[.8],.5,100,120)
        self.assertTrue(selected[0]['degenerate'])
        self.assertEqual(count,1)

    def test_inconsistent_status_rejected(self):
        p=self.prediction();p['status']='schema_failure'
        with self.assertRaises(ValueError):ic.make_record(self.image,Path('a.png'),p,.5,100,120,'medgemma')


class ProtocolTests(Fixture):
    def test_complete_validation_accepted_and_defaults_restored(self):
        r=ic.read_protocol(self.protocol_file(),'yolo26')
        self.assertEqual(ic.option_from_protocol(None,r,('collection','imgsz'),640),960)
        self.assertEqual(ic.option_from_protocol(None,r,('prediction_batch',),1),8)
        self.assertEqual(ic.option_from_protocol(None,None,('prediction_batch',),8),8)
        with self.assertRaisesRegex(ValueError,'differs'):ic.option_from_protocol(640,r,('collection','imgsz'),640)

    def test_reject_test_smoke_model_and_bad_rules(self):
        for key,val in [('split','test'),('status','smoke')]:
            r=self.report();r[key]=val
            with self.assertRaises(ValueError):ic.read_protocol(self.protocol_file(r),'yolo26')
        with self.assertRaises(ValueError):ic.read_protocol(self.protocol_file(),'medgemma')
        r=self.report();r['protocol']['identity']['confidence_comparison']='score >= cutoff'
        with self.assertRaises(ValueError):ic.read_protocol(self.protocol_file(r),'yolo26')

    def test_reject_bad_cutoffs(self):
        for value in [None,True,-.1,2,float('nan')]:
            r=self.report();r['protocol']['confidence_cutoff']=value
            with self.assertRaises(ValueError):ic.read_protocol(self.protocol_file(r),'yolo26')

    def test_checkpoint_settings_torch_and_helper_hash(self):
        r=self.report();identity=r['protocol']['identity'];p={'weights_sha256':ic.sha256(self.checkpoint)}
        settings=copy.deepcopy(identity['inference']);versions={'torch':'fake-version'};helpers={'checkpoints.py':iy.checkpoints.__file__}
        ic.verify_protocol(r,'yolo26',p,settings,versions,helpers)
        for changed in ['checkpoint','settings','torch','helper']:
            rr=copy.deepcopy(r);pp=dict(p);ss=copy.deepcopy(settings);vv=dict(versions)
            if changed=='checkpoint':pp['weights_sha256']='wrong'
            if changed=='settings':ss['prediction_batch']=1
            if changed=='torch':vv['torch']='wrong'
            if changed=='helper':rr['protocol']['identity']['source_sha256']['checkpoints.py']='wrong'
            with self.subTest(changed=changed),self.assertRaises(ValueError):ic.verify_protocol(rr,'yolo26',pp,ss,vv,helpers)

    def test_local_rejects_hf_options_and_nan_conf(self):
        args=SimpleNamespace(device=0,hf=False,hf_repo=None,hf_revision=None,hf_run=None,hf_filename=None,conf=.5,protocol=None)
        ic.validate_args(args,'medgemma')
        args.hf_run='run'
        with self.assertRaises(ValueError):ic.validate_args(args,'medgemma')
        args.hf_run=None;args.conf=float('nan')
        with self.assertRaises(ValueError):ic.validate_args(args,'medgemma')


class FakeArray:
    ndim=3
    def __init__(self):self.shape=(120,100,3);self.draws=[]
    def copy(self):return FakeArray()


class WriterTests(Fixture):
    def writer(self, annotate=False):
        return ic.InferenceWriter(self.output,'medgemma',self.source,[self.image],.5,{}, {},None,annotate)

    def test_complete_jsonl_no_source_change(self):
        before=self.image.read_bytes()
        with self.writer() as w:r=w.add(self.image,FakeArray(),self.prediction())
        meta=json.loads((self.output/'inference.json').read_text())
        self.assertEqual(meta['status'],'complete');self.assertEqual(meta['processed_images'],1)
        self.assertEqual(meta['predictions_sha256'],ic.sha256(self.output/'predictions.jsonl'))
        self.assertEqual(self.image.read_bytes(),before);self.assertEqual(r['detections'][0]['xyxy'],[10.,20.,50.,60.])
        self.assertFalse(meta['ground_truth_used']);self.assertFalse(meta['metrics_computed'])

    def test_infrastructure_error_marks_failed_and_propagates(self):
        with self.assertRaisesRegex(RuntimeError,'CUDA fixture'):
            with self.writer() as w:
                w.add(self.image,FakeArray(),self.prediction())
                raise RuntimeError('CUDA fixture')
        meta=json.loads((self.output/'inference.json').read_text())
        self.assertEqual(meta['status'],'failed');self.assertEqual(meta['processed_images'],1)

    def test_incomplete_coverage_is_failed(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            with self.writer():pass
        self.assertEqual(json.loads((self.output/'inference.json').read_text())['status'],'failed')

    def test_duplicate_result_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):
            with self.writer() as w:
                w.add(self.image,FakeArray(),self.prediction());w.add(self.image,FakeArray(),self.prediction())

    def test_annotation_path_and_invalid_banner(self):
        captured=[]
        cv=SimpleNamespace(FONT_HERSHEY_SIMPLEX=0,LINE_AA=1,
            rectangle=lambda *a:captured.append(('rectangle',a)),
            putText=lambda *a:captured.append(('text',a)),
            imwrite=lambda p,a:Path(p).write_bytes(b'fake annotation')>0)
        original=FakeArray()
        with patch.dict(sys.modules,{'cv2':cv}),self.writer(True) as w:
            r=w.add(self.image,original,{'status':'format_failure','pred_xyxy':[],'scores':[]})
        self.assertEqual(r['annotated_image'],'annotated/a.png.png')
        self.assertIn('INVALID',captured[0][1][1]);self.assertIsNot(captured[0][1][0],original)
        self.assertTrue((self.output/r['annotated_image']).exists())


class Tensor:
    def __init__(self,values):self.values=values
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.values


class EntrypointTests(Fixture):
    def runtime(self, stack, model, predictions=None, fail_on=None):
        calls=[];closed=[]
        cv=SimpleNamespace(__version__='5.0.0',IMREAD_COLOR=1,COLOR_BGR2RGB=2,
                           imread=lambda p,flag:FakeArray(),imcount=lambda p:1,cvtColor=lambda a,code:a)
        torch=SimpleNamespace(__version__='2.11.0+cu128',cuda=SimpleNamespace(
            is_available=lambda:True,device_count=lambda:1,set_device=lambda d:calls.append(('device',d)),empty_cache=lambda:None))
        stack.enter_context(patch.dict(sys.modules,{'cv2':cv,'torch':torch}))
        provenance={'weights_sha256':ic.sha256(self.checkpoint)}
        if model=='yolo26':
            class Detector:
                names={0:'fracture'}
                predictor=SimpleNamespace(model=SimpleNamespace(fp16=True,end2end=False))
                def __init__(self,weights):calls.append(('load',weights))
                def predict(self,**kwargs):
                    calls.append(('predict',kwargs))
                    return [SimpleNamespace(orig_shape=x.shape[:2],boxes=SimpleNamespace(cls=Tensor([0]),xyxy=Tensor([[10,20,50,60]]),conf=Tensor([.8]))) for x in kwargs['source']]
            stack.enter_context(patch.dict(sys.modules,{'ultralytics':SimpleNamespace(__version__='8.4.152',YOLO=Detector)}))
            stack.enter_context(patch.object(iy.checkpoints,'resolve_yolo',return_value=(self.checkpoint,provenance)))
        else:
            class Pil:
                mode='RGB'
                def __init__(self):self.size=(100,120)
                def close(self):closed.append(True)
            stack.enter_context(patch.dict(sys.modules,{'PIL':SimpleNamespace(Image=SimpleNamespace(fromarray=lambda a:Pil()))}))
            cases=iter(predictions or [self.prediction(),self.prediction([])])
            class Predictor:
                def __init__(self,run,adapter,device,tokens):
                    calls.append(('init',device,tokens));self.settings={'device_index':device,'max_new_tokens':tokens or 768}
                def predict(self,image):
                    calls.append(('predict',image.mode))
                    if fail_on and sum(c[0]=='predict' for c in calls)==fail_on:raise RuntimeError('CUDA fixture failure')
                    return next(cases)
            stack.enter_context(patch.object(im.checkpoints,'resolve_medgemma',return_value=(self.root,self.root,provenance)))
            stack.enter_context(patch.object(im.medgemma_model,'MedGemmaPredictor',Predictor))
        return calls,closed

    def run_args(self,entry,extra=()):
        return [entry,'--checkpoint',str(self.checkpoint),'--source',str(self.source),'--conf','.5','--no-annotate','--output',str(self.output),*extra]

    def test_yolo_mocked_batches_and_collection_settings(self):
        with ExitStack() as stack:
            calls,_=self.runtime(stack,'yolo26');stack.enter_context(patch.object(sys,'argv',self.run_args('infer_yolo26.py',['--batch','1','--imgsz','960'])))
            iy.main()
        predicts=[c[1] for c in calls if c[0]=='predict'];self.assertEqual(len(predicts),2)
        for p in predicts:
            self.assertEqual((p['conf'],p['quantize'],p['rect'],p['nms'],p['imgsz']),(.001,16,False,None,960))
            self.assertIsInstance(p['source'][0],FakeArray)
        self.assertEqual(json.loads((self.output/'inference.json').read_text())['status'],'complete')

    def test_medgemma_mocked_invalid_output_stays_invalid(self):
        invalid={'status':'schema_failure','pred_xyxy':[],'scores':[],'generated_text':'wrong schema','error':'fixture'}
        with ExitStack() as stack:
            calls,closed=self.runtime(stack,'medgemma',[self.prediction(),invalid]);stack.enter_context(patch.object(sys,'argv',self.run_args('infer_medgemma.py')))
            im.main()
        rows=[json.loads(x) for x in (self.output/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows],['boxes_above_cutoff','invalid_model_output'])
        self.assertEqual(len(closed),2);self.assertEqual(rows[1]['generated_text'],'wrong schema')

    def test_medgemma_cuda_failure_aborts_not_empty_prediction(self):
        with ExitStack() as stack:
            calls,closed=self.runtime(stack,'medgemma',fail_on=2);stack.enter_context(patch.object(sys,'argv',self.run_args('infer_medgemma.py')))
            with self.assertRaisesRegex(RuntimeError,'CUDA fixture'):im.main()
        meta=json.loads((self.output/'inference.json').read_text())
        self.assertEqual((meta['status'],meta['processed_images']),('failed',1));self.assertEqual(len(closed),2)

    def test_multiframe_tiff_and_bad_decode_rejected(self):
        cv=SimpleNamespace(IMREAD_COLOR=1,imcount=lambda p:2,imread=lambda p,f:None)
        with patch.dict(sys.modules,{'cv2':cv}):
            with self.assertRaisesRegex(ValueError,'single-frame'):ic.read_bgr('a.tiff')
            with self.assertRaisesRegex(ValueError,'Cannot decode'):ic.read_bgr('a.png')

    def test_cli_help_no_model_loading(self):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory(dir=os.environ.get('INFERENCE_TEST_TMP')) as cwd:
            for name in ['infer_yolo26.py','infer_medgemma.py']:
                r=subprocess.run([sys.executable,'-B',str(HERE/name),'--help'],capture_output=True,text=True,env=env,cwd=cwd)
                self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--checkpoint',r.stdout);self.assertIn('--protocol',r.stdout)

    def test_cli_requires_cutoff_choice_and_rejects_both(self):
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        base=[sys.executable,'-B',str(HERE/'infer_yolo26.py'),'--checkpoint','x.pt','--source','x.png']
        for extra in [[],['--conf','.5','--protocol','val.json']]:
            r=subprocess.run(base+extra,capture_output=True,text=True,env=env)
            self.assertEqual(r.returncode,2)

    def test_sources_compile_without_importing_gpu_libraries(self):
        for source in [HERE/'inference_common.py',HERE/'infer_yolo26.py',HERE/'infer_medgemma.py',Path(__file__)]:
            compile(source.read_text(),str(source),'exec')


class SavedOutputRegressionTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('FRACTURE_RESULTS'), 'Optional saved-results regression fixture not supplied')
    def test_filtering_matches_all_completed_saved_evaluations(self):
        root=Path(os.environ['FRACTURE_RESULTS'])
        checked=0
        for report_path in sorted(root.rglob('evaluation.json')):
            report=json.loads(report_path.read_text())
            if report.get('status')!='complete':continue
            output=report_path.parent
            metrics={r.split(',')[0]:float(r.split(',')[1]) for r in (output/'metrics.csv').read_text().splitlines()[1:]}
            expected=round(metrics['lesion_recall']*report['counts']['fracture_boxes'])+round(metrics['false_positives_per_image']*report['counts']['images'])
            selected=0;images=0
            with (output/'predictions.jsonl').open() as handle:
                for line in handle:
                    r=json.loads(line)
                    detections,_,_=ic.select_detections(r['pred_xyxy'],r['scores'],report['protocol']['confidence_cutoff'],r['width'],r['height'])
                    selected+=len(detections);images+=1
            with self.subTest(run=output.name):
                self.assertEqual(selected,expected)
                self.assertEqual(images,report['counts']['images'])
            checked+=1
        self.assertGreaterEqual(checked,5)

    @unittest.skipUnless(os.environ.get('FRACTURE_RESULTS'), 'Optional saved-results regression fixture not supplied')
    def test_existing_validation_protocol_and_helper_identity(self):
        root=Path(os.environ['FRACTURE_RESULTS']);checked=0
        for report_path in sorted(root.rglob('evaluation.json')):
            report=json.loads(report_path.read_text())
            if report.get('status')!='complete' or report.get('split')!='val':continue
            model=report['model'];r=ic.read_protocol(report_path,model)
            helpers={'checkpoints.py':iy.checkpoints.__file__}
            if model=='medgemma':helpers['medgemma_model.py']=im.medgemma_model.__file__
            ic.verify_protocol(r,model,r['checkpoint_source'],r['protocol']['identity']['inference'],
                               {'torch':r['protocol']['identity']['versions']['torch']},helpers)
            checked+=1
        self.assertGreaterEqual(checked,5)


if __name__=='__main__':
    unittest.main(verbosity=2)
