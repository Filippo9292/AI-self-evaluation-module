import tempfile
import unittest
from pathlib import Path
import yaml
from ai_self_evaluation.config import Config,PRESETS,load_yaml
from ai_self_evaluation.policy import replay
from ai_self_evaluation.demo import run_demo
from ai_self_evaluation.engine import Engine
from ai_self_evaluation.monitor import TrainingMonitor
from ai_self_evaluation.providers import payload,validate_response
from ai_self_evaluation.util import read_json

class PolicyTests(unittest.TestCase):
    def test_elapsed_epochs_hysteresis_and_reset(self):
        rule=load_yaml(PRESETS/'policy.yaml');rule.update(smoothing_horizon_epochs=2,confirmation_epochs=4)
        ev=[dict(epoch=e,quality=1.02,defect=.7,progress=.5) for e in range(2,11,2)]
        self.assertEqual(replay(ev,rule)[-1]['first_stop_epoch'],8)
        ev[2]['quality']=.8;ev[3]['quality']=.8
        self.assertIsNone(replay(ev,rule)[-1]['first_stop_epoch'])
        self.assertFalse(replay([ev[0],ev[-1]],rule)[-1]['eligible'])

    def test_no_stop_without_temporal_comparison(self):
        rule=load_yaml(PRESETS/'policy.yaml')
        rows=[dict(epoch=e,quality=2,defect=2,progress=None) for e in range(2,302,2)]
        self.assertFalse(replay(rows,rule)[-1]['should_stop'])

class WorkflowTests(unittest.TestCase):
    def test_end_to_end_cache_assets_reports_and_mutation_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'demo';result=run_demo(root)
            self.assertEqual(result['network_requests_made'],0)
            self.assertTrue(Path(result['training_report']).is_file())
            self.assertTrue((Path(result['training_report']).parent/'curves.png').is_file())
            cfg=Config(root/'evaluation.yaml');eng=Engine(cfg)
            before=eng.provider.usage()['completed_requests'];eng.baseline();self.assertEqual(before,eng.provider.usage()['completed_requests'])
            c=eng.positive[0];pos,neg=eng.references(c,'case',0)
            self.assertNotIn(c.sha256,[x.sha256 for x in pos])
            body,_,bindings=payload(cfg,c,pos,neg,eng.negative[0])
            self.assertEqual(set(body['questions']),{'quality','presence','relative_forward'})
            self.assertEqual(set(body['state']),{'candidate'})
            self.assertNotIn(bindings['{negative_references}'][1],str(body['questions']['quality']))
            reverse,_,_=payload(cfg,c,pos,neg,eng.negative[0],True)
            self.assertEqual(set(reverse['questions']),{'relative_reverse'})
            self.assertEqual(len(reverse['assets']),7)
            panel=root/'panels/12';checkpoint=root/'checkpoint.bin'
            again=TrainingMonitor(cfg).observe(12,panel,checkpoint,True)
            self.assertTrue(again['evaluated'])
            from PIL import Image
            Image.new('RGB',(24,24),'black').save(panel/'0.png')
            with self.assertRaises(Exception):TrainingMonitor(cfg).observe(12,panel,checkpoint,True)
            saved=(cfg.output_dir/'identity.json').read_text()
            self.assertNotIn('api_key',saved.replace('api_key_env','environment'))

    def test_bad_response_rejected(self):
        body={'model':'pinned','questions':{'presence':{'type':'noul'}}}
        for answer in [True,float('nan'),1.1]:
            with self.assertRaises(Exception):validate_response(body,{'model':'pinned','answers':{'presence':{'type':'noul','noul':answer}}})

    def test_reporting_can_change_but_measurement_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'demo';run_demo(root)
            path=root/'evaluation.yaml';raw=yaml.safe_load(path.read_text())
            raw['reporting']['dpi']=100;path.write_text(yaml.safe_dump(raw))
            Engine(Config(path)).baseline()
            raw['policy']['progress_max']=.6;path.write_text(yaml.safe_dump(raw))
            with self.assertRaises(Exception):Engine(Config(path))

    def test_duplicate_generated_files_remain_individual_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'demo';run_demo(root);panel=root/'panels/12'
            (panel/'copy.png').write_bytes((panel/'0.png').read_bytes())
            cfg=Config(root/'evaluation.yaml');result=Engine(cfg).evaluate(panel,'copies')
            self.assertEqual(result['images'],4)
            with self.assertRaises(Exception):Engine(cfg).evaluate(panel,'../outside')

    def test_provider_identity_and_choice_probabilities_checked(self):
        body={'model':'pinned','questions':{'relative':{'type':'choice'}}}
        with self.assertRaises(Exception):validate_response(body,{'model':'other','answers':{'relative':{'type':'choice','probabilities':{'A':.5,'B':.5}}}})
        with self.assertRaises(Exception):validate_response(body,{'model':'pinned','answers':{'relative':{'type':'choice','probabilities':{'A':.8,'B':.8}}}})

    def test_jev_small_inputs_prepared_before_transport(self):
        import io
        from PIL import Image
        from ai_self_evaluation.images import prepared_image,scan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'target';control=root/'control';target.mkdir();control.mkdir()
            for i in range(6):
                Image.new('RGB',(24,24),(150+i,100,100)).save(target/f'{i}.png')
                Image.new('RGB',(24,24),(40+i,100,100)).save(control/f'{i}.png')
            path=root/'evaluation.yaml';path.write_text(yaml.safe_dump(dict(provider='jev',target_image_dirs=['target'],control_image_dirs=['control'])))
            cfg=Config(path);record=scan([target])[0][0]
            with Image.open(io.BytesIO(prepared_image(record,cfg.settings['preprocessing']))) as image:self.assertEqual(image.size,(280,280))
            eng=Engine(cfg);pos,neg=eng.references(record,'case',0)
            cfg.settings['preprocessing'].update(min_side=None,resize=[24,24])
            with self.assertRaises(Exception):payload(cfg,record,pos,neg)

if __name__=='__main__':unittest.main()
