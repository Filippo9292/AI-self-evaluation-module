"""Checkpoint observation; trainer owns generation and optimizer updates."""
import shutil
from pathlib import Path
from .config import Config
from .engine import Engine,summarize
from .images import scan,sample_candidates
from .errors import EvaluationError
from .policy import replay
from .util import RunLock,read_json,write_json,file_digest,digest

class TrainingMonitor:
    def __init__(self,config):
        self.config=config if isinstance(config,Config) else Config(config)

    def due(self,epoch):
        return epoch>0 and epoch%self.config.settings['policy']['audit_every_epochs']==0

    def observations(self):
        return [read_json(p) for p in sorted((self.config.output_dir/'training').glob('epoch_*/observation.json'))]

    def observe(self,epoch,images,checkpoint,monitor_only=False):
        if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<1:raise EvaluationError('Epoch must be a positive integer.')
        if not self.due(epoch):return dict(evaluated=False,should_stop=False)
        checkpoint=Path(checkpoint).resolve()
        if not checkpoint.is_file():raise EvaluationError('Checkpoint must exist.')
        with RunLock(self.config.output_dir):
            engine=Engine(self.config);baseline=engine.baseline();observations=self.observations()
            stage='training/epoch_'+str(epoch).zfill(6);folder=engine.root/stage
            checkpoint_hash=file_digest(checkpoint)
            frozen_checkpoint=folder/'checkpoint_identity.json'
            if frozen_checkpoint.exists() and read_json(frozen_checkpoint)['sha256']!=checkpoint_hash:raise EvaluationError('Checkpoint changed in a partial observation.')
            write_json(frozen_checkpoint,dict(sha256=checkpoint_hash))
            if observations and epoch<=observations[-1]['epoch']:
                if epoch==observations[-1]['epoch'] and checkpoint_hash==observations[-1]['checkpoint_sha256']:
                    current,_=scan([images]);original=read_json(folder/'input_manifest.json')
                    if [r.manifest() for r in current]!=original:raise EvaluationError('Completed image panel changed.')
                    return observations[-1]['decision']
                raise EvaluationError('Epochs must increase.')
            if observations and observations[-1]['decision']['should_stop'] and not monitor_only:raise EvaluationError('Stop already requested; use monitor_only to continue.')
            source=Path(images).resolve()
            if engine.root.is_relative_to(source):raise EvaluationError('Output directory must not be inside input panel.')
            records,_=scan([source]);manifest=[r.manifest() for r in records]
            if (folder/'input_manifest.json').exists() and read_json(folder/'input_manifest.json')!=manifest:raise EvaluationError('Partial panel changed.')
            write_json(folder/'input_manifest.json',manifest)
            all_cases={r.path.relative_to(source).as_posix():r for r in records}
            panel_plan=engine.root/'training/panel_cases.json'
            if panel_plan.exists():
                cases=read_json(panel_plan)
                if not set(cases)<=set(all_cases):raise EvaluationError('Stable case filenames are missing.')
            else:
                selected=sample_candidates(records,self.config.settings['sampling']['checkpoint_max_images'],self.config.settings['sampling']['seed'],'checkpoint-panel')
                cases=sorted(r.path.relative_to(source).as_posix() for r in selected);write_json(panel_plan,cases)
            saved=folder/'images'
            for case in cases:
                target=saved/case;target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists() and file_digest(target)!=all_cases[case].sha256:raise EvaluationError('Saved panel changed.')
                if not target.exists():shutil.copy2(all_cases[case].path,target)
            snapshot,_=scan([saved]);bycase={r.path.relative_to(saved).as_posix():r for r in snapshot}
            lag=self.config.settings['policy']['comparison_lag_epochs'];previous=None
            previous_path=engine.root/'training'/('epoch_'+str(epoch-lag).zfill(6))/'images'
            if previous_path.is_dir():
                rr,_=scan([previous_path]);previous={r.path.relative_to(previous_path).as_posix():r for r in rr}
                if set(previous)!=set(cases):raise EvaluationError('Temporal panel filenames differ.')
            rows=engine.score([(case,bycase[case]) for case in cases],stage,self.config.settings['sampling']['visits_per_image'],previous)
            event=dict(epoch=epoch,**summarize(rows,baseline['normalizers']))
            trace=replay([o['event'] for o in observations]+[event],self.config.settings['policy']);decision=trace[-1]
            # Checkpoint ranking is separate from stop: prioritize maturity, then closeness to real anchors.
            selection=self.config.settings['selection']
            rank=[int(selection['prefer_mature'] and not decision['mature']),selection['quality_distance_weight']*abs(event['quality']-1)+selection['presence_distance_weight']*abs(event['defect']-1)]
            best_path=engine.root/'best/selection.json';best=read_json(best_path) if best_path.exists() else None
            if best is None or rank<best['rank']:
                target=engine.root/'best/checkpoint.bin';target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(checkpoint,target)
                best=dict(epoch=epoch,rank=rank,checkpoint=str(target),checkpoint_sha256=checkpoint_hash);write_json(best_path,best)
            decision=dict(decision,evaluated=True,best=best)
            observation=dict(epoch=epoch,checkpoint_sha256=checkpoint_hash,event=event,decision=decision)
            write_json(folder/'observation.json',observation)
            from .reporting import training_report
            report=self.config.settings['reporting']
            if (len(observations)+1)%report['render_every_observations']==0 or decision['should_stop']:
                training_report(engine.root,baseline,observations+[observation],report)
            return decision
