import argparse
import json
import re
import sys
from pathlib import Path
from .config import Config,PRESETS,load_yaml
from .engine import Engine
from .monitor import TrainingMonitor
from .images import scan,real_sample_size
from .errors import EvaluationError
from .util import RunLock,read_json

def main(argv=None):
    parser=argparse.ArgumentParser(description='Reference-based image evaluation and temporal early stopping.')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('providers')
    demo=commands.add_parser('demo');demo.add_argument('--output',type=Path,default=Path('demo-workspace'))
    for command in ('plan','baseline','evaluate','monitor','report'):
        p=commands.add_parser(command);p.add_argument('--config',required=True,type=Path)
        if command=='plan':p.add_argument('--max-epochs',type=int,default=800)
        if command=='evaluate':p.add_argument('--images',type=Path);p.add_argument('--id',default='inference')
        if command=='monitor':
            p.add_argument('--images',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--epoch',required=True,type=int)
            p.add_argument('--monitor-only',action='store_true');p.add_argument('--stop-exit-code',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='providers':result=[load_yaml(p) for p in sorted((PRESETS/'providers').glob('*.yaml'))]
        elif args.command=='demo':
            from .demo import run_demo
            result=run_demo(args.output)
        else:
            cfg=Config(args.config)
            if args.command=='plan':
                target,_=scan(cfg.real_dirs,deduplicate=True);control,_=scan(cfg.control_dirs,deduplicate=True)
                s=cfg.settings['sampling'];p=cfg.settings['policy'];baseline=sum(real_sample_size(len(pool),s) for pool in (target,control))*s['baseline_visits']
                epochs=range(p['audit_every_epochs'],args.max_epochs+1,p['audit_every_epochs'])
                training=sum(s['checkpoint_max_images']*s['visits_per_image']*(2 if e-p['comparison_lag_epochs']>=p['audit_every_epochs'] else 1) for e in epochs)
                result=dict(target_images=len(target),control_images=len(control),baseline_max_calls=baseline,training_max_calls=training,network_requests_made=0,note='Upper bound assumes a full panel at every audit; excludes retries and completed cache hits.')
            elif args.command=='monitor':
                result=TrainingMonitor(cfg).observe(args.epoch,args.images,args.checkpoint,args.monitor_only)
                print(json.dumps(result,indent=2));return 10 if args.stop_exit_code and result['should_stop'] else 0
            elif args.command=='report':
                from .reporting import training_report,inference_report
                with RunLock(cfg.output_dir):
                    if read_json(cfg.output_dir/'identity.json')['configuration']['protocol_id']!=cfg.protocol_id:raise EvaluationError('Measurement configuration changed; restore it before reporting.')
                    baseline=read_json(cfg.output_dir/'baseline/baseline.json')
                    observations=TrainingMonitor(cfg).observations()
                    if observations:training_report(cfg.output_dir,baseline,observations,cfg.settings['reporting'],cfg.settings['policy'])
                    for folder in (cfg.output_dir/'inference').glob('*'):
                        if (folder/'scores.json').exists():inference_report(folder,baseline,read_json(folder/'scores.json'),cfg.settings['reporting'])
                result=dict(reports_regenerated=True,network_requests_made=0)
            else:
                with RunLock(cfg.output_dir):
                    engine=Engine(cfg)
                    if args.command=='baseline':result=engine.baseline()['normalizers']
                    else:
                        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',args.id):raise EvaluationError('Invalid inference ID.')
                        directory=args.images or cfg.generated_dir
                        if not directory:raise EvaluationError('Set generated_image_dir or --images.')
                        result=engine.evaluate(directory,args.id)
        print(json.dumps(result,indent=2));return 0
    except (EvaluationError,OSError,ValueError,KeyError,TypeError) as exc:
        print('Evaluation stopped: '+str(exc),file=sys.stderr);return 2

def self_evaluation():return main(['evaluate',*sys.argv[1:]])
def early_stopping():return main(['monitor',*sys.argv[1:]])
