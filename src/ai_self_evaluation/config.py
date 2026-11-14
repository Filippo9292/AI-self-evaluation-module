"""Separate YAML responsibilities, resolved relative to the main configuration."""
import copy
import math
import os
from pathlib import Path
import yaml
from .errors import EvaluationError
from .util import digest

PRESETS = Path(__file__).parent / 'presets'

def load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):raise EvaluationError('YAML must contain a mapping.')
    return value

def merge(base, update):
    result = copy.deepcopy(base)
    for k,v in update.items():
        result[k] = merge(result[k],v) if isinstance(v,dict) and isinstance(result.get(k),dict) else copy.deepcopy(v)
    return result

class Config:
    def __init__(self, path):
        self.path = Path(path).resolve();raw = load_yaml(self.path)
        self.resolve = lambda p: (self.path.parent / Path(p).expanduser()).resolve()
        name=raw.get('provider','jev')
        if name not in ('jev','mock') and not raw.get('provider_file'):raise EvaluationError('Use a supported preset or provider_file.')
        self.provider = load_yaml(self.resolve(raw['provider_file']) if raw.get('provider_file') else PRESETS/'providers'/(name+'.yaml'))
        if self.provider.get('adapter') not in ('mock','typesafe_assets'):raise EvaluationError('Supported adapters: typesafe_assets, mock.')
        if self.provider['adapter'] != 'mock' and not self.provider['endpoint'].startswith('https://'):raise EvaluationError('Live endpoints must use HTTPS.')
        self.protocol = load_yaml(self.resolve(raw['protocol_file']) if raw.get('protocol_file') else PRESETS/'protocol.yaml')
        self.settings = {}
        for part in ('policy','selection','sampling','reporting','preprocessing','api'):
            default = load_yaml(PRESETS/(part+'.yaml'))
            if raw.get(part+'_file'):default = merge(default,load_yaml(self.resolve(raw[part+'_file'])))
            self.settings[part] = merge(default,raw.get(part,{}))
        for name in ('target_image_dirs','control_image_dirs'):
            if not isinstance(raw.get(name),list) or not raw[name] or not all(isinstance(p,str) for p in raw[name]):raise EvaluationError(name+' must list original-real directories.')
        self.real_dirs = [self.resolve(p) for p in raw['target_image_dirs']]
        self.control_dirs = [self.resolve(p) for p in raw['control_image_dirs']]
        self.generated_dir = self.resolve(raw['generated_image_dir']) if raw.get('generated_image_dir') else None
        self.output_dir = self.resolve(raw.get('output_dir','outputs/evaluation'))
        for p in self.real_dirs+self.control_dirs+([self.generated_dir] if self.generated_dir else []):
            if self.output_dir.is_relative_to(p):raise EvaluationError('Keep output_dir outside input directories.')
        self.api_key = raw.get('api_key') or os.environ.get(self.provider.get('api_key_env',''),'')
        if not isinstance(self.api_key,str):raise EvaluationError('api_key must be a string.')
        self._validate()
        public_provider={k:v for k,v in self.provider.items() if k not in ('headers','api_key')}
        self.protocol_id=digest(dict(provider=public_provider,protocol=self.protocol,settings={k:v for k,v in self.settings.items() if k not in ('api','reporting')}))

    def _validate(self):
        p,s,r=self.settings['policy'],self.settings['sampling'],self.settings['reporting']
        def positive(v,name):
            if isinstance(v,bool) or not isinstance(v,int) or v<1:raise EvaluationError(name+' must be a positive integer.')
        for k in ('audit_every_epochs','comparison_lag_epochs','smoothing_horizon_epochs','confirmation_epochs'):positive(p[k],k)
        for k in ('real_min_images','real_max_images','baseline_visits','visits_per_image','checkpoint_max_images'):positive(s[k],k)
        for k in ('boxplot_every_epochs','render_every_observations','dpi'):positive(r[k],k)
        if s['references_per_visit'] != 5:raise EvaluationError('Exactly five references per group are required.')
        if s['real_min_images'] > s['real_max_images']:raise EvaluationError('real_min_images exceeds real_max_images.')
        for k in ('target_max_images','control_max_images'):
            if s.get(k) is not None:positive(s[k],k)
        if not isinstance(s['seed'],int) or isinstance(s['seed'],bool):raise EvaluationError('seed must be an integer.')
        if not math.isfinite(s['real_sqrt_multiplier']) or s['real_sqrt_multiplier']<=0:raise EvaluationError('Invalid sampling multiplier.')
        cadence=p['audit_every_epochs']
        if any(p[k]%cadence for k in ('comparison_lag_epochs','smoothing_horizon_epochs','confirmation_epochs')):raise EvaluationError('Lag, smoothing and confirmation must be multiples of cadence.')
        for k,v in p.items():
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise EvaluationError('Invalid policy value: '+k)
        if not 0<p['quality_real_quantile']<1 or not 0<=p['progress_max']<=1:raise EvaluationError('Invalid quantile or progress threshold.')
        if not 0<p['quality_exit_ratio']<=p['quality_enter_ratio'] or not 0<=p['defect_exit_normalized']<=p['defect_enter_normalized']:raise EvaluationError('Invalid entry/retention thresholds.')
        q=self.protocol.get('questions',{})
        if set(q)!= {'quality','presence','relative'} or q['quality']['type']!='score' or len(q['quality']['criteria'])!=5 or q['presence']['type']!='noul' or q['relative']['type']!='choice' or set(q['relative']['criteria'])!={'A','B'}:raise EvaluationError('Protocol requires quality score, presence noul, relative A/B choice.')
        if '{candidate}' not in str(self.protocol.get('state')) or '{positive_references}' not in str(q) or '{negative_references}' not in str(q) or '{previous}' not in str(q['relative']):raise EvaluationError('Missing protocol placeholders.')
        api=self.settings['api']
        if api['timeout_seconds']<=0 or api['max_request_bytes']<=0:raise EvaluationError('Invalid API limits.')
        if api['max_calls'] is not None:positive(api['max_calls'],'max_calls')
        pre=self.settings['preprocessing'];positive(pre['max_side'],'max_side')
        if pre.get('min_side') is not None:
            positive(pre['min_side'],'min_side')
            if pre['min_side']>pre['max_side']:raise EvaluationError('min_side exceeds max_side.')
        if pre['interpolation'] not in ('nearest','bilinear','bicubic','lanczos'):raise EvaluationError('Unknown interpolation.')
        if pre.get('resize') is not None and (len(pre['resize'])!=2 or any(not isinstance(v,int) or v<1 for v in pre['resize'])):raise EvaluationError('resize must be [width,height].')
        if not set(r['formats'])<= {'png','pdf','html'} or not r['formats']:raise EvaluationError('Unknown reporting formats.')
        selection=self.settings['selection']
        if not isinstance(selection['prefer_mature'],bool):raise EvaluationError('selection.prefer_mature must be boolean.')
        weights=[selection[k] for k in ('quality_distance_weight','presence_distance_weight')]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in weights) or sum(weights)<=0:raise EvaluationError('Selection weights must be nonnegative with a positive sum.')

    def public(self):
        return dict(protocol_id=self.protocol_id,provider={k:v for k,v in self.provider.items() if k not in ('headers','api_key')},protocol=self.protocol,settings=self.settings)
