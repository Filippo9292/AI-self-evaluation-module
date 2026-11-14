"""Frozen real banks, fixed reference panels, cached typed evaluations."""
import math
import re
from pathlib import Path
import numpy as np
from .errors import EvaluationError
from .images import scan, sample_candidates, sample_references
from .providers import Provider
from .util import digest,read_json,write_json,write_csv,file_digest

def summarize(rows,norm):
    q=float(np.median([r['QI'] for r in rows]));d=float(np.mean([r['D2'] for r in rows]))
    ps=[r['R1_balanced'] for r in rows]
    return dict(n=len(rows),quality_raw=q,defect_raw=d,quality=q/norm['quality_reference'],defect=(d-norm['healthy_reference'])/norm['defect_span'],progress=None if any(v is None for v in ps) else float(np.mean(ps)))

class Engine:
    def __init__(self,config):
        self.config=config;self.root=config.output_dir;self.root.mkdir(parents=True,exist_ok=True)
        self.positive,_=scan(config.real_dirs,deduplicate=True);self.negative,_=scan(config.control_dirs,deduplicate=True)
        if min(len(self.positive),len(self.negative))<6:raise EvaluationError('At least six unique originals per real bank are required.')
        if {r.sha256 for r in self.positive}&{r.sha256 for r in self.negative}:raise EvaluationError('Target and control banks overlap by hash.')
        self.identity=digest([config.protocol_id,[r.manifest() for r in self.positive],[r.manifest() for r in self.negative]])
        identity_path=self.root/'identity.json'
        if identity_path.exists() and read_json(identity_path)['identity']!=self.identity:raise EvaluationError('Judge, real banks or policy changed; use a new output directory.')
        write_json(identity_path,dict(identity=self.identity,configuration=config.public()))
        self.provider=Provider(config)

    def references(self,candidate,key,visit):
        s=self.config.settings['sampling'];seed=s['seed']
        return (sample_references(self.positive,candidate,seed,['positive',key,visit]),sample_references(self.negative,candidate,seed,['negative',key,visit]))

    def score(self,candidates,stage,visits,previous=None):
        folder=self.root/stage
        manifest=[dict(case=case,**r.manifest()) for case,r in candidates]
        source=dict(images=manifest,previous=[] if previous is None else [dict(case=k,**v.manifest()) for k,v in sorted(previous.items())])
        if (folder/'source.json').exists() and read_json(folder/'source.json')!=source:raise EvaluationError('Image panel changed in an existing stage.')
        write_json(folder/'source.json',source)
        if (folder/'scores.json').exists():return read_json(folder/'scores.json')
        rows=[];reference_rows=[]
        for case,c in candidates:
            pp=None if previous is None else previous.get(case)
            if previous is not None and pp is None:raise EvaluationError('Temporal cases do not match.')
            vv=[]
            for visit in range(visits):
                pos,neg=self.references(c,case,visit)
                f=self.provider.evaluate(c,pos,neg,pp)
                r=self.provider.evaluate(c,pos,neg,pp,reverse=True) if pp else None
                forward=None if pp is None else f['relative_forward']['A'];reverse=None if r is None else r['relative_reverse']['B']
                vv.append(dict(QI=f['quality'],D2=f['presence'],R1_current_as_A=forward,R1_current_as_B=reverse,R1_balanced=None if pp is None else (forward+reverse)/2))
                reference_rows.append(dict(case=case,visit=visit,positive=[r.manifest() for r in pos],negative=[r.manifest() for r in neg]))
            row=dict(case=case,image_sha256=c.sha256,visits=visits)
            for k in vv[0]:row[k]=None if vv[0][k] is None else float(np.mean([v[k] for v in vv]))
            rows.append(row)
        write_json(folder/'references.json',reference_rows);write_json(folder/'scores.json',rows);write_csv(folder/'scores.csv',rows)
        write_json(self.root/'usage.json',self.provider.usage())
        return rows

    def baseline(self):
        path=self.root/'baseline/baseline.json'
        if path.exists():return read_json(path)
        s=self.config.settings['sampling'];banks={}
        for name,pool in [('target',self.positive),('control',self.negative)]:
            count=min(len(pool),s['real_max_images'],max(s['real_min_images'],math.ceil(s['real_sqrt_multiplier']*math.sqrt(len(pool)))))
            cap=s.get(name+'_max_images')
            if cap is not None:count=min(count,cap)
            selected=sample_candidates(pool,count,s['seed'],['baseline',name])
            banks[name]=self.score([(name+'/'+r.sha256,r) for r in selected],'baseline/'+name,s['baseline_visits'])
        quality=float(np.quantile([r['QI'] for r in banks['target']],self.config.settings['policy']['quality_real_quantile']))
        healthy=float(np.median([r['D2'] for r in banks['control']]));defective=float(np.median([r['D2'] for r in banks['target']]))
        if quality<=0 or defective-healthy<=1e-8:raise EvaluationError('Real baseline cannot normalize these prompts: nonpositive quality or target/control separation.')
        baseline=dict(**banks,normalizers=dict(quality_reference=quality,healthy_reference=healthy,defective_reference=defective,defect_span=defective-healthy))
        write_json(path,baseline);return baseline

    def evaluate(self,directory,identifier='inference'):
        if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',identifier):raise EvaluationError('Invalid inference ID.')
        baseline=self.baseline();images,_=scan([directory]);root=Path(directory).resolve()
        if self.root.is_relative_to(root):raise EvaluationError('Output must be outside generated-image directory.')
        candidates=[(r.path.relative_to(root).as_posix(),r) for r in images]
        rows=self.score(candidates,'inference/'+identifier,self.config.settings['sampling']['visits_per_image'])
        summary=summarize(rows,baseline['normalizers']);write_json(self.root/'inference'/identifier/'summary.json',summary)
        from .reporting import inference_report
        inference_report(self.root/'inference'/identifier,baseline,rows,self.config.settings['reporting'])
        return dict(images=len(rows),summary=summary,report=str(self.root/'inference'/identifier/'report.html'))
