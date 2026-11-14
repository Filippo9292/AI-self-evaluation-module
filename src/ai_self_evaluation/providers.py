"""Question-scoped assets and typed decisions; credentials never enter journals."""
import base64
import copy
import io
import json
import math
import numpy as np
import requests
from PIL import Image
from .errors import EvaluationError
from .images import prepared_image
from .util import digest, append_json, records

def render(value, bindings):
    if isinstance(value,str):
        if value in bindings:return bindings[value]
        for token,replacement in bindings.items():
            if isinstance(replacement,str):value=value.replace(token,replacement)
        return value
    if isinstance(value,list):return [render(v,bindings) for v in value]
    if isinstance(value,dict):return {k:render(v,bindings) for k,v in value.items()}
    return value

def payload(config,candidate,positive,negative,previous=None,reverse=False):
    a,b=(previous,candidate) if reverse else (candidate,previous)
    records_by_hash={r.sha256:r for r in [a,*positive,*negative]+([b] if b else [])}
    ids={h:'asset-'+h for h in records_by_hash}
    bindings={'{candidate}':ids[a.sha256],'{positive_references}':[ids[r.sha256] for r in positive],'{negative_references}':[ids[r.sha256] for r in negative],'{previous}':ids[b.sha256] if b else ''}
    q=copy.deepcopy(config.protocol['questions'])
    if reverse:q={'relative_reverse':q['relative']}
    elif b:q['relative_forward']=q.pop('relative')
    else:q.pop('relative')
    body=dict(model=config.provider['model'],state=render(config.protocol['state'],bindings),questions=render(q,bindings))
    serialized=json.dumps(body);used={h:r for h,r in records_by_hash.items() if ids[h] in serialized}
    raw={ids[h]:prepared_image(r,config.settings['preprocessing']) for h,r in used.items()}
    limits=config.provider.get('image_limits')
    if limits:
        for data in raw.values():
            with Image.open(io.BytesIO(data)) as image:
                if min(image.size)<limits['min_side'] or max(image.size)>limits['max_side']:raise EvaluationError('Prepared image is outside the provider size limits; adjust preprocessing YAML.')
    body['assets']={key:dict(type='image',format='png',data_base64=base64.b64encode(v).decode()) for key,v in raw.items()}
    return body,raw,bindings

def validate_response(body,result):
    if not isinstance(result,dict) or result.get('model')!=body['model']:raise EvaluationError('Response model differs from the pinned model.')
    if not isinstance(result.get('answers'),dict) or set(result['answers'])!=set(body['questions']):raise EvaluationError('Response question identifiers differ.')
    values={}
    for qid,q in body['questions'].items():
        a=result['answers'][qid];kind=q['type']
        if a.get('type')!=kind:raise EvaluationError('Unexpected answer type.')
        key='score' if kind=='score' else 'noul' if kind=='noul' else None
        if key:
            v=a.get(key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=(4 if kind=='score' else 1):raise EvaluationError('Invalid numeric answer.')
            values[qid]=float(v)
        else:
            probs=a.get('probabilities',{})
            if set(probs)!={'A','B'} or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probs.values()) or abs(sum(probs.values())-1)>.03:raise EvaluationError('Invalid A/B probabilities.')
            values[qid]={k:float(v) for k,v in probs.items()}
    return values

class Provider:
    def __init__(self,config):
        self.config=config;self.session=requests.Session();self.journal=config.output_dir/'requests.jsonl'
        self.cache={r['request_id']:r for r in records(self.journal)}

    def evaluate(self,candidate,positive,negative,previous=None,reverse=False):
        body,raw,bindings=payload(self.config,candidate,positive,negative,previous,reverse);rid=digest(body)
        if rid in self.cache:return validate_response(body,self.cache[rid]['response'])
        api=self.config.settings['api'];encoded=json.dumps(body).encode()
        if len(encoded)>api['max_request_bytes']:raise EvaluationError('Request exceeds max_request_bytes.')
        attempts=records(self.config.output_dir/'attempts.jsonl')
        if api['max_calls'] is not None and len(attempts)>=api['max_calls']:raise EvaluationError('Configured max_calls reached.')
        if self.config.provider['adapter']=='mock':
            def mean(key):
                with Image.open(io.BytesIO(raw[key])) as im:return float(np.asarray(im,dtype=float).mean())
            m=mean(bindings['{candidate}']);target=float(np.mean([mean(x) for x in bindings['{positive_references}']])) if not reverse else 180
            answers={}
            for qid,q in body['questions'].items():
                if q['type']=='score':answers[qid]=dict(type='score',score=float(np.clip(3.4-abs(m-target)/80,0,4)))
                elif q['type']=='noul':answers[qid]=dict(type='noul',noul=float(np.clip(m/255,0,1)))
                else:
                    old=mean(bindings['{previous}']);v=float(np.clip(.5+(m-old)/400,0,1));answers[qid]=dict(type='choice',probabilities={'A':v,'B':1-v})
            result=dict(model=body['model'],answers=answers,usage=dict(input_tokens=0,output_tokens=0),synthetic=True)
        else:
            if not self.config.api_key:raise EvaluationError('Set api_key or its environment variable.')
            append_json(self.config.output_dir/'attempts.jsonl',dict(request_id=rid))
            try:response=self.session.post(self.config.provider['endpoint'],json=body,headers={**self.config.provider.get('headers',{}),'Authorization':'Bearer '+self.config.api_key,'Content-Type':'application/json'},timeout=api['timeout_seconds'])
            except requests.RequestException:raise EvaluationError('Transport failure; attempt recorded, no automatic retry. Re-running can incur a repeated charge.') from None
            if response.status_code!=200:raise EvaluationError('Provider HTTP '+str(response.status_code)+'; no automatic retry.')
            try:result=response.json()
            except ValueError:raise EvaluationError('Provider returned non-JSON content.') from None
        values=validate_response(body,result)
        if self.config.api_key:
            result=json.loads(json.dumps(result).replace(self.config.api_key,'[REDACTED]'))
        row=dict(request_id=rid,response=result);append_json(self.journal,row);self.cache[rid]=row
        return values

    def usage(self):
        return dict(completed_requests=len(self.cache),http_attempts=len(records(self.config.output_dir/'attempts.jsonl')),input_tokens=sum(r['response'].get('usage',{}).get('input_tokens',0) for r in self.cache.values()),output_tokens=sum(r['response'].get('usage',{}).get('output_tokens',0) for r in self.cache.values()))
