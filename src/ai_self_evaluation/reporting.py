"""Reusable boxplot output: distributions, real anchors, means and controller curves."""
import base64
import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .util import read_json,write_csv

GENERATED='#c55e26'
GENERATED_FILL='#f4b183'
EPOCH_FILLS=('#f6bd60','#f7d9c4','#e78b73','#e8c56d','#d9a066','#f3a84b','#ecc1ae')
REAL_TARGET='#cad7e2'
REAL_CONTROL='#dfe8df'
INK='#43352e'
EDGE='#9b8d82'

def style_figure(fig):
    fig.set_facecolor('#fffaf5')
    for ax in fig.axes:
        ax.set_facecolor('white');ax.set_axisbelow(True)
        for side in ('top','right'):ax.spines[side].set_visible(False)
        for side in ('left','bottom'):
            ax.spines[side].set_color('#d2c6bc');ax.spines[side].set_linewidth(.8)
        ax.tick_params(colors=INK,labelsize=9)
        ax.title.set_color(INK);ax.xaxis.label.set_color(INK);ax.yaxis.label.set_color(INK)
        ax.grid(axis='y',color='#ece4dc',alpha=.8,linewidth=.7)
        legend=ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('white');legend.get_frame().set_edgecolor('#e5d9ce');legend.get_frame().set_alpha(.95)
    if fig._suptitle:fig._suptitle.set_color(INK)

def normalized(rows,norm,key):
    if key=='quality':return [r['QI']/norm['quality_reference'] for r in rows]
    if key=='defect':return [(r['D2']-norm['healthy_reference'])/norm['defect_span'] for r in rows]
    return [r['R1_balanced'] for r in rows if r['R1_balanced'] is not None]

def box_stats(values):
    q1,med,q3=np.quantile(values,[.25,.5,.75]);v=np.array(values);inside=v[(v>=q1-1.5*(q3-q1))&(v<=q3+1.5*(q3-q1))]
    return dict(n=len(v),q1=float(q1),median=float(med),q3=float(q3),mean=float(v.mean()),whisker_low=float(inside.min()),whisker_high=float(inside.max()))

def finish(fig,folder,settings,title):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);buffer=io.BytesIO()
    style_figure(fig)
    fig.savefig(buffer,format='png',dpi=settings['dpi'])
    if 'png' in settings['formats']:(folder/'boxplots.png').write_bytes(buffer.getvalue())
    if 'pdf' in settings['formats']:fig.savefig(folder/'boxplots.pdf')
    if 'html' in settings['formats']:
        image=base64.b64encode(buffer.getvalue()).decode()
        (folder/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{font:16px system-ui;margin:25px}img{max-width:100%}</style><h1>'+title+'</h1><p>Boxes show image distributions. Colored dots show means for D/P. Curves show the controller aggregates after temporal smoothing. Reference boxes contain original-real images.</p><img alt="Evaluation boxplots" src="data:image/png;base64,'+image+'">',encoding='utf-8')
    plt.close(fig)

def training_report(root,baseline,observations,settings,policy=None):
    root=Path(root);norm=baseline['normalizers'];fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained')
    keys=['quality','defect','progress'];titles=['Normalized quality Q','Normalized feature presence D','Temporal preference P']
    threshold=policy or read_json(root/'identity.json')['configuration']['settings']['policy']
    selected=[o for o in observations if o['epoch']%settings['boxplot_every_epochs']==0 or settings['include_first_epoch'] and o is observations[0]]
    stats=[]
    for k,(ax,key,title) in enumerate(zip(axes,keys,titles)):
        for bank,x in ([('target',-40)] if k==0 else [('control',-90),('target',-40)] if k==1 else []):
            values=normalized(baseline[bank],norm,key)
            ax.boxplot([values],positions=[x],widths=22,manage_ticks=False,patch_artist=True,boxprops=dict(facecolor=REAL_TARGET if bank=='target' else REAL_CONTROL,edgecolor=EDGE,linewidth=.9),medianprops=dict(color=INK,linewidth=1.5),whiskerprops=dict(color=EDGE),capprops=dict(color=EDGE),flierprops=dict(marker='o',markersize=2,markerfacecolor=EDGE,markeredgecolor='none',alpha=.5))
            stats.append(dict(metric=key,epoch='',group='real_'+bank,**box_stats(values)))
        positions=[];data=[]
        for o in observations:
            rows=read_json(root/'training'/('epoch_'+str(o['epoch']).zfill(6))/'scores.json');values=normalized(rows,norm,key)
            if values:stats.append(dict(metric=key,epoch=o['epoch'],group='generated',**box_stats(values)))
            if o in selected and values:positions.append(o['epoch']);data.append(values)
        if data:
            boxes=ax.boxplot(data,positions=positions,widths=min(settings['boxplot_every_epochs']*.5,24),manage_ticks=False,patch_artist=True,boxprops=dict(edgecolor='#aa8064',linewidth=.9),medianprops=dict(color=INK,linewidth=1.4),whiskerprops=dict(color=EDGE),capprops=dict(color=EDGE),flierprops=dict(marker='o',markersize=2,markerfacecolor=EDGE,markeredgecolor='none',alpha=.45))
            for box,epoch in zip(boxes['boxes'],positions):
                box.set_facecolor(EPOCH_FILLS[(epoch//settings['boxplot_every_epochs'])%len(EPOCH_FILLS)])
            if k>0:ax.scatter(positions,[np.mean(v) for v in data],color=GENERATED,s=18,edgecolors='white',linewidths=.45,zorder=4,label='Per-epoch mean')
        curve=[o for o in observations if o['decision'][key+'_smooth'] is not None]
        if curve:ax.plot([o['epoch'] for o in curve],[o['decision'][key+'_smooth'] for o in curve],color=GENERATED,label='Smoothed controller aggregate')
        ts=[threshold['quality_enter_ratio'],threshold['quality_exit_ratio']] if k==0 else [threshold['defect_enter_normalized'],threshold['defect_exit_normalized']] if k==1 else [threshold['progress_max']]
        for j,t in enumerate(ts):ax.axhline(t,color='#333' if j==0 else '#888',ls='--' if j==0 else ':',label='Threshold '+str(t))
        stop=next((o['epoch'] for o in observations if o['decision']['first_stop']),None)
        if stop is not None:ax.axvline(stop,color='#c33',ls=':',label='First stop '+str(stop))
        xmax=max(o['epoch'] for o in observations);xticks=list(range(0,xmax+1,max(50,50*((xmax//10)//50))))
        ax.set_xticks(([-40] if k==0 else [-90,-40] if k==1 else [])+xticks,(['Real\ntarget'] if k==0 else ['Real\ncontrol','Real\ntarget'] if k==1 else [])+[str(x) for x in xticks],fontsize=8)
        ax.set_title(title);ax.set_xlabel('Epoch');ax.set_xlim(-120,max(70,xmax+20));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Reference-based evaluation: image distributions and early-stopping signals')
    write_csv(root/'training/boxplot_statistics.csv',stats)
    write_csv(root/'training/epoch_summary.csv',[dict(o['event'],**o['decision']) for o in observations])
    finish(fig,root/'training',settings,'Training evaluation')
    controller_curves(root/'training',observations,settings,threshold)

def controller_curves(folder,observations,settings,policy):
    folder=Path(folder)
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
    for ax,key,title in zip(axes[:3],('quality','defect','progress'),('Normalized quality Q','Normalized feature presence D','Temporal preference P')):
        valid=[o for o in observations if o['decision'][key+'_smooth'] is not None]
        if valid:ax.plot([o['epoch'] for o in valid],[o['decision'][key+'_smooth'] for o in valid],color=GENERATED,label='Smoothed aggregate')
        thresholds=[policy['quality_enter_ratio'],policy['quality_exit_ratio']] if key=='quality' else [policy['defect_enter_normalized'],policy['defect_exit_normalized']] if key=='defect' else [policy['progress_max']]
        for j,t in enumerate(thresholds):ax.axhline(t,color='#333' if j==0 else '#888',ls='--' if j==0 else ':',label='Threshold '+str(t))
        ax.set_ylabel(title);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    axes[3].plot([o['epoch'] for o in observations],[o['decision']['elapsed'] for o in observations],color='#b57932',label='Consecutive confirmation span')
    axes[3].axhline(policy['confirmation_epochs'],color='#333',ls='--',label='Required '+str(policy['confirmation_epochs'])+' epochs')
    axes[3].set_ylabel('Confirmation (epochs)');axes[3].set_xlabel('Epoch');axes[3].grid(alpha=.2);axes[3].legend(fontsize=8,loc='best')
    stop=next((o['epoch'] for o in observations if o['decision']['first_stop']),None)
    if stop is not None:
        for ax in axes:ax.axvline(stop,color='#c33',ls=':')
    fig.suptitle('Controller signals and confirmation'+(' - first stop '+str(stop) if stop is not None else ''))
    style_figure(fig)
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=settings['dpi'])
    if 'png' in settings['formats']:(folder/'curves.png').write_bytes(buffer.getvalue())
    if 'pdf' in settings['formats']:fig.savefig(folder/'curves.pdf')
    if 'html' in settings['formats']:
        report=folder/'report.html';previous=report.read_text(encoding='utf-8')
        report.write_text(previous+'<h2>Controller curves</h2><img alt="Controller signals and confirmation" src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">',encoding='utf-8')
    plt.close(fig)

def inference_report(folder,baseline,rows,settings):
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained');norm=baseline['normalizers'];stats=[]
    for ax,key in zip(axes,['quality','defect']):
        names=['Real target','Generated'] if key=='quality' else ['Real control','Real target','Generated']
        datasets=[normalized(baseline['target'],norm,key),normalized(rows,norm,key)] if key=='quality' else [normalized(baseline['control'],norm,key),normalized(baseline['target'],norm,key),normalized(rows,norm,key)]
        boxes=ax.boxplot(datasets,showmeans=True,patch_artist=True,medianprops=dict(color=INK,linewidth=1.6),whiskerprops=dict(color=EDGE,linewidth=1),capprops=dict(color=EDGE),flierprops=dict(marker='o',markersize=3,markerfacecolor=EDGE,markeredgecolor='none',alpha=.5),meanprops=dict(marker='D',markersize=5,markerfacecolor=GENERATED,markeredgecolor='white',markeredgewidth=.7))
        fills=[REAL_TARGET,GENERATED_FILL] if key=='quality' else [REAL_CONTROL,REAL_TARGET,GENERATED_FILL]
        for i,(box,fill) in enumerate(zip(boxes['boxes'],fills)):
            box.set(facecolor=fill,edgecolor='#aa8064' if i==len(fills)-1 else EDGE,linewidth=1.1)
            if i<len(fills)-1:boxes['means'][i].set_markerfacecolor('#60758d')
        ax.set_xticks(range(1,len(names)+1),names);ax.set_title('Normalized '+key);ax.grid(axis='y',alpha=.2)
        stats.extend(dict(metric=key,group=name,**box_stats(v)) for name,v in zip(names,datasets))
    write_csv(Path(folder)/'boxplot_statistics.csv',stats);finish(fig,folder,settings,'Inference evaluation')
