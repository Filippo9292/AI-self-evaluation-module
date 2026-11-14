"""Synthetic software demonstration; no judge accuracy claim or network calls."""
from pathlib import Path
import yaml
import numpy as np
from PIL import Image
from .config import Config
from .engine import Engine
from .monitor import TrainingMonitor

def run_demo(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Demo output must be empty.')
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42)
    for bank,mean in [('target',180),('control',45)]:
        folder=output/bank;folder.mkdir()
        for i in range(8):Image.fromarray(np.clip(rng.normal(mean,12,(24,24,3)),0,255).astype('uint8')).save(folder/f'{i}.png')
    config=dict(provider='mock',target_image_dirs=['target'],control_image_dirs=['control'],output_dir='results',sampling=dict(real_min_images=6,real_max_images=6,checkpoint_max_images=3,baseline_visits=1),policy=dict(comparison_lag_epochs=2,smoothing_horizon_epochs=2,confirmation_epochs=4),reporting=dict(boxplot_every_epochs=2,formats=['png','html']))
    path=output/'evaluation.yaml';path.write_text(yaml.safe_dump(config));cfg=Config(path);monitor=TrainingMonitor(cfg)
    for e in range(2,13,2):
        panel=output/'panels'/str(e);panel.mkdir(parents=True)
        for i in range(3):Image.fromarray(np.clip(rng.normal(min(180,100+e*15),5,(24,24,3)),0,255).astype('uint8')).save(panel/f'{i}.png')
        checkpoint=output/'checkpoint.bin';checkpoint.write_bytes(str(e).encode());monitor.observe(e,panel,checkpoint,monitor_only=True)
    inferred=Engine(cfg).evaluate(panel)
    return dict(synthetic=True,network_requests_made=0,training_report=str(cfg.output_dir/'training/report.html'),inference_report=inferred['report'])
