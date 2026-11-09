"""Explicit fresh/resume product dispatch; common core still owns all updates."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train',))
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--stop-after', type=int)
    p.add_argument('--selection', choices=('raw',), default='raw')
    a = p.parse_args()
    if a.out.exists():
        p.error('fresh output directory required; never overwrite a previous cohort/attempt')
    from particle_sliders.lifecycle import SessionConfig, atomic_json
    from .current_backend import open_training
    config = SessionConfig.from_dict(json.loads(a.config.read_text()))
    if a.stop_after is not None and a.stop_after < 0:
        p.error('stop-after must be nonnegative')
    a.out.mkdir(parents=True, exist_ok=False)
    # The product dispatcher passes restore intent before prepare/optimizers.
    with open_training(config, resume=a.resume) as session:
        session.train(stop_after=a.stop_after)
        session.checkpoint(a.out/'training.pt')
        session.export_inference(a.out/'adapter-v2.pt', selection='raw')
        atomic_json(a.out/'run.json', dict(config=config.as_dict(),
            completed_steps=session.game.completed_steps, resumed_from=None if a.resume is None else str(a.resume),
            selection='raw', model_weight_EMA=None, pretrained_quality_qualified=False))


if __name__ == '__main__':main()
