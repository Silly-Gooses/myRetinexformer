"""Paired, resumable LOL-v2 Real SAIM experiment. No Test data enters training."""
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from RetinexFormer_arch import RetinexFormer as Baseline
from Proposed_SAIM_arch import RetinexFormer as SAIM, load_baseline_checkpoint
from analysis_utils import find_paired_images, image_metrics, infer_full_image, tensor_to_uint8


DEFAULTS = dict(seed=42, crop_size=128, batch_size=1, lr=2e-4, min_lr=1e-6,
                val_fraction=0.10, val_interval=500, log_interval=20,
                clip_grad=1.0, amp=True, n_feat=40, stage=1, num_blocks=[1, 2, 2])
FIELDS = {
    'learning_curves.csv': ['arm', 'step', 'train_l1', 'lr', 'val_psnr', 'val_ssim'],
    'comparison_by_step.csv': ['step', 'baseline_psnr', 'saim_psnr', 'delta_psnr',
                               'baseline_ssim', 'saim_ssim', 'delta_ssim'],
    'per_image_results.csv': ['split', 'step', 'image', 'baseline_psnr', 'saim_psnr',
                              'delta_psnr', 'baseline_ssim', 'saim_ssim', 'delta_ssim'],
    'best_checkpoint_summary.csv': ['arm', 'step', 'val_psnr', 'val_ssim', 'checkpoint'],
    'training_diagnostics.csv': ['arm', 'step', 'loss', 'lr', 'grad_norm', 'amp_scale'],
}


def _atomic_save(path, obj):
    temp = path.with_name(path.name + '.tmp')
    torch.save(obj, temp)
    os.replace(temp, path)


def _csv(path, rows):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS[path.name])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    os.replace(temp, path)


def _pairs_manifest(pairs):
    return [{'name': name, 'low': str(low.resolve()), 'normal': str(high.resolve())}
            for name, low, high in pairs]


def prepare_split(dataset_root, seed=42, val_fraction=.1):
    root = Path(dataset_root)
    pairs = find_paired_images(root / 'Train/Low', root / 'Train/Normal')
    if len(pairs) < 2:
        raise ValueError('At least two Train pairs are required for holdout validation')
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    count = max(1, round(len(pairs) * val_fraction))
    val_ids = set(order[:count])
    train = [pair for i, pair in enumerate(pairs) if i not in val_ids]
    val = [pair for i, pair in enumerate(pairs) if i in val_ids]
    return train, val


def _rgb(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).float() / 255


def training_batch(pairs, step, seed, crop_size, device):
    """Stateless paired sampling: identical for both arms and across resume."""
    rng = random.Random((seed << 32) + step)
    name, low_path, gt_path = pairs[rng.randrange(len(pairs))]
    low, gt = _rgb(low_path), _rgb(gt_path)
    if low.shape != gt.shape:
        raise ValueError(f'Image dimensions differ for {name}')
    h, w = low.shape[-2:]
    if h < crop_size or w < crop_size:
        raise ValueError(f'{name} is smaller than the {crop_size} pixel crop')
    top, left = rng.randrange(h - crop_size + 1), rng.randrange(w - crop_size + 1)
    low = low[:, top:top+crop_size, left:left+crop_size]
    gt = gt[:, top:top+crop_size, left:left+crop_size]
    if rng.randrange(2):
        low, gt = low.flip(-1), gt.flip(-1)
    if rng.randrange(2):
        low, gt = low.flip(-2), gt.flip(-2)
    turns = rng.randrange(4)
    low, gt = torch.rot90(low, turns, (-2, -1)), torch.rot90(gt, turns, (-2, -1))
    return low.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)


def evaluate_pair(models, pairs, device, step, split='validation'):
    results = []
    cpu_fallbacks = {}
    for model in models.values():
        model.eval()
    with torch.inference_mode():
        for name, low_path, gt_path in pairs:
            low_cpu = _rgb(low_path).unsqueeze(0)
            gt = tensor_to_uint8(_rgb(gt_path))
            row = dict(split=split, step=step, image=name)
            for arm, model in models.items():
                if arm in cpu_fallbacks:
                    prediction = infer_full_image(cpu_fallbacks[arm], low_cpu)[0]
                else:
                    try:
                        prediction = infer_full_image(model, low_cpu.to(device))[0]
                    except torch.cuda.OutOfMemoryError:
                        if device.type != 'cuda':
                            raise
                        torch.cuda.empty_cache()
                        cpu_fallbacks[arm] = copy.deepcopy(model).cpu().eval()
                        prediction = infer_full_image(cpu_fallbacks[arm], low_cpu)[0]
                row[f'{arm}_psnr'], row[f'{arm}_ssim'] = image_metrics(tensor_to_uint8(prediction), gt)
            row['delta_psnr'] = row['saim_psnr'] - row['baseline_psnr']
            row['delta_ssim'] = row['saim_ssim'] - row['baseline_ssim']
            results.append(row)
    return results


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _exports(root, history, best):
    for name in ('learning_curves.csv', 'comparison_by_step.csv', 'per_image_results.csv',
                 'training_diagnostics.csv'):
        _csv(root / name, history[name])
    _csv(root / 'best_checkpoint_summary.csv', [dict(arm=a, **v) for a, v in best.items()])
    from PIL import ImageDraw
    rows = history['learning_curves.csv']
    for metric, filename in [('train_l1', 'loss_comparison.png'),
                             ('val_psnr', 'psnr_comparison.png'),
                             ('val_ssim', 'ssim_comparison.png')]:
        series = {arm: [(int(r['step']), float(r[metric])) for r in rows
                        if r['arm'] == arm and r[metric] != '']
                  for arm in ('baseline', 'saim')}
        values = [point for points in series.values() for point in points]
        if not values:
            continue
        xmax = max(1, max(x for x, _ in values))
        ymin, ymax = min(y for _, y in values), max(y for _, y in values)
        span = ymax - ymin or 1.0
        image = Image.new('RGB', (800, 460), 'white')
        draw = ImageDraw.Draw(image)
        draw.line([(65, 25), (65, 410), (760, 410)], fill='black', width=2)
        draw.text((65, 5), metric, fill='black')
        draw.text((650, 425), 'updates per arm', fill='black')
        draw.text((8, 25), f'{ymax:.4g}', fill='black')
        draw.text((8, 395), f'{ymin:.4g}', fill='black')
        for arm, color, label_y in [('baseline', '#2563eb', 35), ('saim', '#e11d48', 55)]:
            coords = [(65 + round(695*x/xmax), 410 - round(365*(y-ymin)/span))
                      for x, y in series[arm]]
            if len(coords) > 1:
                draw.line(coords, fill=color, width=3)
            for x, y in coords:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            draw.text((650, label_y), arm, fill=color)
        image.save(root / filename)


def _metrics(history, best, models, val_pairs, device, step, root):
    rows = evaluate_pair(models, val_pairs, device, step)
    history['per_image_results.csv'].extend(rows)
    means = {arm: {metric: float(np.mean([r[f'{arm}_{metric}'] for r in rows]))
                   for metric in ('psnr', 'ssim')} for arm in models}
    history['comparison_by_step.csv'].append(dict(
        step=step, baseline_psnr=means['baseline']['psnr'], saim_psnr=means['saim']['psnr'],
        delta_psnr=means['saim']['psnr']-means['baseline']['psnr'],
        baseline_ssim=means['baseline']['ssim'], saim_ssim=means['saim']['ssim'],
        delta_ssim=means['saim']['ssim']-means['baseline']['ssim']))
    for arm in models:
        if means[arm]['psnr'] > best[arm]['val_psnr']:
            filename = f'{arm}_best.pth'
            best[arm] = dict(step=step, val_psnr=means[arm]['psnr'],
                             val_ssim=means[arm]['ssim'], checkpoint=str(root / filename))
            _atomic_save(root / filename, dict(model=models[arm].state_dict(), step=step,
                                               val_psnr=means[arm]['psnr'], val_ssim=means[arm]['ssim']))
        history['learning_curves.csv'].append(dict(arm=arm, step=step, train_l1='', lr='',
                                                    val_psnr=means[arm]['psnr'],
                                                    val_ssim=means[arm]['ssim']))
    return means


def run_experiment(workspace, *, steps=100, dataset_root=None, checkpoint_path=None,
                   result_root=None, resume=False, config=None):
    if steps not in (100, 1000, 5000):
        raise ValueError('steps must be 100, 1000, or 5000')
    workspace = Path(workspace)
    dataset_root = Path(dataset_root or workspace / 'LOLv2/Real_captured')
    checkpoint_path = Path(checkpoint_path or workspace / 'pretrained_weights/LOL_v2_real.pth')
    result_root = Path(result_root or workspace / 'saim_experiment_results')
    root = result_root / f'updates_{steps:04d}'
    cfg = {**DEFAULTS, **(config or {})}
    cfg['val_interval'] = 100 if steps == 100 else 500
    if cfg['batch_size'] != 1:
        raise ValueError('This deterministic schedule requires batch_size=1')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Pretrained baseline checkpoint missing: {checkpoint_path}')
    train_pairs, val_pairs = prepare_split(dataset_root, cfg['seed'], cfg['val_fraction'])
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest = dict(format='saim-v3-1', steps=steps, config=cfg,
                    baseline_checkpoint=str(checkpoint_path.resolve()),
                    baseline_sha256=source_hash, train=_pairs_manifest(train_pairs),
                    validation=_pairs_manifest(val_pairs), test_policy='final evaluation only')
    root.mkdir(parents=True, exist_ok=True)
    last = root / 'last.pth'
    if last.exists() and not resume:
        raise FileExistsError(f'Run exists; set resume=True to continue: {last}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    args = dict(in_channels=3, out_channels=3, n_feat=cfg['n_feat'],
                stage=cfg['stage'], num_blocks=cfg['num_blocks'])
    models = {'baseline': Baseline(**args).to(device), 'saim': SAIM(**args).to(device)}
    report = load_baseline_checkpoint(models['saim'], checkpoint_path, baseline=models['baseline'])
    optimizers = {arm: torch.optim.Adam(model.parameters(), lr=cfg['lr'], betas=(.9, .999))
                  for arm, model in models.items()}
    schedulers = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=cfg['min_lr'])
                  for arm, opt in optimizers.items()}
    scalers = {arm: torch.amp.GradScaler('cuda', enabled=cfg['amp'] and device.type == 'cuda')
               for arm in models}
    history = {name: [] for name in FIELDS if name != 'best_checkpoint_summary.csv'}
    best = {arm: dict(step=-1, val_psnr=-math.inf, val_ssim=-math.inf,
                      checkpoint='') for arm in models}
    completed = 0
    if resume:
        if not last.exists():
            raise FileNotFoundError(last)
        saved = torch.load(last, map_location='cpu', weights_only=False)
        if saved['manifest'] != manifest:
            raise ValueError('Resume manifest differs from checkpoint')
        for arm in models:
            models[arm].load_state_dict(saved['models'][arm], strict=True)
            optimizers[arm].load_state_dict(saved['optimizers'][arm])
            schedulers[arm].load_state_dict(saved['schedulers'][arm])
            scalers[arm].load_state_dict(saved['scalers'][arm])
        history, best, completed = saved['history'], saved['best'], saved['completed']
        _restore_rng(saved['rng'])
    else:
        _json(root / 'run_manifest.json', {**manifest, 'checkpoint_compatibility': report})
        _metrics(history, best, models, val_pairs, device, 0, root)

    def save():
        _atomic_save(last, dict(manifest=manifest, completed=completed, history=history, best=best,
                                models={a: m.state_dict() for a, m in models.items()},
                                optimizers={a: o.state_dict() for a, o in optimizers.items()},
                                schedulers={a: s.state_dict() for a, s in schedulers.items()},
                                scalers={a: s.state_dict() for a, s in scalers.items()},
                                rng=_rng_state()))
        _exports(root, history, best)
    if not resume:
        save()
    start_completed = completed
    start = time.monotonic()
    with (root / 'progress.log').open('a', encoding='utf-8') as log:
        for step in range(completed + 1, steps + 1):
            low, gt = training_batch(train_pairs, step, cfg['seed'], cfg['crop_size'], device)
            for arm, model in models.items():
                model.train()
                optimizer, scaler = optimizers[arm], scalers[arm]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    loss = F.l1_loss(model(low), gt)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{arm} step {step}: non-finite loss')
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'], error_if_nonfinite=True)
                lr = optimizer.param_groups[0]['lr']
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    raise FloatingPointError(f'{arm} step {step}: AMP skipped optimizer update')
                schedulers[arm].step()
                history['training_diagnostics.csv'].append(dict(
                    arm=arm, step=step, loss=float(loss.detach()), lr=lr,
                    grad_norm=float(grad), amp_scale=scaler.get_scale()))
                history['learning_curves.csv'].append(dict(arm=arm, step=step,
                    train_l1=float(loss.detach()), lr=lr, val_psnr='', val_ssim=''))
            completed = step
            if step % cfg['val_interval'] == 0 or step == steps:
                _metrics(history, best, models, val_pairs, device, step, root)
                save()
            if step % cfg['log_interval'] == 0 or step == steps:
                elapsed = time.monotonic() - start
                remaining = int(elapsed / max(1, step - start_completed) * (steps-step))
                eta = f'{remaining//3600:02d}:{remaining//60%60:02d}:{remaining%60:02d}'
                line = f'TRAIN paired step={step}/{steps} loss={float(loss.detach()):.4f} TOTAL_ETA={eta}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
    _exports(root, history, best)
    return root


def evaluate_final_test(run_dir, dataset_root):
    """Call only after training; evaluate selected best checkpoints on held-out Test."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run_manifest.json').read_text(encoding='utf-8'))
    cfg = manifest['config']
    args = dict(in_channels=3, out_channels=3, n_feat=cfg['n_feat'], stage=cfg['stage'],
                num_blocks=cfg['num_blocks'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = {'baseline': Baseline(**args).to(device), 'saim': SAIM(**args).to(device)}
    for arm, model in models.items():
        state = torch.load(run_dir / f'{arm}_best.pth', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
    pairs = find_paired_images(Path(dataset_root) / 'Test/Low', Path(dataset_root) / 'Test/Normal')
    last_path = run_dir / 'last.pth'
    saved = torch.load(last_path, map_location='cpu', weights_only=False)
    if saved['completed'] != manifest['steps']:
        raise ValueError('Complete both model arms before final Test evaluation')
    rows = evaluate_pair(models, pairs, device, -1, split='test')
    saved['history']['per_image_results.csv'] = [
        row for row in saved['history']['per_image_results.csv'] if row['split'] != 'test'
    ] + rows
    _atomic_save(last_path, saved)
    _csv(run_dir / 'per_image_results.csv', saved['history']['per_image_results.csv'])
    summary = {key: float(np.mean([row[key] for row in rows])) for key in
               ('baseline_psnr', 'saim_psnr', 'delta_psnr', 'baseline_ssim', 'saim_ssim', 'delta_ssim')}
    _json(run_dir / 'final_test_summary.json', summary)
    return summary
