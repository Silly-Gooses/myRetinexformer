"""Regenerate the GitHub-connected Model B Colab notebook without output cells."""

import json
from pathlib import Path


def markdown(source):
    return dict(cell_type="markdown", metadata={}, source=source.splitlines(True))


def code(source):
    return dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                source=source.splitlines(True))


cells = [
    markdown("""# Model B Option A: GitHub to Colab reproduction

Runs the repository implementation of the original Batch-8 / Patch-128 / LR 1e-5 experiment. Run top to bottom on a CUDA runtime. The original 69 images are held-out validation images from `Train`, not the official Test set. Only load checkpoints that you created and trust.
"""),
    code("""# 1. Runtime and pinned repository state
import os, subprocess, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > GPU'
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda)
print('GPU capability:', torch.cuda.get_device_capability(0))
from pathlib import Path
REPO = Path('/content/myRetinexformer')
if not (REPO / '.git').exists():
    subprocess.run(['git', 'clone', '--branch', 'model-b', '--single-branch',
                    'https://github.com/Silly-Gooses/myRetinexformer.git', str(REPO)], check=True)
else:
    subprocess.run(['git', 'fetch', 'origin', 'model-b'], cwd=REPO, check=True)
    subprocess.run(['git', 'checkout', 'model-b'], cwd=REPO, check=True)
    subprocess.run(['git', 'reset', '--hard', 'origin/model-b'], cwd=REPO, check=True)
COMMIT = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
print('Repository:', REPO, '| branch: model-b | resolved commit:', COMMIT)
"""),
    code("""# 2. Install the reference notebook's Colab dependencies and import Model B
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'einops', 'Pillow', 'numpy', 'matplotlib', 'opencv-python-headless'],
               check=True)
sys.path.insert(0, str(REPO))
from model_b.model import build_model, ARCH
from model_b.data import load_pairs, evaluate_model
from model_b.experiment import (load_reference_checkpoint, smoke_test,
                                run_option_a, sha256)
from model_b.compare import compare_evaluation, compare_training
model, replaced = build_model()
print('Model B replacements:', len(replaced), replaced)
print('Parameters:', sum(parameter.numel() for parameter in model.parameters()))
del model
"""),
    code("""# 3. Mount Drive and locate the same data and checkpoint ancestry
from google.colab import drive
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive')
DATA_ROOT = DRIVE / 'datasets/LOLv2/Real_captured'
SCREENING = DRIVE / 'RetinexFormer_NAF_Screening'
SPLIT_FILE = SCREENING / 'model_B_v1_full_data_1000steps/split.json'
SOURCE_BEST = SCREENING / 'model_B_batch8_patch128_from_best1500/batch_8_patch_128_lr2e-5_1000steps/best.pth'
REFERENCE_BEST = SCREENING / 'model_B_optionA_batch8_patch128_lr1e-5_from_batch8best/low_lr_1e-5_1000steps/best.pth'
OUTPUT_ROOT = SCREENING / 'model_B_github_reproduction'
RUN_DIR = OUTPUT_ROOT / 'low_lr_1e-5_1000steps'
for path in (SPLIT_FILE, SOURCE_BEST):
    assert path.is_file(), f'Missing required reference file: {path}'
assert DATA_ROOT.is_dir(), f'Missing dataset: {DATA_ROOT}'
assert not OUTPUT_ROOT.resolve().is_relative_to(SOURCE_BEST.parent.resolve())
print('Dataset:', DATA_ROOT)
print('Source:', SOURCE_BEST)
print('Reference Option A best available:', REFERENCE_BEST.is_file())
print('New output:', RUN_DIR)
"""),
    code("""# 4. Validate the exact paired 620/69 split and checkpoint ancestry
train_pairs, val_pairs, train_names, val_names = load_pairs(DATA_ROOT, SPLIT_FILE)
print('Paired split:', len(train_pairs), 'train /', len(val_pairs), 'validation')
print('Official Test images used:', 0)
source_model, source_ckpt, source_replaced = load_reference_checkpoint(
    SOURCE_BEST, train_names, val_names, kind='source')
print('Source local step:', source_ckpt['local_step'])
print('Source SHA256:', sha256(SOURCE_BEST))
print('Strict source load: OK;', len(source_ckpt['model']), 'state tensors')
print('Source ancestry step:', source_ckpt['config']['source_step'])
"""),
    code("""# 5. Batch-8 forward/backward smoke and unchanged 69-image evaluation
smoke = smoke_test(source_model)
print('Smoke:', smoke)
source_model = source_model.cuda()
source_metrics, source_rows = evaluate_model(source_model, val_pairs)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
source_report = compare_evaluation(source_metrics, source_rows,
                                   reference='source', output_dir=OUTPUT_ROOT)
print('Source checkpoint comparison:', source_report)
assert source_report['aggregate_within_tolerance'], 'Source checkpoint evaluation differs from reference'
del source_model
torch.cuda.empty_cache()
if REFERENCE_BEST.is_file():
    reference_model, reference_ckpt, _ = load_reference_checkpoint(
        REFERENCE_BEST, train_names, val_names, kind='option_a')
    reference_model = reference_model.cuda()
    reference_metrics, reference_rows = evaluate_model(reference_model, val_pairs)
    original_report = compare_evaluation(reference_metrics, reference_rows,
                                         reference='best', output_dir=OUTPUT_ROOT)
    print('Original Option A best comparison:', original_report)
    assert original_report['aggregate_within_tolerance'], 'Original best evaluation differs from reference'
    del reference_model
    torch.cuda.empty_cache()
else:
    print('Original Option A best checkpoint unavailable; its direct evaluation is unresolved.')
"""),
    markdown("""## 6. Re-run Option A training

This starts from the verified Batch-8 best **model weights** with fresh Adam. It evaluates local step 0 and then runs 1,000 updates. Existing output checkpoints resume their saved optimizer, scaler, and RNG; the original sampler cursor was not saved. The training module writes checkpoints, CSVs, and a live log to the new Drive folder.
"""),
    code("""RUN_DIR_RESULT = run_option_a(SOURCE_BEST, RUN_DIR, train_pairs, val_pairs,
                              train_names, val_names, DATA_ROOT, COMMIT)
print('Option A run:', RUN_DIR_RESULT)
"""),
    code("""# 7. Final 69-image evaluation and automated comparison report
import json
final_model, final_ckpt, _ = load_reference_checkpoint(
    RUN_DIR_RESULT / 'best.pth', train_names, val_names, kind='option_a')
final_model = final_model.cuda()
final_metrics, final_rows = evaluate_model(final_model, val_pairs)
final_report = compare_evaluation(final_metrics, final_rows,
                                  reference='best', output_dir=RUN_DIR_RESULT)
training_report = compare_training(RUN_DIR_RESULT, output_dir=RUN_DIR_RESULT)
summary = dict(repo_commit=COMMIT, source_sha256=sha256(SOURCE_BEST),
               best_local_step=int(final_ckpt['local_step']),
               final_evaluation=final_report, training_comparison=training_report)
(RUN_DIR_RESULT / 'reproducibility_summary.json').write_text(
    json.dumps(summary, indent=2), encoding='utf-8')
print('Reference: 21.0366 dB PSNR / 0.7355 SSIM')
print('Reproduced:', round(final_metrics['psnr'], 4), 'dB PSNR /',
      round(final_metrics['ssim'], 4), 'SSIM')
print('Differences:', final_report['delta_psnr'], 'dB /',
      final_report['delta_ssim'], 'SSIM')
print('Best step:', training_report['best_local_step'],
      '| reference best step:', training_report['reference_best_local_step'])
print('Comparison files:', RUN_DIR_RESULT)
"""),
]

for index, cell in enumerate(cells):
    cell["id"] = f"model-b-{index:02d}"

notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3",
                 language="python", name="python3"),
                 language_info=dict(name="python"),
                 colab=dict(name="ModelB_GitHub_Batch8_Patch128_LR1e5_Colab.ipynb")),
                nbformat=4, nbformat_minor=5)
target = Path(__file__).with_name("ModelB_GitHub_Batch8_Patch128_LR1e5_Colab.ipynb")
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                  encoding="utf-8")
print(target)
