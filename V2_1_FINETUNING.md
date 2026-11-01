# V2.1 low learning rate fine-tuning

`Adaptive_Edge_V2_1_Colab.ipynb` is the separate Colab workflow. Its setup cell
checks out the `v2.1` branch; push that branch before using the notebook in a
fresh Colab runtime. Mount Drive, set `DATASET_ROOT`, `OUTPUT_ROOT`, and
`ORIGINAL_ROOT`, then run the source and storage preflight. The source must be
the **gated V2 epoch-200 run**. Its selected best snapshot and metadata are
found through `checkpoints/epoch_0200.pth` (or an epoch-200 `last.pth`). No
checkpoint is included in this repository.

The notebook's smoke and both full experiment cells are opt-in. Run the
isolated two-update smoke first. Then enable LR / 5 and LR / 10 separately.
Each starts from the same original best model weights, with a fresh Adam
optimizer; neither starts from the other. Set that experiment's `RESUME_*` flag
after a Colab disconnect to restore its own `latest.pth`. Do not use the
original run's resume checkpoint for low-LR fine-tuning.

The original peak LR is `1e-4`. LR / 5 starts at `2e-5`; LR / 10 starts at
`1e-5`. Each uses a 30-additional-epoch cosine decay to one-tenth of its
starting LR, with validation every five epochs. Maximum epochs, validation
interval, early stopping patience, and minimum improvement are configurable.
Set patience to `0` to disable early stopping.
The original best checkpoint remains the selected fallback if fine-tuning
does not improve validation PSNR. SSIM is reported but does not select a model.

Outputs on Drive are isolated under `v2.1/finetune_lr5` and
`v2.1/finetune_lr10`. Each has `checkpoints/best.pth` and `latest.pth`,
`logs/finetune.csv`, `logs/validation_per_image.csv`, `source.json`, and
`selection.json`, plus four training and validation curves in `visuals/`. The CSV includes the original source epoch, batch, and step,
plus additional and nominal total epochs. If the best snapshot was taken
mid-epoch, `total_epoch` is a label, not an exact count of completed passes.
The source hash is checked again after the run. The notebook's comparison
cell writes `v2.1/comparison/comparison.csv`, JSON, and plots. The CLI
comparison also saves `v2.1/comparison/comparison.png`.

For CLI use in Colab or a machine with the original checkpoint and dataset:

```bash
python finetune_v2_1.py --original-root /drive/original_v2_run --output-root /drive/experiments --preflight
python finetune_v2_1.py --original-root /drive/original_v2_run --output-root /drive/experiments --experiment finetune_lr5 --smoke
python finetune_v2_1.py --original-root /drive/original_v2_run --output-root /drive/experiments --experiment finetune_lr5
python finetune_v2_1.py --original-root /drive/original_v2_run --output-root /drive/experiments --experiment finetune_lr10
python finetune_v2_1.py --original-root /drive/original_v2_run --output-root /drive/experiments --compare
```

To resume, repeat the corresponding experiment command with `--resume` and
identical settings. Smoke output is separate; for repeated CLI smoke runs,
choose a fresh `--output-root`. The full experiments are never launched by
preflight or comparison.

The original 689 Train pairs remain training data, and the original 100 Test
pairs remain the **validation and checkpoint-selection set**. Thus the
historical 20.73 dB and new scores are comparable validation scores, not
independent held-out test estimates. The comparison records the original
score as checkpoint metadata and new scores as measured validation. If a
separate held-out estimate is needed, it requires new untouched data.
