# RetinexFormer Project Memory

## Research context

This project compares the original author-pretrained RetinexFormer model with a
retrained model for low-light image enhancement.  The objective is to compare
not only final enhanced images, but also determine where a performance gap
becomes observable inside the model.

RetinexFormer has this conceptual flow:

```
Low-light input → illumination estimator → lit-up image / light-up feature
→ corruption restorer (IGT) → final enhanced output
```

The illumination estimator produces the lit-up image (`Ilu`) and light-up
feature (`Flu`).  Both feed the restoration process, with `Flu` also guiding
self-attention in the Illumination-Guided Transformer (IGT).

## Analysis performed

Author and retrained models were compared per test image at two stages:

1. Lit-up intermediate image
2. Final enhanced output

The measurements used PSNR, SSIM, and visual inspection.  Lit-up-to-ground
truth PSNR/SSIM are diagnostic measures only: ground truth is not a dedicated
lit-up target (see `EVALUATION.md` for the evaluation implementation details).

## Current evidence and interpretation

The performance gap is already observable at the lit-up stage.  The retrained
model consistently had lower lit-up PSNR and SSIM than the author model.

- Approximate lit-up PSNR delta range: `-0.2 dB` to `-1.4 dB`
- Approximate lit-up SSIM delta range: `-0.01` to `-0.13`
- Average lit-up PSNR delta: approximately `-0.65 dB`
- Average lit-up SSIM delta: approximately `-0.06`

The safe conclusion is:

> The degradation becomes observable at the illumination/light-up stage.

Do **not** claim that the illumination estimator is definitely the root cause:
the metrics establish that the gap is present at that point, not causality.

A stronger but still supportable interpretation is that the retrained model
produces a weaker intermediate lit-up representation, which is propagated to
the restoration stage through the lit-up image and illumination-derived
features.

The final output does not always degrade by as much as its corresponding
lit-up image.  This supports the interpretation that the corruption restorer /
IGT can partially compensate for errors exposed or introduced during light-up,
but does not completely remove the gap.  The paper expresses final enhancement
as residual restoration:

```
Ien = Ilu + Ire
```

Therefore the current conceptual result is:

```
Input → lit-up degradation appears → restoration partly recovers it → final output
```

## Metric interpretation

- Lower PSNR: greater pixel-level reconstruction error relative to ground
  truth at the diagnostic lit-up comparison stage.
- Lower SSIM: reduced structural fidelity, including local structure,
  contrast, edges, and textures.

The concurrent reduction in both PSNR and SSIM suggests the observed gap is
not merely a small global-brightness shift.

## Practical degradation groups

Use absolute deltas when grouping images:

```
|ΔPSNR| = ABS(retrained PSNR − author PSNR)
|ΔSSIM| = ABS(retrained SSIM − author SSIM)
```

| Group | Lit-up PSNR degradation | Lit-up SSIM degradation |
| --- | ---: | ---: |
| Small | `≤ 0.4 dB` | `≤ 0.03` |
| Moderate | `> 0.4` and `≤ 0.8 dB` | `> 0.03` and `≤ 0.07` |
| Large | `> 0.8 dB` | `> 0.07` |

Google Sheets formulas (where `C2` holds the signed delta):

```excel
=IF(ABS(C2)<=0.4,"Small degradation",IF(ABS(C2)<=0.8,"Moderate degradation","Large degradation"))
=IF(ABS(C2)<=0.03,"Small degradation",IF(ABS(C2)<=0.07,"Moderate degradation","Large degradation"))
```

These are practical working thresholds, not final statistical cutoffs.  They
can be refined using the empirical distribution or percentiles.

## Planned visual analysis

Analyze patterns across all images in each degradation group rather than
selecting a few examples.  Inspect:

- overall darkness and extremely dark-region proportion
- illumination uniformity, shadows, and mixed bright/dark regions
- strong local light sources and saturated highlights
- color cast and color variation
- local contrast and reflective surfaces
- complex textures versus low-texture dark regions

Working hypotheses to test, not conclusions:

| Group | Potential characteristics |
| --- | --- |
| Small | More uniform illumination, moderate darkness, fewer extreme dark areas |
| Moderate | Mixed bright/dark regions, uneven illumination, stronger shadows, complex exposure |
| Large | Very dark regions, highly non-uniform illumination, point lights, extreme contrast, difficult shadows or color variation |

The most informative cases for subsequent inspection are images with both
large PSNR and large SSIM degradation.  For example, if most Large-degradation
images have non-uniform lighting and Small-degradation images are mostly
uniformly lit, that would support the narrower hypothesis that the retrained
illumination estimator struggles with complex illumination distributions.

## Supervisor-level summary

> RetinexFormer first estimates illumination and produces a lit-up image and
> light-up feature before passing them to the restoration network. Compared
> with the author model, the retrained model already shows lower PSNR and SSIM
> at the lit-up stage. The performance gap therefore does not appear only at
> the final output stage; it becomes observable during illumination/light-up.
> The later corruption-restoration network appears to recover part of this
> degradation, but not the complete performance gap.

Key statement:

> The degradation becomes observable at the lit-up stage, while the subsequent
> restoration stage partially compensates for the loss.

## Direction for the proposed model

Prioritize improvements to the intermediate illumination/light-up
representation before changing the full architecture.  Investigate:

- illumination estimation and lit-up image quality
- illumination-feature quality
- structural preservation while lighting up
- reduced propagation of illumination-stage errors into IGT
- possible intermediate lit-up supervision or loss functions

Motivation: consistent degradation at the intermediate lit-up stage suggests
that strengthening illumination representation before restoration may improve
final PSNR and SSIM.
