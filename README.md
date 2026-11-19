# Tissue-Guided Bias Field Correction

Reliable-core Gaussian and CUDA RBF bias correction for brain MRI. This revision
ports the `reliable-core-v2` implementation from FastPrep BFC 1.1.0 back into the
original repository. **RBF is the default; it requires CUDA and never silently
falls back to CPU.** Gaussian is an explicit CPU alternative.

This is a research implementation, not a claim of clinical or whole-brain
validation. The previously reviewed Gaussian example still showed local
cerebellar/brainstem overcorrection. RBF improvements in one real T1 example do
not establish correctness for every anatomy, session, or real T2 acquisition.

## What changed

- Fit only independent, physically eroded GM/WM cores, with within-class log-MAD
  outlier rejection; retain the original tissue labels.
- Keep soft tissue weights inside their own reliable cores; do not invent dark
  background observations from unmodeled brain tissue.
- Apply the fitted field continuously throughout the brain mask, including
  tissue label zero. Preserve outside-brain and zero-signal intensities.
- Use actual nearest supported donors for Gaussian residual extrapolation.
- Use stable softmax RBF weights in fitting and export, with export gradients
  disabled and explicit CUDA validation.
- In version 1.1.0, parameterize RBF controls as
  `1e-6 + (100 - 1e-6) * sigmoid(logit)` during both optimization and export.
  This keeps the effective controls positive and bounded, rather than clipping
  a failed field after fitting. The Gaussian algorithm is unchanged in this release.
- Validate grids, finite values, labels and fields, refuse output overwrites,
  and save fitting support and execution provenance.

See [the detailed comparison and migration notes](docs/RELIABLE_CORE_V2.md).
The original scripts and their historical README remain under
`tissue_bfc/_legacy/` and [docs/README-legacy.md](docs/README-legacy.md), respectively.
The original figure and example notebook concern the historical method, not
independent validation of this revision.

## Installation

Use Python 3.10 or newer and a PyTorch installation with CUDA support for RBF:

```bash
python -m pip install .
```

The package declares NumPy, SciPy, nibabel, SimpleITK, scikit-learn and PyTorch.
Installing the package alone does not guarantee a working CUDA runtime; RBF
checks the selected device before creating outputs. Gaussian runs on CPU.
No FreeSurfer, Nextflow, FastPrep checkout or external model weights are needed.

## Input contract

Provide nonnegative, finite, 3D NIfTI images on the same physical grid, with
voxel spacing expressed in millimeters. The brain mask must contain only 0/1.
Tissue labels must be **0 = unmodeled/background, 1 = CSF, 2 = GM, 3 = WM**.
Do not pass raw FreeSurfer/SynthSeg aseg label numbers as these tissue codes.
Label zero inside the brain mask is valid and is corrected, but never fitted.
Both GM and WM must retain enough reliable samples; failure is explicit.

## Usage

RBF, preserving fitted tissue contrast for either T1- or T2-weighted inputs:

```bash
python -m tissue_bfc \
  --input_img T1w.nii.gz --input_lab tissue_labels.nii.gz \
  --input_msk brain_mask.nii.gz \
  --output_img new-rbf/corrected.nii.gz --output_bias new-rbf/field.nii.gz \
  --device cuda:0
```

Gaussian, also preserving fitted contrast by default:

```bash
python -m tissue_bfc --method gaussian \
  --input_img T1w.nii.gz --input_lab tissue_labels.nii.gz \
  --input_msk brain_mask.nii.gz \
  --output_img new-gaussian/corrected.nii.gz --output_bias new-gaussian/field.nii.gz
```

`python bfc_rbf_scatter.py` and `python bfc_gauss_mixture.py` accept the same
arguments and select their respective methods. Installed commands are
`tissue-bfc`, `bfc-rbf`, and `bfc-gaussian`.

| Parameter | Default / meaning |
| --- | --- |
| `--gm-core-mm`, `--wm-core-mm` | 1 / 2 mm; independent tissue-core erosion |
| `--core-log-mad-limit` | 3.5 scaled MAD; fitting outlier rejection, not a QC threshold |
| `--seed` | 0 |
| `--device` | `cuda:0` for RBF; visible-device numbering follows `CUDA_VISIBLE_DEVICES` |
| `--n_points`, `--k_neighbors` | 48 / 48 |
| `--rbf_max_iter`, `--max_iter` | 25; the latter is a compatibility alias |
| `--rbf_sigma`, `--soft_sigma`, `--tol` | 0.05 / 0.5 / 1e-6 |
| `--bias_max_iter`, `--max_kernel` | Gaussian: 10 / 9 |
| `--min_kernel`, `--label_soft_kernel` | Gaussian: spacing-derived / 0.4 |
| `--refine_max_iter` | Must be 0; tissue priors remain fixed |
| `--modality` | `None` preserves contrast; fixed `T1` / `T2` targets are Gaussian-only |

`--modality T1/T2` is not an input-modality declaration: it requests Gaussian
fixed intensity targets and is not supported by RBF. Gaussian smoothing widths
are in voxel units; the tissue-core erosion distances are in millimeters.
Removed historical flags are described in the migration notes and are not
silently ignored.

## Outputs and field conventions

Every attempt writes the requested corrected image and field, plus
`fit-mask.nii.gz`, `fit-labels.nii.gz` and `support.json` next to the corrected
image. RBF also writes `fitted-control-points.npz`. Use a new output directory
for every attempt; existing outputs are never overwritten.

In 1.1.0, the RBF archive retains actual coefficients in `point_values` and adds
the optimized `point_logits`. Provenance records model revision
`bounded-sigmoid-v1` and field bounds. Initialization still corresponds to
coefficient 1. The weighted interpolation is bounded against floating-point
roundoff, and the exported field must be finite, positive and at most 100.
These bounds concern the raw RBF divisor, not the corrected image or an inverse
multiplier; they do not establish safe extrapolation or eliminate the need for QC.

**The raw field conventions differ:**

- RBF exports a **divisor**: `corrected = original / field` inside the brain.
- Gaussian exports a **multiplier**: `corrected = original * field` inside the brain.
- Both export field 1 outside the brain and preserve the original signal there.
- `support.json` records `field_representation`, the actual fitting device,
  parameters, fitting support, status, and `scientific_approval: not_performed`.

The standalone outputs do not include WM110 global normalization. FastPrep's
separate adapter converts the RBF field to a multiplier before publishing a
bias factor; do not confuse that factor with this CLI's raw field.

## Tests

```bash
python -m unittest discover -s tests -v
TISSUE_BFC_TEST_CUDA=cuda:0 python -m unittest discover -s tests -v
```

CUDA integration is opt-in and exercises synthetic T1/T2 images. Unit and
synthetic integration tests are not substitutes for reviewing corrected images
on common grids and display scales. See [validation notes](docs/VALIDATION.md).

## Citation and historical material

The original methods form part of the MacaSurfer cortical analysis pipeline.
Retain the original citation when using this work:

> Wei, Y. et al. *MacaSurfer: unified surface-volume mapping of the macaque brain across the lifespan.* 2026.06.14.732101 Preprint at https://doi.org/10.64898/2026.06.14.732101 (2026).

The original example images, notebook, and figure remain in the repository.
The notebook has not been migrated to the reliable-core CLI. No new license
terms are introduced by this port; the original snapshot contained no separate
license file.
