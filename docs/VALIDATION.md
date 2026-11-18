# Port validation — 2026-11-18

## Source comparison

- Base: `f6983fc0308283b1c4574782613222bfa28cd084` on the original `main` branch.
- Both original solver files match the FastPrep upstream snapshot byte for byte.
- Both files are preserved byte for byte under `tissue_bfc/_legacy/`.
- `support.py`, `validation.py` and `__main__.py` match FastPrep BFC 1.0.0.
- The CLI changes are limited to bundled-source lookup, method-specific entry
  functions, the original RBF `--max_iter` alias, and device help text.
- Existing example resources, the historical notebook and original citation are
  retained. No private MRI inputs or FastPrep run products are included.

## Executed checks

The complete 30-test suite was covered in two runs:

```bash
python -m unittest discover -s tests -v
PYTHONPATH=tests:. TISSUE_BFC_TEST_CUDA=cuda:0 \
  python -m unittest test_cli.SyntheticIntegrationTests.test_rbf_synthetic_t1_t2_on_cuda -v
```

- First run: 30 discovered, **29 passed**, one opt-in CUDA integration test
  skipped, 15.967 seconds.
- Second run: the previously skipped CUDA integration test **passed**, including
  both synthetic T1 and T2 images, 39.825 seconds, on an NVIDIA L20.
- RBF reported `cuda:0` for both fits and nonzero CUDA allocation. GPU numbering
  was remapped with `CUDA_VISIBLE_DEVICES`; this was not a mock GPU test.
- GPU validation acquired the same serialization lock as the existing batch
  pipeline, without changing its configuration or restarting its processes.

Coverage includes independent physical tissue erosion; exclusion of abnormal
core samples; no soft-weight leakage; true nearest donors; correction of
unmodeled brain tissue; zero/outside-brain preservation; finite positive fields;
stable RBF interpolation and gradients; explicit CUDA failures; unchanged input
files; physical-grid and label validation; failed-fit provenance; no output
overwrite; the original script names; and the default/alias CLI behavior.

Synthetic integration uses 64³ images with known smoothly varying multiplicative
bias, T1/T2 tissue contrasts, noise, and internal label-zero regions. Both methods
reduce white-matter CV in these examples, preserve spatial geometry and apply
the exported field according to the documented multiplier/divisor convention.
This checks the implementation, not anatomical or clinical accuracy.

## Independent package check

```bash
python -m pip wheel --no-build-isolation --no-deps --wheel-dir /tmp/bfc-wheels .
```

The wheel built successfully. It was extracted to a temporary directory and
run from outside the source checkout using only the extracted package plus the
existing scientific environment. A real Gaussian fit of a constant-tissue,
anisotropic NIfTI image passed; its provenance pointed to the packaged solver,
not to the source tree or FastPrep. Both solvers and all three console-script
entry definitions were present in the wheel. No models or MRI example resources
are required by the installed package.

## Environment and limits

Tested with Python 3.11, NumPy 1.26.4, SciPy 1.11.4, nibabel 5.3.3,
SimpleITK 2.3.0, scikit-learn 1.9.1, and PyTorch 2.8.0+cu128. These are observed
test versions, not a dependency lock or a guarantee for every version combination.
A fresh dependency installation on a different machine was not tested.

These checks do **not** approve any real subject's image quality. Gaussian
previously retained local cerebellar/brainstem overcorrection in a reviewed real
T1 case. RBF's earlier local improvement does not establish whole-brain,
cross-session, or real-T2 validation. The ongoing batch run is separate from
this repository port and is not evidence of completed scientific review.
