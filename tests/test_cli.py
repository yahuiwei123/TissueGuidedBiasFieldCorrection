"""Portable command, physical-grid and synthetic-image regressions."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODULES = ("numpy", "scipy", "nibabel", "SimpleITK", "sklearn", "torch")
AVAILABLE = all(importlib.util.find_spec(name) for name in MODULES)


class HelpTests(unittest.TestCase):
    def test_all_entrypoints_offer_help_without_scientific_imports(self):
        entries = [["-m", "tissue_bfc"], [str(ROOT / "bfc_gauss_mixture.py")],
                   [str(ROOT / "bfc_rbf_scatter.py")]]
        for entry in entries:
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, "-S", *entry, "--help"], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--gm-core-mm", result.stdout)
                self.assertIn("--max_iter", result.stdout)


@unittest.skipUnless(AVAILABLE, "Command tests need the declared scientific dependencies")
class CommandTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np

        self.temporary = tempfile.TemporaryDirectory(prefix="tissue-bfc-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = np.zeros((32, 32, 32), dtype=np.uint8)
        self.labels[2:15, 2:30, 2:30] = 2
        self.labels[17:30, 2:30, 2:30] = 3
        self.mask = np.zeros_like(self.labels)
        self.mask[1:31, 1:31, 1:31] = 1
        self.image = np.where(self.labels == 2, 60, np.where(self.labels == 3, 110, 20)).astype(np.float32)
        self.image[self.mask == 0] = 7
        self.image[15, 10, 10] = 0
        self.affine = np.array([[-1, 0, 0, 32], [0, 0, 1, -32], [0, -1, 0, 32], [0, 0, 0, 1]], dtype=float)
        self.arguments = []
        for name, values in (("input_img", self.image), ("input_lab", self.labels), ("input_msk", self.mask)):
            path = self.root / (name + ".nii.gz")
            volume = nib.Nifti1Image(values, self.affine)
            volume.header.set_xyzt_units("mm")
            nib.save(volume, path)
            self.arguments += ["--" + name, str(path)]
        self.corrected = self.root / "result/corrected.nii.gz"
        self.field = self.root / "result/field.nii.gz"
        self.arguments += ["--output_img", str(self.corrected), "--output_bias", str(self.field)]

    def run_command(self, entry, *extra, environment=None):
        return subprocess.run([sys.executable, "-B", str(ROOT / entry), *self.arguments, *extra],
                              cwd=self.root, capture_output=True, text=True,
                              env=environment or os.environ.copy())

    def test_gaussian_legacy_filename_uses_reliable_cores_and_refuses_overwrite(self):
        import nibabel as nib
        import numpy as np

        inputs_before = [(self.root / (name + ".nii.gz")).read_bytes()
                         for name in ("input_img", "input_lab", "input_msk")]
        result = self.run_command("bfc_gauss_mixture.py", "--bias_max_iter", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        corrected = nib.load(self.corrected)
        np.testing.assert_allclose(corrected.get_fdata(), self.image, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(corrected.affine, self.affine, rtol=0, atol=1e-5)
        np.testing.assert_allclose(nib.load(self.field).get_fdata(), 1, atol=1e-6)
        report = json.loads((self.corrected.parent / "support.json").read_text())
        self.assertEqual(report["method"], "gaussian")
        self.assertEqual(report["implementation"], "reliable-core-v2")
        self.assertEqual(report["field_representation"], "multiplier")
        self.assertEqual(report["fitting"]["device"], "cpu")
        self.assertEqual(report["scientific_approval"], "not_performed")
        self.assertEqual(report["status"], "completed")
        support = nib.load(self.corrected.parent / "fit-mask.nii.gz").get_fdata() > 0
        self.assertFalse(np.any(support & ~np.isin(self.labels, [2, 3])))
        before = {path.name: path.read_bytes() for path in self.corrected.parent.iterdir()}
        repeated = self.run_command("bfc_gauss_mixture.py", "--bias_max_iter", "2")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("Refusing to overwrite", repeated.stderr)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.corrected.parent.iterdir()})
        for name, contents in zip(("input_img", "input_lab", "input_msk"), inputs_before):
            self.assertEqual((self.root / (name + ".nii.gz")).read_bytes(), contents)

    def test_rbf_alias_and_default_module_fail_without_cuda_before_creating_outputs(self):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
        result = self.run_command("bfc_rbf_scatter.py", "--max_iter", "2", environment=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA device is unavailable", result.stderr)
        self.assertFalse(self.corrected.parent.exists())
        result = subprocess.run([sys.executable, "-m", "tissue_bfc", *self.arguments],
                                cwd=self.root, env=environment, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA device is unavailable", result.stderr)
        self.assertFalse(self.corrected.parent.exists())

    def test_rbf_rejects_cpu_and_fixed_modality_targets(self):
        for extra, error in ((("--device", "cpu"), "CPU fallback is disabled"),
                             (("--modality", "T1"), "fixed modality targets are unsupported")):
            with self.subTest(extra=extra):
                result = self.run_command("bfc_rbf_scatter.py", *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(self.corrected.parent.exists())

    def test_invalid_priors_and_removed_bypass_flags_fail(self):
        for extra, error in ((("--refine_max_iter", "2"), "fixed tissue priors"),
                             (("--use_label_value", "3"), "unrecognized arguments")):
            with self.subTest(extra=extra):
                result = self.run_command("bfc_gauss_mixture.py", *extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(self.corrected.parent.exists())

    def test_grid_and_label_validation(self):
        import nibabel as nib
        import numpy as np
        from tissue_bfc.validation import checked_labels, same_grid

        reference = nib.Nifti1Image(self.image, self.affine)
        shifted = self.affine.copy()
        shifted[0, 3] += 1
        with self.assertRaisesRegex(ValueError, "same physical grid"):
            same_grid(reference, nib.Nifti1Image(self.image, shifted))
        with self.assertRaisesRegex(ValueError, "same physical grid"):
            same_grid(reference, nib.Nifti1Image(self.image[:-1], self.affine))
        invalid = self.image.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            same_grid(reference, nib.Nifti1Image(invalid, self.affine))
        fractional = self.labels.astype(np.float32)
        fractional[4, 4, 4] = 2.5
        with self.assertRaisesRegex(ValueError, "invalid or noninteger"):
            checked_labels(reference, nib.Nifti1Image(fractional, self.affine), [0, 1, 2, 3])
        invalid_mask = self.mask.copy()
        invalid_mask[4, 4, 4] = 2
        with self.assertRaises(ValueError):
            checked_labels(reference, nib.Nifti1Image(invalid_mask, self.affine), [0, 1])

    def test_failed_fit_is_recorded_without_scientific_approval(self):
        from tissue_bfc.cli import main

        with patch("tissue_bfc.support.gaussian_field", side_effect=RuntimeError("synthetic fit failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic fit failure"):
                main([*self.arguments, "--method", "gaussian"])
        report = json.loads((self.corrected.parent / "support.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["scientific_approval"], "not_performed")
        self.assertIn("synthetic fit failure", report["error"])
        self.assertFalse(self.corrected.exists())

    def test_cuda_device_selection_never_falls_back(self):
        from tissue_bfc.support import cuda_device

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(cuda_device("1"), "cuda:1")
            self.assertEqual(cuda_device("cuda"), "cuda:0")
            with self.assertRaises(RuntimeError):
                cuda_device("cuda:2")
            for device in ("cpu", "auto", "cuda:-1"):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    cuda_device(device)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                cuda_device("cuda:0")


@unittest.skipUnless(AVAILABLE, "Synthetic integration needs the declared scientific dependencies")
class SyntheticIntegrationTests(unittest.TestCase):
    def test_gaussian_synthetic_t1_t2(self):
        self.exercise("gaussian")

    @unittest.skipUnless(os.environ.get("TISSUE_BFC_TEST_CUDA"), "Set TISSUE_BFC_TEST_CUDA for real CUDA integration")
    def test_rbf_synthetic_t1_t2_on_cuda(self):
        self.exercise("rbf")

    def exercise(self, method):
        import nibabel as nib
        import numpy as np
        from tissue_bfc.cli import main

        with tempfile.TemporaryDirectory(prefix="tissue-bfc-synthetic-") as temporary:
            root = Path(temporary)
            axes = np.indices((64, 64, 64), dtype=np.float32)
            radius = np.sqrt(np.sum((axes - 31.5) ** 2, axis=0))
            labels = np.zeros(radius.shape, dtype=np.uint8)
            labels[radius < 25] = 1
            labels[radius < 22] = 2
            labels[radius < 16] = 3
            mask = (labels > 0).astype(np.uint8)
            labels[22:27, 28:34, 28:34] = 0
            labels[37:42, 28:34, 28:34] = 0
            unknown = (labels == 0) & (mask > 0)
            white, gray = labels == 3, labels == 2
            affine = np.array([[-1, 0, 0, 32], [0, 0, 1, -32], [0, -1, 0, 32], [0, 0, 0, 1]], dtype=float)
            label_path, mask_path = root / "labels.nii.gz", root / "mask.nii.gz"
            for path, values in ((label_path, labels), (mask_path, mask)):
                volume = nib.Nifti1Image(values, affine)
                volume.header.set_xyzt_units("mm")
                nib.save(volume, path)
            original_labels = label_path.read_bytes()
            for modality, means in (("T1", (20, 60, 110)), ("T2", (220, 80, 35))):
                with self.subTest(method=method, modality=modality):
                    truth = np.where(labels == 1, means[0], np.where(gray, means[1], np.where(white, means[2], means[2] * 1.2)))
                    known_field = np.exp(0.4 * (axes[0] - 31.5) / 31.5 + 0.2 * (axes[2] - 31.5) / 31.5)
                    noise = np.random.default_rng(17).normal(0, 0.4, truth.shape)
                    source = np.where(mask, np.maximum(truth * known_field + noise, 0), 0).astype(np.float32)
                    input_path = root / (modality + ".nii.gz")
                    volume = nib.Nifti1Image(source, affine)
                    volume.header.set_xyzt_units("mm")
                    nib.save(volume, input_path)
                    output = root / modality / "corrected.nii.gz"
                    field_path = output.parent / "field.nii.gz"
                    arguments = ["--method", method, "--input_img", str(input_path), "--input_lab", str(label_path),
                                 "--input_msk", str(mask_path), "--output_img", str(output), "--output_bias", str(field_path)]
                    if method == "rbf":
                        arguments += ["--device", os.environ["TISSUE_BFC_TEST_CUDA"], "--max_iter", "25"]
                    main(arguments)
                    corrected_volume = nib.load(output)
                    corrected = corrected_volume.get_fdata()
                    field = nib.load(field_path).get_fdata()
                    report = json.loads((output.parent / "support.json").read_text())
                    self.assertEqual(report["status"], "completed")
                    self.assertEqual(report["implementation"], "reliable-core-v2")
                    self.assertEqual(report["field_representation"], "divisor" if method == "rbf" else "multiplier")
                    self.assertEqual(report["fitting"]["soft_weight_outside_fit_support"], 0)
                    np.testing.assert_allclose(corrected_volume.affine, affine, rtol=0, atol=1e-5)
                    self.assertTrue(np.isfinite(field).all())
                    self.assertTrue(np.all(field[mask > 0] > 0))
                    np.testing.assert_array_equal(field[mask == 0], 1)
                    np.testing.assert_array_equal(corrected[mask == 0], source[mask == 0])
                    expected = source / field if method == "rbf" else source * field
                    np.testing.assert_allclose(corrected, expected, rtol=1e-5, atol=1e-5)
                    self.assertGreater(float(np.max(np.abs(corrected[unknown] - source[unknown]))), 0.1)
                    before = float(source[white].std() / source[white].mean())
                    after = float(corrected[white].std() / corrected[white].mean())
                    self.assertLess(after, before, (modality, before, after))
                    support = nib.load(output.parent / "fit-mask.nii.gz").get_fdata() > 0
                    self.assertFalse(np.any(support & ~(white | gray)))
                    if method == "rbf":
                        self.assertTrue(report["fitting"]["device"].startswith("cuda:"))
                        self.assertGreater(report["fitting"]["peak_gpu_allocated_bytes"], 0)
                        self.assertTrue((output.parent / "fitted-control-points.npz").is_file())
                    self.assertEqual(label_path.read_bytes(), original_labels)


if __name__ == "__main__":
    unittest.main()
