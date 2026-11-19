"""Portable regressions for fitting support and correction application."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy"))


@unittest.skipUnless(AVAILABLE, "BFC support tests require NumPy and SciPy")
class SupportTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.labels = np.zeros((24, 24, 24), dtype=np.uint8)
        self.labels[2:11, 2:22, 2:22] = 2
        self.labels[13:22, 2:22, 2:22] = 3
        self.labels[11:13, 2:22, 2:22] = 1
        self.brain = np.zeros_like(self.labels)
        self.brain[1:23, 1:23, 1:23] = 1
        self.image = np.where(self.labels == 2, 60, np.where(self.labels == 3, 110, 25)).astype(np.float32)
        self.image[~self.brain.astype(bool)] = 0

    def support(self, **kwargs):
        from tissue_bfc.support import reliable_support

        return reliable_support(self.image, self.labels, self.brain, (1, 1, 1), **kwargs)[0]

    def test_support_is_class_specific_and_cannot_close_over_csf_or_unknown(self):
        import numpy as np

        self.labels[17, 12, 12] = 1
        self.labels[5, 12, 12] = 0
        support = self.support()
        self.assertFalse(np.any(support & ~np.isin(self.labels, [2, 3])))
        self.assertFalse(support[17, 12, 12])
        self.assertFalse(support[18, 12, 12])
        self.assertFalse(support[5, 12, 12])

    def test_erosion_uses_physical_spacing_and_excludes_nonpositive_samples(self):
        from tissue_bfc.support import reliable_support

        self.image[17, 12, 12] = 0
        support = self.support()
        wider, _ = reliable_support(self.image, self.labels, self.brain, (2, 2, 2))
        self.assertGreater(wider.sum(), support.sum())
        self.assertFalse(support[17, 12, 12])
        self.assertFalse(support[18, 12, 12])

    def test_soft_weights_never_leak_into_unmodeled_or_other_tissue(self):
        import numpy as np
        from tissue_bfc.support import restricted_probabilities

        support = self.support()
        weights = restricted_probabilities(self.labels, support, 2.0)
        self.assertTrue(np.all(weights[:, ~support] == 0))
        self.assertTrue(np.all(weights[0, self.labels != 2] == 0))
        self.assertTrue(np.all(weights[1, self.labels != 3] == 0))
        self.assertTrue(np.all(weights.sum(axis=0)[support] > 0))

    def test_low_intensity_mislabeled_wm_core_is_not_a_field_donor(self):
        from tissue_bfc.support import reliable_support

        self.image[17, 12, 12] = 40
        support, report = reliable_support(self.image, self.labels, self.brain, (1, 1, 1))
        self.assertFalse(support[17, 12, 12])
        self.assertEqual(report["classes"]["3"]["intensity_outliers_excluded"], 1)
        self.assertEqual(self.image[17, 12, 12], 40)

    def test_donors_are_actual_nearest_supported_voxels_not_reflected_indices(self):
        import numpy as np
        from tissue_bfc.support import nearest_support_indices

        support = np.zeros((9, 9, 9), dtype=bool)
        support[4, 4, 4] = True
        residual = np.full(support.shape, -0.2)
        residual[support] = 0.1
        donors = nearest_support_indices(support, (1, 1, 1))
        filled = residual[tuple(donors)]
        self.assertTrue(np.all(filled == 0.1))
        self.assertEqual(float(filled[5, 4, 4]), 0.1)

    def test_deep_label_zero_is_corrected_without_clipping_or_restoring_input(self):
        import numpy as np
        from tissue_bfc.support import apply_field

        field = np.full(self.image.shape, 1.5, dtype=np.float32)
        unknown = (self.labels == 0) & (self.brain > 0)
        self.image[unknown] = 300
        result = apply_field(self.image, field, self.brain)
        divided = apply_field(self.image, 1 / field, self.brain, "divisor")
        self.assertTrue(np.all(result[unknown] == 450))
        np.testing.assert_allclose(result, divided, rtol=1e-6)
        np.testing.assert_array_equal(result[self.brain == 0], self.image[self.brain == 0])
        self.assertTrue(np.all(result[self.image == 0] == 0))

    def test_field_validation_uses_application_brain_not_fit_domain_or_entire_box(self):
        import numpy as np
        from tissue_bfc.support import application_field

        field = np.ones(self.image.shape)
        field[self.brain == 0] = -1
        applied = application_field(field, self.brain)
        self.assertTrue(np.all(applied[self.brain == 0] == 1))
        field[1, 1, 1] = 0
        self.assertEqual(self.labels[1, 1, 1], 0)
        self.assertEqual(self.brain[1, 1, 1], 1)
        with self.assertRaisesRegex(ValueError, "application domain"):
            application_field(field, self.brain)

    def test_invalid_inputs_fail_without_support_fallback(self):
        import numpy as np
        from tissue_bfc.support import apply_field, reliable_support, nearest_support_indices

        with self.assertRaisesRegex(ValueError, "Insufficient"):
            self.support(gm_core_mm=100)
        with self.assertRaises(ValueError):
            self.support(wm_core_mm=-1)
        invalid = self.labels.astype(float)
        invalid[5, 5, 5] = 2.5
        with self.assertRaises(ValueError):
            reliable_support(self.image, invalid, self.brain, (1, 1, 1))
        with self.assertRaises(ValueError):
            nearest_support_indices(np.zeros_like(self.brain, dtype=bool), (1, 1, 1))
        for bad in (0, -1, np.nan, np.inf):
            with self.subTest(field_value=bad), self.assertRaises(ValueError):
                apply_field(self.image, np.full(self.image.shape, bad), self.brain)


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec("SimpleITK") and importlib.util.find_spec("sklearn"),
                     "Gaussian regression requires the upstream scientific dependencies")
class GaussianTests(SupportTests):
    def gaussian(self, image, support):
        from tissue_bfc.support import gaussian_field

        specification = importlib.util.spec_from_file_location("gaussian_test", ROOT / "tissue_bfc/_legacy/bfc_gauss_mixture.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return gaussian_field(module, image, self.labels, support, (1, 1, 1), max_iter=3, max_kernel=3, min_kernel=2)[0]

    def test_uniform_tissues_produce_identity_not_nan(self):
        import numpy as np

        field = self.gaussian(self.image, self.support())
        np.testing.assert_allclose(field, 1, atol=1e-6)

    def test_excluded_intensity_changes_cannot_drive_the_fit(self):
        import numpy as np

        support = self.support()
        altered = self.image.copy()
        altered[~support] = 10000
        np.testing.assert_allclose(self.gaussian(self.image, support), self.gaussian(altered, support), atol=1e-7)


@unittest.skipUnless(AVAILABLE and importlib.util.find_spec("torch"), "RBF kernel tests require PyTorch")
class RbfKernelTests(unittest.TestCase):
    def test_control_parameterization_is_positive_and_capped_for_extreme_logits(self):
        import torch
        from tissue_bfc.support import RBF_FIELD_MIN, RBF_FIELD_MAX, rbf_control_values

        logits = torch.tensor([-1e30, -1000.0, -20.0, 0.0, 20.0, 1000.0, 1e30])
        values = rbf_control_values(logits)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.all((values > 0) & (values <= RBF_FIELD_MAX)))
        self.assertEqual(values[0].item(), torch.tensor(RBF_FIELD_MIN).item())
        self.assertEqual(values[-1].item(), RBF_FIELD_MAX)

    def test_bounded_initialization_preserves_unit_control_values(self):
        import torch
        from tissue_bfc.support import RBF_FIELD_MIN, RBF_FIELD_MAX, rbf_control_values

        original = torch.ones(48)
        logits = torch.logit((original - RBF_FIELD_MIN) / (RBF_FIELD_MAX - RBF_FIELD_MIN))
        torch.testing.assert_close(rbf_control_values(logits), original, rtol=1e-6, atol=1e-7)

    def test_bounded_interpolation_remains_valid_outside_fitting_support(self):
        import torch
        from tissue_bfc.support import bounded_rbf_values

        points = torch.tensor([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
        logits = torch.tensor([-1000.0, 1000.0])
        coordinates = torch.tensor([[-10.0, -10.0, -10.0], [0.5, 0.5, 0.5], [10.0, 10.0, 10.0]])
        output = bounded_rbf_values(points, logits, coordinates, 0.05, 2)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all((output > 0) & (output <= 100)))

    def test_upper_bound_survives_weighted_sum_roundoff(self):
        import torch
        from tissue_bfc.support import bounded_rbf_values

        generator = torch.Generator().manual_seed(17)
        points = torch.rand((48, 3), generator=generator)
        logits = torch.full((48,), 1000.0)
        coordinates = torch.rand((97, 3), generator=generator)
        output = bounded_rbf_values(points, logits, coordinates, 0.05, 48)
        self.assertTrue(torch.all(output <= 100))
        torch.testing.assert_close(output, torch.full_like(output, 100.0))

    def test_bounded_parameterization_is_differentiable_during_fitting(self):
        import torch
        from tissue_bfc.support import bounded_rbf_values

        points = torch.tensor([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4]], requires_grad=True)
        logits = torch.tensor([-4.0, -5.0], requires_grad=True)
        coordinates = torch.tensor([[0.25, 0.25, 0.25], [0.35, 0.35, 0.35]])
        output = bounded_rbf_values(points, logits, coordinates, 0.05, 2)
        output.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad > 0))

    def test_optimizer_overshoot_cannot_make_effective_control_values_invalid(self):
        import torch
        from tissue_bfc.support import rbf_control_values

        logits = torch.nn.Parameter(torch.tensor([-4.0, -4.0]))
        optimizer = torch.optim.SGD([logits], lr=1e5)
        optimizer.zero_grad()
        values = rbf_control_values(logits)
        (values[0] - values[1]).backward()
        optimizer.step()
        self.assertLess(logits[0].item(), -100)
        self.assertGreater(logits[1].item(), 100)
        effective = rbf_control_values(logits)
        self.assertTrue(torch.all((effective > 0) & (effective <= 100)))

    def test_constant_control_values_remain_constant_far_outside_fit_support(self):
        import torch
        from tissue_bfc.support import stable_rbf_values

        points = torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
        values = torch.tensor([0.8, 0.8])
        coordinates = torch.tensor([[0.25, 0.25, 0.25], [1.0, 1.0, 1.0], [-10.0, -10.0, -10.0]])
        output = stable_rbf_values(points, values, coordinates, 0.05, 2)
        torch.testing.assert_close(output, torch.full((3,), 0.8))
        self.assertTrue(torch.isfinite(output).all())

    def test_interpolation_is_bounded_and_gradients_remain_finite(self):
        import torch
        from tissue_bfc.support import stable_rbf_values

        points = torch.tensor([[0.2, 0.2, 0.2], [0.3, 0.3, 0.3]], requires_grad=True)
        values = torch.tensor([0.7, 1.3], requires_grad=True)
        coordinates = torch.tensor([[0.25, 0.25, 0.25], [0.4, 0.4, 0.4], [1.0, 1.0, 1.0]])
        output = stable_rbf_values(points, values, coordinates, 0.05, 2)
        self.assertTrue(torch.all((output >= 0.7) & (output <= 1.3)))
        output.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == "__main__":
    unittest.main()
