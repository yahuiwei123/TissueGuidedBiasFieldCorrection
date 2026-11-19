"""Separate reliable tissue fitting from continuous brain-domain correction."""

import numpy as np
from scipy import ndimage


REVISION = "reliable-core-v2"
RBF_MODEL_REVISION = "bounded-sigmoid-v1"
RBF_FIELD_MIN = 1e-6
RBF_FIELD_MAX = 100.0


def validate_arrays(image, labels, brain_mask, spacing):
    image, labels, brain_mask = map(np.asarray, (image, labels, brain_mask))
    spacing = np.asarray(spacing, dtype=float)
    if image.ndim != 3 or image.shape != labels.shape or image.shape != brain_mask.shape:
        raise ValueError("BFC image, labels and brain mask must share a 3D grid")
    if not all(np.isfinite(values).all() for values in (image, labels, brain_mask, spacing)):
        raise ValueError("BFC inputs must be finite")
    if spacing.shape != (3,) or np.any(spacing <= 0) or np.any(image < 0):
        raise ValueError("BFC requires positive spacing and nonnegative intensities")
    if not np.isin(labels, [0, 1, 2, 3]).all() or not np.isin(brain_mask, [0, 1]).all():
        raise ValueError("Expected integer tissue labels 0/1/2/3 and a binary brain mask")
    if not np.any(brain_mask):
        raise ValueError("BFC brain mask is empty")


def reliable_support(image, labels, brain_mask, spacing, gm_core_mm=1.0, wm_core_mm=2.0, log_mad_limit=3.5):
    validate_arrays(image, labels, brain_mask, spacing)
    if not np.isfinite(log_mad_limit) or log_mad_limit <= 0:
        raise ValueError("Core log-MAD limit must be finite and positive")
    support = np.zeros(image.shape, dtype=bool)
    counts = {}
    for tissue, margin in ((2, gm_core_mm), (3, wm_core_mm)):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Tissue erosion margins must be finite and nonnegative")
        eligible = (labels == tissue) & (brain_mask > 0) & (image > 0)
        padded = np.pad(eligible, 1, constant_values=False)
        distances = ndimage.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
        core = eligible & (distances > margin)
        if np.count_nonzero(core) < 2:
            raise ValueError(f"Insufficient reliable tissue {tissue}; no fallback to contaminated samples")
        spatial_count = int(core.sum())
        logarithms = np.log(image[core].astype(np.float64))
        center = float(np.median(logarithms))
        deviation = float(1.4826 * np.median(np.abs(logarithms - center)))
        retained = np.abs(logarithms - center) <= log_mad_limit * max(deviation, np.finfo(np.float32).eps)
        core[core] = retained
        if np.count_nonzero(core) < 2:
            raise ValueError(f"Insufficient robust tissue {tissue}; no sampling fallback")
        support |= core
        counts[str(tissue)] = {"eligible_voxels": int(eligible.sum()), "core_voxels": int(core.sum()),
                              "erosion_mm": float(margin), "spatial_core_voxels": spatial_count,
                              "intensity_outliers_excluded": spatial_count - int(core.sum()),
                              "log_intensity_median": center, "scaled_log_mad": deviation,
                              "log_mad_limit": float(log_mad_limit)}
    return support, {"revision": REVISION, "classes": counts,
                     "fit_voxels": int(support.sum()), "application_voxels": int(np.count_nonzero(brain_mask)),
                     "selection": "Independent physical-distance GM/WM erosion, positive intensities and within-class log-MAD outlier rejection; no closing or class pooling",
                     "independent_anatomical_truth": False}


def restricted_probabilities(labels, support, sigma):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Soft-label sigma must be finite and positive")
    if support.shape != labels.shape or not support.any() or np.any(support & ~np.isin(labels, [2, 3])):
        raise ValueError("Fit support must contain only GM and WM")
    weights = np.stack([ndimage.gaussian_filter((labels == tissue).astype(np.float32), sigma=sigma)
                        * (support & (labels == tissue)) for tissue in (2, 3)])
    return weights


def nearest_support_indices(support, spacing):
    if not np.any(support):
        raise ValueError("Cannot extrapolate from an empty support")
    indices = ndimage.distance_transform_edt(~support, sampling=spacing,
                                           return_distances=False, return_indices=True)
    if not np.all(support[tuple(indices)]):
        raise ValueError("Nearest-field donor escaped the reliable support")
    return indices


def application_field(field, brain_mask):
    field, brain_mask = np.asarray(field), np.asarray(brain_mask)
    if field.ndim != 3 or field.shape != brain_mask.shape or not np.isin(brain_mask, [0, 1]).all():
        raise ValueError("Estimated field and application mask must share a 3D grid")
    if not np.isfinite(field).all() or np.any(field[brain_mask > 0] <= 0):
        raise ValueError("Estimated field must be finite and positive throughout its application domain")
    return np.where(brain_mask > 0, field, 1).astype(np.float32)


def apply_field(image, field, brain_mask, representation="multiplier"):
    image, field, brain_mask = map(np.asarray, (image, field, brain_mask))
    if image.ndim != 3 or image.shape != field.shape or image.shape != brain_mask.shape:
        raise ValueError("Field application requires matching 3D grids")
    if not np.isin(brain_mask, [0, 1]).all() or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Invalid image or application mask")
    if not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError("Correction field must be finite and strictly positive")
    if representation not in {"multiplier", "divisor"}:
        raise ValueError("Unknown field representation")
    active = brain_mask > 0
    result = image.astype(np.float32, copy=True)
    if representation == "multiplier":
        result[active] = image[active] * field[active]
    else:
        result[active] = image[active] / field[active]
    result[image == 0] = 0
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Field application produced invalid intensities")
    return result


def gaussian_field(source, image, labels, support, spacing, *, max_iter=10, max_kernel=9,
                   min_kernel=None, soft_sigma=0.4, modality="None"):
    if max_iter < 1 or not np.isfinite(max_kernel) or max_kernel <= 0:
        raise ValueError("Invalid Gaussian iteration or kernel setting")
    minimum = source.compute_sigma(spacing) if min_kernel is None else (min_kernel,) * 3
    if not np.isfinite(minimum).all() or np.any(np.asarray(minimum) <= 0):
        raise ValueError("Gaussian minimum kernel must be finite and positive")
    probabilities = restricted_probabilities(labels, support, soft_sigma)
    scale = float(image[support].max())
    log_image = np.zeros(image.shape, dtype=np.float64)
    log_image[support] = np.log10(image[support].astype(np.float64) / scale)
    targets = []
    for weights in probabilities:
        targets.append(float(np.sum(log_image * weights) / weights.sum()))
    if modality == "T1":
        targets = np.log10([0.50, 0.95]).tolist()
    elif modality == "T2":
        targets = np.log10([0.35, 0.15]).tolist()
    elif modality != "None":
        raise ValueError("Unknown modality")
    active_classes = (2,) if modality == "T2" else (2, 3)
    active = support & np.isin(labels, active_classes)
    donors = nearest_support_indices(active, spacing)
    bias = np.zeros(image.shape, dtype=np.float64)
    residual = np.zeros_like(bias)
    for iteration in range(max_iter):
        for tissue in active_classes:
            selected = support & (labels == tissue)
            residual[selected] = np.clip(log_image[selected] - bias[selected] - targets[tissue - 2], -0.20, 0.20)
        kernel = source.kernel_scheduler(iteration, max_iter, max_kernel=(max_kernel,) * 3, min_kernel=minimum)
        extrapolated = residual[tuple(donors)]
        bias += ndimage.gaussian_filter(extrapolated, sigma=kernel, mode="nearest")
        print(f"Reliable-core Gaussian iteration {iteration + 1}/{max_iter}; sigma={kernel}", flush=True)
    field = np.power(10.0, -bias).astype(np.float32)
    return field, {"device": "cpu", "fit_rule": "Gaussian class residual conditioned on disjoint hard tissue cores; soft confidence weights for fixed class means",
                   "active_classes": list(active_classes), "initial_log10_class_means": targets,
                   "normalization_scale": scale, "offset": 0.0, "soft_weight_outside_fit_support": 0.0}


def stable_rbf_values(points, point_values, coordinates, sigma, k_neighbors):
    import torch

    if not np.isfinite(sigma) or sigma <= 0 or k_neighbors < 1 or k_neighbors > len(points):
        raise ValueError("Invalid RBF width or neighbor count")
    distances = torch.sum((coordinates.unsqueeze(1) - points.unsqueeze(0)) ** 2, dim=2)
    nearest_distances, nearest_indices = torch.topk(distances, k=k_neighbors, largest=False, dim=1)
    weights = torch.softmax(-nearest_distances / (2 * sigma ** 2), dim=1)
    return torch.sum(weights * point_values[nearest_indices], dim=1)


def cuda_device(device):
    import torch

    device = str(device)
    if device.isdecimal():
        device = "cuda:" + device
    if device == "cuda":
        device = "cuda:0"
    if not device.startswith("cuda:") or not device[5:].isdecimal():
        raise ValueError("RBF requires an explicit CUDA device; CPU fallback is disabled")
    if not torch.cuda.is_available() or int(device[5:]) >= torch.cuda.device_count():
        raise RuntimeError("Requested RBF CUDA device is unavailable; CPU fallback is disabled")
    return str(torch.device(device))


def rbf_control_values(logits):
    import torch

    return RBF_FIELD_MIN + (RBF_FIELD_MAX - RBF_FIELD_MIN) * torch.sigmoid(logits)


def bounded_rbf_values(points, logits, coordinates, sigma, k_neighbors):
    values = stable_rbf_values(points, rbf_control_values(logits), coordinates, sigma, k_neighbors)
    return values.clamp(min=RBF_FIELD_MIN, max=RBF_FIELD_MAX)


def rbf_field(source, image, labels, support, *, device="cuda:0", n_points=48, max_iter=25,
              tol=1e-6, soft_sigma=0.5, rbf_sigma=0.05, k_neighbors=48, seed=0):
    import torch

    device = cuda_device(device)
    if n_points < 1 or n_points > int(support.sum()) or max_iter < 1 or not 1 <= k_neighbors <= n_points:
        raise ValueError("Invalid RBF point or iteration setting")
    if any(not np.isfinite(value) or value <= 0 for value in (tol, soft_sigma, rbf_sigma)):
        raise ValueError("RBF scales and tolerance must be finite and positive")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    weights = restricted_probabilities(labels, support, soft_sigma)
    total = weights.sum(axis=0, keepdims=True)
    probabilities = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)

    class ReliableCorrector(source.ScatterBiasFieldCorrectorFast):
        def _generate_soft_labels(self):
            return probabilities

        def _eval_bias_field_batch(self, coordinates):
            return bounded_rbf_values(self.points, self.point_values, coordinates, self.rbf_sigma, self.k_neighbors)

    scale = float(image[support].max())
    normalized = image.astype(np.float32) / scale
    fit_labels = np.where(support, labels, 0).astype(np.uint8)
    corrector = ReliableCorrector(normalized, fit_labels, mask=support.astype(np.uint8),
                                 n_points=n_points, max_iter=max_iter, tol=tol, soft_sigma=soft_sigma,
                                 rbf_sigma=rbf_sigma, k_neighbors=k_neighbors, device=device)
    if list(corrector.classes) != [2, 3] or np.any(corrector.soft_labels_np[:, ~support] != 0):
        raise ValueError("RBF tissue weights escaped reliable GM/WM support")
    with torch.no_grad():
        initial_fraction = (corrector.point_values - RBF_FIELD_MIN) / (RBF_FIELD_MAX - RBF_FIELD_MIN)
        if not torch.isfinite(initial_fraction).all() or torch.any((initial_fraction <= 0) | (initial_fraction >= 1)):
            raise ValueError("Initial RBF control values must lie strictly inside the configured bounds")
        corrector.point_values.copy_(torch.logit(initial_fraction))
    corrector.fit()
    with torch.no_grad():
        if not torch.isfinite(corrector.points).all() or not torch.isfinite(corrector.point_values).all():
            raise ValueError("RBF optimization produced nonfinite control parameters")
        field = corrector.get_bias_field()
    torch.cuda.synchronize(torch.device(device))
    if not np.isfinite(field).all() or np.any(field <= 0) or np.any(field > RBF_FIELD_MAX):
        raise ValueError("Bounded RBF field must be finite, strictly positive and at most 100")
    fitted = {"points": corrector.points.detach().cpu().numpy(),
              "point_values": rbf_control_values(corrector.point_values.detach()).cpu().numpy(),
              "point_logits": corrector.point_values.detach().cpu().numpy(),
              "initial_points": corrector.init_points, "shape": np.array(image.shape)}
    return field, {"device": device, "gpu_name": torch.cuda.get_device_name(torch.device(device)),
                   "normalization_scale": scale, "offset": 0.0, "active_classes": [2, 3],
                   "model_revision": RBF_MODEL_REVISION,
                   "field_bounds": {"minimum": RBF_FIELD_MIN, "maximum": RBF_FIELD_MAX},
                   "control_parameterization": "minimum + (maximum - minimum) * sigmoid(logit); used in every fitting and export evaluation",
                   "rbf_weight_normalization": "Log-domain softmax in both fitting and export; no additive denominator epsilon",
                   "soft_weight_outside_fit_support": float(probabilities[:, ~support].sum()),
                   "export_autograd": False, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device))}, fitted
