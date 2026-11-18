"""Run BFC with reliable tissue cores and a separate brain application domain."""

import argparse
import importlib.util
import json
from pathlib import Path
import time


def main(argv=None, *, default_method="rbf"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-script", help="Optional override of the bundled upstream implementation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=("gaussian", "rbf"), default=default_method)
    for name in ("input_img", "input_lab", "input_msk", "output_img", "output_bias"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gm-core-mm", type=float, default=1.0)
    parser.add_argument("--wm-core-mm", type=float, default=2.0)
    parser.add_argument("--core-log-mad-limit", type=float, default=3.5)
    parser.add_argument("--bias_max_iter", type=int, default=10)
    parser.add_argument("--max_kernel", type=float, default=9)
    parser.add_argument("--min_kernel", type=float)
    parser.add_argument("--refine_max_iter", type=int, default=0)
    parser.add_argument("--label_soft_kernel", type=float, default=0.4)
    parser.add_argument("--modality", choices=("None", "T1", "T2"), default="None")
    parser.add_argument("--device", default="cuda:0", help="RBF CUDA device; Gaussian always uses CPU")
    parser.add_argument("--n_points", type=int, default=48)
    parser.add_argument("--rbf_max_iter", "--max_iter", dest="rbf_max_iter", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--rbf_sigma", type=float, default=0.05)
    parser.add_argument("--soft_sigma", type=float, default=0.5)
    parser.add_argument("--k_neighbors", type=int, default=48)
    arguments = parser.parse_args(argv)
    expected = "bfc_gauss_mixture.py" if arguments.method == "gaussian" else "bfc_rbf_scatter.py"
    source = (Path(arguments.source_script) if arguments.source_script else
              Path(__file__).resolve().parent / "_legacy" / expected).resolve(strict=True)
    if source.name != expected or arguments.refine_max_iter != 0:
        raise ValueError("Expected matching upstream source and fixed tissue priors")
    import nibabel as nib
    import numpy as np
    from . import __version__
    from .support import REVISION, application_field, apply_field, cuda_device, gaussian_field, rbf_field, reliable_support
    from .validation import checked_labels, same_grid

    if arguments.method == "rbf":
        if arguments.modality != "None":
            raise ValueError("RBF preserves fitted tissue contrast; fixed modality targets are unsupported")
        arguments.device = cuda_device(arguments.device)

    outputs = [arguments.output_img, arguments.output_bias]
    target = arguments.output_img.parent
    outputs += [target / "fit-mask.nii.gz", target / "fit-labels.nii.gz", target / "support.json"]
    if arguments.method == "rbf":
        outputs.append(target / "fitted-control-points.npz")
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite BFC outputs; use a new attempt directory")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    reference = nib.load(arguments.input_img)
    image = same_grid(reference, reference)
    labels = checked_labels(reference, nib.load(arguments.input_lab), [0, 1, 2, 3])
    brain = checked_labels(reference, nib.load(arguments.input_msk), [0, 1])
    spacing = reference.header.get_zooms()[:3]
    support, provenance = reliable_support(image, labels, brain, spacing, arguments.gm_core_mm, arguments.wm_core_mm,
                                           arguments.core_log_mad_limit)

    def save(values, path):
        output = nib.Nifti1Image(values, reference.affine)
        output.header.set_xyzt_units("mm")
        nib.save(output, path)

    save(support.astype(np.uint8), target / "fit-mask.nii.gz")
    save(np.where(support, labels, 0).astype(np.uint8), target / "fit-labels.nii.gz")
    specification = importlib.util.spec_from_file_location("bfc_upstream", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    np.random.seed(arguments.seed)
    started = time.monotonic()
    report = {"toolkit_version": __version__, "method": arguments.method,
              "implementation": REVISION, "status": "running", "support": provenance,
              "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()},
              "source": str(source), "application": "Continuous multiplicative correction across brain, including tissue label zero; input preserved outside brain and at zero signal",
              "scientific_approval": "not_performed"}
    report_path = target / "support.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if arguments.method == "gaussian":
            field, details = gaussian_field(module, image, labels, support, spacing,
                                           max_iter=arguments.bias_max_iter, max_kernel=arguments.max_kernel,
                                           min_kernel=arguments.min_kernel, soft_sigma=arguments.label_soft_kernel,
                                           modality=arguments.modality)
            representation = "multiplier"
        else:
            field, details, fitted = rbf_field(module, image.transpose(2, 1, 0), labels.transpose(2, 1, 0),
                                               support.transpose(2, 1, 0), device=arguments.device,
                                               n_points=arguments.n_points, max_iter=arguments.rbf_max_iter,
                                               tol=arguments.tol, soft_sigma=arguments.soft_sigma,
                                               rbf_sigma=arguments.rbf_sigma, k_neighbors=arguments.k_neighbors,
                                               seed=arguments.seed)
            field = field.transpose(2, 1, 0)
            details["fitting_array_axes"] = "ITK z,y,x; exported field restored to input voxel axes"
            np.savez(target / "fitted-control-points.npz", **fitted)
            representation = "divisor"
        details["nonpositive_estimates_outside_application_domain"] = int(np.sum((brain == 0) & (field <= 0)))
        field = application_field(field, brain)
        corrected = apply_field(image, field, brain, representation)
        save(field, arguments.output_bias)
        save(corrected, arguments.output_img)
        same_grid(reference, nib.load(arguments.output_img))
        report.update(status="completed", fitting=details, field_representation=representation)
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")


def gaussian_main(argv=None):
    main(argv, default_method="gaussian")


def rbf_main(argv=None):
    main(argv, default_method="rbf")


if __name__ == "__main__":
    main()
