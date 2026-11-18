"""Standalone volume checks without a FastPrep runtime dependency."""

import numpy as np


def same_grid(reference, image):
    for volume in (reference, image):
        spacing = np.asarray(volume.header.get_zooms()[:3])
        if (len(volume.shape) != 3 or not np.isfinite(volume.affine).all()
                or not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or abs(np.linalg.det(volume.affine[:3, :3])) < 1e-12):
            raise ValueError("Expected a finite nonsingular 3D physical grid")
    if reference.shape != image.shape or not np.allclose(reference.affine, image.affine, rtol=0, atol=1e-5):
        raise ValueError("BFC volumes do not share the same physical grid")
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("BFC image contains nonfinite values")
    return values


def checked_labels(reference, image, allowed):
    values = same_grid(reference, image)
    if not np.isin(values, allowed).all():
        raise ValueError("BFC label image contains invalid or noninteger codes")
    return values.astype(np.uint8)
