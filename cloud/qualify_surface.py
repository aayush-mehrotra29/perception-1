"""Export template surfaces and test native TRIBE target ordering on CPU."""
import modal
from cloud.modal_brain import tribe_base_image

app = modal.App("perception-surface-qualification")
surface_image = tribe_base_image.pip_install("nilearn==0.12.1").add_local_python_source("brain")


@app.function(image=surface_image, cpu=2, memory=8192, timeout=300, retries=0, max_containers=1)
def qualify():
    import inspect
    import io
    import json
    import zipfile
    import importlib.metadata
    from pathlib import Path
    from unittest.mock import patch
    import numpy as np
    from nilearn import datasets, surface
    from tribev2.utils_fmri import TribeSurfaceProjector
    from brain.worker_contract import file_sha256

    n = 10242
    projector = TribeSurfaceProjector(mesh="fsaverage5")

    class Recording:
        def __init__(self, values):
            self.values, self.shape = values, values.shape

        def get_fdata(self):
            return self.values

    original = np.arange(2 * n * 2, dtype=float).reshape(2 * n, 2)
    np.testing.assert_array_equal(projector.apply(Recording(original)), original)
    high_resolution = np.arange(2 * 40962 * 2, dtype=float).reshape(2 * 40962, 2)
    expected = np.concatenate([high_resolution[:n], high_resolution[40962:40962 + n]])
    np.testing.assert_array_equal(projector.apply(Recording(high_resolution)), expected)
    # Isolate concatenation from interpolation; this does not test anatomy.
    projector._mesh = {f"{part}_{hemi}": f"{part}_{hemi}"
                       for part in ("pial", "white") for hemi in ("left", "right")}
    def marker_projection(rec, *, surf_mesh, **kwargs):
        return np.full((n, 2), 1 if surf_mesh == "pial_left" else 2, dtype=float)
    with patch("nilearn.surface.vol_to_surf", side_effect=marker_projection):
        result = projector.apply(Recording(np.zeros((2, 2, 2, 2))))
    np.testing.assert_array_equal(result[:n], np.ones((n, 2)))
    np.testing.assert_array_equal(result[n:], np.full((n, 2), 2))

    meshes = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    arrays, sources = {}, {}
    for hemi in ("left", "right"):
        for kind in ("pial", "infl"):
            name = f"{kind}_{hemi}"
            coordinates, faces = surface.load_surf_mesh(meshes[name])
            coordinates, faces = np.asarray(coordinates), np.asarray(faces)
            if (coordinates.shape != (n, 3) or not np.isfinite(coordinates).all()
                    or faces.ndim != 2 or faces.shape[1] != 3
                    or faces.min() < 0 or faces.max() >= n):
                raise ValueError("unexpected fsaverage5 geometry")
            arrays[name + "_coordinates"] = coordinates
            arrays[name + "_faces"] = faces
            sources[name] = {"filename": Path(meshes[name]).name, "sha256": file_sha256(Path(meshes[name]))}
    data = io.BytesIO()
    np.savez(data, **arrays)
    import hashlib
    direct = json.loads(importlib.metadata.distribution("tribev2").read_text("direct_url.json"))
    revision = direct["vcs_info"]["commit_id"]
    if revision != "af58661791a351a448a489042a28f6c37e1c14b7":
        raise ValueError("native source revision changed")
    report = {"schema": "perception-template-surface-v1", "mesh": "fsaverage5",
        "vertices_per_hemisphere": n, "prediction_columns": 2 * n,
        "column_order": ["left", "right"], "within_hemisphere_order": "native fsaverage5 vertex index",
        "upstream_git_revision": revision,
        "upstream_projector_sha256": file_sha256(Path(inspect.getsourcefile(TribeSurfaceProjector))),
        "native_projector_index_tests_passed": True,
        "tests": ["same-resolution identity", "fsaverage6 downsampling indices", "mocked volume hemisphere concatenation"],
        "geometry_source": "nilearn.datasets.fetch_surf_fsaverage(fsaverage5)",
        "nilearn_version": importlib.metadata.version("nilearn"), "source_files": sources,
        "arrays_sha256": hashlib.sha256(data.getvalue()).hexdigest(),
        "geometry_role": "generic template visualization; not the original training MNI-space projection mesh",
        "checkpoint_anatomical_correspondence_independently_validated": False,
        "anatomical_region_labels_included": False, "individual_anatomy": False,
        "neural_prediction_accuracy_validated": False}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("surface.npz", data.getvalue())
        archive.writestr("manifest.json", json.dumps(report, indent=2) + "\n")
    return output.getvalue()
