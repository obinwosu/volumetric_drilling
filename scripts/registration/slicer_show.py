"""3D Slicer startup script: show one MRI -> CT registration from the launcher.

Started by the launcher's "Register MRI" tab:
    Slicer --python-script slicer_show.py <work folder>
(the folder is also in FIVRS_REGISTRATION_DIR). Loads ct.nii.gz, mri.nii.gz and
mri_to_ct.tfm, puts the MRI under the transform, and shows the MRI over the CT
(bone window) in the four-up layout, centred on the first temporal bone.

To adjust by hand: Transforms module -> "MRI to CT" -> move the sliders, then
File -> Save Data -> save the transform over mri_to_ct.tfm and press "Reload after
Slicer" in the launcher.
"""
import json
import os
import sys

import slicer

work = os.environ.get("FIVRS_REGISTRATION_DIR") or (sys.argv[-1] if len(sys.argv) > 1 else "")
if not os.path.isdir(work):
    slicer.util.errorDisplay(f"No registration folder: {work!r}")
else:
    ct = slicer.util.loadVolume(os.path.join(work, "ct.nii.gz"), {"name": "CT"})
    mri = slicer.util.loadVolume(os.path.join(work, "mri.nii.gz"), {"name": "MRI"})
    # ITK files hold the resampling transform (CT point -> MRI point); Slicer reads it as the
    # MRI's "from parent" transform, so the MRI under it lands on the CT.
    transform = slicer.util.loadTransform(os.path.join(work, "mri_to_ct.tfm"))
    transform.SetName("MRI to CT")
    mri.SetAndObserveTransformNodeID(transform.GetID())

    tumor_file = os.path.join(work, "tumor", "tumor_on_ct.nii.gz")
    if os.path.isfile(tumor_file):  # the Register MRI tab's tumor parts, on the CT grid
        labels = slicer.util.loadLabelVolume(tumor_file, {"name": "Tumor labels", "show": False})
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Tumor")
        segmentation.CreateDefaultDisplayNodes()
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(ct)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labels, segmentation)
        colors = [(1.0, 0.5, 0.1), (1.0, 0.85, 0.15), (0.25, 0.78, 1.0), (1.0, 0.31, 0.78)]
        segments = segmentation.GetSegmentation()
        for n in range(segments.GetNumberOfSegments()):
            segment = segments.GetNthSegment(n)
            segment.SetName(f"Tumor part {n + 1}")
            segment.SetColor(*colors[n % len(colors)])
        segmentation.CreateClosedSurfaceRepresentation()
        slicer.mrmlScene.RemoveNode(labels)

    display = ct.GetDisplayNode()
    display.SetAutoWindowLevel(False)
    display.SetWindowLevel(2400, 400)  # bone window
    slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    slicer.util.setSliceViewerLayers(background=ct, foreground=mri, foregroundOpacity=0.5, fit=True)

    view = slicer.app.layoutManager().threeDWidget(0).threeDView()
    view.resetFocalPoint()  # the 3D view shows the tumor parts, if any

    try:  # centre the slices on the first temporal bone (LPS -> RAS)
        info = json.load(open(os.path.join(work, "registration.json")))
        x, y, z = info["temporal_bones_mm"][0]
        slicer.modules.markups.logic().JumpSlicesToLocation(-x, -y, z, True)
    except Exception:  # noqa: BLE001 - a view nicety, not worth failing for
        pass
