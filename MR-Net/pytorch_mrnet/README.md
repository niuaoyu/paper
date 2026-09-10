# PyTorch MR-Net feasibility milestones

This directory is an isolated, minimal feasibility implementation. It does not modify or
depend on the legacy TensorFlow 1.x implementation.

Milestones:

1. `check_alignment`: transform physical contour points with `pose + unit_sphere`, compare
   them with canonical meshes, and export GLB overlays.
2. `check_dataset`: verify `[B, 3000, 3]` inputs and fixed-size surface targets.
3. Build a two-component, fixed-topology template from case 0 ED labels.
4. Encode the input with a minimal PointNet and deform the template with graph convolutions.
5. Optimize Chamfer, edge-length, and graph-Laplacian losses.
6. `train_feasibility --mode single`: overfit case 0 ED.
7. `train_feasibility --mode five-cases`: train cases 0-4, ED and ES.

`baseline` performs the next case-disjoint test. By default it trains on cases 0-79,
selects the best checkpoint on cases 80-89, tests once on cases 90-99, and repeats this for
three random seeds. It also compares against the undeformed template, checks output diversity,
ED/ES enclosed-volume direction, mesh quality, and exports orthographic PNG overlays.

The trainable model now includes the complete architectural path used by MR-Net: hierarchical
point extraction, a multi-scale 64-cubed occupancy CNN, point/volume projection at graph
vertices, and a three-stage residual graph decoder. The public model call still returns only
the final mesh unless `return_stages=True`, so the case-disjoint evaluation protocol remains
unchanged.

## Complete MR-Net migration order

Add the remaining modules one at a time and retain the preceding acceptance tests:

1. The global PointNet has been replaced with a PointNet++-style hierarchy using deterministic
   farthest-point sampling, kNN grouping, relative coordinates, shared local MLPs, and max
   pooling. The public `MinimalMRNet` interface and baseline protocol remain unchanged.
2. Canonical contours are rasterized into a differentiable-input-independent binary 64-cubed
   occupancy volume and encoded at 64, 32, 16, and 8 voxel resolutions.
3. Trilinear volume sampling and inverse-distance point-neighbor interpolation project all
   feature scales onto every current graph vertex. Gradient tests cover both encoders and the
   projection adapters.
4. The decoder has three residual deformation stages with 14, 15, and 16 graph-convolution
   layers. Training applies stage weights 0.1, 0.3, and 0.6 while evaluation consumes stage 3.
5. Add the original incomplete-slice simulation only after full-contour reconstruction works
   on held-out cases.
6. Reconcile point and mesh label semantics, then add LV/RV-partitioned Chamfer supervision.
7. If per-vertex L1 supervision is required, register every target to a shared topology first;
   never apply it directly to the current variable-topology meshes.

The current tests prove that the transformed data, fixed-template decoder, and surface losses
form a trainable loop. They do not yet prove generalization or reproduce the paper's accuracy.
