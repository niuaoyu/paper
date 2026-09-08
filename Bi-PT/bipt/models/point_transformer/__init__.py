"""Point Transformer backbone for Bi-PT.

Contains the Point Transformer encoder / decoder, the vector self-attention
``TransformerBlock``, the pairwise cross-attention core (``cr_att``), and the
PointNet++ set-abstraction / feature-propagation utilities.

Submodules keep their original names so the model code can reference the classes
as ``PTEncoder.PTEncoder``, ``PTDecoder.PTDecoder`` and
``cr_att.CrossAttentionModule``.
"""

from . import cr_att, PTEncoder, PTDecoder, transformer, pointnet_util  # noqa: F401

__all__ = ["cr_att", "PTEncoder", "PTDecoder", "transformer", "pointnet_util"]
