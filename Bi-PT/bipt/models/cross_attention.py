"""Cross-attention modules for Bi-PT.

``SingleCrossAttentionModule`` performs a single atlas<-SPC cross-attention (the
sCA ablation); ``DoubleCrossAttentionModule`` performs the *bidirectional*
atlas<->SPC cross-attention (paper Sec. 2.2) that gives Bi-PT its name.
"""
import torch
import torch.nn as nn

from .point_transformer import PTEncoder, PTDecoder, cr_att


class SingleCrossAttentionModule(nn.Module):
    """
    Single-stage cross-attention:
      q <- x

    Supports different d_points for atlas(q_in) vs spc(x_in).

    d_points = feature_dim - 3 (xyz part is first 3 dims)
      - spc BN8: feature_dim=8 => d_points_x=5
      - atlas BN9: feature_dim=9 => d_points_q=6
      - no-labels BN6: feature_dim=6 => d_points=3 (also works)
    """
    def __init__(
        self,
        d_model=512,
        d_points_q=6,   # atlas / q_in
        d_points_x=5,   # spc   / x_in
        nblocks=2,
        nneighbor=16,
        norm_type="bn",
        dec_fc_norm_type: str = None,
        dec_fp_norm_type: str = None,
    ):
        super().__init__()
        self.d_points_q = int(d_points_q)
        self.d_points_x = int(d_points_x)

        self.q_encoder = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_q,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.k_encoder = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_x,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.v_encoder = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_x,
            transformer_dim=d_model, norm_type=norm_type
        )

        self.cross = cr_att.CrossAttentionModule(d_model=d_model)

        # IMPORTANT: same decoder interface you use in DoubleCrossAttentionModule:
        # decoder(xyz0, tokens, xyz_and_feats) -> (pts_feat, gap_code)
        self.decoder = PTDecoder.PTDecoder(
            nblocks=nblocks, nneighbor=nneighbor, transformer_dim=d_model,
            norm_type=norm_type,
            fc_norm_type=dec_fc_norm_type,
            fp_norm_type=dec_fp_norm_type,
        )

    def forward(self, q_in, x_in):
        # q_in: (B,Na,3+d_points_q)
        # x_in: (B,Ns,3+d_points_x)
        q_out = self.q_encoder(q_in)
        k_out = self.k_encoder(x_in)
        v_out = self.v_encoder(x_in)

        attn_out = self.cross(q_out[1], k_out[1], v_out[1])
        xyz0, xyz_and_feats = q_out[0], q_out[2]

        pts_feat, gap_code = self.decoder(xyz0, attn_out, xyz_and_feats)
        return pts_feat, gap_code


class DoubleCrossAttentionModule(nn.Module):
    """
    NEW: supports different d_points for atlas(q_in) vs spc(x_in).

    d_points = feature_dim - 3 (xyz part is first 3 dims)
      - spc BN8: feature_dim=8 => d_points_x=5
      - atlas BN9: feature_dim=9 => d_points_q=6
    """
    def __init__(
        self,
        d_model=512,
        d_points_q=6,   # atlas / q_in
        d_points_x=5,   # spc   / x_in
        nblocks=2,
        nneighbor=16,
        norm_type="bn",
        dec_fc_norm_type: str = None,
        dec_fp_norm_type: str = None,
    ):
        super().__init__()
        self.d_points_q = int(d_points_q)
        self.d_points_x = int(d_points_x)

        # Stage 1: q <- x
        self.q_encoder1 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_q,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.k_encoder1 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_x,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.v_encoder1 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_x,
            transformer_dim=d_model, norm_type=norm_type
        )

        # Stage 2: x <- q
        self.q_encoder2 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_x,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.k_encoder2 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_q,
            transformer_dim=d_model, norm_type=norm_type
        )
        self.v_encoder2 = PTEncoder.PTEncoder(
            nblocks=nblocks, nneighbor=nneighbor, d_points=self.d_points_q,
            transformer_dim=d_model, norm_type=norm_type
        )

        self.cross1 = cr_att.CrossAttentionModule(d_model=d_model)
        self.cross2 = cr_att.CrossAttentionModule(d_model=d_model)

        self.decoder1 = PTDecoder.PTDecoder(
            nblocks=nblocks, nneighbor=nneighbor, transformer_dim=d_model,
            norm_type=norm_type,
            fc_norm_type=dec_fc_norm_type,
            fp_norm_type=dec_fp_norm_type,
        )
        self.decoder2 = PTDecoder.PTDecoder(
            nblocks=nblocks, nneighbor=nneighbor, transformer_dim=d_model,
            norm_type=norm_type,
            fc_norm_type=dec_fc_norm_type,
            fp_norm_type=dec_fp_norm_type,
        )

    def forward(self, q_in, x_in):
        # q_in: (B,Na,3+d_points_q)
        # x_in: (B,Ns,3+d_points_x)
        q_out1 = self.q_encoder1(q_in)
        k_out1 = self.k_encoder1(x_in)
        v_out1 = self.v_encoder1(x_in)

        attn_out_atlas = self.cross1(q_out1[1], k_out1[1], v_out1[1])
        xyz0_1, xyz_and_feats1 = q_out1[0], q_out1[2]
        pts_feats1, _ = self.decoder1(xyz0_1, attn_out_atlas, xyz_and_feats1)

        q_out2 = self.q_encoder2(x_in)
        k_out2 = self.k_encoder2(q_in)
        v_out2 = self.v_encoder2(q_in)

        attn_out_sparse = self.cross2(q_out2[1], k_out2[1], v_out2[1])
        xyz0_2, xyz_and_feats2 = q_out2[0], q_out2[2]
        _, gap_code = self.decoder2(xyz0_2, attn_out_sparse, xyz_and_feats2)

        return pts_feats1, gap_code



__all__ = [
    "SingleCrossAttentionModule",
    "DoubleCrossAttentionModule",
]
