import argparse
import os
import enum
from typing import Optional

import torch
import numpy as np

import coremltools as ct
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools import ComputeUnit
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.mil import Builder as mb

# If your local package is just "sam2_video_predictor.py", adjust import accordingly.
from sam2.sam2_video_predictor import SAM2VideoPredictor

################################################################################
# Configuration: We use a smaller 256x256 instead of 1024x1024 for demonstration
################################################################################
SAM2_HW = (256, 256)  # <--- You can restore (1024,1024) if you prefer large images

################################################################################
# Op registration for bicubic upsampling
################################################################################
@register_torch_op
def upsample_bicubic2d(context, node):
    x = context[node.inputs[0]]
    output_size = context[node.inputs[1]].val
    scale_factor_height = output_size[0] / x.shape[2]
    scale_factor_width = output_size[1] / x.shape[3]
    align_corners = context[node.inputs[2]].val

    x = mb.upsample_bilinear(
        x=x,
        scale_factor_height=scale_factor_height,
        scale_factor_width=scale_factor_width,
        align_corners=align_corners,
        name=node.name,
    )
    context.add(x)

################################################################################
# CLI Argument Parsing
################################################################################
class SAM2Variant(enum.Enum):
    Tiny = "tiny"
    Small = "small"
    BasePlus = "base-plus"
    Large = "large"

    def fmt(self):
        if self == SAM2Variant.BasePlus:
            return "BasePlus"
        return self.value.capitalize()

def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    parser.add_argument(
        "--variant",
        type=lambda x: getattr(SAM2Variant, x),
        choices=[variant for variant in SAM2Variant],
        default=SAM2Variant.Small,
        help="SAM2 variant to export.",
    )
    parser.add_argument(
        "--min-deployment-target",
        type=lambda x: getattr(AvailableTarget, x),
        choices=[target for target in AvailableTarget],
        default=AvailableTarget.iOS17,
        help="Minimum deployment target for CoreML model.",
    )
    parser.add_argument(
        "--compute-units",
        type=lambda x: getattr(ComputeUnit, x),
        choices=[cu for cu in ComputeUnit],
        default=ComputeUnit.ALL,
        help="Which compute units to target.",
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(ComputePrecision, x),
        choices=[p for p in ComputePrecision],
        default=ComputePrecision.FLOAT16,
        help="Precision for quantization.",
    )
    return parser

################################################################################
# PyTorch Wrappers for Video Model
################################################################################
class SAM2VideoEncoder(torch.nn.Module):
    """
    Takes a video [T, 3, H, W] and returns
    (video_embedding, memory_embedding, feats_s0, feats_s1).
    """
    def __init__(self, model: SAM2VideoPredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, video):
        # video: [T, 3, H, W]
        video_embedding, memory_embedding, feats_s0, feats_s1 = self.model.encode_video_raw(video)
        return video_embedding, memory_embedding, feats_s0, feats_s1


class SAM2PointsEncoder(torch.nn.Module):
    """
    For encoding prompt points, shape:
      points: [1, N, 2]
      labels: [1, N]
    """
    def __init__(self, model: SAM2VideoPredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, points, labels):
        # Reuse same method from model (like single-image version),
        # but ensure shapes match the “video” dimension usage if needed.
        sparse_embeddings, dense_embeddings = self.model.encode_points_raw(points, labels)
        return sparse_embeddings, dense_embeddings


class SAM2VideoMaskDecoderWrapper(torch.nn.Module):
    """
    Wraps the mask decoding process. It does:
      1) get_conditioned_embedding(...)
      2) decode_masks_video_raw(...)
    """
    def __init__(self, video_model):
        super().__init__()
        self.video_model = video_model

    @torch.no_grad()
    def forward(
        self,
        video_embedding,   # [1, 256, H', W']
        memory_embedding,  # [1, 256, H', W']
        sparse_embedding,  # [1, M, 256]
        dense_embedding,   # [1, 256, H', W']
        feats_s0,         # [1, 32, 4H', 4W']   (example shape)
        feats_s1          # [1, 64, 2H', 2W']
    ):
        conditioned_embedding = self.video_model.get_conditioned_embedding(
            video_embedding=video_embedding,
            memory_embedding=memory_embedding,
            sparse_embedding=sparse_embedding,
            dense_embedding=dense_embedding
        )
        low_res_masks, iou_scores = self.video_model.decode_masks_video_raw(
            conditioned_embedding,
            None,  # memory_embeddings if needed
            sparse_embedding,
            dense_embedding,
            [feats_s0, feats_s1]
        )
        return low_res_masks, iou_scores

################################################################################
# Validation Functions
################################################################################
def validate_video_encoder(
    model: ct.models.MLModel,
    ground_model: SAM2VideoPredictor,
    prepared_video: torch.Tensor,
):
    """
    Checks that the CoreML model's outputs are close to the PyTorch outputs.
    """
    # Convert PyTorch tensor => NumPy for model.predict
    predictions = model.predict({"video": prepared_video.cpu().numpy()})
    with torch.no_grad():
        gt_vid_embed, gt_mem_embed, gt_s0, gt_s1 = ground_model.encode_video_raw(prepared_video)

    diff_vid = np.mean(np.abs(predictions["video_embedding"] - gt_vid_embed.cpu().numpy()))
    diff_mem = np.mean(np.abs(predictions["memory_embedding"] - gt_mem_embed.cpu().numpy()))
    diff_s0 = np.mean(np.abs(predictions["feats_s0"] - gt_s0.cpu().numpy()))
    diff_s1 = np.mean(np.abs(predictions["feats_s1"] - gt_s1.cpu().numpy()))

    print(f"Video Embedding Avg Diff: {diff_vid:.5f}")
    print(f"Memory Embedding Avg Diff: {diff_mem:.5f}")
    print(f"Feats S0 Avg Diff: {diff_s0:.5f}")
    print(f"Feats S1 Avg Diff: {diff_s1:.5f}")


def validate_prompt_encoder(
    model: ct.models.MLModel,
    ground_model: SAM2VideoPredictor,
    unnorm_coords, labels
):
    predictions = model.predict({
        "points": unnorm_coords.cpu().numpy(),
        "labels": labels.cpu().numpy()
    })
    with torch.no_grad():
        gt_sparse, gt_dense = ground_model.encode_points_raw(unnorm_coords, labels)

    diff_sparse = np.mean(np.abs(predictions["sparse_embeddings"] - gt_sparse.cpu().numpy()))
    diff_dense = np.mean(np.abs(predictions["dense_embeddings"] - gt_dense.cpu().numpy()))

    print(f"Sparse Embeddings Avg Diff: {diff_sparse:.5f}")
    print(f"Dense Embeddings Avg Diff: {diff_dense:.5f}")


def validate_mask_decoder(
    model: ct.models.MLModel,
    ground_model: SAM2VideoPredictor,
    video_embed, memory_embed, sparse_embed, dense_embed, s0, s1
):
    inputs = {
        "video_embedding": video_embed.cpu().numpy(),
        "memory_embedding": memory_embed.cpu().numpy(),
        "sparse_embedding": sparse_embed.cpu().numpy(),
        "dense_embedding": dense_embed.cpu().numpy(),
        "feats_s0": s0.cpu().numpy(),
        "feats_s1": s1.cpu().numpy(),
    }
    predictions = model.predict(inputs)

    with torch.no_grad():
        conditioned_embedding = ground_model.get_conditioned_embedding(
            video_embed, memory_embed, sparse_embed, dense_embed
        )
        gt_masks, gt_scores = ground_model.decode_masks_video_raw(
            conditioned_embedding,
            None,
            sparse_embed,
            dense_embed,
            [s0, s1]
        )

    diff_masks = np.mean(np.abs(predictions["low_res_masks"] - gt_masks.cpu().numpy()))
    print(f"Masks Avg Diff: {diff_masks:.5f}")
    print("Scores (CoreML):", predictions["scores"])
    print("Scores (PyTorch):", gt_scores.cpu().numpy())

################################################################################
# Export Functions
################################################################################
def export_video_encoder(
    video_predictor: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    """
    Export the 'video encoder' portion: encode_video_raw -> ONNX -> CoreML
    """
    # Use a smaller T=2 for the dummy input, shape [T, 3, H, W].
    dummy_video = torch.randn(2, 3, SAM2_HW[0], SAM2_HW[1], dtype=torch.float32)

    traced_model = torch.jit.trace(SAM2VideoEncoder(video_predictor).eval(), dummy_video)
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="video",
                shape=(2, 3, SAM2_HW[0], SAM2_HW[1]),
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="video_embedding"),
            ct.TensorType(name="memory_embedding"),
            ct.TensorType(name="feats_s0"),
            ct.TensorType(name="feats_s1"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    # Validate
    validate_video_encoder(mlmodel, video_predictor, dummy_video)

    # Save
    out_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}VideoEncoder{precision.value.upper()}")
    mlmodel.save(out_path + ".mlpackage")
    print(f"Saved video encoder to: {out_path}.mlpackage")


def export_points_prompt_encoder(
    video_predictor: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    """
    Export the 'points prompt encoder' portion, analogous to single-image prompt encoder.
    """
    # Dummy single point [1,1,2], label [1,1]
    dummy_points = torch.tensor([[[100.0, 200.0]]], dtype=torch.float32)
    dummy_labels = torch.tensor([[1]], dtype=torch.int32)

    traced_model = torch.jit.trace(
        SAM2PointsEncoder(video_predictor).eval(),
        (dummy_points, dummy_labels)
    )

    points_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16), 2))
    labels_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16)))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="points", shape=points_shape),
            ct.TensorType(name="labels", shape=labels_shape),
        ],
        outputs=[
            ct.TensorType(name="sparse_embeddings"),
            ct.TensorType(name="dense_embeddings"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    # Validate
    validate_prompt_encoder(mlmodel, video_predictor, dummy_points, dummy_labels)

    # Save
    out_path = os.path.join(output_dir, f"SAM2{variant.fmt()}VideoPromptEncoder{precision.value.upper()}")
    mlmodel.save(out_path + ".mlpackage")
    print(f"Saved video prompt encoder to: {out_path}.mlpackage")


def export_mask_decoder(
    video_predictor: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    """
    Export the 'mask decoder' portion. We feed:
      - video_embedding: [1,256,64,64]
      - memory_embedding: [1,256,64,64]
      - sparse_embedding: [1,M,256]
      - dense_embedding: [1,256,64,64]
      - feats_s0: [1,32,256,256]
      - feats_s1: [1,64,128,128]
    """
    # Dummy shapes
    video_embedding = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    memory_embedding = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    sparse_embedding = torch.randn(1, 3, 256, dtype=torch.float32)  # M=3 example
    dense_embedding = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    feats_s0 = torch.randn(1, 32, 256, 256, dtype=torch.float32)
    feats_s1 = torch.randn(1, 64, 128, 128, dtype=torch.float32)

    traced_model = torch.jit.trace(
        SAM2VideoMaskDecoderWrapper(video_predictor).eval(),
        (video_embedding, memory_embedding, sparse_embedding, dense_embedding, feats_s0, feats_s1)
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="video_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(name="memory_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(name="sparse_embedding", shape=[1, 3, 256]),
            ct.TensorType(name="dense_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(name="feats_s0", shape=[1, 32, 256, 256]),
            ct.TensorType(name="feats_s1", shape=[1, 64, 128, 128]),
        ],
        outputs=[
            ct.TensorType(name="low_res_masks"),
            ct.TensorType(name="scores"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    # Validate
    validate_mask_decoder(
        mlmodel,
        video_predictor,
        video_embedding,
        memory_embedding,
        sparse_embedding,
        dense_embedding,
        feats_s0,
        feats_s1
    )

    # Save
    out_path = os.path.join(output_dir, f"SAM2{variant.fmt()}VideoMaskDecoder{precision.value.upper()}")
    mlmodel.save(out_path + ".mlpackage")
    print(f"Saved video mask decoder to: {out_path}.mlpackage")

################################################################################
# Master export function that ties everything together
################################################################################
def export(
    output_dir: str,
    variant: SAM2Variant,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Build the SAM2 video model from HF or local checkpoint
    sam2_checkpoint = f"facebook/sam2.1-hiera-{variant.value}"
    video_predictor = SAM2VideoPredictor.from_pretrained(sam2_checkpoint, device=device)
    video_predictor.eval()

    # Export the submodules
    export_video_encoder(video_predictor, variant, output_dir, min_target, compute_units, precision)
    export_points_prompt_encoder(video_predictor, variant, output_dir, min_target, compute_units, precision)
    export_mask_decoder(video_predictor, variant, output_dir, min_target, compute_units, precision)

################################################################################
# CLI
################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Video -> CoreML CLI")
    parser = parse_args(parser)
    args = parser.parse_args()

    export(
        args.output_dir,
        args.variant,
        args.min_deployment_target,
        args.compute_units,
        args.precision,
    )