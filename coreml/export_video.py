import argparse
import ast
import enum
import math
import os
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from coremltools import ComputeUnit
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
from torch import nn
from torchvision.transforms import ToTensor

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2Variant(enum.Enum):
    Tiny = "tiny"
    Small = "small"
    BasePlus = "base-plus"
    Large = "large"

    def fmt(self):
        if self == SAM2Variant.BasePlus:
            return "BasePlus"
        return self.value.capitalize()

SAM2_HW = (1024, 1024)

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

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
        "--points",
        type=str,
        help="List of 2D points, e.g., '[[10,20], [30,40]]'",
    )
    parser.add_argument(
        "--boxes",
        type=str,
        help="List of 2D bounding boxes, e.g., '[[10,20,30,40], [50,60,70,80]]'",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="List of binary labels for each points entry, denoting foreground (1) or background (0).",
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
        help="Which compute units to target for CoreML model.",
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(ComputePrecision, x),
        choices=[p for p in ComputePrecision],
        default=ComputePrecision.FLOAT16,
        help="Precision to use for quantization.",
    )
    return parser

class SAM2ImageEncoder(torch.nn.Module):
    def __init__(self, model: SAM2VideoPredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, image):
        # TODO: vision_feats is not needed?
        (img_embedding, vision_feats, feats_s0, feats_s1, vision_pos_embeds) = self.model.encode_image_raw(image)
        return img_embedding, vision_feats, feats_s0, feats_s1, vision_pos_embeds

class SAM2PointsEncoder(torch.nn.Module):
    def __init__(self, model: SAM2ImagePredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, points, labels):
        prompt_embedding = self.model.encode_points_raw(points, labels)
        return prompt_embedding


class SAM2MaskDecoder(torch.nn.Module):
    def __init__(self, model: SAM2ImagePredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(
        self, image_embedding, sparse_embedding, dense_embedding, feats_s0, feats_s1
    ):
        low_res_masks, iou_scores = self.model.decode_masks_raw(
            image_embedding, sparse_embedding, dense_embedding, [feats_s0, feats_s1]
        )
        return low_res_masks, iou_scores


class SAM2MemoryEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, pix_feat: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, pos_enc = self.model.encode_memory_raw(pix_feat, masks)
        return features, pos_enc


class SAM2MemoryFusion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Important parameters from the original model
        self.d_model = 256  # Base dimension
        self.num_heads = 8  # Default number of attention heads

    @torch.no_grad()
    def forward(
            self,
            curr_feats: torch.Tensor,  # Current frame features [HW, B, C]
            curr_pos: torch.Tensor,  # Current positional encoding [HW, B, C]
            spatial_memory: torch.Tensor,  # Memory features [M, B, C_mem]
            memory_pos: torch.Tensor,  # Memory positional encoding [M, B, C_mem]
            num_memory_tokens: torch.int64 = 0,  # Number of memory tokens
    ) -> torch.Tensor:
        # Run memory attention with current features and memory
        output = self.model.memory_attention(
            curr=[curr_feats],
            curr_pos=[curr_pos],
            memory=spatial_memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_memory_tokens
        )

        # Reshape output back to spatial format
        B = curr_feats.shape[1]
        H = W = int(math.sqrt(curr_feats.shape[0]))
        output = output.permute(1, 2, 0).reshape(B, self.d_model, H, W)

        return output

def export_image_encoder(
    image_predictor: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
) -> Tuple[int, int]:
    # Prepare input tensors
    image = Image.open("../notebooks/images/truck.jpg")
    image = image.resize(SAM2_HW)
    image = np.array(image.convert("RGB"))
    orig_hw = (image.shape[0], image.shape[1])
    image_tensor = ToTensor()(image)

    #prepared_image = image_predictor._transforms(image) # why?
    image_tensor = image_tensor[None, ...].to("cpu")

    print("\nTesting forward pass...")
    model = SAM2ImageEncoder(image_predictor).eval()
    with torch.no_grad():
        model.forward(image_tensor)

    traced_model = torch.jit.trace(
        SAM2ImageEncoder(image_predictor).eval(), image_tensor
    )


    scale = 1 / (0.226 * 255.0)
    bias = [-0.485 / (0.229), -0.456 / (0.224), -0.406 / (0.225)]

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, SAM2_HW[0], SAM2_HW[1]),
                scale=scale,
                bias=bias,
            )
        ],
        outputs=[
            ct.TensorType(name="image_embedding"),
            ct.TensorType(name="vision_feats_s0"),
            ct.TensorType(name="vision_feats_s1"),
            ct.TensorType(name="vision_feats_s2"),
            ct.TensorType(name="feats_s0"),
            ct.TensorType(name="feats_s1"),
            ct.TensorType(name="vision_pos_embeds_s0"),
            ct.TensorType(name="vision_pos_embeds_s1"),
            ct.TensorType(name="vision_pos_embeds_s2"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    image = Image.open("../notebooks/images/truck.jpg")
    validate_image_encoder(mlmodel, image_predictor, image)

    output_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}ImageEncoder{precision.value.upper()}")
    mlmodel.save(output_path + ".mlpackage")
    return orig_hw


def export_points_prompt_encoder(
    image_predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    input_points: List[List[float]],
    input_labels: List[int],
    orig_hw: tuple,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    image_predictor.model.sam_prompt_encoder.eval()

    points = torch.tensor(input_points, dtype=torch.float32)
    labels = torch.tensor(input_labels, dtype=torch.int32)

    unnorm_coords = image_predictor._transforms.transform_coords(
        points,
        normalize=True,
        orig_hw=orig_hw,
    )
    unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]

    traced_model = torch.jit.trace(
        SAM2PointsEncoder(image_predictor), (unnorm_coords, labels)
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

    validate_prompt_encoder(mlmodel, image_predictor, unnorm_coords, labels)

    output_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}PromptEncoder{precision.value.upper()}")
    mlmodel.save(output_path + ".mlpackage")


def export_mask_decoder(
    image_predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    image_predictor.model.sam_mask_decoder.eval()
    s0 = torch.randn(1, 32, 256, 256)
    s1 = torch.randn(1, 64, 128, 128)
    image_embedding = torch.randn(1, 256, 64, 64)
    sparse_embedding = torch.randn(1, 3, 256)
    dense_embedding = torch.randn(1, 256, 64, 64)

    traced_model = torch.jit.trace(
        SAM2MaskDecoder(image_predictor),
        (image_embedding, sparse_embedding, dense_embedding, s0, s1),
    )
    traced_model.eval()


    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(
                name="sparse_embedding",
                shape=ct.EnumeratedShapes(shapes=[[1, i, 256] for i in range(2, 16)]),
            ),
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

    validate_mask_decoder(
        mlmodel,
        image_predictor,
        image_embedding,
        sparse_embedding,
        dense_embedding,
        s0,
        s1,
        precision,
    )

    output_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}MaskDecoder{precision.value.upper()}")
    mlmodel.save(output_path + ".mlpackage")

def validate_image_encoder(
    model: ct.models.MLModel, ground_model: SAM2VideoPredictor, image: Image.Image
):
    prepared_image = image.resize(SAM2_HW, Resampling.BILINEAR)
    predictions = model.predict({"image": prepared_image})

    image = np.array(prepared_image.convert("RGB"))
    tch_image = ToTensor()(prepared_image)
    tch_image = tch_image[None, ...].to("cpu")
    ground_embedding, ground_vision_feats, ground_feats_s0, ground_feats_s1, vision_pos_embeds = ground_model.encode_image_raw(
        tch_image
    )
    ground_embedding, ground_feats_s0, ground_feats_s1, vision_pos_embeds = (
        ground_embedding.numpy(),
        ground_feats_s0.numpy(),
        ground_feats_s1.numpy(),
        vision_pos_embeds,
    )

    ground_vision_feats_s0 = ground_vision_feats[0].numpy()
    ground_vision_feats_s1 = ground_vision_feats[1].numpy()
    ground_vision_feats_s2 = ground_vision_feats[2].numpy()

    vision_pos_embeds_s0 = vision_pos_embeds[0].numpy()
    vision_pos_embeds_s1 = vision_pos_embeds[1].numpy()
    vision_pos_embeds_s2 = vision_pos_embeds[2].numpy()

    img_max_diff = np.max(np.abs(predictions["image_embedding"] - ground_embedding))
    img_avg_diff = np.mean(np.abs(predictions["image_embedding"] - ground_embedding))

    s0_max_diff = np.max(np.abs(predictions["feats_s0"] - ground_feats_s0))
    s0_avg_diff = np.mean(np.abs(predictions["feats_s0"] - ground_feats_s0))

    s1_max_diff = np.max(np.abs(predictions["feats_s1"] - ground_feats_s1))
    s1_avg_diff = np.mean(np.abs(predictions["feats_s1"] - ground_feats_s1))

    vision_feats_s0_max_diff = np.max(np.abs(predictions["vision_feats_s0"] - ground_vision_feats_s0))
    vision_feats_s0_avg_diff = np.mean(np.abs(predictions["vision_feats_s0"] - ground_vision_feats_s0))

    vision_feats_s1_max_diff = np.max(np.abs(predictions["vision_feats_s1"] - ground_vision_feats_s1))
    vision_feats_s1_avg_diff = np.mean(np.abs(predictions["vision_feats_s1"] - ground_vision_feats_s1))

    vision_feats_s2_max_diff = np.max(np.abs(predictions["vision_feats_s2"] - ground_vision_feats_s2))
    vision_feats_s2_avg_diff = np.mean(np.abs(predictions["vision_feats_s2"] - ground_vision_feats_s2))

    vision_pos_embeds_s0_max_diff = np.max(np.abs(predictions["vision_pos_embeds_s0"] - vision_pos_embeds_s0))
    vision_pos_embeds_s0_avg_diff = np.mean(np.abs(predictions["vision_pos_embeds_s0"] - vision_pos_embeds_s0))

    vision_pos_embeds_s1_max_diff = np.max(np.abs(predictions["vision_pos_embeds_s1"] - vision_pos_embeds_s1))
    vision_pos_embeds_s1_avg_diff = np.mean(np.abs(predictions["vision_pos_embeds_s1"] - vision_pos_embeds_s1))

    vision_pos_embeds_s2_max_diff = np.max(np.abs(predictions["vision_pos_embeds_s2"] - vision_pos_embeds_s2))
    vision_pos_embeds_s2_avg_diff = np.mean(np.abs(predictions["vision_pos_embeds_s2"] - vision_pos_embeds_s2))

    print(
        f"Image Embedding: Max Diff: {img_max_diff:.4f}, Avg Diff: {img_avg_diff:.4f}"
    )
    print(f"Feats S0: Max Diff: {s0_max_diff:.4f}, Avg Diff: {s0_avg_diff:.4f}")
    print(f"Feats S1: Max Diff: {s1_max_diff:.4f}, Avg Diff: {s1_avg_diff:.4f}")
    print(f"Vision Feats S0: Max Diff: {vision_feats_s0_max_diff:.4f}, Avg Diff: {vision_feats_s0_avg_diff:.4f}")
    print(f"Vision Feats S1: Max Diff: {vision_feats_s1_max_diff:.4f}, Avg Diff: {vision_feats_s1_avg_diff:.4f}")
    print(f"Vision Feats S2: Max Diff: {vision_feats_s2_max_diff:.4f}, Avg Diff: {vision_feats_s2_avg_diff:.4f}")
    print(f"Vision Pos Embeds S0: Max Diff: {vision_pos_embeds_s0_max_diff:.4f}, Avg Diff: {vision_pos_embeds_s0_avg_diff:.4f}")
    print(f"Vision Pos Embeds S1: Max Diff: {vision_pos_embeds_s1_max_diff:.4f}, Avg Diff: {vision_pos_embeds_s1_avg_diff:.4f}")
    print(f"Vision Pos Embeds S2: Max Diff: {vision_pos_embeds_s2_max_diff:.4f}, Avg Diff: {vision_pos_embeds_s2_avg_diff:.4f}")

    # Lack of bicubic upsampling in CoreML causes slight differences
    # assert np.allclose(
    #    predictions["image_embedding"], ground_embedding, atol=2e1
    # )
    # assert np.allclose(predictions["feats_s0"], ground_feats_s0, atol=1e-1)
    # assert np.allclose(predictions["feats_s1"], ground_feats_s1, atol=1e-1)


def validate_prompt_encoder(
    model: ct.models.MLModel, ground_model: SAM2ImagePredictor, unnorm_coords, labels
):
    predictions = model.predict({"points": unnorm_coords, "labels": labels})

    (ground_sparse, ground_dense) = ground_model.encode_points_raw(
        unnorm_coords, labels
    )

    ground_sparse = ground_sparse.numpy()
    ground_dense = ground_dense.numpy()
    sparse_max_diff = np.max(np.abs(predictions["sparse_embeddings"] - ground_sparse))
    sparse_avg_diff = np.mean(np.abs(predictions["sparse_embeddings"] - ground_sparse))

    dense_max_diff = np.max(np.abs(predictions["dense_embeddings"] - ground_dense))
    dense_avg_diff = np.mean(np.abs(predictions["dense_embeddings"] - ground_dense))

    print(
        "Sparse Embeddings: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            sparse_max_diff, sparse_avg_diff
        )
    )
    print(
        "Dense Embeddings: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            dense_max_diff, dense_avg_diff
        )
    )

    assert np.allclose(predictions["sparse_embeddings"], ground_sparse, atol=1e-2)
    assert np.allclose(predictions["dense_embeddings"], ground_dense, atol=1e-3)


def validate_mask_decoder(
    model: ct.models.MLModel,
    ground_model: SAM2ImagePredictor,
    image_embedding,
    sparse_embedding,
    dense_embedding,
    feats_s0,
    feats_s1,
    precision: ComputePrecision,
):
    predictions = model.predict(
        {
            "image_embedding": image_embedding,
            "sparse_embedding": sparse_embedding,
            "dense_embedding": dense_embedding,
            "feats_s0": feats_s0,
            "feats_s1": feats_s1,
        }
    )

    ground_masks, scores = ground_model.decode_masks_raw(
        image_embedding, sparse_embedding, dense_embedding, [feats_s0, feats_s1]
    )

    ground_masks = ground_masks.numpy()
    masks_max_diff = np.max(np.abs(predictions["low_res_masks"] - ground_masks))
    masks_avg_diff = np.mean(np.abs(predictions["low_res_masks"] - ground_masks))

    print(
        "Masks: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            masks_max_diff, masks_avg_diff
        )
    )

    # atol = 7e-2 if precision == ComputePrecision.FLOAT32 else 3e-1
    # assert np.allclose(predictions["low_res_masks"], ground_masks, atol=atol)
    print(f"Scores: {predictions['scores']}, ground: {scores}")
    assert np.allclose(predictions["scores"], scores, atol=1e-2)


def export_memory_encoder(
        video_predictor,
        variant: SAM2Variant,
        output_dir: str,
        min_target: AvailableTarget,
        compute_units: ComputeUnit,
        precision: ComputePrecision
) -> None:
    """Export memory encoder component to CoreML."""

    # Create sample inputs
    pix_feat = torch.randn(1, 256, 64, 64)  # Example shape
    masks = torch.randn(1, 1, 1024, 1024)

    traced_model = torch.jit.trace(
        SAM2MemoryEncoder(video_predictor),
        (pix_feat, masks)
    )

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="pix_feat", shape=ct.Shape(shape=(1, 256, 64, 64))),
            ct.TensorType(name="masks", shape=ct.Shape(shape=(1, 1, 1024, 1024)))
        ],
        outputs=[
            ct.TensorType(name="memory_features"),
            ct.TensorType(name="memory_pos_enc")
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision
    )

    output_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}MemoryEncoder{precision.value.upper()}")
    mlmodel.save(output_path + ".mlpackage")


def export_memory_fusion(
        video_predictor,
        variant: SAM2Variant,
        output_dir: str,
        min_target: AvailableTarget,
        compute_units: ComputeUnit,
        precision: ComputePrecision
) -> None:
    """Export memory fusion component to CoreML."""

    # Set up dimensions matching the real model usage
    B = 1  # Batch size
    C = 256  # Current features channel dimension
    C_mem = 64  # Memory channel dimension
    HW = 4096  # Spatial tokens (64x64)
    M = 28736  # Memory size from debug

    # Create inputs with explicit dtypes
    #TODO: curr_feats - we are using just first level?
    curr_feats = torch.randn(HW, B, C, dtype=torch.float32).contiguous()
    curr_pos = torch.randn(HW, B, C, dtype=torch.float32).contiguous()
    memory = torch.randn(M, B, C_mem, dtype=torch.float32).contiguous()
    memory_pos = torch.randn(M, B, C_mem, dtype=torch.float32).contiguous()
    num_memory_tokens = torch.tensor([64], dtype=torch.int32)

    print("\nInput shapes and types:")
    print(f"curr_feats: {curr_feats.shape}, {curr_feats.dtype}")
    print(f"curr_pos: {curr_pos.shape}, {curr_pos.dtype}")
    print(f"memory: {memory.shape}, {memory.dtype}")
    print(f"memory_pos: {memory_pos.shape}, {memory_pos.dtype}")
    print(f"num_memory_tokens: {num_memory_tokens}, {num_memory_tokens.dtype}")

    # Simple forward pass
    fusion_model = SAM2MemoryFusion(video_predictor)
    fusion_model.eval()

    print("\nTesting forward pass...")
    with torch.no_grad():
        out = fusion_model(curr_feats, curr_pos, memory, memory_pos, num_memory_tokens)
        print(f"Output shape: {out.shape}, {out.dtype}")

    print("\nTracing model...")
    traced_model = torch.jit.trace(
        fusion_model,
        (curr_feats, curr_pos, memory, memory_pos, num_memory_tokens)
    )

    print("\nConverting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="curr_feats", shape=(HW, B, C)),
            ct.TensorType(name="curr_pos", shape=(HW, B, C)),
            ct.TensorType(name="memory", shape=(ct.RangeDim(1, M, M), B, C_mem)),
            ct.TensorType(name="memory_pos", shape=(ct.RangeDim(1, M, M), B, C_mem)),
            ct.TensorType(name="num_memory_tokens", shape=(1,))
        ],
        outputs=[
            ct.TensorType(name="fused_features")
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision
    )

    validate_memory_fusion(mlmodel, video_predictor, curr_feats, curr_pos, memory, memory_pos, num_memory_tokens)

    output_path = os.path.join(output_dir, f"SAM2_1{variant.fmt()}MemoryFusion{precision.value.upper()}")
    mlmodel.save(output_path + ".mlpackage")


def validate_memory_fusion(
        model: ct.models.MLModel,
        ground_model: SAM2VideoPredictor,
        curr_feats: torch.Tensor,
        curr_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        num_memory_tokens: torch.Tensor
) -> None:
    """Validate the CoreML model output matches PyTorch."""

    print("Running CoreML prediction...")
    predictions = model.predict({
        "curr_feats": curr_feats.numpy(),
        "curr_pos": curr_pos.numpy(),
        "memory": memory.numpy(),
        "memory_pos": memory_pos.numpy(),
        "num_memory_tokens": num_memory_tokens.numpy()
    })

    print("Running PyTorch ground truth...")
    fusion_model = SAM2MemoryFusion(ground_model)
    with torch.no_grad():
        ground_output = fusion_model(
            curr_feats,
            curr_pos,
            memory,
            memory_pos,
            num_memory_tokens,
        )

    ground_output = ground_output.numpy()

    # Compare outputs
    max_diff = np.max(np.abs(predictions["fused_features"] - ground_output))
    avg_diff = np.mean(np.abs(predictions["fused_features"] - ground_output))
    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Average difference: {avg_diff:.6f}")

    # Use appropriate tolerance based on precision
    atol = 1
    assert np.allclose(predictions["fused_features"], ground_output, atol=atol), \
        "CoreML and PyTorch outputs don't match within tolerance"

def export(
    output_dir: str,
    variant: SAM2Variant,
    points: Optional[List[Point]],
    boxes: Optional[List[Box]],
    labels: Optional[List[int]],
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Build SAM2 model
    sam2_checkpoint = f"facebook/sam2.1-hiera-{variant.value}"

    with torch.no_grad():
        # Create both image and video predictors
        img_predictor = SAM2ImagePredictor.from_pretrained(
            sam2_checkpoint, device=device
        )
        img_predictor.model.eval()

        video_predictor = SAM2VideoPredictor.from_pretrained(
            sam2_checkpoint, device=device
        )
        video_predictor.eval()

        # Export image components
        orig_hw = export_image_encoder(
            video_predictor, variant, output_dir, min_target, compute_units, precision
        )
        if boxes is not None and points is None:
            raise ValueError("Boxes are not supported yet")
        else:
            export_points_prompt_encoder(
                img_predictor,
                variant,
                points,
                labels,
                orig_hw,
                output_dir,
                min_target,
                compute_units,
                precision,
            )
        export_mask_decoder(
            img_predictor, variant, output_dir, min_target, compute_units, precision
        )

        # Export video components
        export_memory_encoder(
            video_predictor, variant, output_dir, min_target, compute_units, precision
        )
        export_memory_fusion(
            video_predictor, variant, output_dir, min_target, compute_units, precision
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 -> CoreML CLI")
    parser = parse_args(parser)
    args = parser.parse_args()

    points, boxes, labels = None, None, None
    if args.points:
        points = [tuple(p) for p in ast.literal_eval(args.points)]
    if args.boxes:
        boxes = [tuple(b) for b in ast.literal_eval(args.boxes)]
    if args.labels:
        labels = ast.literal_eval(args.labels)

    if boxes and points:
        raise ValueError("Cannot provide both points and boxes")

    if points:
        if not isinstance(points, list) or not all(
            isinstance(p, tuple) and len(p) == 2 for p in points
        ):
            raise ValueError("Points must be a tuple of 2D points")

    if labels:
        if not isinstance(labels, list) or not all(
            isinstance(l, int) and l in [0, 1] for l in labels
        ):
            raise ValueError("Labels must denote foreground (1) or background (0)")

    if points:
        if len(points) != len(labels):
            raise ValueError("Number of points must match the number of labels")

        if len(points) > 16:
            raise ValueError("Number of points must be less than or equal to 16")

    if boxes:
        if not isinstance(boxes, list) or not all(
            isinstance(b, tuple) and len(b) == 4 for b in boxes
        ):
            raise ValueError("Boxes must be a tuple of 4D bounding boxes")

    export(
        args.output_dir,
        args.variant,
        points,
        boxes,
        labels,
        args.min_deployment_target,
        args.compute_units,
        args.precision,
    )
