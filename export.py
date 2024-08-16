import argparse
import os
import ast
import torch
from typing import List, Tuple
import numpy as np
import enum
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import coremltools as ct
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools import ComputeUnit
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.mil import Builder as mb

class SAM2Variant(enum.Enum):
    Tiny = "tiny"
    Small = "small"
    BasePlus = "base_plus"
    Large = "large"

    def cfg(self):
        match self:
            case SAM2Variant.BasePlus:
                return "sam2_hiera_b+.yaml"
            case _:
                return f"sam2_hiera_{self.value[0]}.yaml"


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
        required=True,
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="List of binary labels for each points entry, denoting foreground (1) or background (0).",
        required=True,
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
    return parser


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


class SAM2ImageEncoder(torch.nn.Module):
    def __init__(self, model: SAM2ImagePredictor):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, image):
        (img_embedding, feats_s0, feats_s1) = self.model.encode_image_raw(image)
        return img_embedding, feats_s0, feats_s1


def validate_image_encoder(model: ct.models.MLModel, prepared_image: np.ndarray):
    predictions = model.predict({"image": prepared_image})

    ground_embedding = np.load("notebooks/image_embed.npy")
    ground_feats_s0 = np.load("notebooks/feats_s0.npy")
    ground_feats_s1 = np.load("notebooks/feats_s1.npy")

    img_max_diff = np.max(np.abs(predictions["image_embedding"] - ground_embedding))
    img_avg_diff = np.mean(np.abs(predictions["image_embedding"] - ground_embedding))

    s0_max_diff = np.max(np.abs(predictions["feats_s0"] - ground_feats_s0))
    s0_avg_diff = np.mean(np.abs(predictions["feats_s0"] - ground_feats_s0))

    s1_max_diff = np.max(np.abs(predictions["feats_s1"] - ground_feats_s1))
    s1_avg_diff = np.mean(np.abs(predictions["feats_s1"] - ground_feats_s1))

    print(
        f"Image Embedding: Max Diff: {img_max_diff:.4f}, Avg Diff: {img_avg_diff:.4f}"
    )
    print(f"Feats S0: Max Diff: {s0_max_diff:.4f}, Avg Diff: {s0_avg_diff:.4f}")
    print(f"Feats S1: Max Diff: {s1_max_diff:.4f}, Avg Diff: {s1_avg_diff:.4f}")

    assert np.allclose(predictions["image_embedding"], ground_embedding, atol=2e1) #ERROR: 2e1
    assert np.allclose(predictions["feats_s0"], ground_feats_s0, atol=5e-2)
    assert np.allclose(predictions["feats_s1"], ground_feats_s1, atol=6e-2)


def validate_prompt_encoder(model: ct.models.MLModel, points, labels):
    predictions = model.predict({"points": points, "labels": labels})

    ground_sparse = np.load("notebooks/sparse_embeddings.npy")
    ground_dense = np.load("notebooks/dense_embeddings.npy")

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

    assert np.allclose(predictions["sparse_embeddings"], ground_sparse, atol=6e-3)
    assert np.allclose(predictions["dense_embeddings"], ground_dense, atol=1e-3)


def validate_mask_decoder(
    model: ct.models.MLModel,
    image_embedding,
    sparse_embedding,
    dense_embedding,
    feats_s0,
    feats_s1,
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

    ground_masks = np.load("notebooks/low_res_masks.npy")

    masks_max_diff = np.max(np.abs(predictions["low_res_masks"] - ground_masks))
    masks_avg_diff = np.mean(np.abs(predictions["low_res_masks"] - ground_masks))

    print(
        "Masks: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            masks_max_diff, masks_avg_diff
        )
    )

    assert np.allclose(predictions["low_res_masks"], ground_masks, atol=7e-2)


class SAM2PromptEncoder(torch.nn.Module):
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
        low_res_masks = self.model.decode_masks_raw(
            image_embedding, sparse_embedding, dense_embedding, [feats_s0, feats_s1]
        )
        return low_res_masks


def export_image_encoder(
    image_predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
) -> Tuple[int, int]:
    # Prepare input tensors
    image = Image.open("notebooks/images/truck.jpg")
    image = np.array(image.convert("RGB"))
    orig_hw = (image.shape[0], image.shape[1])

    prepared_images = image_predictor._transforms(image)
    prepared_images = prepared_images[None, ...].to("cpu")

    traced_model = torch.jit.trace(SAM2ImageEncoder(image_predictor), prepared_images)

    output_path = os.path.join(output_dir, f"sam2_{variant.value}_image_encoder")

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image", shape=(1, 3, 1024, 1024)),
        ],
        outputs=[
            ct.TensorType(name="image_embedding"),
            ct.TensorType(name="feats_s0"),
            ct.TensorType(name="feats_s1"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
    )

    if variant == SAM2Variant.Small:
        validate_image_encoder(mlmodel, prepared_images)

    mlmodel.save(output_path + ".mlpackage")
    return orig_hw


def export_prompt_encoder(
    image_predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    input_points: List[List[float]],
    input_labels: List[int],
    orig_hw: tuple,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
):
    image_predictor.model.sam_prompt_encoder.eval()

    points = torch.tensor(input_points, dtype=torch.float32)
    labels = torch.tensor(input_labels, dtype=torch.int32)

    unnorm_coords = image_predictor._transforms.transform_coords(
        points,
        normalize=True,
        orig_hw=orig_hw,  # todo: avoid hardcoding truck.jpg
    )
    unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]

    traced_model = torch.jit.trace(
        SAM2PromptEncoder(image_predictor), (unnorm_coords, labels)
    )

    output_path = os.path.join(output_dir, f"sam2_{variant.value}_prompt_encoder")

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
    )

    if variant == SAM2Variant.Small:
        validate_prompt_encoder(mlmodel, unnorm_coords, labels)

    mlmodel.save(output_path + ".mlpackage")


def export_mask_decoder(
    image_predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
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

    output_path = os.path.join(output_dir, f"sam2_{variant.value}_mask_decoder")

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
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
    )


    if variant == SAM2Variant.Small:
        image_embedding = np.load("notebooks/image_embed.npy")
        sparse_embedding = np.load("notebooks/sparse_embeddings.npy")
        dense_embedding = np.load("notebooks/dense_embeddings.npy")
        s0 = np.load("notebooks/feats_s0.npy")
        s1 = np.load("notebooks/feats_s1.npy")
        validate_mask_decoder(
            mlmodel, image_embedding, sparse_embedding, dense_embedding, s0, s1
        )

    mlmodel.save(output_path + ".mlpackage")


def export(
    output_dir: str,
    variant: SAM2Variant,
    points: list,
    labels: list,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    # Build SAM2 model
    sam2_checkpoint = f"./checkpoints/sam2_hiera_{variant.value}.pt"
    model_cfg = variant.cfg()

    with torch.no_grad():
        model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        img_predictor = SAM2ImagePredictor(model)
        img_predictor.model.eval()

        orig_hw = export_image_encoder(img_predictor, variant, output_dir, min_target, compute_units)
        export_prompt_encoder(
          img_predictor, variant, points, labels, orig_hw, output_dir, min_target, compute_units
        )
        export_mask_decoder(img_predictor, variant, output_dir, min_target, compute_units)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 -> CoreML CLI")
    parser = parse_args(parser)
    args = parser.parse_args()

    # Process points and labels
    try:
        points = ast.literal_eval(args.points)
        labels = ast.literal_eval(args.labels)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid format for points or labels: {e}")

    if not isinstance(points, list) or not all(
        isinstance(p, list) and len(p) == 2 for p in points
    ):
        raise ValueError("Points must be a list of [x, y] coordinates")

    if not isinstance(labels, list) or not all(
        isinstance(l, int) and l in [0, 1] for l in labels
    ):
        raise ValueError("Labels must denote foreground (1) or background (0)")

    if len(points) != len(labels):
        raise ValueError("Number of points must match the number of labels")

    if len(points) > 16:
        raise ValueError("Number of points must be less than or equal to 16")

    export(
        args.output_dir,
        args.variant,
        points,
        labels,
        args.min_deployment_target,
        args.compute_units,
    )
