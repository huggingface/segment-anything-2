import argparse
import ast
import numpy as np
from PIL import Image
from sam2_coreml import SAM2CoreMLPredictor, show_masks, SAM2Variant


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 CoreML CLI")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory containing the exported CoreML models.",
    )
    parser.add_argument(
        "--variant",
        type=lambda v: getattr(SAM2Variant, v),
        choices=[v for v in SAM2Variant],
        default=SAM2Variant.Small,
        help="SAM2 variant to use.",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input image.",
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
        "--mask-threshold",
        type=float,
        default=0.0,
        help="Threshold for mask binarization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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

    predictor = SAM2CoreMLPredictor(
        variant=args.variant.name,
        model_dir=args.model_dir,
        mask_threshold=args.mask_threshold,
    )

    image = Image.open(args.image_path)
    masks = predictor.predict(args.image_path, np.array(points), np.array(labels))

    show_masks(
        image,
        masks,
        [1, 1, 1],
        point_coords=np.array(points),
        input_labels=np.array(labels),
    )


if __name__ == "__main__":
    main()
