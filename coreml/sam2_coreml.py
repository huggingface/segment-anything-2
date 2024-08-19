import numpy as np
from PIL import Image
import coremltools as ct
from sam2.utils.transforms import SAM2Transforms
import torch
import matplotlib.pyplot as plt

class SAM2CoreMLPredictor:
    def __init__(self, variant="small", model_dir=".", mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0):
        self.variant = variant
        self.model_dir = model_dir
        self._transforms = SAM2Transforms(
            resolution=1024,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self.mask_threshold = mask_threshold
        self.orig_hw = (1200, 1800)
        self.load_models()

    def load_models(self):
        self.image_encoder = ct.models.MLModel(f"{self.model_dir}/sam2_{self.variant}_image_encoder.mlpackage")
        self.prompt_encoder = ct.models.MLModel(f"{self.model_dir}/sam2_{self.variant}_prompt_encoder.mlpackage")
        self.mask_decoder = ct.models.MLModel(f"{self.model_dir}/sam2_{self.variant}_mask_decoder.mlpackage")

    def encode_image(self, image):
        output = self.image_encoder.predict({"image": image})
        return output["image_embedding"], output["feats_s0"], output["feats_s1"]

    def encode_prompt(self, points, labels):
        output = self.prompt_encoder.predict({"points": points, "labels": labels})
        return output["sparse_embeddings"], output["dense_embeddings"]

    def decode_mask(self, image_embedding, sparse_embeddings, dense_embeddings, feats_s0, feats_s1):
        output = self.mask_decoder.predict({
            "image_embedding": image_embedding,
            "sparse_embedding": sparse_embeddings,
            "dense_embedding": dense_embeddings,
            "feats_s0": feats_s0,
            "feats_s1": feats_s1
        })
        return output["low_res_masks"]

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        self.orig_hw = (image.shape[0], image.shape[1])

        transformed = self._transforms(image)
        return transformed[None, ...].to("cpu")

    def preprocess_prompt(self, points, labels):
        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int32)

        unnorm_coords = self._transforms.transform_coords(points, normalize=True, orig_hw=self.orig_hw)
        return unnorm_coords[None, ...], labels[None, ...]

    def predict(self, image_path, points, labels):
        input_image = self.load_image(image_path)
        image_embedding, feats_s0, feats_s1 = self.encode_image(input_image)
        
        input_points, input_labels = self.preprocess_prompt(points, labels)
        sparse_embeddings, dense_embeddings = self.encode_prompt(input_points, input_labels)
        
        # Decode mask
        low_res_masks = self.decode_mask(image_embedding, sparse_embeddings, dense_embeddings, feats_s0, feats_s1)
        
        # Postprocess mask
        low_res_masks = torch.tensor(low_res_masks, dtype=torch.float32)
        masks = self._transforms.postprocess_masks(low_res_masks, self.orig_hw)
        if not False:
            masks = masks > self.mask_threshold
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np 


def show_mask(mask, ax, borders = True):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        print("Shape before show_mask: ", mask.shape)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            print("Point coords: ", point_coords)
            print("Input labels: ", input_labels)
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"CoreML\nMask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"mask_{i}.png")

# Example usage
if __name__ == "__main__":
    predictor = SAM2CoreMLPredictor(variant="small")
    
    image_path = "../notebooks/images/truck.jpg"

    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    masks = predictor.predict(image_path, input_point, input_label)
    show_masks(image, masks, [1,1,1], point_coords=input_point, input_labels=input_label, borders=True)
