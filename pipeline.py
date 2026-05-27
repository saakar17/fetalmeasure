"""
UltraMeasure — Inference Pipeline
Handles preprocessing, classification, segmentation, and AC estimation.
"""

import os
from os import path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE         = 224
MEAN             = 0.11916620971431377
STD              = 0.1479186227862317
SCANNER_SPACING  = 0.28                                # mm/pixel (original)
ORIGINAL_SIZE    = 744                                 # max side after pad
PIXEL_SPACING_MM = SCANNER_SPACING / (IMG_SIZE / ORIGINAL_SIZE)  # ~0.930


# ── Preprocessing ──────────────────────────────────────────────────────────

def pad_to_square(img: Image.Image) -> Image.Image:
    """Zero-pad shorter side to square — preserves ellipse geometry."""
    w, h     = img.size
    max_side = max(w, h)
    pad_l = (max_side - w) // 2;  pad_r = max_side - w - pad_l
    pad_t = (max_side - h) // 2;  pad_b = max_side - h - pad_t
    return TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)


def preprocess(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalised [3, 224, 224] tensor."""
    img = img.convert("L")
    img = pad_to_square(img)
    img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
    t   = TF.to_tensor(img).repeat(3, 1, 1)
    t   = TF.normalize(t, mean=[MEAN] * 3, std=[STD] * 3)
    return t


# ── Classifier ─────────────────────────────────────────────────────────────

class Classifier(nn.Module):
    """ResNet-34 with replaced FC head for binary classification."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet34(pretrained=False)
        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        self.model  = backbone

    def forward(self, x):
        return self.model(x)


# ── U-Net components ───────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Upsample → concatenate skip → Conv-BN-ReLU × 2."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNetResNet34(nn.Module):
    """U-Net with pretrained ResNet-34 encoder. Output: [B, 1, 224, 224] logits."""

    def __init__(self):
        super().__init__()
        bb = models.resnet34(pretrained=False)
        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.enc1 = nn.Sequential(bb.maxpool, bb.layer1)
        self.enc2 = bb.layer2
        self.enc3 = bb.layer3
        self.enc4 = bb.layer4
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128,  64,  64)
        self.dec1 = DecoderBlock( 64,  64,  32)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x);  e1 = self.enc1(e0)
        e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        d  = self.dec4(e4, e3); d = self.dec3(d, e2)
        d  = self.dec2(d,  e1); d = self.dec1(d, e0)
        d  = F.interpolate(d, scale_factor=2,
                           mode='bilinear', align_corners=False)
        return self.output_conv(self.final_upsample(d))


# ── AC Estimation ──────────────────────────────────────────────────────────

def estimate_ac(binary_mask: np.ndarray,
                pixel_spacing_mm: float = PIXEL_SPACING_MM):
    """
    Fit ellipse to the largest contour in a binary mask and compute AC
    using Ramanujan's approximation.

    Args:
        binary_mask      : float32 numpy array [H, W], values 0.0 / 1.0
        pixel_spacing_mm : effective mm/pixel in the 224×224 space

    Returns:
        ac_mm   : estimated circumference in mm, or None on failure
        ellipse : cv2 ellipse tuple, or None
        contour : largest contour, or None
    """
    mask_u8    = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, None

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None, None, largest

    try:
        ellipse  = cv2.fitEllipse(largest)
        a_mm     = (ellipse[1][0] / 2.0) * pixel_spacing_mm
        b_mm     = (ellipse[1][1] / 2.0) * pixel_spacing_mm
        h        = ((a_mm - b_mm) ** 2) / ((a_mm + b_mm) ** 2)
        ac_mm    = np.pi * (a_mm + b_mm) * (
            1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))
        )
        return float(ac_mm), ellipse, largest
    except cv2.error:
        return None, None, largest


# ── Inference Pipeline ─────────────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end pipeline: PIL images → classification → segmentation → AC.

    Usage:
        pipeline = InferencePipeline('models/best_model1.pth',
                                     'models/best_seg_model.pth')
        results  = pipeline.run(pil_images_list)
    """

    def __init__(self, cls_path: str, seg_path: str):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.classifier = self._load_classifier(cls_path)
        self.segmenter  = self._load_segmenter(seg_path)

    def _load_classifier(self, path: str) -> nn.Module:
        model = Classifier()
        state = torch.load(path, map_location=self.device, weights_only=False)

        # best_model1.pth was saved from a plain ResNet-34 with replaced FC,
        # not wrapped in a Classifier() class. Keys are 'layer1.x', 'fc.x' etc.
        # Try direct load first, fall back to remapping if it fails.
        try:
            model.model.load_state_dict(state)
        except RuntimeError:
        # If keys have a 'model.' prefix already, load directly
            model.load_state_dict(state)

        return model.eval().to(self.device)

 

    def _load_segmenter(self, path: str) -> nn.Module:
        model = UNetResNet34()
        state = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(state)
        return model.eval().to(self.device)

    def classify_all(self, images: list) -> list:
        """
        Classify all images and return sorted by probability descending.
        Returns: [(pil_image, probability), ...]
        """
        results = []
        with torch.no_grad():
            for img in images:
                t    = preprocess(img).unsqueeze(0).to(self.device)
                prob = torch.sigmoid(self.classifier(t)).item()
                results.append((img, prob))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def segment(self, pil_image: Image.Image):
        """
        Segment a single PIL image.
        Returns: (prob_map [H,W], binary_mask [H,W])
        """
        t      = preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.segmenter(t)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
        return prob, (prob > 0.5).astype(np.float32)

    def make_overlay(self, pil_image: Image.Image,
                     binary_mask: np.ndarray) -> Image.Image:
        """Overlay teal segmentation mask on grayscale image."""
        arr     = np.array(pil_image.convert("L"))
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        h, w    = arr.shape

        mask_rs = cv2.resize(
            binary_mask.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        overlay          = np.zeros_like(img_bgr)
        overlay[mask_rs > 0] = [0, 180, 120]            # teal
        result = cv2.addWeighted(img_bgr, 0.72, overlay, 0.28, 0)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def draw_ellipse(self, pil_image: Image.Image,
                     ellipse) -> Image.Image:
        """Draw fitted ellipse on the image, scaled to original dimensions."""
        arr     = np.array(pil_image.convert("L"))
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        h, w    = arr.shape

        # Scale ellipse from 224×224 → original image size
        sx = w / IMG_SIZE;  sy = h / IMG_SIZE
        cx, cy   = ellipse[0][0] * sx, ellipse[0][1] * sy
        ax, ay   = ellipse[1][0] * sx, ellipse[1][1] * sy
        scaled   = ((cx, cy), (ax, ay), ellipse[2])

        cv2.ellipse(img_bgr, scaled, (0, 200, 255), 2)
        return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def run(self, images: list) -> dict:
        """
        Full pipeline on a list of PIL images.
        Returns result dict with all intermediate outputs.
        """
        classified      = self.classify_all(images)
        best_img, prob  = classified[0]

        if prob < 0.5:
            return {'error': 'No abdomen-containing slice detected.'}

        prob_map, mask  = self.segment(best_img)
        overlay         = self.make_overlay(best_img, mask)
        ac_mm, ellipse, _ = estimate_ac(mask)
        ellipse_img     = (self.draw_ellipse(best_img, ellipse)
                           if ellipse is not None else overlay)

        return {
            'all_results'  : classified,
            'best_image'   : best_img,
            'best_prob'    : prob,
            'prob_map'     : prob_map,
            'binary_mask'  : mask,
            'overlay'      : overlay,
            'ellipse'      : ellipse,
            'ellipse_img'  : ellipse_img,
            'ac_mm'        : ac_mm,
            'error'        : None,
        }
