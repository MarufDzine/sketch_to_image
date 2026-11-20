"""
metrics.py - how I measure the results.

Sketch preservation (does the generated image keep the structure of my sketch?)
  1. Skeletonise the sketch line mask to get 1-px centre lines S.
  2. Find edges E in the generated image with Canny (thresholds from the median).
  3. A distance transform of E gives, for every pixel, the distance to the
     nearest generated edge. I read it at the sketch pixels S:
       - recall     : fraction of sketch pixels with a generated edge within 5 px (~1% of 512 px)
       - chamfer_px : mean distance (px) from sketch pixels to the nearest generated edge
  Higher recall / lower chamfer = structure better kept.
  I only measure in one direction on purpose: a good result should ADD detail
  (texture, shading), so extra edges in the output are not punished.

Prompt adherence
  clip_score = 100 * cosine(CLIP image embedding, CLIP text embedding)
  (Hessel et al., 2021), with openai/clip-vit-base-patch16.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def auto_canny(rgb, sigma=0.33):
    gray = cv2.cvtColor(np.array(rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    m = float(np.median(gray))
    lo, hi = int(max(0, (1 - sigma) * m)), int(min(255, (1 + sigma) * m))
    return cv2.Canny(gray, lo, hi)


def structure_metrics(lines, generated, tol=5):
    """lines: uint8 mask (255 = sketch line) at generation size; generated: PIL image."""
    if generated.size != (lines.shape[1], lines.shape[0]):
        generated = generated.resize((lines.shape[1], lines.shape[0]))
    skel = skeletonize(lines > 0)
    edges = auto_canny(generated)
    # distanceTransform measures the distance to the nearest ZERO pixel, so edges must be 0
    if edges.max() == 0:  # no edges at all -> nothing of the sketch is kept
        return {"recall": 0.0, "chamfer_px": float(np.hypot(*lines.shape))}
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    d = dist[skel]
    if d.size == 0:
        return {"recall": float("nan"), "chamfer_px": float("nan")}
    return {"recall": float((d <= tol).mean()), "chamfer_px": float(d.mean())}


class ClipScorer:
    def __init__(self, model_id="openai/clip-vit-base-patch16", device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, image, text):
        inp = self.proc(text=[text], images=image, return_tensors="pt",
                        padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        img, txt = out.image_embeds, out.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return float(100 * (img * txt).sum().clamp(min=0))
