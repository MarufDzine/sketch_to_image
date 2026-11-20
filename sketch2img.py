"""
sketch2img.py - main part of my project.

It has three parts:
  1. Sketch preprocessing: phone photo of a pen sketch -> clean binary line mask
  2. Control images: line mask -> the input image that each ControlNet expects
  3. Generator: Stable Diffusion 1.5 + ControlNet (and img2img as a baseline)

How the sketch controls the output:
  ControlNet is a trainable copy of the SD UNet encoder. It gets my sketch as
  an extra input, and its outputs are added (through zero-initialised 1x1
  convolutions) to the skip connections of the frozen SD UNet at every
  denoising step. cond_scale multiplies these outputs, so it sets how strongly
  the sketch is followed (0 = normal text-to-image, sketch ignored).

Run on one image:
  python sketch2img.py --sketch sketches/house.jpeg \
      --prompt "a small house at night, photorealistic" --out out.png
"""

import argparse
import time

import cv2
import numpy as np
from PIL import Image

# Model IDs on the Hugging Face Hub (all for SD 1.5)
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNETS = {
    "scribble": "lllyasviel/control_v11p_sd15_scribble",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}
NEGATIVE_PROMPT = (
    "lowres, blurry, bad anatomy, deformed, disfigured, extra limbs, "
    "watermark, text, signature, jpeg artifacts, worst quality, low quality"
)


# --- 1. Sketch preprocessing -------------------------------------------------
def load_and_resize(path_or_img, size=512):
    """Load an image and resize it so the longest side is `size`.
    Both sides are rounded down to a multiple of 8 because the SD VAE downsamples by 8."""
    img = Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img
    img = img.convert("RGB")
    w, h = img.size
    s = size / max(w, h)
    new_w, new_h = int(w * s) // 8 * 8, int(h * s) // 8 * 8
    return img.resize((new_w, new_h), Image.LANCZOS)


def sketch_to_lines(img, block_size=31, c=12, min_area=30):
    """Turn a phone photo of a pen sketch into a binary mask (lines = 255, background = 0).

    I use adaptive thresholding because my photos have shadows and uneven paper
    brightness. Small blobs (paper grain, dust, faint pencil marks) are removed.
    """
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lines = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


# --- 2. Control images (ControlNet v1.1 wants white lines on black) ----------
def make_control_image(img, lines, control):
    """img: resized sketch, lines: mask from sketch_to_lines.
    control: 'scribble', 'lineart', 'canny' or 'img2img'."""
    if control == "scribble":
        # the scribble model was trained on thick rough strokes, so I make the lines thicker
        out = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=2)
    elif control == "lineart":
        out = lines  # thin clean lines
    elif control == "canny":
        # normal Canny on the grey sketch (gives both edges of every stroke)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        out = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    elif control == "img2img":
        # img2img starts from the image itself, so here I keep black lines on white
        out = 255 - lines
    else:
        raise ValueError(f"unknown control type: {control}")
    return Image.fromarray(out).convert("RGB")


def prepare(sketch, control, size=512):
    """Full preprocessing: path -> (resized sketch, line mask, control image)."""
    img = load_and_resize(sketch, size)
    lines = sketch_to_lines(img)
    return img, lines, make_control_image(img, lines, control)


# --- 3. Generator ------------------------------------------------------------
class Generator:
    """Loads SD 1.5 one time. All ControlNet and img2img pipelines share the same
    VAE, text encoder and UNet, so each extra ControlNet only costs about 0.7 GB."""

    def __init__(self, base_model=BASE_MODEL, controlnets=None, device=None):
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
        from diffusers import UniPCMultistepScheduler

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.controlnet_ids = controlnets or CONTROLNETS
        self.base_model = base_model
        self._pipes = {}

        first = next(iter(self.controlnet_ids))
        cn = ControlNetModel.from_pretrained(self.controlnet_ids[first], torch_dtype=self.dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, controlnet=cn, torch_dtype=self.dtype,
            safety_checker=None, requires_safety_checker=False,
        )
        # UniPC sampler gives good images in 20-30 steps
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self._pipes[first] = pipe.to(self.device)
        # everything except the ControlNet is shared between pipelines
        self._shared = {k: v for k, v in pipe.components.items() if k != "controlnet"}

    def _pipe(self, control):
        if control in self._pipes:
            return self._pipes[control]
        if control == "img2img":
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline(**self._shared, requires_safety_checker=False)
        else:
            from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
            cn = ControlNetModel.from_pretrained(
                self.controlnet_ids[control], torch_dtype=self.dtype
            ).to(self.device)
            pipe = StableDiffusionControlNetPipeline(**self._shared, controlnet=cn,
                                                     requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        self._pipes[control] = pipe
        return pipe

    def generate(self, control_image, prompt, control="scribble", negative_prompt=NEGATIVE_PROMPT,
                 steps=25, guidance=7.5, cond_scale=1.0, strength=0.8, seed=42):
        g = self.torch.Generator(device=self.device).manual_seed(int(seed))
        pipe = self._pipe(control)
        common = dict(prompt=prompt, negative_prompt=negative_prompt,
                      num_inference_steps=int(steps), guidance_scale=float(guidance), generator=g)
        if control == "img2img":
            return pipe(image=control_image, strength=float(strength), **common).images[0]
        w, h = control_image.size
        return pipe(image=control_image, width=w, height=h,
                    controlnet_conditioning_scale=float(cond_scale), **common).images[0]


class FakeGenerator:
    """Used with --dry-run to test the whole pipeline without a GPU.
    It just returns the blurred sketch with a colour tint (weaker when cond_scale is low)."""

    def generate(self, control_image, prompt, control="scribble", cond_scale=1.0,
                 strength=0.8, seed=42, **_):
        rng = np.random.default_rng(int(seed))
        a = np.array(control_image).astype(np.float32)
        if control != "img2img":
            a = 255 - a  # back to dark lines on light background
        k = 1.0 if control != "img2img" else 1.0 - strength
        a = cv2.GaussianBlur(a, (5, 5), 0) * min(float(cond_scale), 1.0) * k + 255 * (
            1 - min(float(cond_scale), 1.0) * k)
        a = a * rng.uniform(0.6, 1.0, size=3) + rng.normal(0, 12, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# --- Command line: one image -------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Sketch -> image with SD 1.5 + ControlNet")
    p.add_argument("--sketch", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--control", default="scribble", choices=list(CONTROLNETS) + ["img2img"])
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--cond-scale", type=float, default=1.0)
    p.add_argument("--strength", type=float, default=0.8, help="img2img only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out", default="out.png")
    a = p.parse_args()

    _, _, ctrl = prepare(a.sketch, a.control, a.size)
    gen = Generator()
    t = time.time()
    img = gen.generate(ctrl, a.prompt, control=a.control, steps=a.steps, guidance=a.guidance,
                       cond_scale=a.cond_scale, strength=a.strength, seed=a.seed)
    img.save(a.out)
    print(f"saved {a.out} ({time.time() - t:.1f} s)")


if __name__ == "__main__":
    main()
