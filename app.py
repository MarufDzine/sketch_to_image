"""
app.py - small web UI (Gradio) for my sketch-to-image pipeline.

  python app.py            # local: http://127.0.0.1:7860
  python app.py --share    # on Colab/Kaggle: gives a public *.gradio.live link
"""

import argparse

import gradio as gr

from metrics import structure_metrics
from sketch2img import CONTROLNETS, Generator, make_control_image, load_and_resize, sketch_to_lines

STYLES = {
    "none": "",
    "photo": "photorealistic, professional photograph, natural lighting, highly detailed, 8k",
    "concept art": "digital concept art, dramatic cinematic lighting, artstation, vibrant colors",
    "watercolor": "watercolor painting, soft pastel colors, paper texture",
    "clay 3D": "3d render, claymation style, soft studio lighting",
}

gen = None  # the model loads on the first click, so the UI opens fast


def run(sketch, prompt, style, control, cond_scale, guidance, steps, strength, seed):
    global gen
    if sketch is None:
        raise gr.Error("Upload or draw a sketch first.")
    if gen is None:
        gen = Generator()
    img = load_and_resize(sketch, 512)
    lines = sketch_to_lines(img)
    ctrl = make_control_image(img, lines, control)
    full_prompt = ", ".join(p for p in [prompt.strip(), STYLES[style]] if p)
    out = gen.generate(ctrl, full_prompt, control=control, steps=steps, guidance=guidance,
                       cond_scale=cond_scale, strength=strength, seed=seed)
    m = structure_metrics(lines, out)
    info = (f"**Prompt:** {full_prompt}  \n**Sketch recall @5px:** {m['recall']:.2f} · "
            f"**Chamfer:** {m['chamfer_px']:.2f} px")
    return ctrl, out, info


def build():
    with gr.Blocks(title="Sketch to Image") as demo:
        gr.Markdown("# Sketch → Image\nStable Diffusion 1.5 + ControlNet. "
                    "Upload a photo of a pen sketch (dark lines on light paper).")
        with gr.Row():
            with gr.Column():
                sketch = gr.Image(type="pil", label="Sketch", sources=["upload", "clipboard"])
                prompt = gr.Textbox(label="What is drawn?", value="a futuristic building")
                style = gr.Dropdown(list(STYLES), value="photo", label="Style")
                control = gr.Radio(list(CONTROLNETS) + ["img2img"], value="scribble",
                                   label="Conditioning method")
                with gr.Accordion("Settings", open=False):
                    cond = gr.Slider(0, 2, 1.0, step=0.05, label="ControlNet conditioning scale")
                    cfg = gr.Slider(1, 15, 7.5, step=0.5, label="Guidance scale (CFG)")
                    steps = gr.Slider(10, 50, 25, step=1, label="Steps")
                    strength = gr.Slider(0.3, 1.0, 0.8, step=0.05, label="img2img strength")
                    seed = gr.Number(42, precision=0, label="Seed")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                ctrl_out = gr.Image(label="Control image (what the model sees)")
                out = gr.Image(label="Generated image")
                info = gr.Markdown()
        btn.click(run, [sketch, prompt, style, control, cond, cfg, steps, strength, seed],
                  [ctrl_out, out, info])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    build().launch(share=ap.parse_args().share)
