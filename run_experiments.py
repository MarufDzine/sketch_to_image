"""
run_experiments.py - runs all experiments from config.yaml and saves the metrics.

  python run_experiments.py              # real run (needs a GPU)
  python run_experiments.py --dry-run    # no models, only tests the code

Outputs in outputs/:
  control/<sketch>_<control>.png   control images that go into the model
  images/<sketch>/<settings>.png   all generated images (if an image already exists
                                   it is re-used, so a Colab disconnect costs nothing)
  results.csv                      one row per (experiment, sketch, value)
"""

import argparse
import os
import time

import pandas as pd
import yaml
from PIL import Image

from metrics import ClipScorer, structure_metrics
from sketch2img import NEGATIVE_PROMPT, FakeGenerator, Generator, prepare

PARAMS = ["control", "style", "steps", "guidance", "cond_scale", "strength", "seed", "size"]


def image_name(p):
    s = f"{p['control']}_{p['style']}_st{p['steps']}_g{p['guidance']}_seed{p['seed']}"
    if p["control"] == "img2img":
        return s + f"_str{p['strength']}.png"
    return s + f"_cs{p['cond_scale']}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--only", nargs="*", help="run only these experiments")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    os.makedirs(f"{a.out}/control", exist_ok=True)

    if a.dry_run:
        gen, clip = FakeGenerator(), None
    else:
        gen = Generator(cfg["models"]["base"], cfg["models"]["controlnets"])
        clip = ClipScorer(cfg["models"]["clip"])

    rows, cache = [], {}
    for exp_name, exp in cfg["experiments"].items():
        if a.only and exp_name not in a.only:
            continue
        for sk in cfg["sketches"]:
            sk_name = os.path.splitext(os.path.basename(sk["file"]))[0]
            os.makedirs(f"{a.out}/images/{sk_name}", exist_ok=True)
            for v in exp["values"]:
                p = {**cfg["defaults"], **exp.get("fixed", {}), exp["vary"]: v}
                prompt = f"{sk['subject']}, {cfg['styles'][p['style']]}"

                # preprocessing is done once per sketch and control type
                key = (sk["file"], p["control"], p["size"])
                if key not in cache:
                    cache[key] = prepare(sk["file"], p["control"], p["size"])
                    cache[key][2].save(f"{a.out}/control/{sk_name}_{p['control']}.png")
                _, lines, ctrl = cache[key]

                path = f"{a.out}/images/{sk_name}/{image_name(p)}"
                t = time.time()
                if os.path.exists(path):
                    img, secs = Image.open(path).convert("RGB"), float("nan")
                else:
                    img = gen.generate(ctrl, prompt, control=p["control"],
                                       negative_prompt=NEGATIVE_PROMPT, steps=p["steps"],
                                       guidance=p["guidance"], cond_scale=p["cond_scale"],
                                       strength=p["strength"], seed=p["seed"])
                    img.save(path)
                    secs = time.time() - t

                m = structure_metrics(lines, img)
                m["clip_score"] = clip(img, prompt) if clip else float("nan")
                rows.append({"experiment": exp_name, "sketch": sk_name, "varied": exp["vary"],
                             "value": v, **{k: p[k] for k in PARAMS}, "prompt": prompt,
                             "image": path, **m, "seconds": secs})
                print(f"[{exp_name}] {sk_name} {exp['vary']}={v}: "
                      f"recall={m['recall']:.3f} chamfer={m['chamfer_px']:.2f} "
                      f"clip={m['clip_score']:.1f}")

    df = pd.DataFrame(rows)
    csv = f"{a.out}/results.csv"
    if a.only and os.path.exists(csv):  # keep results of experiments that were not re-run
        old = pd.read_csv(csv)
        df = pd.concat([old[~old.experiment.isin(a.only)], df], ignore_index=True)
    df.to_csv(csv, index=False)
    print(f"\n{len(df)} rows -> {csv}")


if __name__ == "__main__":
    main()
