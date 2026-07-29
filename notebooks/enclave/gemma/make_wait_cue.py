"""Generate the "wait for the other data owner" notebook cues.

Each cue is a 760x280 card: a figure holding an hourglass, and a speech bubble
reading "Wait for the other data owner connected to the enclave <action>".
The bubble text is auto-fitted, so cues with different action wording still
render at exactly the same size.

To reword a cue, edit CUES below and re-run:

    uv run --with pillow --with matplotlib python make_wait_cue.py

The cues live only inside the notebooks, as inlined base64 PNGs — Colab cannot
read files from the repo, and its markdown renderer does not reliably render
data-URI SVG. So this script renders through a temp dir and rewrites the img
tags in ./colab in place; nothing else is checked in. Re-running with no wording
change is a no-op. Pass --save-svg DIR to keep the SVGs to look at.

Needs rsvg-convert and ImageMagick on PATH.
"""

import argparse
import base64
import pathlib
import re
import shutil
import subprocess
import tempfile

import matplotlib
from PIL import ImageFont

HERE = pathlib.Path(__file__).parent
NB_DIR = HERE / "colab"
FONT = (
    pathlib.Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
)

CARD_W, CARD_H = 760, 280
BUBBLE_X, BUBBLE_Y, BUBBLE_W, BUBBLE_H = 241, 44, 496, 188
PADDING = 18
LEAD = 1.42  # line height as a multiple of the font size

LEAD_IN = ["Wait for the other data owner", "connected to the enclave"]

# Each cue names what the *other* notebook's owner has to finish.
CUES = {
    "submit_inference_job": "to submit the inference job",
    "approve_peer_request": "to approve your peer request",
    "submit_syft_restrict_job": "to submit the syft-restrict job",
    "approve_inference_job": "to approve the inference job",
}

# Which cue goes where, in the order the cues appear in each notebook.
NOTEBOOKS = {
    "1. DO-model-owner-gemma-restrict.ipynb": ["submit_inference_job"],
    "2. DO-benchmark-owner-gemma-restrict.ipynb": [
        "approve_peer_request",
        "submit_syft_restrict_job",
        "approve_inference_job",
    ],
}

# Any cue already inlined by this script, or the original "take a break" one it
# replaced. Group 1 keeps whatever trailing newline the source line had.
CUE_IMG = re.compile(
    r'"<img src=\\"data:image/png;base64,[A-Za-z0-9+/=]+\\" alt=\\"'
    r"(?:Wait for the other data owner|Time to take a break)"
    r'[^\\]*\\" width=\\"620\\"/>((?:\\n)?)"'
)

# fmt: off
ART = """  <rect x="0.5" y="0.5" width="759" height="279" rx="6" fill="#fefefe" stroke="#e8eaed"/>

  <!-- person -->
  <g>
    <path d="M78 245 L78 190 a47 47 0 0 1 47 -47 a47 47 0 0 1 47 47 L172 245 Z" fill="#4b87d1"/>
    <path d="M150 240 L150 196 a22 22 0 0 1 22 -22 l6 0 a20 20 0 0 1 0 40 l-6 0 a22 22 0 0 0 -22 22 Z" fill="#4b87d1"/>
    <rect x="158" y="140" width="30" height="42" rx="14" fill="#f6cda4"/>
    <rect x="82" y="60" width="76" height="82" rx="26" fill="#f6cda4"/>
    <path d="M82 92 a38 34 0 0 1 76 0 l0 -6 a38 34 0 0 0 -76 0 Z" fill="#4a3728"/>
    <path d="M80 90 c0 -26 18 -44 45 -44 c27 0 45 18 45 40 c-8 -14 -20 -22 -34 -22 c-20 0 -30 12 -56 26 Z" fill="#4a3728"/>
    <path d="M96 104 q7 -9 14 0" fill="none" stroke="#3b3b3b" stroke-width="3.4" stroke-linecap="round"/>
    <path d="M126 104 q7 -9 14 0" fill="none" stroke="#3b3b3b" stroke-width="3.4" stroke-linecap="round"/>
    <!-- hourglass -->
    <rect x="150" y="88" width="48" height="7" rx="3.5" fill="#4a5568"/>
    <rect x="150" y="151" width="48" height="7" rx="3.5" fill="#4a5568"/>
    <path d="M156 95 L192 95 L177 123 L192 151 L156 151 L171 123 Z" fill="#eef1f6" stroke="#4a5568" stroke-width="3.5" stroke-linejoin="round"/>
    <path d="M160 99 L188 99 L174 122 Z" fill="#f0b429"/>
    <path d="M174 133 L186 147 L162 147 Z" fill="#f0b429"/>
    <line x1="174" y1="124" x2="174" y2="140" stroke="#f0b429" stroke-width="3" stroke-linecap="round"/>
    <!-- waiting motion marks -->
    <path d="M212 96 q9 12 0 24" fill="none" stroke="#c6ccd6" stroke-width="4" stroke-linecap="round"/>
    <path d="M70 74 q-9 12 0 24" fill="none" stroke="#c6ccd6" stroke-width="4" stroke-linecap="round"/>
  </g>

  <!-- speech bubble -->
  <path d="M242 118 L214 138 L242 158 Z" fill="#eaeff7" stroke="#ccd8e8" stroke-width="2"/>
  <rect x="241" y="44" width="496" height="188" rx="14" fill="#eaeff7" stroke="#ccd8e8" stroke-width="2"/>
  <rect x="243" y="120" width="4" height="36" fill="#eaeff7"/>
"""
# fmt: on


def greedy_wrap(text, font, max_width):
    """Fewest lines that each measure within max_width, or None if a word alone
    is too wide."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or font.getlength(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if any(font.getlength(line) > max_width for line in lines):
        return None
    return lines


def wrap_to_width(text, font, max_width):
    """Wrap on the fewest lines, then even the lines out by tightening the width
    until another line would be needed."""
    best = greedy_wrap(text, font, max_width)
    if best is None:
        return None
    for width in range(int(max_width), 40, -4):
        lines = greedy_wrap(text, font, width)
        if lines is None or len(lines) > len(best):
            break
        best = lines
    return best


def fit_text(action):
    """Largest font size at which the whole message fits the bubble."""
    max_w = BUBBLE_W - 2 * PADDING
    max_h = BUBBLE_H - 2 * PADDING
    for size in range(30, 17, -1):
        font = ImageFont.truetype(str(FONT), size)
        action_lines = wrap_to_width(action, font, max_w)
        if action_lines is None:
            continue
        lines = LEAD_IN + action_lines
        if any(font.getlength(line) > max_w for line in lines):
            continue
        if len(lines) * LEAD * size <= max_h:
            return lines, size
    raise ValueError(f"cannot fit: {action!r}")


def render_svg(action):
    lines, size = fit_text(action)
    lead = LEAD * size
    # vertically centre the block in the bubble, using a cap-height-ish offset
    first_baseline = BUBBLE_Y + BUBBLE_H / 2 - (len(lines) * lead) / 2 + size * 0.78
    cx = BUBBLE_X + BUBBLE_W / 2
    texts = "\n".join(
        f'    <text x="{cx:g}" y="{first_baseline + i * lead:.1f}">{line}</text>'
        for i, line in enumerate(lines)
    )
    alt = " ".join(LEAD_IN) + " " + action
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}" height="{CARD_H}"'
        f' viewBox="0 0 {CARD_W} {CARD_H}" role="img" aria-label="{alt}">\n'
        f"{ART}\n"
        f"  <g fill=\"#2c3e50\" font-family=\"'DejaVu Sans','Open Sans',Verdana,sans-serif\""
        f' font-size="{size}" font-weight="bold" text-anchor="middle">\n'
        f"{texts}\n"
        f"  </g>\n"
        f"</svg>\n"
    ), alt


def to_inline_png(svg_path):
    """Render at the card's native width, quantised to keep notebooks small."""
    raw = svg_path.with_suffix(".raw.png")
    out = svg_path.with_suffix(".opt.png")
    subprocess.run(
        ["rsvg-convert", "-w", str(CARD_W), str(svg_path), "-o", str(raw)], check=True
    )
    subprocess.run(
        [
            "magick",
            str(raw),
            "-strip",
            "-colors",
            "96",
            "-dither",
            "None",
            f"PNG8:{out}",
        ],
        check=True,
    )
    encoded = base64.b64encode(out.read_bytes()).decode()
    raw.unlink()
    out.unlink()
    return encoded


def build_cues(save_svg_to=None):
    """Render every cue and return {slug: (base64 png, alt text)}."""
    built = {}
    with tempfile.TemporaryDirectory() as tmp:
        for slug, action in CUES.items():
            svg, alt = render_svg(action)
            path = pathlib.Path(tmp) / f"wait_{slug}.svg"
            path.write_text(svg)
            built[slug] = (to_inline_png(path), alt)
            if save_svg_to is not None:
                shutil.copy(path, save_svg_to / path.name)
            print(f"{path.name}: {alt}")
    return built


def inline_into_notebooks(built):
    """Swap each notebook's cue images for the freshly rendered ones, in order."""
    for name, slugs in NOTEBOOKS.items():
        path = NB_DIR / name
        text = path.read_text()
        found = len(CUE_IMG.findall(text))
        if found != len(slugs):
            raise ValueError(f"{name}: found {found} cue(s), expected {len(slugs)}")
        remaining = iter(slugs)

        def replace(match):
            encoded, alt = built[next(remaining)]
            return (
                f'"<img src=\\"data:image/png;base64,{encoded}\\" '
                f'alt=\\"{alt}\\" width=\\"620\\"/>{match.group(1)}"'
            )

        path.write_text(CUE_IMG.sub(replace, text))
        print(f"{name}: inlined {len(slugs)} cue(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-svg",
        metavar="DIR",
        type=pathlib.Path,
        help="also write the SVGs here, to look at them (not checked in)",
    )
    args = parser.parse_args()
    if args.save_svg is not None:
        args.save_svg.mkdir(parents=True, exist_ok=True)
    inline_into_notebooks(build_cues(args.save_svg))


if __name__ == "__main__":
    main()
