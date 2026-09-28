"""Generate the README animation using the application's rendering code."""

from math import cos, pi, sin
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compiled  # Make the locally built Cython modules importable.
import numpy as np
from PIL import Image, ImageDraw
from bezier import generate_vertices
from data_structures import Point
from rendering import render_to_image
from rotation import rotate_control_points, rotate_vertices
from triangulation import triangulate_grid


def main():
    controls = [
        np.loadtxt(ROOT / name).reshape(4, 4, 3)
        for name in ("control_points.txt", "control_points2.txt")
    ]
    meshes = [generate_vertices(points, 14) for points in controls]
    triangles = [triangulate_grid(vertices, size) for vertices, size in meshes]
    frames = []
    for index in range(90):
        phase = 2 * pi * index / 90
        angles = [
            (25 + 35 * sin(phase), 20 + 25 * cos(phase)),
            (-20 + 30 * sin(phase), -65 + 20 * cos(phase)),
        ]
        rotated = []
        for points, (vertices, _), (alpha, beta) in zip(controls, meshes, angles):
            rotate_vertices(vertices, alpha, beta)
            rotated.append(rotate_control_points(points, alpha, beta))
        frame = Image.new("RGB", (960, 440), "white")
        for panel, wireframe in enumerate((True, False)):
            view = render_to_image(
                triangles, rotated, 480, 400, 115,
                False, wireframe, not wireframe,
                "constant", (78, 155, 210), None,
                0.75, 0.3, 35, (1.0, 1.0, 1.0), Point(0, 0, 5),
                False, None,
            )
            frame.paste(view, (480 * panel, 40))
        draw = ImageDraw.Draw(frame)
        draw.text((24, 15), "Triangle mesh", fill="#334155", font_size=18)
        draw.text((504, 15), "Shaded surfaces", fill="#334155", font_size=18)
        draw.line((480, 16, 480, 424), fill="#e2e8f0")
        frames.append(frame)

    # One palette for the whole loop avoids colour flicker between frames.
    sample = Image.new("RGB", (960, 440 * 9))
    for row, frame in enumerate(frames[::10]):
        sample.paste(frame, (0, 440 * row))
    palette = sample.quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output = ROOT / "docs" / "images" / "surfaces.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    indexed[0].save(
        output, save_all=True, append_images=indexed[1:],
        duration=100, loop=0, optimize=True, disposal=1,
    )
    print(f"Saved {output.name}: {len(frames)} frames, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
