# Bezier Surface Renderer

A desktop application for visualizing two bicubic Bezier surfaces with adjustable
rotation, tessellation, lighting, textures, and normal mapping. The interface uses
Tkinter, while performance-sensitive geometry and rasterization code is compiled
from Cython.

## Requirements

- Python 3.8 or newer with Tkinter
- A C/C++ compiler supported by Cython (Microsoft C++ Build Tools on Windows)

## Setup

From the repository root, install the build dependencies and the project:

```powershell
python -m pip install setuptools wheel Cython numpy
python -m pip install . --no-build-isolation
```

Then start the application:

```powershell
python main.py
```

The example surfaces are loaded from `control_points.txt` and
`control_points2.txt`. Texture and normal-map images can be selected from the

