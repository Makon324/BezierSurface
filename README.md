# Bezier Surface Renderer

An interactive desktop application for visualizing two bicubic Bezier surfaces with
configurable tessellation, rotation, lighting, textures, and normal mapping. The
Tkinter interface is backed by Cython implementations of performance-sensitive
geometry and rasterization routines.

## Requirements

- Python 3.8 or later, including Tkinter
- A C/C++ compiler supported by Cython (Microsoft C++ Build Tools on Windows)

## Setup

From the repository root, install the build dependencies, then install the project:

```powershell
python -m pip install setuptools wheel Cython numpy
python -m pip install . --no-build-isolation
```

Run the application:

```powershell
python main.py
```

By default, the application loads example surfaces from `control_points.txt` and
`control_points2.txt`. Texture and normal-map images can be selected through the
application controls.
