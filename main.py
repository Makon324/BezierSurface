import compiled  # <-- this triggers __init__.py → adds folder to path
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox
import numpy as np
from bezier import generate_vertices
from triangulation import triangulate_grid
from rotation import rotate_vertices, rotate_control_points
from rendering import render_to_image
from PIL import Image, ImageTk
from light import LightManager
from normal_map import NormalMapManager
from color_texture import ColorTextureManager
from typing import Optional
import os


def load_control_points(filename: str) -> np.ndarray:
    """
    Load control points from a text file.

    Args:
        filename: Path to the file containing control points.

    Returns:
        A 4x4x3 NumPy array of control points.

    Raises:
        ValueError: If the file format is invalid.
        IOError: If there's an issue reading the file.
    """
    try:
        data = np.zeros((4, 4, 3), dtype=float)
        with open(filename, "r") as f:
            lines = f.readlines()
        if len(lines) != 16:
            raise ValueError("File must have exactly 16 lines")
        idx = 0
        for i in range(4):
            for j in range(4):
                parts = lines[idx].strip().split()
                if len(parts) != 3:
                    raise ValueError("Each line must have 3 floats")
                data[i, j, :] = list(map(float, parts))
                idx += 1
        return data
    except (IOError, ValueError) as e:
        messagebox.showerror("Error Loading Control Points", str(e))
        raise


class BezierApp:
    """Main application class for rendering and interacting with a Bézier surface."""

    def __init__(self, root: tk.Tk, control_points: np.ndarray):
        """
        Initialize the BezierApp.

        Args:
            root: The root Tkinter window.
            control_points: 4x4x3 NumPy array of Bézier control points.
        """
        self.root = root
        self.control_points = control_points
        self.alpha: int = 0
        self.beta: int = 0
        self.divisions: int = 10  # Default accuracy

        self.vertices, self.grid_size = generate_vertices(self.control_points, self.divisions)
        self.triangles = triangulate_grid(self.vertices, self.grid_size)

        self.photo: Optional[ImageTk.PhotoImage] = None  # To hold image reference

        self._init_gui()
        self.update()

    def _init_gui(self) -> None:
        """Initialize the GUI components."""
        # Canvas
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.pack(side=tk.TOP, fill="both", expand=True)

        # Bind to resize
        self.root.bind("<Configure>", self.on_resize)

        # Controls frame at bottom with padding
        control_frame = tk.Frame(self.root, padx=10, pady=10, bg="#f0f0f0")
        control_frame.pack(side=tk.BOTTOM, fill="x")

        # Section for rotations and divisions
        rotations_frame = tk.LabelFrame(
            control_frame, text="Rotations & Divisions", padx=5, pady=5, bg="#f0f0f0"
        )
        rotations_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)

        # Sliders in grid for better alignment
        tk.Label(rotations_frame, text="Alpha (Z rotation)", bg="#f0f0f0").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.alpha_slider = tk.Scale(
            rotations_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self.update
        )
        self.alpha_slider.set(0)
        self.alpha_slider.grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(rotations_frame, text="Beta (X rotation)", bg="#f0f0f0").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.beta_slider = tk.Scale(
            rotations_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self.update
        )
        self.beta_slider.set(0)
        self.beta_slider.grid(row=1, column=1, sticky="ew", pady=2)

        tk.Label(rotations_frame, text="Divisions", bg="#f0f0f0").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.div_slider = tk.Scale(
            rotations_frame, from_=1, to=50, orient=tk.HORIZONTAL, command=self.update
        )
        self.div_slider.set(self.divisions)
        self.div_slider.grid(row=2, column=1, sticky="ew", pady=2)

        # Display options
        display_frame = tk.LabelFrame(
            control_frame, text="Display Options", padx=5, pady=5, bg="#f0f0f0"
        )
        display_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)

        self.show_control = tk.BooleanVar(value=False)
        tk.Checkbutton(
            display_frame, text="Show Control Mesh", variable=self.show_control, command=self.update
        ).pack(side=tk.LEFT, padx=5)

        self.show_mesh = tk.BooleanVar(value=True)
        tk.Checkbutton(
            display_frame, text="Show Triangle Mesh", variable=self.show_mesh, command=self.update
        ).pack(side=tk.LEFT, padx=5)

        self.show_filled = tk.BooleanVar(value=True)
        tk.Checkbutton(
            display_frame, text="Fill Triangles", variable=self.show_filled, command=self.update
        ).pack(side=tk.LEFT, padx=5)

        # Lighting section
        light_frame = tk.LabelFrame(
            control_frame, text="Lighting", padx=5, pady=5, bg="#f0f0f0"
        )
        light_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        self.light_manager = LightManager(light_frame, self.update)
        light_elements = self.light_manager.build_gui_elements()
        for idx, elem in enumerate(light_elements):
            elem.grid(row=idx // 2, column=idx % 2, sticky="ew", padx=5, pady=2)

        # Normal mapping section
        normal_frame = tk.LabelFrame(
            control_frame, text="Normal Mapping", padx=5, pady=5, bg="#f0f0f0"
        )
        normal_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        self.normal_manager = NormalMapManager(normal_frame, self.update)
        normal_elements = self.normal_manager.build_gui_elements()
        for elem in normal_elements:
            elem.pack(side=tk.LEFT, padx=5)

        # Color & texture section
        color_frame = tk.LabelFrame(
            control_frame, text="Color & Texture", padx=5, pady=5, bg="#f0f0f0"
        )
        color_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        self.color_texture_manager = ColorTextureManager(color_frame, self.update)
        color_elements = self.color_texture_manager.build_gui_elements()
        for elem in color_elements:
            elem.pack(side=tk.LEFT, padx=5)

    def on_resize(self, event: tk.Event) -> None:
        """Handle window resize event by updating the rendering."""
        self.update()

    def update(self, _=None) -> None:
        """Update the rendering based on current parameters."""
        self.alpha = self.alpha_slider.get()
        self.beta = self.beta_slider.get()
        new_div = self.div_slider.get()
        if new_div != self.divisions:
            self.divisions = new_div
            self.vertices, self.grid_size = generate_vertices(
                self.control_points, self.divisions
            )
            self.triangles = triangulate_grid(self.vertices, self.grid_size)

        rotate_vertices(self.vertices, self.alpha, self.beta)
        rotated_controls = rotate_control_points(
            self.control_points, self.alpha, self.beta
        )

        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        scale = min(width, height) / 4.0

        color_mode, const_col, texture = (
            self.color_texture_manager.get_color_and_texture()
        )
        k_d, k_s, m, I_L, light_pos = self.light_manager.get_lighting_params()

        normal_enabled, normal_texture = self.normal_manager.get_normal_map_params()

        img = render_to_image(
            self.triangles,
            rotated_controls,
            width,
            height,
            scale,
            self.show_control.get(),
            self.show_mesh.get(),
            self.show_filled.get(),
            color_mode,
            const_col,
            texture,
            k_d,
            k_s,
            m,
            I_L,
            light_pos,
            normal_enabled,
            normal_texture,
        )
        self.photo = ImageTk.PhotoImage(img)  # Keep reference to avoid GC
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")


if __name__ == "__main__":
    CONTROL_POINTS_FILE = "control_points.txt"
    CONTROL_POINTS2_FILE = "control_points2.txt"

    control_points1 = load_control_points(CONTROL_POINTS_FILE)
    control_points2 = load_control_points(CONTROL_POINTS2_FILE)
    root = tk.Tk()
    root.title("Bézier Surface Renderer")
    app = BezierApp(root, control_points1)#, control_points2)  # Pass both
    root.mainloop()
