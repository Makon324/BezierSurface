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
import random


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

    def __init__(self, root: tk.Tk, control_points1: np.ndarray, control_points2: np.ndarray):
        self.root = root
        self.control_points1 = control_points1
        self.control_points2 = control_points2
        self.alpha1: int = 0  # First surface
        self.beta1: int = 0
        self.alpha2: int = 0  # Second surface
        self.beta2: int = -90  # Initial beta = -90 for second
        self.divisions: int = 10

        # Generate initial for both surfaces
        self.vertices1, self.grid_size1 = generate_vertices(self.control_points1, self.divisions)
        self.triangles1 = triangulate_grid(self.vertices1, self.grid_size1)
        self.vertices2, self.grid_size2 = generate_vertices(self.control_points2, self.divisions)
        self.triangles2 = triangulate_grid(self.vertices2, self.grid_size2)

        # Random rotation variables
        self.target_alpha1 = self.alpha1
        self.target_beta1 = self.beta1
        self.target_alpha2 = self.alpha2
        self.target_beta2 = self.beta2
        self.lerp_speed = 0.15  # Fraction per step (0.0–1.0); higher = faster/smoother arrival
        self.target_threshold = 1.0  # Degrees; when closer than this, pick new target
        self.animation_interval_ms = 30  # Smoother than 100ms

        self.photo: Optional[ImageTk.PhotoImage] = None
        self.random_rotating = False  # For random rotation toggle

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
            rotations_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self._update_sliders
        )
        self.alpha_slider.set(0)
        self.alpha_slider.grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(rotations_frame, text="Beta (X rotation)", bg="#f0f0f0").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.beta_slider = tk.Scale(
            rotations_frame, from_=-90, to=90, orient=tk.HORIZONTAL, command=self._update_sliders
        )
        self.beta_slider.set(0)
        self.beta_slider.grid(row=1, column=1, sticky="ew", pady=2)

        tk.Label(rotations_frame, text="Divisions", bg="#f0f0f0").grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.div_slider = tk.Scale(
            rotations_frame, from_=1, to=50, orient=tk.HORIZONTAL, command=self._update_sliders
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

        # Random rotation section
        random_frame = tk.LabelFrame(
            control_frame, text="Random Rotation", padx=5, pady=5, bg="#f0f0f0"
        )
        random_frame.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        random_button = tk.Button(random_frame, text="Start Random Rotation", command=self.toggle_random_rotation)
        random_button.pack(side=tk.LEFT, padx=5)

    def on_resize(self, event: tk.Event) -> None:
        """Handle window resize event by updating the rendering."""
        self.update()

    def _update_sliders(self, _=None):
        self.alpha1 = self.alpha_slider.get()
        self.beta1 = self.beta_slider.get()
        self.update()

    def toggle_random_rotation(self):
        self.random_rotating = not self.random_rotating
        if self.random_rotating:
            # Reset targets to current on start
            self.target_alpha1 = self.alpha1
            self.target_beta1 = self.beta1
            self.target_alpha2 = self.alpha2
            self.target_beta2 = self.beta2
            self._random_rotate_step()
        # If stopping, no need to reset angles—they stay where they are

    def _random_rotate_step(self):
        if not self.random_rotating:
            return

        # Lerp each angle toward its target
        self.alpha1 = self._lerp(self.alpha1, self.target_alpha1, self.lerp_speed)
        self.beta1 = self._lerp(self.beta1, self.target_beta1, self.lerp_speed)
        self.alpha2 = self._lerp(self.alpha2, self.target_alpha2, self.lerp_speed)
        self.beta2 = self._lerp(self.beta2, self.target_beta2, self.lerp_speed)

        # Check if close to targets and pick new random ones if needed
        if abs(self.alpha1 - self.target_alpha1) < self.target_threshold:
            self.target_alpha1 += random.uniform(-45, 45)  # Larger range for more dynamic motion
        if abs(self.beta1 - self.target_beta1) < self.target_threshold:
            self.target_beta1 += random.uniform(-45, 45)
        if abs(self.alpha2 - self.target_alpha2) < self.target_threshold:
            self.target_alpha2 += random.uniform(-45, 45)  # Or sync with alpha1 if you want identical motion
        if abs(self.beta2 - self.target_beta2) < self.target_threshold:
            self.target_beta2 += random.uniform(-45, 45)

        # Update the rendering
        self.update()

        # Schedule next step
        self.root.after(self.animation_interval_ms, self._random_rotate_step)

    def _lerp(self, current: float, target: float, t: float) -> float:
        """Linear interpolation from current to target by fraction t."""
        return current + (target - current) * t

    def update(self, _=None) -> None:
        """Update the rendering based on current parameters."""
        new_div = self.div_slider.get()
        if new_div != self.divisions:
            self.divisions = new_div
            self.vertices1, self.grid_size1 = generate_vertices(self.control_points1, self.divisions)
            self.triangles1 = triangulate_grid(self.vertices1, self.grid_size1)
            self.vertices2, self.grid_size2 = generate_vertices(self.control_points2, self.divisions)
            self.triangles2 = triangulate_grid(self.vertices2, self.grid_size2)

        rotate_vertices(self.vertices1, self.alpha1, self.beta1)
        rotated_controls1 = rotate_control_points(self.control_points1, self.alpha1, self.beta1)
        rotate_vertices(self.vertices2, self.alpha2, self.beta2)
        rotated_controls2 = rotate_control_points(self.control_points2, self.alpha2, self.beta2)

        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        scale = min(width, height) / 4.0

        color_mode, const_col, texture = self.color_texture_manager.get_color_and_texture()
        k_d, k_s, m, I_L, light_pos = self.light_manager.get_lighting_params()
        normal_enabled, normal_texture = self.normal_manager.get_normal_map_params()

        # Render both surfaces
        img = render_to_image(
            [self.triangles1, self.triangles2],  # List of triangle lists
            [rotated_controls1, rotated_controls2],  # List of rotated controls
            width, height, scale,
            self.show_control.get(), self.show_mesh.get(), self.show_filled.get(),
            color_mode, const_col, texture, k_d, k_s, m, I_L, light_pos,
            normal_enabled, normal_texture
        )
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")


if __name__ == "__main__":
    CONTROL_POINTS_FILE = "control_points.txt"
    CONTROL_POINTS2_FILE = "control_points2.txt"

    control_points1 = load_control_points(CONTROL_POINTS_FILE)
    control_points2 = load_control_points(CONTROL_POINTS2_FILE)
    root = tk.Tk()
    root.title("Bézier Surface Renderer")
    app = BezierApp(root, control_points1, control_points2)
    root.mainloop()
