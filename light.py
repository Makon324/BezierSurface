import tkinter as tk
from tkinter import colorchooser
from data_structures import Point
from math import sin, cos, pi
from typing import List, Tuple, Callable


class LightManager:
    """Manager for handling light parameters and animation in a GUI application."""

    def __init__(self, parent_frame: tk.Frame, update_callback: Callable[[], None]):
        """
        Initialize the LightManager.

        Args:
            parent_frame: The parent Tkinter frame.
            update_callback: Function to call when light parameters change.
        """
        self.parent_frame = parent_frame
        self.update_callback = update_callback
        self.I_L: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Default white
        self.k_d: float = 0.7
        self.k_s: float = 0.3
        self.m: int = 50
        self.light_z: float = 5.0
        self.animate: bool = False
        self.theta: float = 0.0
        self.light_pos: Point = Point(0, 0, self.light_z)

    def _update_light_z(self, v: str) -> None:
        """Update light_z and light_pos.z, then trigger callback."""
        self.light_z = float(v)
        self.light_pos.z = self.light_z
        self.update_callback()

    def choose_light_color(self) -> None:
        """Open color chooser dialog and update light color if selected."""
        initial_color = tuple(int(c * 255) for c in self.I_L)
        color = colorchooser.askcolor(color=initial_color)[0]
        if color:
            self.I_L = tuple(c / 255.0 for c in color)
            self.update_callback()

    def toggle_animate(self) -> None:
        """Toggle light animation on or off."""
        self.animate = not self.animate
        if self.animate:
            self.animate_light()

    def animate_light(self) -> None:
        """Animate the light position in a spiral pattern."""
        if not self.animate:
            return
        self.theta += 0.1
        r = self.theta * 0.05
        if r > 10:
            self.theta = 0
            r = 0
        x = r * cos(self.theta)
        y = r * sin(self.theta)
        z = self.light_z
        self.light_pos = Point(x, y, z)
        self.update_callback()
        self.parent_frame.after(50, self.animate_light)

    def build_gui_elements(self) -> List[tk.Widget]:
        """Build and return the list of GUI elements for light controls."""
        # k_d slider
        kd_label = tk.Label(self.parent_frame, text="k_d")
        kd_slider = tk.Scale(
            self.parent_frame,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=lambda v: setattr(self, "k_d", float(v)) or self.update_callback(),
        )
        kd_slider.set(self.k_d)

        # k_s slider
        ks_label = tk.Label(self.parent_frame, text="k_s")
        ks_slider = tk.Scale(
            self.parent_frame,
            from_=0,
            to=1,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=lambda v: setattr(self, "k_s", float(v)) or self.update_callback(),
        )
        ks_slider.set(self.k_s)

        # m slider
        m_label = tk.Label(self.parent_frame, text="m")
        m_slider = tk.Scale(
            self.parent_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            command=lambda v: setattr(self, "m", int(v)) or self.update_callback(),
        )
        m_slider.set(self.m)

        # light_z slider
        z_label = tk.Label(self.parent_frame, text="Light Z")
        z_slider = tk.Scale(
            self.parent_frame,
            from_=-10,
            to=10,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            command=self._update_light_z,
        )
        z_slider.set(self.light_z)

        # Animate checkbox
        animate_check = tk.Checkbutton(
            self.parent_frame, text="Animate Light", command=self.toggle_animate
        )

        # Light color button
        light_color_button = tk.Button(
            self.parent_frame,
            text="Choose Light Color",
            command=self.choose_light_color,
        )

        return [
            kd_label,
            kd_slider,
            ks_label,
            ks_slider,
            m_label,
            m_slider,
            z_label,
            z_slider,
            animate_check,
            light_color_button,
        ]

    def get_lighting_params(self) -> Tuple[float, float, int, Tuple[float, float, float], Point]:
        """Return the current lighting parameters."""
        return self.k_d, self.k_s, self.m, self.I_L, self.light_pos