import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
from typing import List, Tuple


class NormalMapManager:
    """Manager for handling normal map enabling and loading in a GUI application."""

    def __init__(self, parent_frame: tk.Frame, update_callback: callable):
        """
        Initialize the NormalMapManager.

        Args:
            parent_frame: The parent Tkinter frame.
            update_callback: Function to call when normal map parameters change.
        """
        self.parent_frame = parent_frame
        self.update_callback = update_callback
        self.enabled = tk.BooleanVar(value=False)
        self.normal_texture = self.create_default_normal_map()  # Default flat map

    def create_default_normal_map(self) -> Image.Image:
        """Create a default flat normal map where all values are (128, 128, 255) corresponding to (0, 0, 1) in tangent space."""
        size = 256
        img = Image.new("RGB", (size, size), (128, 128, 255))
        return img

    def load_normal_map(self) -> None:
        """Open file dialog to load a normal map image and update if selected."""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp")]
        )
        if filename:
            try:
                self.normal_texture = Image.open(filename)
                self.update_callback()
            except (IOError, Image.DecompressionBombError) as e:
                # Handle potential errors in loading image
                print(f"Error loading normal map: {e}")
                # Optionally, show a messagebox error in GUI

    def build_gui_elements(self) -> List[tk.Widget]:
        """Build and return the list of GUI elements for normal map controls."""
        # Checkbox to enable
        enable_check = tk.Checkbutton(
            self.parent_frame,
            text="Enable Normal Mapping",
            variable=self.enabled,
            command=self.update_callback,
        )

        # Button to load
        load_button = tk.Button(
            self.parent_frame, text="Load Normal Map", command=self.load_normal_map
        )

        return [enable_check, load_button]

    def get_normal_map_params(self) -> Tuple[bool, Image.Image]:
        """Return the current normal map parameters: enabled status and texture."""
        return self.enabled.get(), self.normal_texture