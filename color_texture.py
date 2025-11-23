import tkinter as tk
from tkinter import filedialog, colorchooser
from PIL import Image
from typing import Tuple, Optional, List


class ColorTextureManager:
    """Manager for handling constant color or texture selection in a GUI application."""

    def __init__(self, parent: tk.Widget, update_callback: callable):
        """
        Initialize the ColorTextureManager.

        Args:
            parent: The parent Tkinter widget.
            update_callback: Function to call when color or texture changes.
        """
        self.parent = parent
        self.update_callback = update_callback
        self.mode: str = "constant"  # Default to constant color mode
        self.const_col: Tuple[int, int, int] = (255, 255, 255)  # Default white
        self.texture: Optional[Image.Image] = None

        # Buttons
        self.choose_color_btn = tk.Button(
            parent, text="Choose Color", command=self.choose_color
        )
        self.load_texture_btn = tk.Button(
            parent, text="Load Texture", command=self.load_texture
        )

    def build_gui_elements(self) -> List[tk.Widget]:
        """Return the list of GUI elements (buttons) for this manager."""
        return [self.choose_color_btn, self.load_texture_btn]

    def choose_color(self) -> None:
        """Open color chooser dialog and update constant color if selected."""
        color = colorchooser.askcolor()[0]  # Returns RGB tuple or None
        if color:
            self.const_col = tuple(map(int, color))  # Ensure integers
            self.mode = "constant"
            self.update_callback()

    def load_texture(self) -> None:
        """Open file dialog to load texture image and update if selected."""
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if filename:
            try:
                self.texture = Image.open(filename)
                self.mode = "texture"
                self.update_callback()
            except (IOError, Image.DecompressionBombError) as e:
                # Handle potential errors in loading image
                print(f"Error loading texture: {e}")
                # Optionally, show a messagebox error in GUI

    def get_color_and_texture(self) -> Tuple[str, Tuple[int, int, int], Optional[Image.Image]]:
        """Return the current mode, constant color, and texture."""
        return self.mode, self.const_col, self.texture