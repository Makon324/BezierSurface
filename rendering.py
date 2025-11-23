import compiled  # <-- this triggers __init__.py → adds folder to path
import numpy as np
from PIL import Image
from rasterize import fill_triangle, draw_line
from data_structures import Point  # For draw_line compatibility
from typing import List, Tuple, Optional, Any


def render_to_image(
    triangles: List[Any],
    rotated_controls: np.ndarray,
    width: int,
    height: int,
    scale: float,
    show_control: bool,
    show_mesh: bool,
    show_filled: bool,
    color_mode: str,
    const_col: Tuple[int, int, int],
    texture: Optional[Image.Image],
    k_d: float,
    k_s: float,
    m: int,
    I_L: Tuple[float, float, float],
    light_pos: Point,
    normal_enabled: bool,
    normal_texture: Optional[Image.Image],
) -> Image.Image:
    """
    Render the Bézier surface to an image based on the provided parameters.

    Args:
        triangles: List of triangles to render.
        rotated_controls: Rotated control points array (4x4x3).
        width: Width of the output image.
        height: Height of the output image.
        scale: Scaling factor for rendering.
        show_control: Whether to show the control mesh.
        show_mesh: Whether to show the triangle mesh.
        show_filled: Whether to fill the triangles.
        color_mode: Color mode ('constant' or 'texture').
        const_col: Constant color tuple (R, G, B).
        texture: Optional texture image.
        k_d: Diffuse reflection coefficient.
        k_s: Specular reflection coefficient.
        m: Specular exponent.
        I_L: Light intensity (R, G, B).
        light_pos: Position of the light source.
        normal_enabled: Whether normal mapping is enabled.
        normal_texture: Optional normal map texture.

    Returns:
        PIL Image object of the rendered scene.
    """
    # Define colors as constants for clarity
    WHITE = [255, 255, 255]
    BLACK = (0, 0, 0)
    BLUE = (0, 0, 255)

    img_array = np.full((height, width, 3), WHITE, dtype=np.uint8)  # White background

    offset_x = width / 2.0
    offset_y = height / 2.0

    tex_array = np.array(texture.convert("RGB")) if texture is not None else None
    normal_tex_array = (
        np.array(normal_texture.convert("RGB")) if normal_enabled else None
    )

    if show_filled:
        for tri in triangles:
            fill_triangle(
                img_array,
                tri,
                scale,
                offset_x,
                offset_y,
                color_mode,
                const_col,
                tex_array,
                k_d,
                k_s,
                m,
                I_L,
                light_pos,
                normal_enabled,
                normal_tex_array,
            )

    if show_mesh:
        for tri in triangles:
            for i in range(3):
                v1 = tri.vertices[i]
                v2 = tri.vertices[(i + 1) % 3]
                draw_line(
                    img_array,
                    v1.P_post,
                    v2.P_post,
                    scale,
                    offset_x,
                    offset_y,
                    *BLACK,  # Black
                )

    if show_control:
        # Draw horizontal lines (constant i)
        for i in range(4):
            for j in range(3):
                p1 = Point(
                    rotated_controls[i, j, 0],
                    rotated_controls[i, j, 1],
                    rotated_controls[i, j, 2],
                )
                p2 = Point(
                    rotated_controls[i, j + 1, 0],
                    rotated_controls[i, j + 1, 1],
                    rotated_controls[i, j + 1, 2],
                )
                draw_line(
                    img_array,
                    p1,
                    p2,
                    scale,
                    offset_x,
                    offset_y,
                    *BLUE,  # Blue
                )

        # Draw vertical lines (constant j)
        for j in range(4):
            for i in range(3):
                p1 = Point(
                    rotated_controls[i, j, 0],
                    rotated_controls[i, j, 1],
                    rotated_controls[i, j, 2],
                )
                p2 = Point(
                    rotated_controls[i + 1, j, 0],
                    rotated_controls[i + 1, j, 1],
                    rotated_controls[i + 1, j, 2],
                )
                draw_line(
                    img_array,
                    p1,
                    p2,
                    scale,
                    offset_x,
                    offset_y,
                    *BLUE,  # Blue
                )

    img = Image.fromarray(img_array, "RGB")
    return img