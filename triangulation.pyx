import cython

from data_structures cimport Triangle, Vertex


@cython.boundscheck(False)
@cython.wraparound(False)
def triangulate_grid(list vertices, int grid_size):
    """
    Triangulate a regular grid of vertices into a list of triangles.

    The input vertices are assumed to be in row-major order (u increases first, then v),
    forming a grid of size (grid_size x grid_size).

    Each grid cell (quad) is split into two triangles with consistent winding order:
    - Upper-right: v1 -> v2 -> v4
    - Lower-left:  v1 -> v4 -> v3

    Args:
        vertices: List of Vertex objects in row-major order.
        grid_size: Number of vertices along one side of the grid (must be >= 2).

    Returns:
        List of Triangle objects covering the entire grid.

    Raises:
        ValueError: If the number of vertices does not match grid_size**2 or grid_size < 2.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    expected_len = grid_size * grid_size
    if len(vertices) != expected_len:
        raise ValueError(
            f"Expected {expected_len} vertices for grid_size={grid_size}, got {len(vertices)}"
        )

    cdef int num_cells = grid_size - 1
    cdef int num_triangles = 2 * num_cells * num_cells
    triangles = [None] * num_triangles

    cdef int i, j, idx = 0
    cdef Vertex v1, v2, v3, v4
    cdef int row_start, next_row_start

    for i in range(num_cells):
        row_start = i * grid_size
        next_row_start = (i + 1) * grid_size
        for j in range(num_cells):
            v1 = <Vertex>vertices[row_start + j]
            v2 = <Vertex>vertices[row_start + j + 1]
            v3 = <Vertex>vertices[next_row_start + j]
            v4 = <Vertex>vertices[next_row_start + j + 1]

            # Upper-right triangle
            triangles[idx] = Triangle(v1, v2, v4)
            idx += 1
            # Lower-left triangle
            triangles[idx] = Triangle(v1, v4, v3)
            idx += 1

    return triangles