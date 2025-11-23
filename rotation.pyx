import cython
cimport cython

import numpy as np
cimport numpy as np

from data_structures cimport Point, Vertex
from libc.math cimport cos, sin

cdef double PI = 3.14159265358979323846  # More precise value for PI

cdef double deg2rad(double deg):
    """Convert degrees to radians."""
    return deg * PI / 180.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void rotation_matrix_z(double alpha_deg, double[:, :] mat):
    """Compute the 3x3 rotation matrix around the Z-axis for the given angle in degrees."""
    cdef double alpha = deg2rad(alpha_deg)
    cdef double c = cos(alpha)
    cdef double s = sin(alpha)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[0, 2] = 0
    mat[1, 0] = s
    mat[1, 1] = c
    mat[1, 2] = 0
    mat[2, 0] = 0
    mat[2, 1] = 0
    mat[2, 2] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void rotation_matrix_x(double beta_deg, double[:, :] mat):
    """Compute the 3x3 rotation matrix around the X-axis for the given angle in degrees."""
    cdef double beta = deg2rad(beta_deg)
    cdef double c = cos(beta)
    cdef double s = sin(beta)
    mat[0, 0] = 1
    mat[0, 1] = 0
    mat[0, 2] = 0
    mat[1, 0] = 0
    mat[1, 1] = c
    mat[1, 2] = -s
    mat[2, 0] = 0
    mat[2, 1] = s
    mat[2, 2] = c

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point apply_rotation(Point point, double[:, :] rot_z, double[:, :] rot_x):
    """Apply Z and then X rotations to the given point using the provided matrices."""
    cdef double x = point.x
    cdef double y = point.y
    cdef double z = point.z
    # Apply rot_z
    cdef double tx = rot_z[0, 0] * x + rot_z[0, 1] * y + rot_z[0, 2] * z
    cdef double ty = rot_z[1, 0] * x + rot_z[1, 1] * y + rot_z[1, 2] * z
    cdef double tz = rot_z[2, 0] * x + rot_z[2, 1] * y + rot_z[2, 2] * z
    # Apply rot_x
    x = rot_x[0, 0] * tx + rot_x[0, 1] * ty + rot_x[0, 2] * tz
    y = rot_x[1, 0] * tx + rot_x[1, 1] * ty + rot_x[1, 2] * tz
    z = rot_x[2, 0] * tx + rot_x[2, 1] * ty + rot_x[2, 2] * tz
    return Point(x, y, z)

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_vertices(list vertices, double alpha, double beta):
    """Rotate the list of vertices around Z by alpha and X by beta degrees, updating post-rotation fields."""
    cdef double[:, :] rot_z = np.empty((3, 3), dtype=np.float64)
    cdef double[:, :] rot_x = np.empty((3, 3), dtype=np.float64)
    rotation_matrix_z(alpha, rot_z)
    rotation_matrix_x(beta, rot_x)
    cdef Vertex vertex
    for v in vertices:
        vertex = <Vertex> v
        vertex.P_post = apply_rotation(vertex.P_pre, rot_z, rot_x)
        vertex.Pu_post = apply_rotation(vertex.Pu_pre, rot_z, rot_x)
        vertex.Pv_post = apply_rotation(vertex.Pv_pre, rot_z, rot_x)
        vertex.N_post = apply_rotation(vertex.N_pre, rot_z, rot_x)

@cython.boundscheck(False)
@cython.wraparound(False)
def rotate_control_points(double[:, :, :] control_points, double alpha, double beta):
    """Rotate the 4x4x3 control points array around Z by alpha and X by beta degrees."""
    cdef np.ndarray[np.float64_t, ndim=3] rotated = np.empty((4, 4, 3), dtype=np.float64)
    cdef double[:, :] rot_z = np.empty((3, 3), dtype=np.float64)
    cdef double[:, :] rot_x = np.empty((3, 3), dtype=np.float64)
    rotation_matrix_z(alpha, rot_z)
    rotation_matrix_x(beta, rot_x)
    cdef int i, j
    cdef double x, y, z, tx, ty, tz
    for i in range(4):
        for j in range(4):
            x = control_points[i, j, 0]
            y = control_points[i, j, 1]
            z = control_points[i, j, 2]
            tx = rot_z[0, 0] * x + rot_z[0, 1] * y + rot_z[0, 2] * z
            ty = rot_z[1, 0] * x + rot_z[1, 1] * y + rot_z[1, 2] * z
            tz = rot_z[2, 0] * x + rot_z[2, 1] * y + rot_z[2, 2] * z
            x = rot_x[0, 0] * tx + rot_x[0, 1] * ty + rot_x[0, 2] * tz
            y = rot_x[1, 0] * tx + rot_x[1, 1] * ty + rot_x[1, 2] * tz
            z = rot_x[2, 0] * tx + rot_x[2, 1] * ty + rot_x[2, 2] * tz
            rotated[i, j, 0] = x
            rotated[i, j, 1] = y
            rotated[i, j, 2] = z
    return rotated