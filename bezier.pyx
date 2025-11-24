import cython
cimport cython

from data_structures cimport Point, Vertex
from libc.math cimport pow, sqrt

cdef double PI = 3.14159265358979323846  # More precise value for PI

cdef double deg2rad(double deg):
    """Convert degrees to radians."""
    return deg * PI / 180.0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double bernstein_poly(int i, double t):
    """Compute the Bernstein polynomial of degree 3 for basis i at parameter t."""
    if i == 0:
        return pow(1 - t, 3)
    elif i == 1:
        return 3 * t * pow(1 - t, 2)
    elif i == 2:
        return 3 * pow(t, 2) * (1 - t)
    elif i == 3:
        return pow(t, 3)
    else:
        raise ValueError("Bernstein polynomial index i must be between 0 and 3 inclusive")

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double bernstein_poly_deg2(int i, double t):
    """Compute the Bernstein polynomial of degree 2 for basis i at parameter t."""
    if i == 0:
        return pow(1 - t, 2)
    elif i == 1:
        return 2 * t * (1 - t)
    elif i == 2:
        return pow(t, 2)
    else:
        raise ValueError("Bernstein polynomial index i must be between 0 and 2 inclusive")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point bezier_surface_point(double[:, :, :] control_points, double u, double v):
    """Evaluate the Bézier surface point at parameters u and v."""
    cdef double point_x = 0.0
    cdef double point_y = 0.0
    cdef double point_z = 0.0
    cdef int i, j
    cdef double b_u, b_v
    for i in range(4):
        b_u = bernstein_poly(i, u)
        for j in range(4):
            b_v = bernstein_poly(j, v)
            point_x += b_u * b_v * control_points[i, j, 0]
            point_y += b_u * b_v * control_points[i, j, 1]
            point_z += b_u * b_v * control_points[i, j, 2]
    return Point(point_x, point_y, point_z)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point bezier_derivative_u(double[:, :, :] control_points, double u, double v):
    """Compute the partial derivative of the Bézier surface with respect to u at (u, v)."""
    cdef double der_x = 0.0
    cdef double der_y = 0.0
    cdef double der_z = 0.0
    cdef int n = 3
    cdef double b_u, b_v
    cdef int i, j
    for i in range(n):
        b_u = n * bernstein_poly_deg2(i, u)
        for j in range(n + 1):
            b_v = bernstein_poly(j, v)
            der_x += b_u * b_v * (control_points[i + 1, j, 0] - control_points[i, j, 0])
            der_y += b_u * b_v * (control_points[i + 1, j, 1] - control_points[i, j, 1])
            der_z += b_u * b_v * (control_points[i + 1, j, 2] - control_points[i, j, 2])
    return Point(der_x, der_y, der_z)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point bezier_derivative_v(double[:, :, :] control_points, double u, double v):
    """Compute the partial derivative of the Bézier surface with respect to v at (u, v)."""
    cdef double der_x = 0.0
    cdef double der_y = 0.0
    cdef double der_z = 0.0
    cdef int n = 3
    cdef double b_u, b_v
    cdef int i, j
    for i in range(n + 1):
        b_u = bernstein_poly(i, u)
        for j in range(n):
            b_v = n * bernstein_poly_deg2(j, v)
            der_x += b_u * b_v * (control_points[i, j + 1, 0] - control_points[i, j, 0])
            der_y += b_u * b_v * (control_points[i, j + 1, 1] - control_points[i, j, 1])
            der_z += b_u * b_v * (control_points[i, j + 1, 2] - control_points[i, j, 2])
    return Point(der_x, der_y, der_z)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point compute_normal(Point pu, Point pv):
    """Compute the normal vector from the cross product of partial derivatives pu and pv."""
    cdef double cx = pv.y * pu.z - pv.z * pu.y
    cdef double cy = pv.z * pu.x - pv.x * pu.z
    cdef double cz = pv.x * pu.y - pv.y * pu.x

    cdef double norm = sqrt(cx * cx + cy * cy + cz * cz)
    if norm > 1e-10:
        cx /= norm
        cy /= norm
        cz /= norm
    else:
        cx = 0.0
        cy = 0.0
        cz = 1.0  # Default to (0,0,1) in degenerate cases

    return Point(cx, cy, cz)

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_vertices(double[:, :, :] control_points, int divisions):
    """Generate a grid of vertices on the Bézier surface with the given number of divisions."""
    if divisions < 1:
        raise ValueError("Divisions must be at least 1")

    vertices = []
    cdef double du = 1.0 / divisions
    cdef double dv = 1.0 / divisions
    cdef int i, j
    cdef double u, v
    cdef Vertex vertex
    for i in range(divisions + 1):
        u = i * du
        for j in range(divisions + 1):
            v = j * dv
            vertex = Vertex(u, v)
            vertex.P_pre = bezier_surface_point(control_points, u, v)
            vertex.Pu_pre = bezier_derivative_u(control_points, u, v)
            vertex.Pv_pre = bezier_derivative_v(control_points, u, v)
            vertex.N_pre = compute_normal(vertex.Pu_pre, vertex.Pv_pre)
            vertices.append(vertex)
    return vertices, divisions + 1