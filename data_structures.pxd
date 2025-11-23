cdef class Point:
    cdef public double x, y, z

cdef class Vertex:
    cdef public double u, v
    cdef public Point P_pre, Pu_pre, Pv_pre, N_pre
    cdef public Point P_post, Pu_post, Pv_post, N_post

cdef class Triangle:
    cdef public list vertices

cdef struct Point2D:
    double x
    double y

cdef struct Edge:
    int y_min
    int y_max
    double x_start
    double dx_dy