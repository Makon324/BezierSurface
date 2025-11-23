import cython
cimport cython

import numpy as np
cimport numpy as np

from data_structures cimport Point, Triangle, Vertex, Point2D, Edge
from libc.math cimport floor, ceil, fabs, round, sqrt, pow, fmax, fmin
from libc.stdlib cimport abs as c_abs

cdef Point2D project(Point p, double scale, double offset_x, double offset_y):
    cdef Point2D res
    res.x = p.x * scale + offset_x
    res.y = p.y * scale + offset_y
    return res

cdef void set_pixel(np.ndarray[np.uint8_t, ndim=3] img, int x, int y, int r, int g, int b):
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    if 0 <= x < width and 0 <= y < height:
        img[y, x, 0] = r
        img[y, x, 1] = g
        img[y, x, 2] = b

@cython.boundscheck(False)
@cython.wraparound(False)
def fill_triangle(np.ndarray[np.uint8_t, ndim=3] img_array, Triangle tri, double scale, double offset_x, double offset_y, str color_mode, tuple const_col, object tex_array, double k_d, double k_s, int m, object I_L, Point light_pos, bint normal_enabled, object normal_tex_array):
    # Project vertices to 2D
    cdef Vertex v0 = tri.vertices[0]
    cdef Vertex v1 = tri.vertices[1]
    cdef Vertex v2 = tri.vertices[2]
    cdef Point2D p0 = project(v0.P_post, scale, offset_x, offset_y)
    cdef Point2D p1 = project(v1.P_post, scale, offset_x, offset_y)
    cdef Point2D p2 = project(v2.P_post, scale, offset_x, offset_y)

    # Sort points by y (p0.y <= p1.y <= p2.y) and corresponding vertices
    cdef Point2D temp_p
    cdef Vertex temp_v
    if p0.y > p1.y:
        temp_p = p0
        p0 = p1
        p1 = temp_p
        temp_v = v0
        v0 = v1
        v1 = temp_v
    if p0.y > p2.y:
        temp_p = p0
        p0 = p2
        p2 = temp_p
        temp_v = v0
        v0 = v2
        v2 = temp_v
    if p1.y > p2.y:
        temp_p = p1
        p1 = p2
        p2 = temp_p
        temp_v = v1
        v1 = v2
        v2 = temp_v

    # Skip degenerate (flat line)
    if fabs(p0.y - p2.y) < 1e-6:
        return

    cdef double denom = p0.x * (p1.y - p2.y) + p1.x * (p2.y - p0.y) + p2.x * (p0.y - p1.y)
    if fabs(denom) < 1e-6:
        return

    cdef int y_start = <int>ceil(p0.y)
    cdef int y_end = <int>floor(p2.y)

    if y_start > y_end:
        return  # No pixels to fill

    # Declare all Edge variables upfront
    cdef Edge long_edge
    cdef Edge e1
    cdef Edge e2

    # Prepare edge table
    cdef dict edge_table = {}

    long_edge = make_edge(p0, p2)  # Long edge always
    if long_edge.y_min <= long_edge.y_max:
        if long_edge.y_min not in edge_table:
            edge_table[long_edge.y_min] = []
        edge_table[long_edge.y_min].append(long_edge)

    if fabs(p0.y - p1.y) > 1e-6:
        e1 = make_edge(p0, p1)
        if e1.y_min <= e1.y_max:
            if e1.y_min not in edge_table:
                edge_table[e1.y_min] = []
            edge_table[e1.y_min].append(e1)

    if fabs(p1.y - p2.y) > 1e-6:
        e2 = make_edge(p1, p2)
        if e2.y_min <= e2.y_max:
            if e2.y_min not in edge_table:
                edge_table[e2.y_min] = []
            edge_table[e2.y_min].append(e2)

    cdef list active_edges = []
    cdef int y = y_start
    cdef double x_left, x_right
    cdef int x_start_pix, x_end_pix
    cdef int r_pix, g_pix, b_pix
    cdef Edge edge
    cdef np.ndarray[np.uint8_t, ndim=3] tex
    cdef int tex_width, tex_height
    cdef double uu, vv
    cdef int tx, ty
    cdef double l0, l1, l2
    cdef double px, py

    cdef bint is_texture = color_mode == "texture" and tex_array is not None
    if is_texture:
        tex = tex_array
        tex_height = tex.shape[0]
        tex_width = tex.shape[1]

    cdef bint use_normal_map = normal_enabled and normal_tex_array is not None
    cdef np.ndarray[np.uint8_t, ndim=3] normal_tex
    cdef int normal_tex_width, normal_tex_height
    if use_normal_map:
        normal_tex = normal_tex_array
        normal_tex_height = normal_tex.shape[0]
        normal_tex_width = normal_tex.shape[1]

    cdef double I_L_r = <double>I_L[0]
    cdef double I_L_g = <double>I_L[1]
    cdef double I_L_b = <double>I_L[2]

    cdef Point V = Point(0, 0, 1)

    cdef Point pos = Point()
    cdef Point N = Point()
    cdef Point L_vec = Point()
    cdef Point L = Point()
    cdef Point R = Point()
    cdef double norm_N, norm_L, norm_R
    cdef double dot_NL, dot_VR
    cdef double I_r, I_g, I_b
    cdef double I_O_r, I_O_g, I_O_b
    cdef double spec

    cdef double Pu_x, Pu_y, Pu_z, Pv_x, Pv_y, Pv_z
    cdef double T_x, T_y, T_z, B_x, B_y, B_z, N_x, N_y, N_z
    cdef double norm_Pu, norm_Pv
    cdef double normal_map_x, normal_map_y, normal_map_z

    while y <= y_end:
        # Add new edges starting at this y from edge_table
        if y in edge_table:
            active_edges.extend(edge_table[y])

        # Sort active edges by current x
        active_edges.sort(key=lambda e: (<Edge>e).x_start)

        # Fill between pairs (for triangle always <=2 active edges)
        if len(active_edges) >= 2:
            x_left = (<Edge>active_edges[0]).x_start
            x_right = (<Edge>active_edges[1]).x_start
            if x_left > x_right:
                x_left, x_right = x_right, x_left

            x_start_pix = <int>ceil(x_left)
            x_end_pix = <int>floor(x_right)

            for x in range(x_start_pix, x_end_pix + 1):
                px = <double>x + 0.5
                py = <double>y + 0.5
                l0 = (p1.x * (p2.y - py) + p2.x * (py - p1.y) + px * (p1.y - p2.y)) / denom
                l1 = (p2.x * (p0.y - py) + p0.x * (py - p2.y) + px * (p2.y - p0.y)) / denom
                l2 = (p0.x * (p1.y - py) + p1.x * (py - p0.y) + px * (p0.y - p1.y)) / denom

                # Interpolate position
                pos.x = l0 * v0.P_post.x + l1 * v1.P_post.x + l2 * v2.P_post.x
                pos.y = l0 * v0.P_post.y + l1 * v1.P_post.y + l2 * v2.P_post.y
                pos.z = l0 * v0.P_post.z + l1 * v1.P_post.z + l2 * v2.P_post.z

                # Interpolate normal
                N.x = l0 * v0.N_post.x + l1 * v1.N_post.x + l2 * v2.N_post.x
                N.y = l0 * v0.N_post.y + l1 * v1.N_post.y + l2 * v2.N_post.y
                N.z = l0 * v0.N_post.z + l1 * v1.N_post.z + l2 * v2.N_post.z

                norm_N = sqrt(N.x * N.x + N.y * N.y + N.z * N.z)
                if norm_N > 0:
                    N.x /= norm_N
                    N.y /= norm_N
                    N.z /= norm_N
                else:
                    set_pixel(img_array, x, y, 0, 0, 0)
                    continue

                # Interpolate Pu and Pv for tangent space
                Pu_x = l0 * v0.Pu_post.x + l1 * v1.Pu_post.x + l2 * v2.Pu_post.x
                Pu_y = l0 * v0.Pu_post.y + l1 * v1.Pu_post.y + l2 * v2.Pu_post.y
                Pu_z = l0 * v0.Pu_post.z + l1 * v1.Pu_post.z + l2 * v2.Pu_post.z
                norm_Pu = sqrt(Pu_x * Pu_x + Pu_y * Pu_y + Pu_z * Pu_z)
                if norm_Pu > 0:
                    T_x = Pu_x / norm_Pu
                    T_y = Pu_y / norm_Pu
                    T_z = Pu_z / norm_Pu
                else:
                    T_x = T_y = T_z = 0.0  # Fallback, though rare

                Pv_x = l0 * v0.Pv_post.x + l1 * v1.Pv_post.x + l2 * v2.Pv_post.x
                Pv_y = l0 * v0.Pv_post.y + l1 * v1.Pv_post.y + l2 * v2.Pv_post.y
                Pv_z = l0 * v0.Pv_post.z + l1 * v1.Pv_post.z + l2 * v2.Pv_post.z
                norm_Pv = sqrt(Pv_x * Pv_x + Pv_y * Pv_y + Pv_z * Pv_z)
                if norm_Pv > 0:
                    Pv_x /= norm_Pv
                    Pv_y /= norm_Pv
                    Pv_z /= norm_Pv

                # Compute orthogonal bitangent B = N x T (for right-handed TBN)
                N_x = N.x
                N_y = N.y
                N_z = N.z
                B_x = N_y * T_z - N_z * T_y
                B_y = N_z * T_x - N_x * T_z
                B_z = N_x * T_y - N_y * T_x

                # If normal mapping enabled, perturb N
                if use_normal_map:
                    uu = l0 * v0.u + l1 * v1.u + l2 * v2.u
                    vv = l0 * v0.v + l1 * v1.v + l2 * v2.v
                    tx = <int>round(uu * (normal_tex_width - 1))
                    ty = <int>round(vv * (normal_tex_height - 1))
                    if tx < 0: tx = 0
                    elif tx >= normal_tex_width: tx = normal_tex_width - 1
                    if ty < 0: ty = 0
                    elif ty >= normal_tex_height: ty = normal_tex_height - 1

                    # Sample normal map (0-1 range, then to -1 to 1)
                    normal_map_x = (normal_tex[ty, tx, 0] / 255.0) * 2.0 - 1.0
                    normal_map_y = (normal_tex[ty, tx, 1] / 255.0) * 2.0 - 1.0
                    normal_map_z = (normal_tex[ty, tx, 2] / 255.0) * 2.0 - 1.0

                    # Apply perturbation in world space via TBN
                    N.x = normal_map_x * T_x + normal_map_y * B_x + normal_map_z * N_x
                    N.y = normal_map_x * T_y + normal_map_y * B_y + normal_map_z * N_y
                    N.z = normal_map_x * T_z + normal_map_y * B_z + normal_map_z * N_z

                    # Renormalize perturbed N
                    norm_N = sqrt(N.x * N.x + N.y * N.y + N.z * N.z)
                    if norm_N > 0:
                        N.x /= norm_N
                        N.y /= norm_N
                        N.z /= norm_N

                # Compute L
                L_vec.x = light_pos.x - pos.x
                L_vec.y = light_pos.y - pos.y
                L_vec.z = light_pos.z - pos.z
                norm_L = sqrt(L_vec.x * L_vec.x + L_vec.y * L_vec.y + L_vec.z * L_vec.z)
                if norm_L > 0:
                    L.x = L_vec.x / norm_L
                    L.y = L_vec.y / norm_L
                    L.z = L_vec.z / norm_L
                else:
                    set_pixel(img_array, x, y, 0, 0, 0)
                    continue

                dot_NL = N.x * L.x + N.y * L.y + N.z * L.z
                dot_NL = fmax(0.0, dot_NL)

                # Compute R
                R.x = 2 * dot_NL * N.x - L.x
                R.y = 2 * dot_NL * N.y - L.y
                R.z = 2 * dot_NL * N.z - L.z
                norm_R = sqrt(R.x * R.x + R.y * R.y + R.z * R.z)
                if norm_R > 0:
                    R.x /= norm_R
                    R.y /= norm_R
                    R.z /= norm_R

                dot_VR = V.x * R.x + V.y * R.y + V.z * R.z
                dot_VR = fmax(0.0, dot_VR)

                spec = pow(dot_VR, m)

                # Get I_O
                if is_texture:
                    uu = l0 * v0.u + l1 * v1.u + l2 * v2.u
                    vv = l0 * v0.v + l1 * v1.v + l2 * v2.v
                    tx = <int>round(uu * (tex_width - 1))
                    ty = <int>round(vv * (tex_height - 1))
                    if tx < 0: tx = 0
                    elif tx >= tex_width: tx = tex_width - 1
                    if ty < 0: ty = 0
                    elif ty >= tex_height: ty = tex_height - 1
                    I_O_r = tex[ty, tx, 0] / 255.0
                    I_O_g = tex[ty, tx, 1] / 255.0
                    I_O_b = tex[ty, tx, 2] / 255.0
                else:
                    I_O_r = const_col[0] / 255.0
                    I_O_g = const_col[1] / 255.0
                    I_O_b = const_col[2] / 255.0

                # Compute I
                I_r = k_d * I_L_r * I_O_r * dot_NL + k_s * I_L_r * I_O_r * spec
                I_g = k_d * I_L_g * I_O_g * dot_NL + k_s * I_L_g * I_O_g * spec
                I_b = k_d * I_L_b * I_O_b * dot_NL + k_s * I_L_b * I_O_b * spec

                # Clamp to 0-1
                I_r = fmax(0.0, fmin(1.0, I_r))
                I_g = fmax(0.0, fmin(1.0, I_g))
                I_b = fmax(0.0, fmin(1.0, I_b))

                # To 0-255
                r_pix = <int>(I_r * 255 + 0.5)
                g_pix = <int>(I_g * 255 + 0.5)
                b_pix = <int>(I_b * 255 + 0.5)

                set_pixel(img_array, x, y, r_pix, g_pix, b_pix)

        # Advance y
        y += 1

        # Remove ended edges
        active_edges = [e for e in active_edges if (<Edge>e).y_max >= y]

        # Update x for remaining active edges
        for i in range(len(active_edges)):
            edge = active_edges[i]
            edge.x_start += edge.dx_dy
            active_edges[i] = edge

cdef Edge make_edge(Point2D a, Point2D b):
    if a.y > b.y:
        a, b = b, a
    cdef Edge e
    e.y_min = <int>ceil(a.y)
    e.y_max = <int>floor(b.y)
    e.x_start = a.x
    dy = b.y - a.y
    if dy != 0:
        e.dx_dy = (b.x - a.x) / dy
    else:
        e.dx_dy = 0.0
    if e.y_min > a.y:
        e.x_start += e.dx_dy * (e.y_min - a.y)
    return e

@cython.boundscheck(False)
@cython.wraparound(False)
def draw_line(np.ndarray[np.uint8_t, ndim=3] img_array, Point p_start, Point p_end, double scale, double offset_x, double offset_y, int r, int g, int b):
    cdef double x1_d = p_start.x * scale + offset_x
    cdef double y1_d = p_start.y * scale + offset_y
    cdef double x2_d = p_end.x * scale + offset_x
    cdef double y2_d = p_end.y * scale + offset_y

    cdef int x1 = <int>round(x1_d)
    cdef int y1 = <int>round(y1_d)
    cdef int x2 = <int>round(x2_d)
    cdef int y2 = <int>round(y2_d)

    cdef int dx = c_abs(x2 - x1)
    cdef int dy = c_abs(y2 - y1)
    cdef int sx = 1 if x1 < x2 else -1
    cdef int sy = 1 if y1 < y2 else -1
    cdef int err = dx - dy
    cdef int e2

    while True:
        set_pixel(img_array, x1, y1, r, g, b)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy