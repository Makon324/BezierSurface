import cython

cdef class Point:
    """Represents a 3D point with x, y, z coordinates."""

    def __init__(self, double x=0.0, double y=0.0, double z=0.0):
        """
        Initialize a Point instance.

        Args:
            x: The x-coordinate (default 0.0).
            y: The y-coordinate (default 0.0).
            z: The z-coordinate (default 0.0).
        """
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self) -> str:
        """Return a string representation of the Point."""
        return f"Point({self.x}, {self.y}, {self.z})"

cdef class Vertex:
    """Represents a vertex on a surface with parameters and points."""

    def __init__(self, double u, double v):
        """
        Initialize a Vertex instance.

        Args:
            u: The u parameter.
            v: The v parameter.
        """
        self.u = u
        self.v = v
        self.P_pre = Point()
        self.Pu_pre = Point()
        self.Pv_pre = Point()
        self.N_pre = Point()
        self.P_post = Point()
        self.Pu_post = Point()
        self.Pv_post = Point()
        self.N_post = Point()

cdef class Triangle:
    """Represents a triangle composed of three vertices."""

    def __init__(self, Vertex v1, Vertex v2, Vertex v3):
        """
        Initialize a Triangle instance.

        Args:
            v1: First vertex.
            v2: Second vertex.
            v3: Third vertex.
        """
        self.vertices = [v1, v2, v3]