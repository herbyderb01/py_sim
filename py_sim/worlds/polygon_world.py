"""polygon_world.py Provides classes and function for a world made of convex polygons
"""

import copy
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
from py_sim.path_planning.graph_search import UndirectedPathGraph
from py_sim.tools.sim_types import TwoDimArray
from scipy.spatial import Voronoi


class ConvexPolygon():
    """Stores the data required for convex polygon definition and evaluation

        Properties:
            vertices: 2xm matrix of points where each column is a point and m is the number of points
            normal_matrix: mx2 matrix of normal vectors to polygon sides where each normal vector is represented as a row
            offset: mx1 column vector giving the offset of each point

                Note that for two points, p1 and p2, the normal to these points can be calculated as
                    n = J@(p2-p1), where J is a 90 degree rotation matrix rotating the vectors clockwise

                A third point can be determined to be in the halfplane of the normal given that
                    n'@(p3-p1) > 0

                The normal_matrix stores n' on each row
                The offset matrix stores -n'p1 for each row

                Thus, a point, p3 can be determined to be inside the convex polygon if b has all positive elements for
                    b = normal_matrix@p3 + c

            edges: list of 2x2 matrices where the first column corresponds to the
                   starting point and the second to the ending point

    """
    def __init__(self, vertices: npt.NDArray[Any]) -> None:
        """ Initializes the polygon from the vertices provided.

        Points should not be repeated and the first point should not be equal to the last point.
        A polygon is defined by connecting the points sequentially with the last point connected
        to the first point.
        The vertices should be defined traversing counter clockwise around the polygon from a
        top down view.

            Inputs:
                vertices: 2xm matrix of points where each column consists of a single point.
        """
        # Check the vertices shape
        if vertices.shape[0] != 2 or vertices.shape[1] < 3:
            raise ValueError("The input vertices should have two rows and at least three columns/poitns")

        # Add first point to end of the vertices array for easier iteration through the edges
        self.points = copy.deepcopy(vertices)
        vertices = np.concatenate((vertices, vertices[:,0:1]), axis=1)

        # Loop through and calculate the normal and offset values
        self.normal_matrix = np.zeros((self.points.shape[1], 2)) # Matrix where the normal of each line is given in the row
        self.offset = np.zeros((self.points.shape[1], 1)) # Column vector where the offset for each normal difference is given
        self.edges: list[npt.NDArray[Any]] = []
        for k in range(self.points.shape[1]):
            # Extract the two points that make up the line
            p1 = vertices[:,k:k+1]
            p2 = vertices[:,k+1:k+2]

            # Create the normal vector
            J = np.array([[0., -1.], # CCW rotation matrix
                          [1., 0.]])
            n = J@(p2-p1)
            dist = np.linalg.norm(n)
            if dist < 1e-6: # Nearly zero
                raise ValueError("Points cannot be duplicated")
            n = n/dist

            # Store the unit vector and offset vector
            self.normal_matrix[k:k+1, :] = n.transpose()
            c = -n.transpose()@p1
            self.offset[k:k+1, :] = c

            # Check all points to see if the are in the halfplane defined by the normal and offset
            for i in range(self.points.shape[1]):
                # Extract point
                p = vertices[:, i:i+1]

                # Verify that it is not p1 or p2
                if np.linalg.norm(p-p1) < 1e-6 or np.linalg.norm(p-p2) < 1e-6:
                    continue

                # Check to see if it is in the halfplane
                if n.transpose()@p + c < 0.:
                    raise ValueError("Points must be defined in counter-clockwise order")

            # Store the edge
            edge = vertices[:, k:k+2]
            self.edges.append(edge)


    def inside_polygon(self, point: npt.NDArray[Any]) -> bool:
        """ Returns true if the given point is inside the polygon

            Inputs:
                point: 2x1 point to be evaluated
        """
        # Ensure that the point is the correct shape
        point = np.reshape(point, (2,1))

        # Evaluate the point against all of the normals
        result = self.normal_matrix@point + self.offset

        # Inside obstacle => result >=0
        for k in range(result.shape[0]):
            if result.item(k) < 0.:
                return False

        return True

class PolygonWorld():
    """Defines a world made up of distinct polygons"""
    def __init__(self, vertices: list[npt.NDArray[Any]]) -> None:
        """Creates a world of polygons given the list of matrices where each matrix defines a series of points

            Inputs:
                verticies: List of matrices where each matrix has 2 rows and at least 3 columns. Each matrix defines a convex
                polygon where vertices are defined in a counter-clockwise fashion (see notes to ConvexPolygon() above)
        """
        # Create a list of convex polygons
        self.polygons: list[ConvexPolygon] = []
        for poly_vertices in vertices:
            self.polygons.append(ConvexPolygon(vertices=poly_vertices))

        # Create an aggregate list of edges
        self.edges: list[npt.NDArray[Any]] = []
        for polygon in self.polygons:
            for edge in polygon.edges:
                self.edges.append(edge)

    def inside_obstacle(self, point: npt.NDArray[Any]) -> bool:
        """Returns true if the given point is inside any of the polygons defining the world

            Inputs:
                Point: 2x1 point to be evaluated
        """
        # Loop through each polygon and determine if the point is inside the polygon
        for polygon in self.polygons:
            if polygon.inside_polygon(point=point):
                return True

        return False

    def find_closest_obstacle(self, edge: npt.NDArray[Any]) -> Optional[tuple[float, npt.NDArray[Any]]]:
        """ Finds the closest intersection point between an edge and the edges forming the obstacles.
            Proximity is based on the location of the first point in the given edge

        Inputs:
            edge: 2x2 matrix of points where each point is a column
        """
        # Loop through each edge and find the closest
        return find_closest_intersection(edge1=edge, edge_list=self.edges)

    def intersects_obstacle(self, edge: npt.NDArray[Any], shrink_edge: bool = False, edge_shrink_dist: float = 1.e-3) -> bool:
        """ Determines whether an edge intersects an obstacle, returns true if it does

            Inputs:
                edge: 2x2 matrix of points where each point is a column
                shrink_edge: True implies that the edge points will be moved towards each
                  other before checking for intersection. This allows, for example, the edge
                  points to lie on the obstacle
                edge_shrink_dist: The distance that the edges will be moved towards each other

            Returns:
                True if the edge intersects the obstacles, false if it does not
        """
        # Shrink the edge
        if shrink_edge:
            q = (edge[:,1:2]-edge[:,0:1]) # direction vector
            q_mag = np.linalg.norm(q) # magnitude of the direction vector
            if q_mag < 2*edge_shrink_dist:
                return True # Edge is too small to shrink, so consider it an intersection
            q_hat = q/q_mag
            edge[:,0:1] = edge[:,0:1] + edge_shrink_dist*q_hat # Move edge 1 point towards point 2
            edge[:,1:2] = edge[:,1:2] - edge_shrink_dist*q_hat # Move edge 2 point towards point 1

        # Determine the intersection
        result = find_closest_intersection(edge1=edge, edge_list=self.edges)

        # Return result indicating whether or not the intersection exists
        if result is not None:
            return True

        # Determine if the edge points are inside the obstacle
        if self.inside_obstacle(point=edge[:,0:1]):
            return True
        if self.inside_obstacle(point=edge[:,1:2]):
            return True
        return False



def line_segment_intersection(edge1: npt.NDArray[Any], edge2: npt.NDArray[Any]) -> Optional[npt.NDArray[Any]]:
    """ Determines the intersection point of an two line segments or None if the intersection does not exist.

        The intersection is found by determining the scaling along edge1 of the point of intersection

        Inputs:
            edge1: 2x2 matrix of points where each point is a column
            edge2: 2x2 matrix of points where each point is a column

        Return:
            None if no intersection exists, the intersection point as a column vector if it does
    """
    # Extract the (x,y) coordinates of each point
    x1 = edge1[0,0] # Edge 1 start
    y1 = edge1[1,0]
    x2 = edge1[0,1] # Edge 1 end
    y2 = edge1[1,1]
    x3 = edge2[0,0] # Edge 2 start
    y3 = edge2[1,0]
    x4 = edge2[0,1] # Edge 2 end
    y4 = edge2[1,1]

    # Calculate the denominator
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return np.array([[x],[y]])

def find_closest_intersection(edge1: npt.NDArray[Any],
                              edge_list: list[npt.NDArray[Any]]) -> Optional[tuple[float, npt.NDArray[Any]]]:
    """ Finds the closest intersection point between an edge and an edge list.
        Proximity is based on the location of the first point in edge 1

        Inputs:
            edge1: 2x2 matrix of points where each point is a column
            edge_list: list of 2x2 matrices consisting of edges

        Returns:
            Null if no intersection found
            (inter_dist, inter_point) if an intersection is found
                inter_dist is the distance from the first point on edge 1 to the intersection point
                inter_point: 2x1 matrix representing the position of intersection
    """
    # Extract the point of interest
    p1 = edge1[:,0:1]

    # Initialize the result
    inter_dist = np.inf     # Distance to the point of intersection
    inter_point = np.zeros((2,1))

    # Loop through and find the closest intersection to p1
    for edge in edge_list:
        result = line_segment_intersection(edge1=edge1, edge2=edge)
        if result is not None:
            # Calculate the distance to the result
            dist = cast(float, np.linalg.norm(p1-result))

            # Store the distance and point if better than previous
            if dist < inter_dist:
                inter_dist = dist
                inter_point = result

    # Return the result
    if np.isinf(inter_dist):
        return None
    return (inter_dist, inter_point)

def generate_world_obstacles() -> PolygonWorld:
    """Generates a world with three different obstacles"""
    V1 = np.array([[6., 5., 1.],
                     [1., 6., 4.]])
    V2 = np.array([[17., 21., 21., 17.],
                     [1.,   1.,  7., 7.]])
    V3 = np.array([[7.,    8., 13.,  14., 13., 8.],
                     [-1.5, -3., -3., -1.5,  0., 0.]])
    return PolygonWorld(vertices=[V1, V2, V3])

def topology_world_obstacles() -> UndirectedPathGraph:
    """Generates a topology graph for the world obstacles world"""
    graph = UndirectedPathGraph()

    # Add the nodes
    graph.add_node(position=TwoDimArray(x=0., y=0.)) # Node 0
    graph.add_node(position=TwoDimArray(x=-2., y=4.)) # Node 1
    graph.add_node(position=TwoDimArray(x=6., y=0.)) # Node 2
    graph.add_node(position=TwoDimArray(x=7., y=-4.)) # Node 3
    graph.add_node(position=TwoDimArray(x=15., y=-4.)) # Node 4
    graph.add_node(position=TwoDimArray(x=15., y=0.)) # Node 5
    graph.add_node(position=TwoDimArray(x=10., y=3.)) # Node 6
    graph.add_node(position=TwoDimArray(x=22., y=0.)) # Node 7
    graph.add_node(position=TwoDimArray(x=22., y=8.)) # Node 8
    graph.add_node(position=TwoDimArray(x=15., y=8.)) # Node 9
    graph.add_node(position=TwoDimArray(x=5., y=7.)) # Node 10
    graph.add_node(position=TwoDimArray(x=7., y=1.)) # Node 11


    # Add the edges
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 6)
    graph.add_edge(4, 7)
    graph.add_edge(5, 7)
    graph.add_edge(7, 8)
    graph.add_edge(8, 9)
    graph.add_edge(5, 9)
    graph.add_edge(6, 9)
    graph.add_edge(1, 10)
    graph.add_edge(6, 10)
    graph.add_edge(9, 10)
    graph.add_edge(2, 11)
    graph.add_edge(5, 11)
    graph.add_edge(6, 11)
    graph.add_edge(10, 11)

    return graph

def generate_non_convex_obstacles() -> PolygonWorld:
    """Generates a simple world that is non-convex and bad for greedy planners"""
    V1 = np.array([[6., 5., 1.],
                     [1., 6., 4.]])
    V2 = np.array([[6., 3., 7.],
                     [2., -3., -3.]])
    V3 = np.array([[10., 10., 12., 12.],
                   [12., 0., 0., 12.]])
    return PolygonWorld(vertices=[V1, V2, V3])

def topology_non_convex_obstacles() -> UndirectedPathGraph:
    """Generates a topology graph for the non convex obstacles world"""
    graph = UndirectedPathGraph()

    # Add the nodes
    graph.add_node(position=TwoDimArray(x=0., y=0.)) # Node 0
    graph.add_node(position=TwoDimArray(x=4.75, y=0.8)) # Node 1
    graph.add_node(position=TwoDimArray(x=0., y=4.5)) # Node 2
    graph.add_node(position=TwoDimArray(x=0., y=-4.)) # Node 3
    graph.add_node(position=TwoDimArray(x=7.2, y=8.)) # Node 4
    graph.add_node(position=TwoDimArray(x=8.5, y=-4.)) # Node 5
    graph.add_node(position=TwoDimArray(x=8.25, y=-0.75)) # Node 6
    graph.add_node(position=TwoDimArray(x=13., y=-1.)) # Node 7
    graph.add_node(position=TwoDimArray(x=18., y=6.)) # Node 8

    # Add the edges
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 6)
    graph.add_edge(5, 6)
    graph.add_edge(5, 7)
    graph.add_edge(7, 8)

    return graph

def create_visibility_graph(world: PolygonWorld) -> UndirectedPathGraph:
    """ Creates a visibility graph from the polygon world
    """

    # Add in all edges corresponding to edges in the polygon
    graph = UndirectedPathGraph()
    for polygon in world.polygons: # Loop through all of the convex polygons
        node_ind_prev = 0 # Stores the first node index
        first_ind = 0 # Stores the index of the first node added
        n_edges = len(polygon.edges)
        for index, edge in enumerate(polygon.edges): # Loop through each edge
            # Special case: first edge - you need to add in the first vertex
            if index == 0:
                node_ind_prev = graph.add_node(position=TwoDimArray(x=edge[0,0], y=edge[1,0]))
                first_ind = node_ind_prev

            # Special case: last edge - no need to add the node and the edge will go to the first node
            if index >= n_edges-1:
                graph.add_edge(node_1=node_ind_prev, node_2=first_ind)
            else:
                # Add the node on the second edge
                node_ind = graph.add_node(position=TwoDimArray(x=edge[0,1], y=edge[1,1]))

                # Add in the edge
                graph.add_edge(node_1=node_ind_prev, node_2=node_ind)

            # Set the node_ind_prev for the next iteration
            node_ind_prev = node_ind

    # Add in edges between each node and each other node if no intersection exists
    nodes = list(graph.node_location.keys())
    for (index, node_1) in enumerate(nodes):
        # Initialize the edge
        edge = np.zeros((2,2))
        edge[:,0] = graph.node_location[node_1]

        # Check all possible nodes
        for k in range(index+1, graph.node_count()):
            # Create the edge
            node_2 = nodes[k]
            edge[:,1] = graph.node_location[node_2]

            # Check the edge for collision
            if not world.intersects_obstacle(edge=edge, shrink_edge=True):
                graph.add_edge(node_1=node_1, node_2=node_2)

    return graph


def create_voronoi_graph(world: PolygonWorld,
                         x_limits: tuple[float, float],
                         y_limits: tuple[float, float],
                         resolution: float = 1.) -> UndirectedPathGraph:
    """ Creates a voronoi graph from the polygon world
    """
    # Create points along the boundary
    points = np.zeros((2,0))
    x_vals = np.arange(start=x_limits[0], stop=x_limits[1],step=resolution).tolist()
    y_vals = np.arange(start=y_limits[0], stop=y_limits[1],step=resolution).tolist()
    for x in x_vals: # top and bottom rows
        points_tmp = np.array([[x, x],[y_limits[0], y_limits[1]]])
        points = np.concatenate((points, points_tmp), axis=1)
    for y in y_vals: # sides
        points_tmp = np.array([[x_limits[0], x_limits[1]],[y, y]])
        points = np.concatenate((points, points_tmp), axis=1)

    # Collect all of the points in the polygon world
    for polygon in world.polygons:
        points = np.concatenate((points, polygon.points), axis=1)

    # Sample between points in the polygon world
    for polygon in world.polygons:
        # Create a set of points that wraps back around to the start of the polygon
        vertices = np.concatenate((polygon.points, polygon.points[:,0:1]), axis=1)

        # Loop through and add points to the polygon
        point_a = vertices[:,0:1]
        for k in range(1, vertices.shape[1]):
            # Extract a point from the vertices
            point_b = vertices[:,k:k+1]

            # Get the direction vector
            dist =np.linalg.norm(point_b-point_a)
            q = (point_b - point_a)/dist

            # Add points until along the unit vector
            distances = np.arange(start=resolution, stop=dist-resolution, step=resolution).tolist()
            for distance in distances:
                point = point_a + distance*q
                points = np.concatenate((points, point), axis=1)

            # Update for next iteration
            point_a = point_b
    points = np.transpose(points)

    # Form a voronoi diagram
    vor = Voronoi(points=points)

    # Add the voronoi points to the graph
    graph = UndirectedPathGraph()
    vor_to_graph: dict[int, int] = {}
    for k in range(vor.vertices.shape[0]):
        # Check to ensure that the node is outside the obstacles
        position=TwoDimArray(x=vor.vertices[k,0], y=vor.vertices[k,1])
        if world.inside_obstacle(point=position.state):
            continue

        # Check to ensure that the node is inside the bounds
        if position.x < x_limits[0] or position.x > x_limits[1] or \
           position.y < y_limits[0] or position.y > y_limits[1]:
            continue

        # Add the node to the graph
        vor_to_graph[k] = graph.add_node(position=position)


    # Add in the edges for each of the voronoi edges
    for vertices in vor.ridge_vertices:
        if len(vertices) > 2:
            raise ValueError("Expecting only two nodes. Need to reimplement for more")
        node_1 = vertices[0]
        node_2 = vertices[1]

        # Skip if either of the points is equal to -1
        if node_1 < 0 or node_2 < 0:
            continue

        # Skip if either of the nodes is not in the graph
        if node_1 not in vor_to_graph or node_2 not in vor_to_graph:
            continue

        # Create the edge
        edge = np.zeros((2,2))
        edge[:,0] = graph.node_location[vor_to_graph[node_1] ]
        edge[:,1] = graph.node_location[vor_to_graph[node_2] ]

        # Check the edge for collision
        if not world.intersects_obstacle(edge=edge):
            graph.add_edge(node_1=vor_to_graph[node_1], node_2=vor_to_graph[node_2])

    return graph
