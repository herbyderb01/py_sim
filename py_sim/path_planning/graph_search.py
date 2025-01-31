""" graph_search.py provides the utilities required for searching a graph to find a plan

Attributes:
    GraphType(TypeVar): Defines a type for using a graph with bound of nx.Graph

"""

from typing import Any, Generic, Optional, Protocol, TypeVar, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import TwoDimArray
from rtree import index


class World(Protocol):
    """Defines the intersection method required for the path graph"""
    def intersects_obstacle(self, edge: npt.NDArray[Any], shrink_edge: bool = False, edge_shrink_dist: float = 1.e-3) -> bool:
        """ Determines whether an edge intersects an obstacle, returns true if it does

        Args:
            edge: 2x2 matrix of points where each point is a column
            shrink_edge: True implies that the edge points will be moved towards each
                other before checking for intersection. This allows, for example, the edge
                points to lie on the obstacle
            edge_shrink_dist: The distance that the edges will be moved towards each other

        Returns:
            bool: True if the edge intersects the obstacles, false if it does not
        """

GraphType = TypeVar("GraphType", bound=nx.Graph)

class PathGraph(Generic[GraphType]):
    """ Provides an interface to networkx graph creation to ensure that
        node position information and edge weights are appropriately defined
        for graph search methods

    Attributes:
        graph(GraphType): The graph over which the search is performed
        node_location(dict[int, npt.NDArray[Any]]): Maps from the node number to the node position (shape (2,))
        rtree(Index): rtree used for rapid nearest neighbor searching
        _last_index(int): Stores a counter for node indices
    """
    def __init__(self, graph: GraphType) -> None:
        """Initializes the data structures
        """
        self.graph: GraphType = graph
        self.node_location: dict[int, npt.NDArray[Any]] = {}# Maps from the node number to the node position (shape (2,))
        self.rtree = index.Index() # Used for nearest neighbor searching

        # Private variables
        self._last_index: int = 0 # Stores a counter for node indices

    def add_node(self, position: TwoDimArray) -> int:
        """Adds a node given the position. The node location is assumed unique, but not checked

        Args:
            position: the physical (x,y) location of the node

        Returns:
            int: The node index added
        """
        # Add the position to the node location
        node_index = self._last_index
        self.node_location[node_index] = np.array([position.x, position.y])

        # Add the position to the rtree (note that a repeat of the position is used as
        # a point is being added. rtree supports rectangles)
        self.rtree.insert(id=node_index, coordinates=(position.x, position.y, position.x, position.y) )

        # Add the node to the graph
        self.graph.add_node(node_index)
        self._last_index += 1

        return node_index

    def add_node_and_edges(self, position: TwoDimArray, world: World, n_connections: int = 1) -> int:
        """Adds a node as well as edges to all other nodes that do not intersect with the obstacles

        Args:
            position: Node position to be added
            world: Polygon world used to check for obstacles
            n_connections: The number of connections to add

        Returns:
            int: The index for the node added
        """
        # Get the nearest connection indices (times 10 to account for obstacles)
        nearest_points, nearest_ind = self.nearest(point=position, n_nearest=n_connections*10)

        # Add the node to the graph
        ind_node = self.add_node(position=position)

        # Initialize the edge
        edge = np.zeros((2,2))
        edge[0,0] = position.x
        edge[1,0] = position.y

        # Loop through all possible new edges and add them if they do not intersect with obstacle
        edge_added_count = 0
        for node, node_position in zip(nearest_ind, nearest_points):
            # Create the edge using the new position
            edge[:,1] = node_position

            # Check the edge for collisions with obstacles
            if not world.intersects_obstacle(edge=edge, shrink_edge=True):
                self.add_edge(ind_node, node)

            # Update the edge count
            edge_added_count += 1
            if edge_added_count >= n_connections:
                break

        return ind_node

    def add_edge(self, node_1: int, node_2: int, weight: Optional[float] = None) -> bool:
        """ Adds the edge into the graph with the optional weight. If no weight is provided
            then euclidean distance between the nodes is used

        Args:
            node_1: First node in the edge
            node_2: Second node in the edge

        Output:
            bool: True if the edge was added, false if the nodes do no exist in the graph (thus not added)
        """
        # Check to see if the nodes exist in the graph
        if node_1 not in self.node_location or node_2 not in self.node_location:
            return False

        # Calculate the weight if not provided
        if weight is None:
            weight = cast(float, np.linalg.norm(self.node_location[node_1]-self.node_location[node_2]))

        # Add the edge to the graph
        self.graph.add_edge(u_of_edge=node_1, v_of_edge=node_2, weight=weight)

        return True

    def convert_to_cartesian(self, nodes: list[int]) -> tuple[list[float], list[float]]:
        """ Converts a list of nodes into x and y vectors

        Args:
            nodes: Node indices to be converted

        Returns:
            tuple[list[float], list[float]]:
                x_vec: List of x indices

                y_vec: List of y indices
        """
        # Initialize the outputs
        x_vec: list[float] = []
        y_vec: list[float] = []

        # Populate the vectors
        for node in nodes:
            position = self.node_location[node]
            x_vec.append(position.item(0))
            y_vec.append(position.item(1))

        # Return the converted path
        return (x_vec, y_vec)

    def calculate_path_length(self, nodes: list[int]) -> float:
        """Calculates the length of the path through a graph

        Args:
            nodes: ordered list of indices for the nodes in the graph

        Returns:
            float: path length
        """
        # Zero path if the number of nodes is less than two
        if len(nodes) < 2:
            return 0.

        # Convert the path to cartesian coordinates
        x_vec, y_vec = self.convert_to_cartesian(nodes=nodes)

        # Loop through and create the path
        q_prev = np.array([x_vec[0], y_vec[0]])
        length = 0.
        for x,y in zip(x_vec, y_vec):
            # Form the position array
            q = np.array([x,y])

            # Augment the length
            length += cast(float, np.linalg.norm(q-q_prev))

            # Update for the next iteration
            q_prev = q

        return length

    def nearest(self, point: TwoDimArray, n_nearest: int = 1) -> tuple[list[npt.NDArray[Any]], list[int]]:
        """Finds the n_nearest neighbors to the given point

        Args:
            point: Point around which to find the nearest neighbors
            n_nearest: The number of nearest neighbors to find

        Returns:
            tuple[list[npt.NDArray[Any]], list[int]]:
                nearest_points: A list of the nearest neighbors found. Each is a (2,) array
                    Can be more than n_nearest if points are equidistant
                    Can be less than n_nearest if there are less points

                nearest_ind: The corresponding indices
        """
        # Initialize the output
        nearest_points: list[npt.NDArray[Any]] = []

        # Loop through all of the nearest points
        nearest_ind = cast(list[int], list(self.rtree.nearest(coordinates= \
                        (point.x, point.y, point.x, point.y), num_results=n_nearest)) )
        for ind in nearest_ind:
            nearest_points.append(self.node_location[ind])

        return nearest_points, nearest_ind

    def node_count(self) -> int:
        """Returns the number of nodes in the graph"""
        return len(self.node_location)

    def remove(self, node: int) -> None:
        """Removes the node from the graph if it exists

        Args:
            node: Index of the node to be removed
        """
        if node in self.node_location:
            q = self.node_location.pop(node)
            self.graph.remove_node(n=node)
            self.rtree.delete(id=node, coordinates=(q.item(0), q.item(1), q.item(0), q.item(1)))

    def remove_edge(self, node_1: int, node_2: int) -> None:
        """Removes the edge, if it exists

        Args:
            node_1: Index of the first node in the edge to be removed
            node_2: Index of the second node in the edge to be removed

        """
        self.graph.remove_edge(u=node_1, v=node_2)

    def get_node_position(self, node: int) -> TwoDimArray:
        """Returns the position of the given node

        Args:
            node: index of the node being evaluated

        Returns:
            TwoDimArray: The position of the node
        """
        node_loc = self.node_location[node]
        return TwoDimArray(x=node_loc.item(0), y=node_loc.item(1))

    def get_edge_weight(self, node_1: int, node_2: int) -> float:
        """ Returns the stored weight of the edge

        Args:
            node_1: parent node
            node_2: child node

        Returns:
            float: The stored weight of the edge
        """
        return cast(float, self.graph[node_1][node_2]["weight"])


class UndirectedPathGraph(PathGraph[nx.Graph]):
    """An undirected graph default for the PathGraph"""
    def __init__(self) -> None:
        super().__init__(graph=nx.Graph())

class DirectedPathGraph(PathGraph[nx.DiGraph]):
    """A directed graph default for the PathGraph"""
    def __init__(self) -> None:
        super().__init__(graph=nx.DiGraph())
