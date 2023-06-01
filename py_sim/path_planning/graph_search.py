""" graph_search.py provides the utilities required for searching a graph to find a plan
"""

from typing import Any, cast, Optional

import networkx as nx
import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import TwoDimArray


class PathGraph:
    """ Provides an interface to networkx graph creation to ensure that
        node position information and edge weights are appropriately defined
        for graph search methods
    """
    def __init__(self) -> None:
        """Initializes the data structures"""
        self.graph = nx.Graph() # Stores the graph nodes and edges
        self.node_count: int = 0 # Stores the number of nodes in the graph (used for adding nodes)
        self.node_location: dict[int, npt.NDArray[Any]] = {}# Maps from the node number to the node position (shape (2,))

    def add_node(self, position: TwoDimArray) -> None:
        """Adds a node given the position. The node location is assumed unique, but not checked

            Inputs:
                position: the physical (x,y) location of the node
        """
        # Add the position to the node location
        self.node_location[self.node_count] = np.array([position.x, position.y])

        # Add the node to the graph
        self.graph.add_node(self.node_count)
        self.node_count += 1

    # def add_node_and_edges(self, position: TwoDimArray, max_edges: int = -1) -> int:
    #     """Adds a node and an edge to every possible node
    #     """

    def add_edge(self, node_1: int, node_2: int, weight: Optional[float] = None) -> bool:
        """ Adds the edge into the graph with the optional weight. If no weight is provided
            then euclidean distance between the nodes is used

            Inputs:
                node_1: First node in the edge
                node_2: Second node in the edge

            Output:
                True if the edge was added, false if the nodes do no exist in the graph (thus not added)
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

            Inputs:
                nodes: Node indices to be converted
            Outputs:
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
