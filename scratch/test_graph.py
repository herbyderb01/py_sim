"""test_graph.py tests functions and attributes of import networkx as nx
"""
import networkx as nx

def get_directed_plan() -> None:
    """Test traversing through a tree
    """
    # Create a simple graph
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    graph.add_edge(5,3)

    # Traverse up through the graph
    nodes = [0]
    parent_itr: iter = graph.predecessors(n=nodes[0])
    try:
        while True:
            n = parent_itr.__next__()
            nodes.append(n)
            parent_itr = graph.predecessors(n=n)
    except:
        pass

    # Print the nodes
    print("path: ", nodes)



if __name__ == "__main__":
    get_directed_plan()