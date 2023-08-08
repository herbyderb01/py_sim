"""test_graph.py tests functions and attributes of import networkx as nx
"""
import networkx as nx
from matplotlib import pyplot as plt, animation
import random

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


def test_plotting() -> None:
    """Tests the repeated drawing of a graph

    """
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    fig = plt.figure()

    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2, 3, 4])

    nx.draw(G, with_labels=True)

    def animate(frame):
        fig.clear()
        num1 = random.randint(0, 4)
        num2 = random.randint(0, 4)
        G.add_edges_from([(num1, num2)])
        nx.draw(G, with_labels=True)

    ani = animation.FuncAnimation(fig, animate, frames=6, interval=1000, repeat=True)

    plt.show()



if __name__ == "__main__":
    #get_directed_plan()
    test_plotting()