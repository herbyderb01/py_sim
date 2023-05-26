"""simple_priority_queue.py Defines a very simple priority queue based on the heapq library
"""

from heapq import heappush, heappop, heapify

class SimplePriorityQueue:
    """Defines a priority queue where the first element is a cost and the second an index"""
    def __init__(self) -> None:
        # Initialize an empty queue
        self.q: list[tuple[float, int]] = []
        self.cost_dict: dict[int, float] = {}


    def push(self, cost: float, index: int) -> None:
        """Pushes the index onto the queue with a priority of cost. This does not check for duplicates.

            Inputs:
                cost: The priority of the index
                index: Index being added
        """
        heappush(self.q, (cost, index))
        self.cost_dict[index] = cost

    def peek(self) -> int:
        """Peeks at the lowest cost index in the queue
        """
        (_, val) = self.q[0]
        return val

    def pop(self) -> int:
        """Pops the lowest cost element from the queue
        """
        # Get the lowest cost element
        (_, index) = heappop(self.q)

        # Remove the element from the dictionary
        self.cost_dict.pop(index)

        return index

    def contains(self, index: int) -> bool:
        """Returns true if the index is in the priority queue"""
        return index in self.cost_dict

    def update(self, cost: float, index: int) -> None:
        """Updates the cost associated with the given index, does nothing if the index is not in the queue

        Note that this function is rather inefficient due to the resorting. A more
        efficient technique uses bubble sort algorithms to adjust the position of
        the element being replaced
        """
        # Check to see if the index is in the queue
        if not self.contains(index):
            return

        # Determine the index of the element in the queue
        cost_prev = self.cost_dict[index]
        ind = self.q.index((cost_prev, index))

        # Update the cost
        self.q[ind] = (cost, index)
        self.cost_dict[index] = cost

        # Resort
        heapify(self.q)

    def count(self) -> int:
        """Returns the number of elements in the queue"""
        return len(self.q)
