from heapq import heappush, heappop, heapify
from collections import deque
import random

def print_queue(q):
    for (cost, val) in q:
        print("Queue cost: ", cost, ", val: ", val)






def display_queue(queue: SimplePriorityQueue):
    # Loop through and pop each element off and display the cost and index
    while queue.count() > 0:
        # Get the cost
        ind_peek = queue.peek()
        cost = queue.cost_dict[ind_peek]

        # Display the cost and index
        print("Cost: ", cost, ", Index: ", queue.pop())



def test_priority_queue():
    """Tests the priority queue with random numbers and displays the result"""
    # ######## Test sorting ############
    # # Loop through and add indices with random costs
    # queue = SimplePriorityQueue()
    # for k in range(500):
    #     queue.push(cost=random.random(), index=k)

    # # Display the results
    # display_queue(queue=queue)

    ######## Test updating the cost ##########
    # Loop through and add indices with random costs
    queue = SimplePriorityQueue()
    cost = 0.1
    for k in range(200):
        queue.push(cost=random.random(), index=k)
        #queue.push(cost=cost, index=k)
        #cost += 0.1

    # Update the costs
    queue.update(cost=1.2, index=3)
    queue.update(cost=1.1, index=1)

    # Display the results
    display_queue(queue=queue)



if __name__ == "__main__":
    # q = []
    # heappush(q, (1, 'first'))
    # heappush(q, (0, 'Second'))

    # q.append((-1, "append manual"))

    # q.insert(0, (3, "prepended"))

    # heappush(q, (5, "last"))

    # ind_found = q.index((3, "prepended"))
    # print("index found: ", ind_found)

    # for (ind, name) in q:
    #     print("Name: ", name, ", with index: ", ind)

    test_priority_queue()