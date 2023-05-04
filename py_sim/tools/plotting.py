'''This is a single class code to implement the robot sim to a single gotogoal
run this code to see the sim and change parameters below'''
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
from robot_sim.robot_sim.parameters import (dot_path, field, scanClass,
                                            stateClass, variableList)
from robot_sim.scenario.map import mapClass
from robot_sim.vector_field.vector_functions import vectorfield


def makegraph(state: stateClass, xdif: float, ydif:float, variable_array: variableList, field: field, scan: scanClass, map: mapClass):
    """make and show the sim for each time through the loop
    Args:
        delta x
        delta y
        obj variable arrays
        vector field parameters

    Returns:
        plotted graph
    """
    plt.figure(1)
    plt.clf()
    plt.title("Robot Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")

    #  #Change vector field if way point is meet
    # if(field.dot_plan.path_planning):
    #     #plot the waypoint goal
    #     x = [field.dot_plan.dot_goals[1][0]]
    #     y = [field.dot_plan.dot_goals[1][1]]
    #     plt.plot(x, y, color='b', alpha=0.5, marker='o')
    #     check_for_change_in_field(state, variable_array, field.dot_plan)

    # #print out current vector in vector field
    # graph_vector_field(variable_array, field, scan)
    # map.declare_map(scan, declare=False)
    # plot_lines_to_obstacle(state, scan)


    # #plot the blue dots
    # Xmin, Ymin, Xmax, Ymax = squareview(variable_array.xlist[0], variable_array.ylist[0], field.goal_params.x_goal, field.goal_params.y_goal)
    # if(variable_array.xlist[-1] > Xmax):
    #     Xmax = variable_array.xlist[-1]
    # if(variable_array.xlist[-1] < Xmin):
    #     Xmin = variable_array.xlist[-1]
    # if(variable_array.ylist[-1] > Ymax):
    #     Ymax = variable_array.ylist[-1]
    # if(variable_array.ylist[-1] < Ymin):
    #     Ymin = variable_array.ylist[-1]

    # if((Ymax-Ymin) > (Xmax-Xmin)):
    #     plt.axis([Ymin, Ymax, Ymin, Ymax])
    # else:
    #     plt.axis([Xmin, Xmax, Xmin, Xmax])
    # plt.scatter(variable_array.xlist, variable_array.ylist, color = 'blue', s = 5) #,s=1 for smaller dots

    #show the arrow on the robot
    create_robot_arrow(xdif, ydif, variable_array.xlist, variable_array.ylist, field.goal_params.x_goal, field.goal_params.y_goal)

# def plot_lines_to_obstacle(state: stateClass, scan: scanClass):
#     """plot the lines to obstacle if within distance
#     Args:
#         delta x
#         delta y
#         obj variable arrays
#         vector field parameters

#     Returns:
#         plotted graph
#     """
#     theta = state.theta
#     for scan_item in scan.scanresults:
#         x_values = [state.x_pos, state.x_pos+math.cos(theta)*scan_item]
#         y_values = [state.y_pos, state.y_pos+math.sin(theta)*scan_item]
#         # else:
#         #     x_values = [state.x_pos, state.x_pos+math.cos(theta)*scan.distance_dectection]
#         #     y_values = [state.y_pos, state.y_pos+math.sin(theta)*scan.distance_dectection]
#         plt.plot(x_values, y_values, color='b')
#         theta += scan.theta_diffrence

#graphs the vector field each itteration of while loop
# def graph_vector_field(variable_array: variableList, field:field, scan: scanClass):
#     """make and show the sim for each time through the loop
#     Args:
#         delta x
#         delta y
#         obj variable arrays
#         vector field parameters

#     Returns:
#         plotted graph
#     """
#     if (variable_array.changefield == True):
#         map_out_vectorfield(variable_array, field, scan)
#         print("changed vector field")
#         variable_array.changefield = False
#     #place here an update for the scan vector field induced by the scanner
#     plt.quiver(variable_array.X, variable_array.Y, variable_array.u, variable_array.v, zorder=1, color = 'green', angles='xy', scale_units='xy', scale=2*field.goal_params.velocity_nominal)

# def map_out_vectorfield(variable_array: variableList, field:field, scan: scanClass):
#     """make and show the sim for each time through the loop
#     Args:
#         delta x
#         delta y
#         obj variable arrays
#         vector field parameters

#     Returns:
#         plotted vecotr field
#     """
#     Xmin, Ymin, Xmax, Ymax = squareview(variable_array.xlist[0], variable_array.ylist[0], field.goal_params.x_goal, field.goal_params.y_goal)

#     if((Ymax-Ymin) > (Xmax-Xmin)):
#         x = np.arange(Ymin*2, Ymax*2, 0.5)
#         y = np.arange(Ymin*2, Ymax*2, 0.5)
#     else:
#         x = np.arange(Xmin*2, Xmax*2, 0.5)
#         y = np.arange(Xmin*2, Xmax*2, 0.5)


#     #create arrows at all points on veiwable graph
#     variable_array.X, variable_array.Y = np.meshgrid(x, y)
#     variable_array.u = [[0]*len(variable_array.Y[0]) for i in range(len(variable_array.X))]
#     variable_array.v = [[0]*len(variable_array.Y[0]) for i in range(len(variable_array.X))]
#     for i in range(len(variable_array.X)):
#         for j in range(len(variable_array.Y[0])):
#             variable_array.u[i][j], variable_array.v[i][j], _ = vectorfield(variable_array.X[i][j], variable_array.Y[i][j], field, scan)


#function used to graph the red arrow infront of robot
def create_robot_arrow(xdif, ydif, xlist, ylist, Xg, Yg):
    """create a red arrow at the current step in the animation
    Args:
        delta x
        delta y
        obj variable arrays
        vector field parameters

    Returns:
        plotted quiver
    """
    Xmin, Ymin, Xmax, Ymax = squareview(xlist[0], ylist[0], Xg, Yg)
    unitvectormag = math.sqrt(xdif**2 + ydif**2)
    if(unitvectormag == 0):
        unitvectormag = 1
    plt.quiver(xlist[-1], ylist[-1], xdif/unitvectormag, ydif/unitvectormag, color='r',
    minshaft= 0.1, minlength=0.1, headwidth=10, headlength=4, headaxislength=2, zorder = 10
    ,angles='xy', scale_units='xy', pivot='middle',scale=10, width=0.005)
    # /(Xmax-Xmin+0.0001)*50)



#function used to decide the box the graph should show
def squareview(X1, Y1, X2, Y2):
    """create a box around the animation
    Args:
        starting position
        ending position

    Returns:
        minimum and maximum x and y to the new veiw
    """
    offset = 0.2
    if X1 > X2:
        closepoint = [X2, Y2]
        farpoint = [X1, Y1]
        if Y1 < Y2:
            closepoint = [X2, Y1]
            farpoint = [X1, Y2]
    else:
        if Y2 < Y1:
            closepoint = [X1, Y2]
            farpoint = [X2, Y1]
        else:
            closepoint = [X1, Y1]
            farpoint = [X2, Y2]

    totwidth = (farpoint[0] - closepoint [0]) / (1 - 2 * offset)
    totheight = (farpoint[1] - closepoint [1]) / (1 - 2 * offset)

    Xmin = closepoint[0] - (totwidth * offset)
    Ymin = closepoint[1] - (totheight * offset)
    Xmax = farpoint[0] + (totwidth * offset)
    Ymax = farpoint[1] + (totheight * offset)
    return (Xmin, Ymin, Xmax, Ymax)


def check_for_change_in_field(state: stateClass, variable_array: variableList, dot_params: dot_path):
    """check if a threshold has been crossed
    Args:
        state
        goal plots

    Returns:
        returns if vector field needs to be changed
    """
    obstacle_crossed: bool = False
    Point2 = np.array([dot_params.dot_goals[1][0],dot_params.dot_goals[1][1]])
    Point1 = np.array([dot_params.dot_goals[0][0],dot_params.dot_goals[0][1]])
    Point_robot = np.array([state.x_pos, state.y_pos])
    q_p = Point1 - Point2
    q_r = Point_robot - Point2
    if(np.dot(q_p, q_r) < 0):
        obstacle_crossed = True
    if (len(dot_params.dot_goals) > 2 and obstacle_crossed):
        dot_params.dot_goals.pop(0)
        variable_array.changefield = True