import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import networkx as nx
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import math
from student_utils import*
from utils import*
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    home_indices = convert_locations_to_indices(list_of_homes,list_of_locations)
    start_index = convert_locations_to_indices([starting_car_location],list_of_locations)[0]
    G = adjacency_matrix_to_graph(adjacency_matrix)[0]
    carPath,dropoffs = solverTrialAndError(G,home_indices,start_index,100)
    finaldropoffs = {}
    for dLoc in dropoffs:
        if len(dropoffs[dLoc]) > 0:
            finaldropoffs[dLoc] = dropoffs[dLoc]
    for dLoc in finaldropoffs:
        finaldropoffs[dLoc] = list(set(finaldropoffs[dLoc]))
    return carPath, finaldropoffs



"""
Helper functions for the project
"""

#graph solver function
def solverDijkstraChoices(G,home_indices_master,start_index,choices):
    def pathLength(dest): return nx.dijkstra_path_length(G,currLoc,dest)
    home_indices = home_indices_master[:]
    dropoffs= {}
    currLoc = start_index
    carPath = []
    prevLoc = None
    pathToClosest = None
    returning = False
    while len(home_indices) > 0:
        #Find closest home
        closestHomes = closestNeighbors(G,currLoc,home_indices,choices)
        closest = chooseClosest(G,currLoc,closestHomes)
        intermediate = chokepoint(G,prevLoc,currLoc,closest,carPath,pathToClosest)
        if intermediate is not None:
            pathToClosest = rectifyPath(G,carPath,intermediate,currLoc,closest,dropoffs)
        else:
            pathToClosest = nx.dijkstra_path(G,currLoc, closest)
        carPath.extend(pathToClosest)
        carPath.pop()
        prevLoc = currLoc
        currLoc = closest
        if currLoc in home_indices_master:
            if currLoc in dropoffs:
                dropoffs[currLoc].append(currLoc)
            else:
                dropoffs[currLoc] = [currLoc]
        home_indices.remove(closest)
        if (len(home_indices) <= 0) and not returning:
            home_indices.append(start_index)
            returning=True
    carPath.append(start_index)
    return carPath, dropoffs

def chokepoint(G,prevLoc,currLoc,nextLoc,carPath,pathToCurr):
    if prevLoc == None or pathToCurr == None:
        return
    prevToCurr = pathToCurr
    currToNext = nx.dijkstra_path(G,currLoc, nextLoc)
    for v in prevToCurr:
        intermediate = nx.dijkstra_path(G,v, currLoc)
        if set(intermediate) <= set(currToNext) and len(intermediate) > 1:
            return intermediate[0]

        
def rectifyPath(G,carPath,intermediate,currLoc,nextLoc,dropoffs):
    intStart = intermediate
    dropoffs[currLoc].remove(currLoc)
    if intStart in dropoffs:
        dropoffs[intStart].append(currLoc)
    else:
        dropoffs[intStart] = [currLoc]
    dropoffLoc = carPath.pop()
    while dropoffLoc != intStart:
        dropoffLoc = carPath.pop()
    return nx.dijkstra_path(G,intermediate,nextLoc)
    
    

def solverTrialAndError(G,home_indices_master,start_index,iters):
    home_indices = home_indices_master[:]
    dropoffs = {}
    currLoc = start_index
    carPath = []
    minPath,minDropoffs = solverDijkstraChoices(G,home_indices,start_index,1)
    minCost = cost_of_solution(G,minPath,minDropoffs)[0]
    #Trial and error solver
    prevLoc = None;
    pathToClosest = None;
    for _ in range(iters):
        carPath,dropoffs = solverDijkstraChoices(G,home_indices,start_index,4)
        currCost = cost_of_solution(G,carPath,dropoffs)[0]
        if type(minCost) == type(currCost) and minCost > currCost:
                minPath = carPath
                minDropoffs = dropoffs
        dropoffs = {}
        carPath = []
        home_indices = home_indices_master[:]
        currLoc = start_index
        prevLoc = None;
        pathToClosest = None;
    return minPath,minDropoffs


def chooseClosest(G,currLoc,candidates):
    #make weights for each candidate
    weights = {}
    totalWeight = 0
    if len(candidates) == 1:
        return candidates[0]
    if currLoc in candidates:
        return currLoc
    for v in candidates:
        currWeight = 1/(nx.dijkstra_path_length(G,currLoc,v))
        weights[v] = currWeight
        totalWeight += currWeight
    probList = []
    for v in candidates:
        probList.append(weights[v]/totalWeight)
    return np.random.choice(candidates,p=probList)
        
           
def closestNeighbors(G,currLoc,home_indices,numNeighbors):
    #minKey Lambda
    def pathLength(dest): 
        return nx.dijkstra_path_length(G,currLoc,dest)
    home_indices_temp = home_indices[:]
    closest = []
    for _ in range(numNeighbors):
        currClosest = min(home_indices_temp,key=pathLength)
        closest.append(currClosest)
        home_indices_temp.remove(currClosest)
        if len(home_indices_temp) ==0:
            break
    return closest


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
