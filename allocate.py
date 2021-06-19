from os import minor
import numpy as np
from numpy import unravel_index
from simulator import Simulator

def Manhattan_distance(pos1, pos2):
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1]) + abs(pos2[0]-pos2[2]) + abs(pos2[1]-pos2[3])

def get_allocate_matrix(robots, targets):
    distance_matrix = np.zeros((len(robots), len(targets)))
    for robot in robots:
        for target in targets:
            distance_matrix[robot][target] = Manhattan_distance(robots[robot], targets[target])
    pairs = find_pairs(distance_matrix)
    return pairs

def find_pairs(matrix):
    pairs = []
    while len(pairs) < len(matrix):
        if len(matrix) == 1:
            return [(0,0)]
        if (matrix == 0).all():
            miss_line = np.array([True for i in range(matrix.shape[0])])
            miss_row =  np.array([True for i in range(matrix.shape[0])])
            for i in range(len(pairs)):
                miss_line[pairs[i][0]] = False
                miss_row[pairs[i][1]] = False
            pairs.append((miss_line.argmax(), miss_row.argmax()))
        matrix = delete_longest(matrix)
        pairs_new, matrix = check(matrix)
        pairs.extend(pairs_new)
        print(pairs)
    return pairs

def delete_longest(matrix):
    index = unravel_index(matrix.argmax(), matrix.shape)
    matrix[index[0],index[1]] = 0
    return matrix

def check(matrix):
    pairs = []
    # checkline
    for i in range(len(matrix)):
        if sum(matrix[i]==0) == len(matrix[i])-1:
            row = matrix[i].argmax()
            pairs.append((i, row))
            matrix[i,row] = 0
            for line in matrix:
                line[row] = 0
    # checkrow
    for i in range(len(matrix[0])):
        if sum(matrix[:,i]==0) == len(matrix)-1:
            line = matrix[:,i].argmax()
            pairs.append((line, i))
            matrix[line, i] = 0
            for k in range(len(matrix[line])):
                matrix[line, k] = 0

    return pairs, matrix


if __name__ == "__main__":
    env = Simulator((601,601,3), 10)
    robots, targets = env.information()
    print(get_allocate_matrix(robots, targets)) # first robot num, second box num
    