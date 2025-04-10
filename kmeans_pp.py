import mykmeanssp as KM
import numpy as np
import pandas as pd
import math
import sys


class Point:
    def __init__(self, coords, dim, cluster, index):
        self.coords = coords
        self.dim = dim
        self.cluster = cluster
        self.index = index

def main():
    np.random.seed(1234)
    DEF_MAX_ITER = 300

    argc = len(sys.argv)
    
    def is_valid_integer(value):
        try:
            float_value = float(value)
            return float_value.is_integer() and 1 < float_value
        except ValueError:
            return False
    
    if(argc == 5):  # Iter not given
        try:
            eps = float(sys.argv[2])
        except ValueError: 
            print("Invalid epsilon!")
            exit(1)
        if eps < 0:
            print("Invalid epsilon!")
            exit(1)
        iter = DEF_MAX_ITER
        f1 = pd.read_csv(sys.argv[3], header=None)
        f2 = pd.read_csv(sys.argv[4], header=None)
    
    elif(argc == 6):  # Iter given
        if not (is_valid_integer(sys.argv[2]) and float(sys.argv[2]) < 1000):
            print("Invalid maximum iteration!")
            exit(1)
        iter = int(float(sys.argv[2]))  # Convert to integer
        try:
            eps = float(sys.argv[3])
        except ValueError: 
            print("Invalid epsilon!")
            exit(1)
        if eps < 0:
            print("Invalid epsilon!")
            exit(1)
        f1 = pd.read_csv(sys.argv[4], header=None)
        f2 = pd.read_csv(sys.argv[5], header=None)
    
    else:
        print("An Error Has Occurred")
        exit(1)

    file = pd.merge(f1, f2, on=f1.columns[0], how="inner", sort=True)  # Also sorts
    N, d = file.shape  # Set N and d
    d -= 1  # Ignore keys

    if not (is_valid_integer(sys.argv[1]) and float(sys.argv[1]) < N):
        print("Invalid number of clusters!")
        exit(1)
    K = int(float(sys.argv[1]))  # Convert to integer

    # Convert dataframe to matrix of values
    file = list(file.to_numpy())
    for i in range(len(file)):
        file[i] = list(map(float, file[i]))

    # Initialize data to our data in file
    data = [Point([0] * d, d, -1, -1)] * N  # Initialize data array to default values
    for i in range(len(file)):
        data[i] = Point(file[i][1:], d, -1, file[i][0])

    cents = INIT_CENTS(data, d, K)  # Initialize centroids using Kmeans++ algorithm
    
    # Print calculated centroids (Their indices)
    for i in range(len(cents)):
        print(int(cents[i].index), end='')
        if i == len(cents) - 1:
            break
        print(",", end='')
    print()
    
    # Perform K-Means clustering using my module
    mat = KM.fit(K, N, d, iter, eps, data, cents)
    
    # Print output from clustering (Output from fit() is given in a matrix where each row is a centroid)
    for row in mat:
        print(','.join(f'{round(num, 4):.4f}' for num in row))

# Initializes centroids from datapoints, returns centroids
def INIT_CENTS(dp, d, k):
    cents = []
    prob = []

    cents.append(np.random.choice(dp))  # First centroid is chosen uniformly from datapoints
    
    for i in range(1, k):
        prob = computeNewProb(cents, dp, d)  # Update distribution
        cents.append(np.random.choice(dp, p=prob))  # Draw a new centroid according to computed distribution

    return cents

# Computes weighted probability distribution among points - according to distance from other centroids
def computeNewProb(cents, dp, dim):
    DistArr = [0] * len(dp)
    Distsum = 0
    prob = [0] * len(dp)

    for i in range(len(dp)):
        _, DistArr[i] = FindClosestCentroid(dp[i], cents, dim)  # Compute distance from closest centroid
        Distsum += DistArr[i]
    
    for i in range(len(dp)):
        prob[i] = float(DistArr[i]) / float(Distsum)

    return prob

def FindClosestCentroid(x, centroids, dim):
    assigned = 0
    minDist = dist(x, centroids[0], dim)
    for i in range(len(centroids)):
        curDist = dist(x, centroids[i], dim)
        if curDist < minDist:
            minDist = curDist
            assigned = i
    
    return (assigned, minDist)

# Compute distance between two points
def dist(x, y, dim):
    dist = 0
    for i in range(dim):
        dist += pow(x.coords[i] - y.coords[i], 2)

    dist = math.sqrt(dist)
    return dist

if __name__ == '__main__':
    main()
