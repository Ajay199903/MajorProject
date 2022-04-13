import math
import random
import numpy as np
import copy
import csv
import firefly_graph_util
import testing_util
from tqdm import tqdm

def readEdges(fileName):
    edges = []

    with open(fileName, 'r') as file:

        if fileName[-1] == 'v':
            csvFile = csv.reader(file)

            for line in csvFile:
                vals = line.split()
                u, v = int(vals[0]), int(vals[1])
                edges.append([u, v])

        else:
            txtFile = file.readlines()
            for line in txtFile:
                vals = line.split()
                u, v = int(vals[0]), int(vals[1])
                edges.append([u, v])

    return edges

def getLargestComponent(edgeList):

    def dfs(node):
        vis[node] = clr
        for i in adj[node]:
            if vis[i] == 0:
                dfs(i)

    mxcomp = []
    mxcompsize = 0
    mxcompclr = 0
    N = len(get_unique_nodes(edgeList))
    vis = [0 for i in range(N)]
    clr = 1
    adj = [[] for i in range(N)]

    for i in edgeList:
        adj[i[0]].append(i[1])
        adj[i[1]].append(i[0])

    for i in range(N):
        if vis[i] == 0:
            dfs(i)
            clr += 1

    freq = {}
    for i in vis:
        if i in freq: freq[i] += 1
        else: freq[i] = 1

    mxcompsize = max(freq.values())
    for i in freq:
        if freq[i] == mxcompsize:
            mxcompclr = i

    ret = []

    for i in edgeList:
        if vis[i[0]] == mxcompclr and vis[i[1]] == mxcompclr:
            ret.append(i)

    return ret

def indexNodes(edgeList):

    unique_nodes = get_unique_nodes(edgeList)
    node_map = dict()

    for i in range(len(unique_nodes)):
        node_map[unique_nodes[i]] = i

    for i in range(len(edgeList)):
        edgeList[i][0] = node_map[edgeList[i][0]]
        edgeList[i][1] = node_map[edgeList[i][1]]

    return edgeList

def get_unique_nodes(edgeList):
    unique_nodes = set()

    for edge in edgeList:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])

    unique_nodes = sorted(list(unique_nodes))
    return unique_nodes

def Firefly(edgeListTrain,edgeList,num_firefly, max_iterations, alpha, beta ,gamma, rho, eps, f_alpha, f_beta, f_gamma):
    
    GComplete = firefly_graph_util.Graph(len(get_unique_nodes(edgeList)))

    for edge in edgeList:
        GComplete.addEdge(edge[0], edge[1])
        GComplete.addEdge(edge[1], edge[0])
    
    print("Total Nodes", GComplete.numNodes)
    print("Total Edges", len(GComplete.edges))

    print("Graph created!")
    
    G_firefly = firefly_graph_util.Graph(len(get_unique_nodes(edgeList)))
    G_firefly.updateHyperparameters(alpha, beta, gamma, rho, eps, f_alpha, f_beta, f_gamma)
    
    best_firefly = firefly_graph_util.Graph(len(get_unique_nodes(edgeList)))
    best_firefly.updateHyperparameters(alpha, beta, gamma, rho, eps, f_alpha, f_beta, f_gamma)
    best_firefly_score = 0
    
    for edge in edgeListTrain:
        G_firefly.addEdge(edge[0], edge[1])
        G_firefly.addEdge(edge[1], edge[0])
        best_firefly.addEdge(edge[0], edge[1])
        best_firefly.addEdge(edge[1], edge[0])
    
    G_firefly.initializePositions()
            
    time = 1
    print("Firefly Optimization is starting!")
    for itr in range(max_iterations):
        
        print("Time: " + str(time))
        
        for i in tqdm(range(num_firefly)):
            
            curNode = random.randint(0, G_firefly.numNodes - 1)
            path = [curNode]

            for j in range(num_firefly - 1):
                nextNode = G_firefly.randomNextNode(curNode)
                path.append(nextNode)
                if G_firefly.getIntensity(curNode, curNode) < G_firefly.getIntensity(nextNode, nextNode):
                    G_firefly.updatePosition(curNode, nextNode)
                curNode = nextNode
            
            firefly_score = G_firefly.getPathFitness(path)
            if(firefly_score > best_firefly_score):
                best_firefly.copyPosition(G_firefly)
                best_firefly_score = firefly_score   
        time += 1
        precision = best_firefly.precisionScore(GComplete)
        auc = best_firefly.AUC(GComplete)
        print("Precision Score: " + str(precision))
        print("AUC Score: " + str(auc))
    
    print("Firefly Optimization Completed!")
    
    return best_firefly.positionMat


if __name__ == '__main__':
    # edgeList = readEdges('/content/MajorProject/USAir.txt')
    edgeList = readEdges('karate.txt')
    edgeList = indexNodes(edgeList)
    edgeList = getLargestComponent(edgeList)
    edgeList = indexNodes(edgeList)
    random.shuffle(edgeList)

    edgeListTrain, edgeListTest = testing_util.trainTestSplit(edgeList, trainSetSize = 0.9)

    testSetSize = len(edgeListTest)
    for i in range(testSetSize):
        edgeListTest.append([edgeListTest[i][1], edgeListTest[i][0]])

    max_iterations = 50
    
    f_alpha = 0.2
    f_beta = 1
    f_gamma = 1
    alpha = 0.4 
    beta = 0.9
    gamma = 0.8
    rho = 0.8
    eps = 0.01
    print(alpha, beta, gamma, rho, eps)
    
    num_firefly = 50
    
    Firefly(edgeListTrain,edgeList,num_firefly, max_iterations, alpha, beta ,gamma, rho, eps, f_alpha, f_beta, f_gamma)
