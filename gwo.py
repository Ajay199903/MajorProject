import math
import random
import numpy as np
import copy
import gwo_graph_util
from tqdm import tqdm

# Original Gray Wolf Optimizer Alogrithm Implemention

def get_unique_nodes(edgeList):
    unique_nodes = set()

    for edge in edgeList:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])

    unique_nodes = sorted(list(unique_nodes))
    return unique_nodes

def GrayWolfOptimizer(edgeListTrain,edgeList,num_wolf, max_iterations, alpha, beta ,gamma, rho, eps):
    
    GComplete = gwo_graph_util.Graph(len(get_unique_nodes(edgeList)))
    A_pos = gwo_graph_util.Graph(len(get_unique_nodes(edgeList)))
    B_pos = gwo_graph_util.Graph(len(get_unique_nodes(edgeList)))
    D_pos = gwo_graph_util.Graph(len(get_unique_nodes(edgeList)))
    
    frequency = {}
    
    for edge in edgeListTrain:
        
        A_pos.addEdge(edge[0], edge[1])
        A_pos.addEdge(edge[1], edge[0])
        
        B_pos.addEdge(edge[0], edge[1])
        B_pos.addEdge(edge[1], edge[0])
        
        D_pos.addEdge(edge[0], edge[1])
        D_pos.addEdge(edge[1], edge[0])

    for edge in edgeList:
        GComplete.addEdge(edge[0], edge[1])
        GComplete.addEdge(edge[1], edge[0])
    
    O_pos = []
    
    for i in range(num_wolf):
        O_pos.append(gwo_graph_util.Graph(len(get_unique_nodes(edgeList))))
        
    for edge in edgeListTrain:
        for i in range(num_wolf):
            O_pos[i].addEdge(edge[0], edge[1])
            O_pos[i].addEdge(edge[1], edge[0])

    A_pos.initializePositions()
    A_pos.updateHyperparameters(alpha, beta, gamma, rho, eps)

    B_pos.initializePositions()
    B_pos.updateHyperparameters(alpha, beta, gamma, rho, eps)

    D_pos.initializePositions()
    D_pos.updateHyperparameters(alpha, beta, gamma, rho, eps)

    A_cost = - (A_pos.AUC(GComplete) + A_pos.precisionScore(GComplete))
    
    B_cost = - (B_pos.AUC(GComplete) + B_pos.precisionScore(GComplete))
    
    D_cost = - (D_pos.AUC(GComplete) + D_pos.precisionScore(GComplete))

    for i in range(num_wolf):
        O_pos[i].initializePositions()
        O_pos[i].updateHyperparameters(alpha, beta, gamma, rho, eps)

    time = 1
    print("GWO is starting!")
    for itr in range(max_iterations):
        
        print("Time: " + str(time))
        
        a = 2 * (1 - itr / max_iterations)
        
        for i in tqdm(range(num_wolf)):
            
            curNode = random.randint(0, O_pos[i].numNodes - 1)
            path = [curNode]

            for j in range(O_pos[i].numNodes - 1):
                nextNode = O_pos[i].randomNextNode(curNode)
                path.append(nextNode)
                curNode = nextNode
            
            O_cost = - (O_pos[i].AUC(GComplete) + O_pos[i].precisionScore(GComplete))
            # print(O_pos[i].adjMat)
            
            if O_cost < A_cost:
                D_pos.copyPosition(B_pos)
                D_cost = B_cost
                B_pos.copyPosition(A_pos)
                B_cost = A_cost
                A_pos.copyPosition(O_pos[i])
                A_cost = O_cost

            elif O_cost < B_cost:
                D_pos.copyPosition(B_pos)
                D_cost = B_cost
                B_pos.copyPosition(O_pos[i])
                B_cost = O_cost
               
            elif O_cost < D_cost:
                D_pos.copyPosition(O_pos[i])
                D_cost = O_cost

            fitness = A_pos.getPathFitness(path)

            for j in range(1, len(path)):
                
                x = path[j-1]
                y = path[j]
                if (x,y) in frequency:
                    frequency[(x,y)] += 1
                    frequency[(y,x)] += 1
                else:
                    frequency[(x,y)] = 1
                    frequency[(y,x)] = 1
                r1 = random.random()
                r2 = random.random()
                
                A_alpha = a*(2*r1 - 1)
                C_alpha = 2*r2
                
                D_alpha = abs(C_alpha*A_pos.getPosition(x,y) - O_pos[i].getPosition(x,y))
                X1 = abs(A_pos.getPosition(x,y) - A_alpha*D_alpha + fitness*frequency[(x,y)])
                
                r1 = random.random()
                r2 = random.random()
                
                A_beta = a*(2*r1 - 1)
                C_beta = 2*r2
                
                D_beta = abs(C_beta*B_pos.getPosition(x,y) - O_pos[i].getPosition(x,y))
                X2 = abs(B_pos.getPosition(x,y) - A_beta*D_beta + fitness*frequency[(x,y)])
                
                r1 = random.random()
                r2 = random.random()
                
                A_delta = a*(2*r1 - 1)
                C_delta = 2*r2
                
                D_delta = abs(C_delta*D_pos.getPosition(x,y) - O_pos[i].getPosition(x,y))
                X3 = abs(D_pos.getPosition(x,y) - A_delta*D_delta + fitness*frequency[(x,y)])
                
                O_pos[i].updatePosition(x, y, (X1 + X2 + X3)/3)
        
        time += 1
        precision = A_pos.precisionScore(GComplete)
        auc = A_pos.AUC(GComplete)
        print("Precision Score: " + str(precision))
        print("AUC Score: " + str(auc))
    
    print("GWO Completed!")
    
    return A_pos.positionMat
