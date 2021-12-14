import numpy as np
from sklearn.metrics import roc_auc_score, precision_score
import heapq
import random

class Graph:

    def __init__(self, num_nodes):
        self.adjMat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        self.adjList = [[] for i in range(num_nodes)]
        self.edges = []
        self.numNodes = num_nodes
        self.positionMat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.eps = 0
        self.rho = 0

    def updateHyperparameters(self, alpha, beta, gamma, rho, eps):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho 
        self.eps = eps

    def addEdge(self, a, b, w = 1):
        self.adjList[a].append((b, w))
        self.adjMat[a][b] = 1
        self.edges.append([a, b, w])

    def tao(self, isEdge):
        return self.alpha * (isEdge + self.eps)

    def initializePositions(self):
        for i in range(self.numNodes):
            for j in range(i,self.numNodes):
                self.positionMat[i][j] = (self.eta(i,j) + random.random())/(1 + random.random())
                self.positionMat[j][i] = self.positionMat[i][j]

    def updatePosition(self, i, j, val):
        self.positionMat[i][j] = val
        self.positionMat[j][i] = val
    
    def copyPosition(self,G):
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                self.positionMat[i][j] = G.positionMat[i][j]
    
    def getCommonNeigbours(self, a, b):
        neighboursA = set(self.adjList[a])

        commonNeigbours = []

        for i in self.adjList[b]:
            if i in neighboursA:
                commonNeigbours.append(i)

        return commonNeigbours

    def eta(self, a, b):
        return self.gamma * len(self.getCommonNeigbours(a, b))

    def getProb(self, a, b):
        taoVal = self.tao(self.adjMat[a][b])
        etaVal = self.eta(a, b)

        return pow(taoVal, self.alpha) * pow(etaVal, self.beta)
    
    def getPosition(self, a, b):
        return self.positionMat[a][b]

    def getDegree(self, a):
        return len(self.adjList[a])

    def getPathFitness(self, path, C = 0.95):
        # fitness = sum([self.getDegree(i) for i in path]) / self.numNodes
        # return C * fitness
        fitness = 0
        for i in range(1,len(path)):
            fitness += len(self.getCommonNeigbours(path[i-1],path[i])) / self.numNodes
        return fitness
    
    def randomNextNode(self, a):
        probability = [0.0 for i in range(self.numNodes)]

        for i in range(self.numNodes):
            probability[i] = self.getProb(a, i)

        sm = sum(probability)

        if sm != 0:
            probability = [i/sm for i in probability]
            return np.random.choice(range(self.numNodes), p = probability)
        else:
            return np.random.choice(range(self.numNodes))

    def getPredictedLinks(self, topK):
        allPredictedLinks = []
        for i in range(self.numNodes):
            for j in range(i + 1, self.numNodes):
                if i != j and self.adjMat[i][j] == 0:
                    allPredictedLinks.append([-self.positionMat[i][j], i, j])

        allPredictedLinks.sort()

        ret = []
        for i in range(min(topK, len(allPredictedLinks))):
            ret.append([allPredictedLinks[i][1], allPredictedLinks[i][2]])
        return ret
    
    def precisionScore(self, Goriginal):

        auc = self.AUC(Goriginal)
        edgeList = self.getPredictedLinks(41)

        tp, fp = 0, 0 
        mx = 0
        mndiff = 100
        prec_scores = []

        for i in range(len(edgeList)):
            if Goriginal.adjMat[edgeList[i][0]][edgeList[i][1]]:
                tp += 1
            else:
                fp += 1
            prec = tp/(tp + fp)
            prec_scores.append(prec)

        return (tp/(tp + fp))

    def AUC(self, Goriginal):
        trueLabel = []
        score = []

        for i in range(self.numNodes):
            for j in range(self.numNodes):
                if i != j and self.adjMat[i][j] == 0:
                    trueLabel.append(Goriginal.adjMat[i][j])
                    score.append(self.positionMat[i][j])

        return roc_auc_score(trueLabel, score)

