import numpy as np
import heapq

class Graph:

    def __init__(self, num_nodes):
        self.adjMat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        self.adjList = [[] for i in range(num_nodes)]
        self.edges = []
        self.numNodes = num_nodes
        self.pheromoneMat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        self.heuristicMat = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
        self.deltaPherom = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
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

    def initializePheromones(self):
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                self.pheromoneMat[i][j] = self.tao(self.adjMat[i][j])

    def getCommonNeigbours(self, a, b):
        neighboursA = set(self.adjList[a])

        commonNeigbours = []

        for i in self.adjList[b]:
            if i in neighboursA:
                commonNeigbours.append(i)

        return commonNeigbours

    def eta(self, a, b):
        return self.gamma * len(self.getCommonNeigbours(a, b))

    def initializeHeuristic(self):
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                self.heuristicMat[i][j] = self.eta(i, j)

    def getProb(self, a, b):
        taoVal = self.tao(self.adjMat[a][b])
        etaVal = self.eta(a, b)

        return pow(taoVal, self.alpha) * pow(etaVal, self.beta)

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

    def getDegree(self, a):
        return len(self.adjList[a])

    def getPathFitness(self, path, C = 0.95):
        fitness = sum([self.getDegree(i) for i in path]) / self.numNodes
        return C * fitness

    def resetDeltaPherom(self):
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                self.deltaPherom[i][j] = 0

    def updateDeltaPherom(self, a, b, val):
        self.deltaPherom[a][b] += val

    def updatePathPherom(self, path):
        fitness = self.getPathFitness(path)
        for i in range(1, len(path)):
            self.updateDeltaPherom(path[i - 1], path[i], fitness)
            self.updateDeltaPherom(path[i], path[i - 1], fitness)

    def updatePheromoneMat(self, rho = 0.8):
        for i in range(self.numNodes):
            for j in range(self.numNodes):
                newVal = (rho * self.pheromoneMat[i][j])
                newVal += self.deltaPherom[i][j]
                self.pheromoneMat[i][j] = newVal

    def getPredictedLinks(self, topK):
        allPredictedLinks = []
        for i in range(self.numNodes):
            for j in range(i + 1, self.numNodes):
                if i != j and self.adjMat[i][j] == 0:
                    allPredictedLinks.append([-self.pheromoneMat[i][j], i, j])

        allPredictedLinks.sort()

        ret = []
        for i in range(min(topK, len(allPredictedLinks))):
            ret.append([allPredictedLinks[i][1], allPredictedLinks[i][2]])
        return ret
