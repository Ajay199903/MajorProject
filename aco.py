import graph_util
import csv
import random
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


def get_unique_nodes(edgeList):
    unique_nodes = set()

    for edge in edgeList:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])

    unique_nodes = sorted(list(unique_nodes))
    return unique_nodes

def indexNodes(edgeList):

    unique_nodes = get_unique_nodes(edgeList)
    node_map = dict()

    for i in range(len(unique_nodes)):
        node_map[unique_nodes[i]] = i

    for i in range(len(edgeList)):
        edgeList[i][0] = node_map[edgeList[i][0]]
        edgeList[i][1] = node_map[edgeList[i][1]]

    return edgeList

def aco(G, GComplete, max_iterations, numAnts, alpha, beta, gamma, rho, eps):
    global mxauc, mxprecision, res1, res2

    G.initializePheromones()
    G.initializeHeuristic()
    G.updateHyperparameters(alpha, beta, gamma, rho, eps)

    print("Pheromone Matrix and Heuristic Matrix initialized!")

    time = 1

    while time <= max_iterations:

        print("Time: " + str(time))

        for k in tqdm(range(numAnts)):
            # print("Ant: ", k)

            curNode = random.randint(0, G.numNodes - 1)
            path = [curNode]

            for i in range(G.numNodes - 1):
                nextNode = G.randomNextNode(curNode)
                path.append(nextNode)
                curNode = nextNode

            G.updatePathPherom(path)

        G.updatePheromoneMat()
        time += 1
        G.resetDeltaPherom()

        precision = testing_util.precisionScore(G, GComplete)
        auc = testing_util.AUC(G, GComplete)
        print("Precision Score: " + str(precision))
        print("AUC Score: " + str(auc))


    print("ACO Completed!")

    return G.pheromoneMat

if __name__ == '__main__':
    edgeList = readEdges('/content/MajorProject/USAir.txt')
    edgeList = indexNodes(edgeList)
    edgeList = getLargestComponent(edgeList)
    edgeList = indexNodes(edgeList)
    random.shuffle(edgeList)

    edgeListTrain, edgeListTest = testing_util.trainTestSplit(edgeList, trainSetSize = 0.9)
    GComplete = graph_util.Graph(len(get_unique_nodes(edgeList)))
    G = graph_util.Graph(len(get_unique_nodes(edgeList)))

    testSetSize = len(edgeListTest)
    for i in range(testSetSize):
        edgeListTest.append([edgeListTest[i][1], edgeListTest[i][0]])

    for edge in edgeListTrain:
        G.addEdge(edge[0], edge[1])
        G.addEdge(edge[1], edge[0])

    for edge in edgeList:
        GComplete.addEdge(edge[0], edge[1])
        GComplete.addEdge(edge[1], edge[0])

    print("Graph created!")

    mxprecision, mxauc = 0, 0
    res1 = []
    res2 = []

    max_iterations = 50
    num_ants = 100

    alpha = 0.4 
    beta = 0.9
    gamma = 0.8
    rho = 0.8
    eps = 0.01
    print(alpha, beta, gamma, rho, eps)
    aco(G, GComplete, max_iterations, num_ants, alpha, beta, gamma, rho, eps)


'''
1. dont follow same path again.
2. grey wolf implementation.
3. graphs
'''
