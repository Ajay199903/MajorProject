import testing_util
import networkx as nx
import sklearn
from sklearn.metrics import roc_auc_score, precision_score
import random

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

if __name__ == '__main__':
	edgeList = readEdges('USAir.txt')
	edgeList = indexNodes(edgeList)
	edgeList = getLargestComponent(edgeList)
	edgeList = indexNodes(edgeList)
	random.shuffle(edgeList)

	edgeListTrain, edgeListTest = testing_util.trainTestSplit(edgeList, trainSetSize = 0.9)

	G = nx.Graph()
	G.add_edges_from(edgeListTrain)

	# Jaccard Coefficient
	jaccard_predict = list(nx.jaccard_coefficient(G))
	jaccard_predict.sort(key = lambda x : -x[2])
	y_true = []
	y_score = []
	y_predict = [1 for i in range(len(edgeListTest))]
	for i in jaccard_predict:
		if [i[0], i[1]] in edgeListTest or [i[1], i[0]] in edgeListTest:
			y_true.append(1)
		else:
			y_true.append(0)
		y_score.append(i[2])

	y_true2 = y_true[:len(edgeListTest)]

	print("Jaccard Coefficient")
	print(roc_auc_score(y_true, y_score))
	print(precision_score(y_true2, y_predict))
	print("-------------------------------")

	# Resource Allocation Index
	rai_predict = list(nx.resource_allocation_index(G))
	rai_predict.sort(key = lambda x : -x[2])
	y_true = []
	y_score = []
	y_predict = [1 for i in range(len(edgeListTest))]
	for i in rai_predict:
		if [i[0], i[1]] in edgeListTest or [i[1], i[0]] in edgeListTest:
			y_true.append(1)
		else:
			y_true.append(0)
		y_score.append(i[2])

	y_true2 = y_true[:len(edgeListTest)]

	print("Resource Allocation Index")
	print(roc_auc_score(y_true, y_score))
	print(precision_score(y_true2, y_predict))
	print("-------------------------------")

	# Adamic Adar Index
	adamic_predict = list(nx.adamic_adar_index(G))
	adamic_predict.sort(key = lambda x : -x[2])
	y_true = []
	y_score = []
	y_predict = [1 for i in range(len(edgeListTest))]
	for i in adamic_predict:
		if [i[0], i[1]] in edgeListTest or [i[1], i[0]] in edgeListTest:
			y_true.append(1)
		else:
			y_true.append(0)
		y_score.append(i[2])

	y_true2 = y_true[:len(edgeListTest)]

	print("Adamic Adar Index")
	print(roc_auc_score(y_true, y_score))
	print(precision_score(y_true2, y_predict))
	print("-------------------------------")

	# Preferential Attachment
	pref_predict = list(nx.preferential_attachment(G))
	pref_predict.sort(key = lambda x : -x[2])
	y_true = []
	y_score = []
	y_predict = [1 for i in range(len(edgeListTest))]
	for i in pref_predict:
		if [i[0], i[1]] in edgeListTest or [i[1], i[0]] in edgeListTest:
			y_true.append(1)
		else:
			y_true.append(0)
		y_score.append(i[2])

	y_true2 = y_true[:len(edgeListTest)]

	print("Preferential Attachment Index")
	print(roc_auc_score(y_true, y_score))
	print(precision_score(y_true2, y_predict))
	print("-------------------------------")