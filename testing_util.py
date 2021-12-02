from sklearn.metrics import roc_auc_score, precision_score
import matplotlib.pyplot as plt
INF = 10**18 

def trainTestSplit(arr, trainSetSize):
    return [arr[:int(len(arr) * trainSetSize)], arr[int(len(arr) * trainSetSize):]]

def precisionScore(Gnew, Goriginal):

    auc = AUC(Gnew, Goriginal)
    edgeList = Gnew.getPredictedLinks(41)

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

def AUC(Gnew, Goriginal):
    trueLabel = []
    score = []

    for i in range(Gnew.numNodes):
        for j in range(Gnew.numNodes):
            if i != j and Gnew.adjMat[i][j] == 0:
                trueLabel.append(Goriginal.adjMat[i][j])
                score.append(Gnew.pheromoneMat[i][j])

    return roc_auc_score(trueLabel, score)

