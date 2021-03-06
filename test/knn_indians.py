import csv
import random
import math
import operator

# DataSet utilizado http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
#funcao para carregar o dataset.
def loadDataset(filename, split, trainingSet, testSet):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            #range vai de 1 a 4 para nao pegar o primeiro valor do vetor que eh uma string.
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])

#calculo da distancia euclidiana usando um tamanho determinado para nao pegar o
#valor que contem texto.
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1,length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

#pega os vizinhos mais proximos calculando a distancia euclidiana e retorna o
#numero de vizinhos de acordo com o parametro k.
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def getPredictions(trainingSet, testSet, k, predictions):
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> Previsto=' + repr(result) + ', Correto=' + repr(testSet[x][-1]))
    return predictions


def main():
	# preparando os dados
    trainingSet=[]
    testSet=[]
    split = 0.80
    loadDataset('pima-indians-diabetes.csv', split, trainingSet, testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))

    # gerando as predicoes para  k=1, 5 e 9.
    predictions_k_one=[]
    predictions_k_five=[]
    predictions_k_nine=[]

    k = 1
    getPredictions(trainingSet, testSet, k, predictions_k_one)
    k_one_accuracy = getAccuracy(testSet, predictions_k_one)

    k = 5
    getPredictions(trainingSet, testSet, k, predictions_k_five)
    k_five_accuracy = getAccuracy(testSet, predictions_k_five)

    k = 9
    getPredictions(trainingSet, testSet, k, predictions_k_nine)
    k_nine_accuracy = getAccuracy(testSet, predictions_k_nine)

    #mostrando a precisao.
    #precisao tem apresentado resultados entre 60% e 70%
    print 'Precisao para k = 1: ' + repr(k_one_accuracy) + '%'
    #precisao tem apresentado resultados entre 70% e 75%
    print 'Precisao para k = 5: ' + repr(k_five_accuracy) + '%'
    #precisao tem apresentado resultados entre 75% e 80%
    print 'Precisao para k = 9: ' + repr(k_nine_accuracy) + '%'

main()
