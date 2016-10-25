from keras.models import Sequential
from keras.layers import Dense
import pandas
import numpy
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

# Integrantes:
# Gabriel Pasqualini RM:67623
# Diego Cardi RM: 64644
# Jaime Junior RM: 67313
# Andre Bassalo RM: 67264
# Pedro Garcia RM: 67034

# Documentacao da library utilizada: https://keras.io/
# Links uteis: https://keras.io/models/model/
# https://keras.io/#getting-started-30-seconds-to-keras
# https://keras.io/models/sequential/#sequential-model-methods
# https://github.com/fchollet/keras
# http://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#
# Tutoriais em: http://machinelearningmastery.com/

# criacao do Model e adicao das camadas com ativacao Rectifier, que segundo o exemplo
# seguido, tem melhor performance e Sigmoid eh usada na camada de saida
def create_model(hidden_layer_neurons):
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='uniform', activation='relu'))
    model.add(Dense(hidden_layer_neurons, init='uniform', activation='relu'))
    model.add(Dense(3, init='uniform', activation='sigmoid'))
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# DataSet utilizado http://archive.ics.uci.edu/ml/datasets/Balance+Scale
# carregando balance scale dataset e dividindo a classe (Y) dos atributos (X)
dataframe = pandas.read_csv("balance-scale.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:5].astype(float)
Y = dataset[:,0]

#trecho de codigo para converter os strings das classes para dummy variables
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

#criacao dos modelos com numeros de neuronios diferentes na camada oculta
model_one_time   = create_model(4)
model_two_time   = create_model(8)
model_three_time = create_model(16)

# Compila o modelo usando uma funcao de perda indicada na internet e a metrica como a precisao
model_one_time.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_two_time.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_three_time.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.20, random_state=seed)

# Faz o fit ou treinamento dos modelos de rede neural
print "Treinando com 4 neuronios"
model_one_time.fit(X_train, Y_train, validation_data=(X_test,Y_test), nb_epoch=150, batch_size=10)
print "Treinando com 8 neuronios"
model_two_time.fit(X_train, Y_train, validation_data=(X_test,Y_test), nb_epoch=150, batch_size=10)
print "Treinando com 16 neuronios"
model_three_time.fit(X_train, Y_train, validation_data=(X_test,Y_test), nb_epoch=150, batch_size=10)

# Avalia o modelo com os dados de teste
scores = model_one_time.evaluate(X_test, Y_test)
print "\n"
# Media de precisao tem sido de 90% e perda de 20% no treinamento
print "Precisao Media 4 neuronios: %.2f%% and Perda Media: %.2f%% " % (scores[1]*100, scores[0]*100)

# Avalia o modelo com os dados de teste
scores = model_two_time.evaluate(X_test, Y_test)
print "\n"
# Media de precisao tem sido de 92% e perda de 16% no treinamento
print "Precisao Media 8 neuronios: %.2f%% and Perda Media: %.2f%% " % (scores[1]*100, scores[0]*100)

# Avalia o modelo com os dados de teste
scores = model_three_time.evaluate(X_test, Y_test)
print "\n"
# Media de precisao tem sido de 93~94% e perda de 14% no treinamento
print "Precisao Media 16 neuronios: %.2f%% and Perda Media: %.2f%% " % (scores[1]*100, scores[0]*100)


# Exemplo da criacao de predicoes de classes e como imprimilas
# classes = model_one_time.predict_classes(X_test)
# print encoder.inverse_transform(classes)

# print "%s: %.2f%% and %s: %.2f%% " % (model.metrics_names[1], scores[1]*100, model.metrics_names[0], scores[0]*100)
