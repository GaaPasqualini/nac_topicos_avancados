from keras.models import Sequential
from keras.layers import Dense
import numpy

# criacao do Model e adicao das camadas com ativacao Rectifier, que segundo o exemplo
# seguido, tem melhor performance e Sigmoid eh usada na camada de saida
def create_model(hidden_layer_neurons):
    model = Sequential()
    model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(hidden_layer_neurons, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# DataSet utilizado http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
# carregando pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# dividindo em variaveis de input (X) e output (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

# criacao do Model e adicao das camadas com ativacao Rectifier, que segundo o exemplo
# seguido, tem melhor performance e Sigmoid eh usada na camada de saida
model = create_model(8)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
