# Train model and make predictions
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("balance-scale.csv", header=None)
dataset = dataframe.values
X = dataset[:,1:5].astype(float)
Y = dataset[:,0]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.20, random_state=seed)
estimator.fit(X_train, Y_train)
predictions = estimator.predict(X_test)
# model = baseline_model()
# loss_and_metrics = model.evaluate(X_test, Y_test)

# print loss_and_metrics
score = estimator.score(X_test, Y_test)
# print dummy_y
print "Taxa de acerto: %.2f%%" % (score * 100)
# print predictions
print 'Predicao: ' + encoder.inverse_transform(predictions)
print Y_test
# print encoder.inverse_transform(Y_test[1])

# print Y_test

# results = cross_val_score(estimator, X, dummy_y, cv=kfold)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
