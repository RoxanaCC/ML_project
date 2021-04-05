import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold #epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# A: Face prezenta
# B: E de la 8:00
# C: Vine Mariel
# Out: Merg la curs
# A|B|C|Out
# 0|0|0| 0
# 0|0|1| 1
# 0|1|0| 0
# 0|1|1| 0
# 1|0|0| 1
# 1|0|1| 1
# 1|1|0| 0
# 1|1|1| 1

training_inputs = []
training_inputs.append(np.array([0, 0, 0]))
training_inputs.append(np.array([0, 0, 1]))
training_inputs.append(np.array([0, 1, 0]))
training_inputs.append(np.array([0, 1, 1]))
training_inputs.append(np.array([1, 0, 1]))
training_inputs.append(np.array([1, 1, 0]))
training_inputs.append(np.array([1, 1, 1]))

labels = np.array([0, 1, 0, 0, 1, 0, 1])

perceptron = Perceptron(3)
perceptron.train(training_inputs, labels)

inputs = np.array([1, 0, 1])
print("Nu face prezenta, nu e de la 8:00 si vine Mariel => %d" % perceptron.predict(inputs))
#=> Merg

inputs = np.array([0, 1, 1])
print("Nu face prezenta, e de la 8:00 si vine Mariel => %d" % perceptron.predict(inputs))
#=> Nu merg

inputs = np.array([1, 0, 0])
print("Face prezenta, nu e de la 8:00 si nu vine Mariel => %d" % perceptron.predict(inputs))
#(Ex nou) => Merg