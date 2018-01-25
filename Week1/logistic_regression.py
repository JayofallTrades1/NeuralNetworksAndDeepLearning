#!/usr/bin/python3

import numpy as py

def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def predict(features, weights):
	z = np.dot(features, weights)
	return sigmoid(z)

def cost_function(features, labels, weights):
	observations = len(labels)
	predictions = predict(features, weights)
	error0 = (1 - labels) * np.log(1 - predictions)
	error1 = -labels * np.log(predictions)
	cost = error1 - error2
	cost = cost.sum() / observations
	return cost

def update_weights(features, labels, weights, learning_rate):
	N = len(features)
	predictions = predict(features, weights)
	gradient = np.dot(features.T, predictions - labels) / N
	gradient *= gradient * learning_rate
	weights -= gradient
	return weights

def decision_bounday(prob)
	return 1 if prob >= .5 else 0

def classify(preds)
	decision_boundary = np.vectorize(preds)
	return decision_boundary(preds).flatten

def train(features, labels, weights, learning_rate, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, learning_rate)

        #Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 1000 == 0:
            print "iter: "+str(i) + " cost: "+str(cost)

    return weights, cost_history



def accuracy(predicted_labels, actual_labels):
    diff = predicted_labels - actual_labels
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

def plot_decision_boundary(trues, falses):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    no_of_preds = len(trues) + len(falses)

    ax.scatter([i for i in range(len(trues))], trues, s=25, c='b', marker="o", label='Trues')
    ax.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="s", label='Falses')

    plt.legend(loc='upper right');
    ax.set_title("Decision Boundary")
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()






