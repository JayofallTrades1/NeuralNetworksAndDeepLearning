#!/bin/usr/python3
import numpy as np

"""
Simple Linear Regression model y = mx + b with multiple parameters x1, x2, x3,..xn
We are predicting the sales of multiple companies given their radio advertisement and 
sales as weights.
"""

def predict(features, weights):
	"""
	features = number of different companies
	weights = variables 
	"""
	return np.dot(features, weights)

weights = np.array([0.0],[0.0],[0.0])

def cost_function(features, targets, weight):
	"""
	targets = y-value (W1*X1 + W2*X2 + W3*X3)
	"""
	N = len(targets)
	predictions = predict(features, weights)
	squared_err = (predictions - targets)**2
	return 1.0/(2N)*sq_error.sum

def update_weights(features, targets, weights, learning_rate):
	"""
	Extracting features. In this case we have 3 total x1,x2,x3
	gradient = features.T *(predictions - targets) / N
	"""
	predictions = predict(features, weights)

	"""
	calculate error/loss
	"""
	error = targets - predictions
	gradient = np.dot(-X.T, error)
	gradient /= companies
	gradient *= learning_rate
	weights += gradient
	return weights

bias = np.ones(shape=len(features),1))
features = np.append(bias, features, axis=1)


	
