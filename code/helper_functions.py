import numpy as np
import pandas as pd
import glob
import time
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
import ransac
from scipy.stats import multivariate_normal



    

	 
def get_transition_mat(predictors,responses, single_predictor = False):
	"""if single predictor is set to false, all of the predictors are used in predicting each response
	   if single predictor is true, the predictor at position i is used to predict the response at position i. Ignores interaction
	   between parameters.
	"""
	transition_matrix = np.zeros((responses.shape[1],predictors.shape[1]))
	transition_offset = np.zeros(responses.shape[1])
	
	if not single_predictor:
	
		#runs RANSAC to get matrix A and vector b such that C = A*data + b
		offsets                              = np.ones(predictors.shape[0])[...,np.newaxis] 
		transition_matrix, transition_offset = ransac.find_matrix(np.hstack((predictors,offsets)),responses)
	else:
		#
		for i in range(responses.shape[1]):
			ransac_regressor = linear_model.RANSACRegressor(linear_model.LinearRegression(),max_trials=10000)
			ransac_regressor.fit(predictors[:,i][...,np.newaxis],responses[:,i][...,np.newaxis])
			linear_slope = ransac_regressor.estimator_.coef_
			linear_inter = ransac_regressor.estimator_.intercept_
			transition_matrix[i,i]    = linear_slope
			transition_offset[i]      = linear_inter		
	
			
	return transition_matrix, transition_offset	
	


def get_log_transition_mat(predictors,responses, single_predictor = False):
	"""if single predictor is set to false, all of the predictors are used in predicting each response
	   if single predictor is true, the predictor at position i is used to predict the response at position i. Ignores interaction
	   between parameters. Does the same as the "get_transition_mat" except operates in the log space for the 3rd and 4th parameters
	   i.e. k and t0
	"""
	transition_matrix = np.zeros((responses.shape[1],predictors.shape[1]))
	transition_offset = np.zeros(responses.shape[1])
	
	if not single_predictor:
	
		#runs RANSAC to get matrix A and vector b such that C = A*data + b
		offsets                              = np.ones(predictors.shape[0])[...,np.newaxis] 
		transition_matrix, transition_offset = ransac.find_matrix(np.hstack((predictors,offsets)),responses)
	else:
		#
		predictors2 = predictors.copy()
		responses2  = responses.copy()
		for i in range(responses.shape[1]):
			if i in [2,3]:
				predictors2[:,i] = np.log(predictors[:,i])
				responses2[:,i]  = np.log(responses[:,i])
			ransac_regressor = linear_model.RANSACRegressor(linear_model.LinearRegression(),max_trials=10000)
			ransac_regressor.fit(predictors2[:,i][...,np.newaxis],responses2[:,i][...,np.newaxis])
			linear_slope = ransac_regressor.estimator_.coef_
			linear_inter = ransac_regressor.estimator_.intercept_
			transition_matrix[i,i]    = linear_slope
			transition_offset[i]      = linear_inter		
	
			
	return transition_matrix, transition_offset		

def create_linear_function(A,b):
	assert A.shape[0]==len(b), 'A and b size mismatch. %i and %i'%(A.shape[0],len(b))
	def temp(x):
		return np.dot(A,x) + b
	return temp	

	
def find_normalized_RMSE(prediction,gt, with_std = True):
	"finds the normalized Mean Absolute Error between the ground truth and the predictions"
	
	range = max(gt) - min(gt)
	MAE  = np.sqrt(np.mean((prediction-gt)**2))
	if with_std:
		NMAE = MAE/gt.std()
	else:
		NMAE = MAE/range
	return NMAE


		
def get_conditional_params(X1,X2):
	'''computes p(x1|x2) from p(x1,x2), when the p(x1,x2) is a gaussian. Condition on x2
	'''
	X1 = X1.reshape(X1.shape[0],-1)
	X2 = X2.reshape(X2.shape[0],-1)
	n_x1_params   = X1.shape[1]
	n_x2_params   = X2.shape[1]
	n_params      = n_x1_params + n_x2_params	
	combined_data = np.hstack((X1,X2))
	mu_x1         = X1.mean(axis=0)
	mu_x2         = X2.mean(axis=0)
	joint_cov     = np.cov(combined_data.T)
	assert joint_cov.shape == (n_params,n_params)
	sigma_11      = joint_cov[0:n_x1_params,0:n_x1_params]
	sigma_12      = joint_cov[0:n_x1_params,n_x1_params:] 
	sigma_21      = joint_cov[n_x1_params:,0:n_x1_params]
	sigma_22      = joint_cov[n_x1_params:,n_x1_params:]
	return mu_x1, mu_x2, sigma_11, sigma_12, sigma_21, sigma_22
	
def create_gauss_trans_func(X1,X2,trouble_shoot=False):	
	'''computes p(x1|x2) from p(x1,x2), when the p(x1,x2) is a gaussian. Condition on x2
		This is useful for finding the state transition and observation functions used in the particle filter.
	'''
	mu_x1,mu_x2,sigma_11,sigma_12, sigma_21, sigma_22 = get_conditional_params(X1,X2)
	mu_x1 = mu_x1.reshape(mu_x1.shape[0],-1)
	mu_x2 = mu_x2.reshape(mu_x2.shape[0],-1)
	s_bar = sigma_11 - np.dot(sigma_12,np.dot(np.linalg.inv(sigma_22),sigma_21)) #the covariance of the conditional gaussian

	def find_gauss_trans_mean(x,ts=False):

		x = x.reshape(x.shape[0],-1)
		

			
		mu_prod = np.dot(sigma_12,np.linalg.inv(sigma_22))#check if transpose
		trans_mean = mu_x1 + np.dot(mu_prod,(x-mu_x2))
		if ts:
			return mu_x1,mu_prod,mu_x2
		return trans_mean.flatten()
		
	return find_gauss_trans_mean, s_bar		
	




	

	
	
	
