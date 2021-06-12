#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import scipy.signal

def matrix_from_csv_file(file_path):
	csv_data = np.genfromtxt(file_path, delimiter = ',')
	full_matrix = csv_data[1:,]
	return full_matrix

def get_time_slice(full_matrix, start = 0., period = 1.):
	rstart  = full_matrix[0, 0] + start
	index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
	index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
	
	duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
	return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):
	ret = np.mean(matrix, axis = 0).flatten()
	names = ['mean_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_mean_d(h1, h2):
	ret = (feature_mean(h2)[0] - feature_mean(h1)[0]).flatten()
	
	names = ['mean_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names



def feature_mean_q(q1, q2, q3, q4):
	v1 = feature_mean(q1)[0]
	v2 = feature_mean(q2)[0]
	v3 = feature_mean(q3)[0]
	v4 = feature_mean(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	names = []
	for i in range(4):
		names.extend(['mean_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): 
		for j in range((i + 1), 4):
			names.extend(['mean_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
	# print(ret,names)
	return ret, names



def feature_max(matrix):
	ret = np.max(matrix, axis = 0).flatten()
	names = ['max_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_max_d(h1, h2):
	ret = (feature_max(h2)[0] - feature_max(h1)[0]).flatten()
	names = ['max_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_max_q(q1, q2, q3, q4):
	v1 = feature_max(q1)[0]
	v2 = feature_max(q2)[0]
	v3 = feature_max(q3)[0]
	v4 = feature_max(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	names = []
	for i in range(4):
		names.extend(['max_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3):
		for j in range((i + 1), 4):
			names.extend(['max_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
	return ret, names


def feature_min(matrix):
	ret = np.min(matrix, axis = 0).flatten()
	names = ['min_' + str(i) for i in range(matrix.shape[1])]
	return ret, names



def feature_min_d(h1, h2):
	ret = (feature_min(h2)[0] - feature_min(h1)[0]).flatten()
	
	# Fixed naming [fcampelo]
	names = ['min_d_h2h1_' + str(i) for i in range(h1.shape[1])]
	return ret, names


def feature_min_q(q1, q2, q3, q4):
	v1 = feature_min(q1)[0]
	v2 = feature_min(q2)[0]
	v3 = feature_min(q3)[0]
	v4 = feature_min(q4)[0]
	ret = np.hstack([v1, v2, v3, v4, 
				     v1 - v2, v1 - v3, v1 - v4, 
					 v2 - v3, v2 - v4, v3 - v4]).flatten()
	
	
	# Fixed naming [fcampelo]
	names = []
	for i in range(4): # for all quarter-windows
		names.extend(['min_q' + str(i + 1) + "_" + str(j) for j in range(len(v1))])
	
	for i in range(3): # for quarter-windows 1-3
		for j in range((i + 1), 4): # and quarter-windows (i+1)-4
			names.extend(['min_d_q' + str(i + 1) + 'q' + str(j + 1) + "_" + str(k) for k in range(len(v1))])
			 
	return ret, names


def calc_feature_vector(matrix, state):
	h1, h2 = np.split(matrix, [ int(matrix.shape[0] / 2) ])
	q1, q2, q3, q4 = np.split(matrix, 
						      [int(0.25 * matrix.shape[0]), 
							   int(0.50 * matrix.shape[0]), 
							   int(0.75 * matrix.shape[0])])

	var_names = []	
	
	x, v = feature_mean(matrix)
	var_names += v
	var_values = x
	
	x, v = feature_mean_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_mean_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_max(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_max_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_max_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])
	
	x, v = feature_min(matrix)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_min_d(h1, h2)
	var_names += v
	var_values = np.hstack([var_values, x])

	x, v = feature_min_q(q1, q2, q3, q4)
	var_names += v
	var_values = np.hstack([var_values, x])


	
	if state != None:
		var_values = np.hstack([var_values, np.array([state])])
		var_names += ['Label']
	return var_values, var_names

def generate_feature_vectors_from_samples(file_path, nsamples, period, 
										  state = None, 
										  remove_redundant = False,
										  cols_to_ignore = None):
	matrix = matrix_from_csv_file(file_path)
	t = 0.
	previous_vector = None
	ret = None

	while True:
		try:
			s, dur = get_time_slice(matrix, start = t, period = period)
			if cols_to_ignore is not None:
				s = np.delete(s, cols_to_ignore, axis = 1)
		except IndexError as e:
			print (e)
			break
		if len(s) == 0:
			break
		if dur < 0.9 * period:
			break
		ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
								 t = s[:, 0], axis = 0)

		t += 0.5 * period
		r, headers = calc_feature_vector(ry, state)
		
		if previous_vector is not None:
			feature_vector = np.hstack([previous_vector, r])

			if ret is None:
				ret = feature_vector
			else:
				ret = np.vstack([ret, feature_vector])

		previous_vector = r
		if state is not None:
			previous_vector = previous_vector[:-1] 

	feat_names = ["lag1_" + s for s in headers[:-1]] + headers
	
	return ret, feat_names

def generate_feature_vectors_from_data(matrix, nsamples, period, 
										  state = None, 
										  cols_to_ignore = None):
	t = 0.
	previous_vector = None
	ret = None

	while True:
		try:
			s, dur = get_time_slice(matrix, start = t, period = period)
			if cols_to_ignore is not None:
				s = np.delete(s, cols_to_ignore, axis = 1)
		except IndexError as e:
			print (e)
			break
		if len(s) == 0:
			break
		if dur < 0.9 * period:
			break
		ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, 
								 t = s[:, 0], axis = 0)

		t += 0.5 * period
		r, headers = calc_feature_vector(ry, state)
		
		if previous_vector is not None:
			feature_vector = np.hstack([previous_vector, r])

			if ret is None:
				ret = feature_vector
			else:
				ret = np.vstack([ret, feature_vector])

		previous_vector = r
		if state is not None:
			previous_vector = previous_vector[:-1] 

	feat_names = ["lag1_" + s for s in headers[:-1]] + headers
	
	return ret, feat_names