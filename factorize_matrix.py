import numpy as np
import csv
import argparse

def constructArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, default=None,
    help='input path of data file')
  parser.add_argument('-o', '--output', type=str, default=None,
    help='output path of data file')
  args = vars(parser.parse_args())

  return args


def initialize_U_and_V():
  U = np.zeros((n, d))
  V = np.zeros((d, m))

  for i in range(n):
    for j in range(d):
      U[i][j] = np.random.random_sample()

  for i in range(d):
    for j in range(m):
      V[i][j] = np.random.random_sample()

  return U, V


args = constructArguments()

with open(args['input'], 'r') as inf:
  reader = csv.reader(inf, delimiter='\t')
  int_reader_list = [[int(j) for j in i] for i in list(reader)]

  # n = num_users
  # m = num_movies
  d = 2
  n = 3
  m = 3
  x = []

  U, V = initialize_U_and_V()

  # for p in range(20):
  #   for k in range(1, d + 1):
  #     for i in range(1, n + 1):
  #       x[i] = 
