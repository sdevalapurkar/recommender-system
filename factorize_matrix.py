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


def generate_adj_list_and_rev_hash_map(int_reader_list):
  adj_list_dict = {}
  reverse_hash_map = {}

  for value in int_reader_list:
    user = value[0]
    item = value[1]
    rating = value[2]

    if (user in adj_list_dict):
      adj_list_dict[user].append([item, rating])
    else:
      adj_list_dict[user] = [[item, rating]]

    if (item in reverse_hash_map):
      reverse_hash_map[item].append([user, rating])
    else:
      reverse_hash_map[item] = [[user, rating]]

  return adj_list_dict, reverse_hash_map


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

  U, V = initialize_U_and_V()
  adj_list_dict, reverse_hash_map = generate_adj_list_and_rev_hash_map(int_reader_list)
  x = [0 for i in range(n)]

  print(U)
  print(V)

  print(8 ** 2)

  # for p in range(20):
  for k in range(0, d):
    for i in adj_list_dict:
      new_U_of_i = U[i]
      new_V = V
      new_U_of_i_minus_k = np.reshape(np.delete(new_U_of_i, k), (1, d - 1))

      x[i] = -1 * sum(map(lambda j: (np.dot(new_U_of_i_minus_k, np.reshape(np.delete(new_V[:, j[0]], k), (d - 1, 1))) - j[1]) * V[k][j[0]], adj_list_dict[i])) / sum(map(lambda j: V[k][j[0]] ** 2, adj_list_dict[i]))
      # for j in adj_list_dict[i]:
      #   new_V_of_j = V[:, j[0]]

      #   new_U_of_i_minus_k = np.delete(new_U_of_i, k)
      #   new_V_of_j_minus_k = np.delete(new_V_of_j, k)

      #   new_U_of_i_minus_k = np.reshape(new_U_of_i_minus_k, (1, d - 1))
      #   new_V_of_j_minus_k = np.reshape(new_V_of_j_minus_k, (d - 1, 1))

      #   x[i] += (-1 * (np.dot(new_U_of_i_minus_k, new_U_of_i_minus_k) - j[1]) * V[k][j[0]]) / (V[k][j[0]] ** 2)

  print(x)
