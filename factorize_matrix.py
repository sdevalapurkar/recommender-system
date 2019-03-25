from collections import defaultdict
import numpy as np
import argparse
import time
import csv
import sys

def constructArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, default=None,
    help='input path of data file')
  parser.add_argument('-o', '--output', type=str, default=None,
    help='output path of data file')
  args = vars(parser.parse_args())

  return args


def generate_user_and_toy_dict(int_reader_list):
  user_ratings_dict = {}
  toy_ratings_dict = {}

  for value in int_reader_list:
    user = value[0]
    item = value[1]
    rating = value[2]

    if (user in user_ratings_dict):
      user_ratings_dict[user].append([item, rating])
    else:
      user_ratings_dict[user] = [[item, rating]]

    if (item in toy_ratings_dict):
      toy_ratings_dict[item].append([user, rating])
    else:
      toy_ratings_dict[item] = [[user, rating]]

  return user_ratings_dict, toy_ratings_dict


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


def matrix_factorization(M, U, V, T, d, non_zero_entries, alpha=0.01, beta=0.02):
    V = V.T

    # M_transposed = M.T
    #
    # # print('M')
    # # print(M)
    # # print('U')
    # # print(U)
    # # print('V')
    # # print(V)
    #
    # x = [0 for i in range(n)]
    # y = [0 for i in range(m)]
    #
    # # for iter in range(T):
    # for k in range(d):
    #     for i in range(n):
    #         numerator_value = 0
    #         denominator_value = 0
    #
    #         for j in range(m):
    #             if (M[i][j] > 0):
    #                 new_U_of_i_minus_k = np.delete(U[i,:], k)
    #                 new_V_of_j_minus_k = np.delete(V[:,j], k)
    #                 # print(k, i, j)
    #                 # print(new_U_of_i_minus_k, new_V_of_j_minus_k)
    #
    #                 numerator_value = numerator_value + ((np.dot(new_U_of_i_minus_k, new_V_of_j_minus_k) - M[i][j]) * V[k][j])
    #                 denominator_value = denominator_value + (V[k][j] ** 2)
    #
    #         x[i] = (-1 * numerator_value) / denominator_value
    #
    #     for i in range(n):
    #         U[i, k] = x[i]
    #
    # for k in range(d):
    #     for j in range(m):
    #         numerator_value = 0
    #         denominator_value = 0
    #
    #         for i in range(n):
    #             if (M_transposed[j][i] > 0):
    #                 new_U_of_i_minus_k = np.delete(U[i,:], k)
    #                 new_V_of_j_minus_k = np.delete(V[:,j], k)
    #
    #                 numerator_value = numerator_value + ((np.dot(new_U_of_i_minus_k, new_V_of_j_minus_k) - M[i][j]) * U[i][k])
    #                 denominator_value = denominator_value + (U[i][k] ** 2)
    #
    #         y[j] = (-1 * numerator_value) / denominator_value
    #
    #     for j in range(m):
    #         V[k][j] = y[j]
    #
    # return U, V

    for iter in range(T):
        for i in range(n):
            for j in range(m):
                if M[i][j] > 0:
                    eij = M[i][j] - np.dot(U[i,:],V[:,j])
                    for k in range(d):
                        U[i][k] = U[i][k] + alpha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] = V[k][j] + alpha * (2 * eij * U[i][k] - beta * V[k][j])
        # RMSE(U, V.T, M, non_zero_entries)
    return U, V.T


def RMSE(nU, nV, M, non_zero_entries):
    nM = np.dot(nU, nV.T)

    # nM = np.dot(nU, nV)

    sum = 0
    for i in range(n):
        for j in range(m):
            if np.isnan(M[i,j]):
                continue
            sum += (M[i,j] - nM[i,j])**2
    mean = float(sum)/non_zero_entries
    print('The RMSE is: %.4f' % (mean**(0.5)))


if __name__ == "__main__":

    # For Toy Ratings Dataset
    # n = 3
    # m = 3
    # d = 2
    # T = 100

    # For Movie Ratings Dataset
    n = 943
    m = 1682
    d = 20
    T = 60 # T = 60 is gucci number with RMSE of 0.6163 and time of 4:13

    args = constructArguments()
    data = defaultdict(dict)
    all_movies = set()
    all_users = set()
    M = np.empty((n,m))
    M[:] = np.nan

    with open(args['input'], 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        lines = [[int(j) for j in i] for i in list(reader)]

        for line in lines:
            userid = int(line[0])
            if userid > n:
                break
            else:
                movieid = int(line[1])
                rating = int(line[2])
                data[userid][movieid] = rating
                all_movies.add(movieid)
                all_users.add(userid)

        all_movies = sorted(all_movies)
        all_movies = all_movies[:m]
        all_users = sorted(all_users)

        non_zero_entries=0
        for x in range(0,n):
            user = all_users[x]
            for y in range(0,m):
                movie = all_movies[y]
                if movie in data[user]:
                    M[x,y] = data[user][movie]
                    non_zero_entries += 1

    U = np.random.rand(n,d)
    V = np.random.rand(m,d)
    start_time = time.time()
    nU, nV = matrix_factorization(M, U, V, T, d, non_zero_entries)

    # nM = np.dot(nU, nV.T)
    # print("Original M (Ratings):")
    # print(M)
    # print("\n")
    # print("Predicted M (Ratings):")
    # print(nM)
    # print("\n")
    RMSE(nU, nV, M, non_zero_entries)
    elapsed_time = time.time() - start_time
    print("Time elapsed:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    np.savetxt('UT.tsv', nU, fmt='%.18e', delimiter='\t', newline='\n')

    # End of working code
