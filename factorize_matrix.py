from collections import defaultdict
import numpy as np
import argparse
import csv
import sys
import os

def constructArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, default=None,
    help='input path of data file')
  parser.add_argument('-o', '--output', type=str, default=None,
    help='output path of data file')
  args = vars(parser.parse_args())

  return args


def matrix_factorization(M, U, V, T, d, non_zero_entries, alpha=0.01, beta=0.02):
    V = V.T

    for iter in range(T):
        for i in range(n):
            for j in range(m):
                if M[i][j] > 0:
                    eij = M[i][j] - np.dot(U[i,:],V[:,j])
                    for k in range(d):
                        U[i][k] = U[i][k] + alpha * (2 * eij * V[k][j] - beta * U[i][k])
                        V[k][j] = V[k][j] + alpha * (2 * eij * U[i][k] - beta * V[k][j])

    return U, V.T


def RMSE(nU, nV, M, non_zero_entries):
    nM = np.dot(nU, nV.T)
    sum = 0

    for i in range(n):
        for j in range(m):
            if np.isnan(M[i,j]):
                continue
            sum += (M[i,j] - nM[i,j])**2

    mean = float(sum)/non_zero_entries
    print('The RMSE is: %.4f' % (mean**(0.5)))


if __name__ == "__main__":
    n = 943
    m = 1682
    d = 20
    T = 60 # T = 60 is good number with RMSE of 0.6163 and time of 4:13

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
            movieid = int(line[1])
            rating = int(line[2])
            data[userid][movieid] = rating
            all_movies.add(movieid)
            all_users.add(userid)

        all_movies = list(all_movies)
        all_movies = all_movies[:m]
        all_users = list(all_users)

        non_zero_entries=0
        for x in range(0,n):
            user = all_users[x]
            for y in range(0,m):
                movie = all_movies[y]
                if movie in data[user]:
                    M[x,y] = data[user][movie]
                    non_zero_entries += 1

    user_np = np.asarray(all_users, dtype=np.float32)
    movie_np = np.asarray(all_movies, dtype=np.float32)

    U = np.random.rand(n,d)
    V = np.random.rand(m,d)
    nU, nV = matrix_factorization(M, U, V, T, d, non_zero_entries)

    RMSE(nU, nV, M, non_zero_entries)

    user_np = np.reshape(user_np, (n,1))
    UT = np.concatenate((user_np, nU), axis=1)
    os.remove('UT.tsv')
    np.savetxt('UT.tsv', UT, delimiter="\t", fmt="%s")

    movie_np = np.reshape(movie_np, (m,1))
    VT = np.concatenate((movie_np, nV), axis=1)
    os.remove('VT.tsv')
    np.savetxt('VT.tsv', VT, delimiter="\t", fmt="%s")
