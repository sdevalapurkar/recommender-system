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


def factorize():
    x = [0 for i in range(n)]
    y = [0 for i in range(m)]
    
    for iter in range(T):
        for k in range(d):
            for i in range(n):
                numerator_value = 0
                denominator_value = 0
        
                for j in range(m):
                    if (M[i][j] > 0):
                        new_U_of_i_minus_k = np.delete(U[i,:], k)
                        new_V_of_j_minus_k = np.delete(V[:,j], k)
                        # print(k, i, j)
                        # print(new_U_of_i_minus_k, new_V_of_j_minus_k)
        
                        numerator_value = numerator_value + ((np.dot(new_U_of_i_minus_k, new_V_of_j_minus_k) - M[i][j]) * V[k][j])
                        denominator_value = denominator_value + (V[k][j] ** 2)
        
                x[i] = (-1 * numerator_value) / denominator_value
        
            for i in range(n):
                U[i, k] = x[i]
        
        for k in range(d):
            for j in range(m):
                numerator_value = 0
                denominator_value = 0
        
                for i in range(n):
                    if (M_transposed[j][i] > 0):
                        new_U_of_i_minus_k = np.delete(U[i,:], k)
                        new_V_of_j_minus_k = np.delete(V[:,j], k)
        
                        numerator_value = numerator_value + ((np.dot(new_U_of_i_minus_k, new_V_of_j_minus_k) - M[i][j]) * U[i][k])
                        denominator_value = denominator_value + (U[i][k] ** 2)
        
                y[j] = (-1 * numerator_value) / denominator_value
        
            for j in range(m):
                V[k][j] = y[j]
    
    return U, V
