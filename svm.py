import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_training_data_binary(num):
  if num == 1:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, -1]

  elif num == 2:
    data = np.zeros((10,3))
    for i in range(5):
      data[i] = [0, i-5, 1]
      data[i+5] = [0, i+1, -1]

  elif num == 3:
    data = np.zeros((10,3))
    data[0] = [3, 2, 1]
    data[1] = [6, 2, 1]
    data[2] = [3, 6, 1]
    data[3] = [4, 4, 1]
    data[4] = [5, 4, 1]
    data[5] = [-1, -2, -1]
    data[6] = [-2, -4, -1]
    data[7] = [-3, -3, -1]
    data[8] = [-4, -2, -1]
    data[9] = [-4, -4, -1]
  elif num == 4:
    data = np.zeros((10,3))
    data[0] = [-1, 1, 1]
    data[1] = [-2, 2, 1]
    data[2] = [-3, 5, 1]
    data[3] = [-3, -1, 1]
    data[4] = [-2, 1, 1]
    data[5] = [3, -6, -1]
    data[6] = [0, -2, -1]
    data[7] = [-1, -7, -1]
    data[8] = [1, -10, -1]
    data[9] = [0, -8, -1]
  elif num == 5:
    data = np.zeros((4,3))
    data[0] = [-1, -1, -1]
    data[1] = [-1, 1, 1]
    data[2] = [1, 1, -1]
    data[3] = [1, -1, 1]
  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return data

def generate_training_data_multi(num):
  if num == 1:
    data = np.zeros((20,3))
    for i in range(5):
      data[i] = [i-5, 0, 1]
      data[i+5] = [i+1, 0, 2]
      data[i+10] = [0, i-5, 3]
      data[i+15] = [0, i+1, 4]
    C = 4

  elif num == 2:
    data = np.zeros((15,3))
    data[0] = [-5, -5, 1]
    data[1] = [-3, -2, 1]
    data[2] = [-5, -3, 1]
    data[3] = [-5, -4, 1]
    data[4] = [-2, -9, 1]
    data[5] = [0, 6, 2]
    data[6] = [-1, 3, 2]
    data[7] = [-2, 1, 2]
    data[8] = [1, 7, 2]
    data[9] = [1, 5, 2]
    data[10] = [6, 3, 3]
    data[11] = [9, 2, 3]
    data[12] = [10, 4, 3]
    data[13] = [8, 1, 3]
    data[14] = [9, 0, 3]
    C = 3

  else:
    print("Incorrect num", num, "provided to generate_training_data_binary.")
    sys.exit()

  return [data, C]

def plot_training_data_binary(data):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')

  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()


def plot_training_data_multi(data):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])

  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

def distance_point_to_hyperplane(pt, w, b):
    return abs(np.dot(pt[0:2], w) + b)/np.sqrt(w[0]**2 + w[1]**2)

def compute_margin(data, w, b):
    min_margin = distance_point_to_hyperplane(data[0], w, b)
    for pt in data:
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0
        margin = distance_point_to_hyperplane(pt, w, b)
        if margin < min_margin:
            min_margin = margin
    return min_margin

def midpoint(pt1, pt2):
    return [(pt1[0] + pt2[0])/2, (pt1[1] + pt2[1])/2]

def makeline(pt1, pt2):
    return (pt1-pt2)[:2]

def magnitude(line):
    return np.sqrt(line.dot(line))

def svm_test_brute(w, b, x):
    if np.dot(w, x[:2]) + b > 0:
        return 1
    else:
        return -1

def svm_train_brute(training_data):
    pos = training_data[training_data[:, 2] == 1]
    neg = training_data[training_data[:, 2] == -1]
    max_margin = 0
    svm = None
    # 2 +, 1 -
    for p1 in pos:
        for p2 in pos:
            for q in neg:

                # if two points are equidistant to point in other class
                if ((p1[0] != p2[0]) and (p1[1] != p2[1])):
                    line = makeline(p2, p1)
                    v = line/magnitude(line)
                    p = p1[:2] + v.dot((q-p1)[:2])*v #projection of q onto line

                    mid = midpoint(p, q)
                    w = makeline(p, q[:2])
                    b = -np.dot(w, mid)
                    margin = compute_margin(training_data, w, b)
                    if margin >= max_margin:
                        max_margin = margin
                        if svm: # always sets S to the closest points to decision boundary / accounts for python precision / account for data in a line
                            if distance_point_to_hyperplane(p, w, b) < distance_point_to_hyperplane(svm[2][0], w, b):
                                # print('updated')
                                # print(p)
                                S = np.array([q, p1, p2])
                        else:
                            S = np.array([q, p1, p2])
                        svm = [w, b, S]

    # 1 +, 2 -
    for q in pos:
        for n1 in neg:
            for n2 in neg:
                # if two points are equidistant to point in other class
                if n1[0] != n2[0] and n1[1] != n2[1]:
                    line = makeline(n2, n1)
                    v = line/magnitude(line)
                    n = n1[:2] + v.dot((q-n1)[:2])*v #projection of q onto line

                    mid = midpoint(q, n)
                    w = makeline(q[:2], n)
                    b = -np.dot(w, mid)
                    margin = compute_margin(training_data, w, b)
                    if margin >= max_margin:
                        max_margin = margin
                        if svm: # always sets S to the closest points to decision boundary / accounts for python precision / account for data in a line
                            if distance_point_to_hyperplane(n, w, b) < distance_point_to_hyperplane(svm[2][0], w, b):
                                # print('updated')
                                # print(p)
                                S = np.array([q, n1, n2])
                        else:
                            S = np.array([q, n1, n2])
                        svm = [w, b, S]
    # 1 +, 1 -
    for p in pos:
        for n in neg:
            mid = midpoint(p, n)
            w = makeline(p, n)
            sep = [-w[1], w[0]]
            b = -np.dot(w, mid)
            margin = compute_margin(training_data, w, b)
            if margin >= max_margin:
                max_margin = margin
                if svm: # always sets S to the closest points to decision boundary / accounts for python precision / account for data in a line
                    if distance_point_to_hyperplane(p, w, b) < distance_point_to_hyperplane(svm[2][0], w, b):
                        S = np.array([p, n])
                else:
                    S = np.array([p, n])
                svm = [w, b, S]
    return svm

def plot_binary(data, w, b):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')
  x = np.linspace(-10, 10, 100)
  if w[0] == 0:
      plt.axhline(-b/w[1])
  elif w[1] == 0:
      plt.axvline(x = -b/w[0])
  else:
      slope = -w[0]/w[1]
      plt.plot(x, slope*x - b/w[1])
      # plt.plot(x, -x/slope)

  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

######### Problem 1 #########
# 1
# train_data = generate_training_data_binary(1)
# svm = svm_train_brute(train_data)
# plot_binary(train_data, svm[0],svm[1])
# 2
# train_data = generate_training_data_binary(2)
# svm = svm_train_brute(train_data)
# plot_binary(train_data, svm[0],svm[1])
# 3
# train_data = generate_training_data_binary(3)
# svm = svm_train_brute(train_data)
# plot_binary(train_data, svm[0],svm[1])
# 4
# train_data = generate_training_data_binary(4)
# svm = svm_train_brute(train_data)
# plot_binary(train_data, svm[0],svm[1])




import copy

def svm_train_multiclass(training_data):
    w = []
    b = []
    for c in range(1, training_data[1] + 1):
        data = copy.copy(training_data[0])
        for pt in data:
            if pt[2] == c:
                pt[2] = 1
            else:
                pt[2] = -1
        svm = svm_train_brute(data)
        w.append(svm[0])
        b.append(svm[1])

    return np.array([w,b])

def svm_test_multiclass(W,B,x):
    c = -1
    max_margin = 0

    for i in range(W):
        margin = np.dot(W[i], x[:2]) + B[i]
        if margin > 0:
            if c == -1:
                c = i
                max_margin = margin
            elif margin > max_margin:
                max_margin = margin
                c = i
    return c

def plot_multi(data, W, B):
  colors = ['b', 'r', 'g', 'm']
  shapes = ['+', 'o', '*', '.']

  for item in data:
    plt.plot(item[0], item[1], colors[int(item[2])-1] + shapes[int(item[2])-1])
  x = np.linspace(-10,10,1000)
  for c in range(len(W)):
      if W[c][0] == 0:
          plt.axhline(-B[c]/W[c][1])
      elif W[c][1] == 0:
          plt.axvline(x = -B[c]/W[c][0])
      else:
          m = -W[c][0]/W[c][1]
          yint = -B[c]/W[c][1]
          plt.plot(x, m*x + yint)
  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

######### Problem 2 ########
# 1
# training_data = generate_training_data_multi(1)
# svm = svm_train_multiclass(training_data)
# print(svm)
# plot_multi(training_data[0], svm[0], svm[1])

# 2
# training_data = generate_training_data_multi(2)
# svm = svm_train_multiclass(training_data)
# # print(svm)
# plot_multi(training_data[0], svm[0], svm[1])




def transform(data):
    for pt in data:
        pt[1] = pt[0]*pt[1]
    return data

def kernel_svm_train(training_data):
    return svm_train_brute(transform(copy.copy(training_data)))

def plot_kernel(data, w, b):
  for item in data:
    if item[-1] == 1:
      plt.plot(item[0], item[1], 'b+')
    else:
      plt.plot(item[0], item[1], 'ro')

  x = np.linspace(-10, 10, 100)
  if w[0] == 0:
      plt.axhline(-b/w[1])
  elif w[1] == 0:
      plt.axvline(x = -b/w[0])
  else:
      slope = -w[0]/w[1]
      plt.plot(x, slope*x - b/w[1])
      # plt.plot(x, -x/slope)

  m = max(data.max(), abs(data.min()))+1
  plt.axis([-m, m, -m, m])
  plt.show()

def kernel(data):
    new_data = []
    for pt in data:
        new_data.append([pt[0]*pt[1], 0, pt[2]])
    return np.array(new_data)

def kernel_svm_train2(training_data):
    return svm_train_brute(kernel(training_data))

######### Problem 3 ##########
# 1
# training_data = generate_training_data_binary(5)
# plot_training_data_binary(training_data)
# svm = kernel_svm_train(training_data)
# plot_kernel(transform(copy.copy(training_data)), svm[0], svm[1])
# 2
# training_data = generate_training_data_binary(5)
# svm2 = kernel_svm_train2(training_data)
# plot_kernel(kernel(training_data), svm2[0], svm2[1])
