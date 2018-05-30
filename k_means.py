
"""
CS446: Advanced topics in Machine Learning
Author: Israel Bond
Programming Assignment #2
    Assignment #1: K-Means
        standard version of K-Means.
        starting points for K clusters means are a randomly selected data points
        have the option to run algorithm r times from r different chosen initializations
            * generate random number for cluster count K r times
            * run k-means r times with each cluster K
            * produce a cluster count that provides the lowest sum of squares error
            * produce 2-d plot for each iteration with data points and clusters

"""

import pandas as pd
import numpy as np
import sklearn.metrics as met
import random
from matplotlib import pyplot as plt
""" might be used for verification"""
#   from sklearn.cluster import KMeans

#np.set_printoptions(threshold=np.nan)
random.seed(a=None)


""" PREPROCESSING """
# load file into numpy array
data = np.loadtxt('GMM_dataset.txt', dtype= float)
# randomize rows as they came in order.
np.random.shuffle(data)
# plot data in graph
x, y = data.T
plt.scatter(x, y)
plt.show()
#print(data.mean())
#print(np.mean(data))
#print(data[:,0].mean() + data[:,1].mean())
#print(data[:,1].mean())
#print(data[0])
#print(data[0, :])
""" TEST for looping through indices """
#for x in range(1):
#    print("print", x)
#print("done")

"""saves plot to file"""
#plt.savefig('test.png', bbox_inches= 'tight')
""" TEST: getting a random index for the data to initialize centroid value"""
#for i in range(10):
#    index = np.random.randint(0,data.shape[0])
#    centroid = data[index]
#    print("index ", index, "centroid ", centroid)

class cluster:
    def __init__(self):
        # establish Binary Indicator Variable for cluster identification
        self.biv = np.zeros(1500, int)
        # set centroid to randomly selected datum
        self.centroid = data[np.random.randint(0, data.shape[0])]
        self.cluster_mean = np.copy(self.centroid)
        #
""" TEST: one instance of a cluster"""
#k = cluster()
#print(k.biv.shape)
#print(k.biv)
#print(k.centroid)
""" TEST: array of clusters"""
#clusters = [cluster() for i in range(10)]
#for k in clusters:
#    print(k.centroid)
""" TEST: looping through cluster creation """
#for i in range(1,10):
#    Kset = [cluster() for x in range(i)]
#    print("type ", type(Kset)," size ", len(Kset))
#    k_argmins = [float for y in range(i)]
#    print("type ", type(k_argmins)," size ", len(k_argmins))

def k_means_algorithm():
    test_num = 11
    cluster_num = 11

    for test in range(1,test_num):
        print("\t\ttest ", test)
        for clusterI in range(1,cluster_num):
            print("\tcluster index ", clusterI)
            K_set = [cluster() for i in range(clusterI)]
            k_argmins = [np.float64 for j in range(clusterI)]

            # E-setp
            # calculate binary indicator variables for each cluster
            for i in range(data.shape[0]):
                for k in range(clusterI):
                    k_argmins[k] = (data[i][0] - K_set[k].cluster_mean[0])**2
                    k_argmins[k] += (data[i][1] - K_set[k].cluster_mean[1])**2
                # get index for cluster biv to set
                setK_biv = np.argmin(k_argmins)
                K_set[setK_biv].biv[i] = 1
                # set all other cluster biv at this data index to zero
                for zero in range(clusterI):
                    if zero != setK_biv:
                        K_set[zero].biv[i] = 0
            # M-step 
            # recalculate each cluster mean values
            # objective function
            # for each cluster
            k_cluster_sum = 0
            for k in K_set:
                # for each data member
                for i in range(k.biv.size):
                    if k.biv[i] == 1:
                        k_cluster_sum += abs(data[i])
                print("cluster sum x", k_cluster_sum[0])
                print("cluster sum y", k_cluster_sum[1])
            print(k_cluster_sum)




k_means_algorithm()
