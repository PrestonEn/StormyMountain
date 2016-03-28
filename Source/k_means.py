from __future__ import print_function
import numpy as np
import pandas as pd
import math
import pprint
from  matplotlib import pyplot
import sys
import collections
import copy

data_path = "../data/"
f = open('results', 'w')

def euclid_dist(mean, rec):
    p1 = (mean[0] - rec[0]) ** 2
    p2 = (mean[1] - rec[1]) ** 2
    test = math.sqrt(p1 + p2)
    return test

def cheby_dist(mean, rec):
    p1 = abs(mean[0] - rec[0])
    p2 = abs(mean[1] - rec[1])
    return max(p1, p2)

def mikow_dist(mean, rec):
    p1 = abs(mean[0] - rec[0]) ** 3
    p2 = abs(mean[1] - rec[1]) ** 3
    return math.pow(p1 + p2, 1/float(3))




def k_means(k, data_file, dist_function, centroids_file=None):
    data_points = []
    means = []
    centroid_sums = []
    num_in_class = []

    with open(data_path + data_file) as f:
        f.readline()
        for line in f:
            x, y = line.split(',')
            data_points.append([float(x), float(y), -1])

    # if the parameter for a centroid file is not set,
    # just use random datapoints from the list as centroids
    if centroids_file == None:
        for i in xrange(k):
            p = data_points[np.random.random_integers(0, len(data_points))]
            means.append([p[0], p[1], i])
            centroid_sums.append([0, 0, i])
            num_in_class.append(0)

    else:
        with open(data_path + centroids_file) as f:
            count = 0
            for line in f:
                x, y = line.split(',')
                means.append([float(x), float(y), count])
                num_in_class.append(0)
                centroid_sums.append([0, 0, count])
                count += 1

    epocs = 0
    change_made = True
    while change_made:
        epocs += 1

        for centro in centroid_sums:
            centro[0] = 0.0
            centro[1] = 0.0

        for i in xrange(len(num_in_class)):
            num_in_class[i] = 0

        # set flag to test for convergence

        for rec in data_points:

            dist = sys.maxint
            for mean in means:
                test = dist_function(rec, mean)
                if test < dist:
                    rec[2] = mean[2]
                    dist = test

            num_in_class[rec[2]] += 1
            centroid_sums[rec[2]][0] += rec[0]
            centroid_sums[rec[2]][1] += rec[1]

        for i in xrange(len(means)):
            divfac = num_in_class[i]
            if divfac != 0:
                centroid_sums[i][0] = centroid_sums[i][0] / divfac
                centroid_sums[i][1] = centroid_sums[i][1] / divfac
            else:
                centroid_sums[i][0] = 0.0
                centroid_sums[i][1] = 0.0

        if all(means[i] == centroid_sums[i] for i in range(len(means))):
            change_made = False
            break

        means = copy.deepcopy(centroid_sums)

    out_data = [[], means]

    for rec in data_points:
        out_data[0].append(rec)

    return out_data


def dunn_index(points, means, dist_fn):
    """
    Dunn index is defined as the minimal intercluster distance (minimum dist of 2 objects in different clusters)
    divided by the maximum distance of 2 objects in the same cluster

    :param points: the data points in all clusters
    :param means: the centroid data and label for each cluster
    :dist_fn: the function to be used as a distance calculation
    :return: the dunn index value
    """
    max_dist = -1
    min_dist = sys.maxint
    means.sort()
    for mean in means:
        data = points[points.clust == mean].values.tolist()
        others = points[points.clust != mean].values.tolist()

        for x1 in data:
            for x2 in data:
                max_dist = max(dist_fn(x1, x2), max_dist)

        for x1 in data:
            for x2 in others:
                min_dist = min(dist_fn(x1, x2), min_dist)

    return float(min_dist) / float(max_dist)



def makeplot(data, clusters, name):

    pyplot.figure(figsize=(10,10))
    pyplot.axes().set_aspect('equal', 'datalim')
    classes = data['clust'].unique().tolist()

    for x in classes:
        df = data[data['clust'] == x]
        clust = clusters[clusters['clust'] == x]
        pyplot.scatter(df['x'],
                       df['y'],
                       color=(np.random.uniform(0.2, 0.9),np.random.uniform(0.2,0.9),np.random.uniform(0.2,0.9)))

        pyplot.scatter(clust['x'],
                       clust['y'],
                       color='0.0',
                       marker='^',
                       s=150)


    pyplot.savefig("../figs/" + name)
    pyplot.close()


def exprun(k, dist_fun, data_file, plot_name, cluster_file=None):

    dat = k_means(k, data_file, dist_fun, cluster_file)
    cluster_points = pd.DataFrame.from_records(dat[1], columns = ['x', 'y', 'clust'])
    data_points = pd.DataFrame.from_records(dat[0], columns = ['x', 'y', 'clust'])
    data_points.sort_values(['clust'], ascending=True)
    classes = data_points['clust'].unique().tolist()
    makeplot(data_points, cluster_points, plot_name)
    print(str(dunn_index(data_points,classes, dist_fun)) + "\t" + plot_name, file=f)





exprun(15, euclid_dist, "s1.txt", "s1_rand_euclid_15.png")

exprun(10, euclid_dist, "s1.txt", "s1_rand_euclid_10.png")

exprun(20, euclid_dist, "s1.txt", "s1_rand_euclid_20.png")

exprun(15, cheby_dist, "s1.txt", "s1_rand_cheb_15.png")

exprun(15, mikow_dist, "s1.txt", "s1_rand_miko_15.png")

exprun(15, euclid_dist, "s1.txt", "s1_true_euclid_15.png", "s1-cb.txt")

exprun(15, euclid_dist, "s2.txt", "s2_rand_euclid_15.png")
exprun(10, euclid_dist, "s2.txt", "s2_rand_euclid_10.png")

exprun(20, euclid_dist, "s2.txt", "s2_rand_euclid_20.png")
exprun(15, cheby_dist, "s2.txt", "s2_rand_cheb_15.png")
exprun(15, mikow_dist, "s2.txt", "s2_rand_miko_15.png")

exprun(15, euclid_dist, "s2.txt", "s2_true_euclid_15.png", "s2-cb.txt")

exprun(15, euclid_dist, "s3.txt", "s3_rand_euclid_15.png")

exprun(10, euclid_dist, "s3.txt", "s3_rand_euclid_10.png")

exprun(20, euclid_dist, "s3.txt", "s3_rand_euclid_20.png")
exprun(15, cheby_dist, "s3.txt", "s3_rand_cheb_15.png")
exprun(15, mikow_dist, "s3.txt", "s3_rand_miko_15.png")

exprun(15, euclid_dist, "s3.txt", "s3_true_euclid_15.png", "s3-cb.txt")

exprun(15, euclid_dist, "s4.txt", "s4_rand_euclid_15.png")

exprun(10, euclid_dist, "s4.txt", "s4_rand_euclid_10.png")

exprun(20, euclid_dist, "s4.txt", "s4_rand_euclid_20.png")
print("exp21")

exprun(15, cheby_dist, "s4.txt", "s4_cheb_euclid_15.png")
print("exp22")

exprun(15, mikow_dist, "s4.txt", "s4_miko_euclid_15.png")
print ("exp23")

exprun(15, euclid_dist, "s4.txt", "s4_true_euclid_15.png", "s4-cb.txt")
print("exp24")
