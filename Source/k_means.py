import numpy as np
import pandas as pd
import math
import pprint
from  matplotlib import pyplot
import seaborn
import sys
import collections
import copy

data_path = "../data/"


def euclid_dist(mean, rec):
    p1 = (mean[0] - rec[0]) ** 2
    p2 = (mean[1] - rec[1]) ** 2
    test = math.sqrt(p1 + p2)
    return test


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
            print rec[2]
            num_in_class[rec[2]] += 1
            centroid_sums[rec[2]][0] += rec[0]
            centroid_sums[rec[2]][1] += rec[1]

        for i in xrange(len(means)):
            divfac = num_in_class[i]
            centroid_sums[i][0] = centroid_sums[i][0] / divfac
            centroid_sums[i][1] = centroid_sums[i][1] / divfac

        if all(means[i] == centroid_sums[i] for i in range(len(means))):
            change_made = False
            break

        means = copy.deepcopy(centroid_sums)

    out_data = [[], means]

    for rec in data_points:
        out_data[0].append(rec)

    return out_data


dat = k_means(15, "s4.txt", euclid_dist, "s4-cb.txt")
data_points = pd.DataFrame.from_records(dat[0], columns=['x', 'y', 'class'])
cluster_points = pd.DataFrame.from_records(dat[1], columns=['x', 'y', 'class'])
point_cmap = pyplot.get_cmap('gist_ncar')
centroid_cmap = pyplot.get_cmap('Reds')
ax = data_points.plot.scatter(x='x', y='y', c=data_points['class'], cmap=point_cmap)
cluster_points.plot.scatter(x='x', y='y', marker='^', c="class", cmap=centroid_cmap, s=50, ax=ax)
ax.set_aspect('equal')
pyplot.show()


def dunn_index(cluster_and_means):
    print "not implemented"


def db_index(cluster_and_means):
    print "not implemented"


def c_index(cluster_and_means):
    print "not implemented"
