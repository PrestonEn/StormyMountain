import numpy as np
import pandas as pd
import math
import pprint
from  matplotlib import pyplot
import seaborn
import sys
data_path = "../data/"


np.random.seed(5)
def euclid_dist(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + ((p2[1] - p1[1])**2))

def k_means(k, data_file, dist_function):
    data_points = []
    means = []
    centroid_sums = []

    with open(data_path+data_file) as f:
        f.readline()
        for line in f:
            x,y = line.split(',')
            data_points.append([int(x), int(y), -1, sys.maxint])

        # select random data points as mean
        clss = 0
        num_in_class = []
        for mean in xrange(k):
            ex = np.random.random_integers(0, len(data_points))
            p = [data_points[ex][0], data_points[ex][1]]
            means.append([p[0], p[1], clss])
            centroid_sums.append([0, 0, clss])
            num_in_class.append(0)
            clss += 1

        epocs = 0
        change_made = True

        while change_made:
            epocs += 1

            # reset collector for centroid sums
            for centro in centroid_sums:
                centro[0] = 0.0
                centro[1] = 0.0

            for i in xrange(len(num_in_class)):
                num_in_class[i] = 0

            # set flag to test for convergence
            change_made = False

            for rec in data_points:

                for index in xrange(len(means)):
                    dist = dist_function([rec[0],rec[1]], means[index])

                    if dist < rec[3]:
                        change_made = True
                        rec[2] = index
                        rec[3] = dist

                num_in_class[rec[2]] += 1
                centroid_sums[rec[2]][0] += rec[0]
                centroid_sums[rec[2]][1] += rec[1]



            index = 0
            print centroid_sums, "calced centroids"
            print means, "old memes"
            for centro in centroid_sums:
                centro[0] /= num_in_class[index]
                centro[1] /= num_in_class[index]
                means[index][0] = centro[0]
                means[index][1] = centro[1]
                index += 1
            print means, "new centroids"



    out_data = [[], means]

    for rec in data_points:
        del rec[3]
        out_data[0].append(rec)
    return out_data







dat = k_means(3, "s3.txt", euclid_dist)

pprint.pprint(dat[0])
data_points = pd.DataFrame.from_records(dat[0],columns=['x', 'y', 'class'])
cluster_points = pd.DataFrame.from_records(dat[1],columns=['x', 'y', 'class'])

point_cmap = pyplot.get_cmap('gist_ncar')
ax = data_points.plot.scatter(x='x', y='y', c=data_points['class'], cmap=point_cmap)
cluster_points.plot.scatter(x='x', y='y', marker='^', s=500 ,ax=ax)
ax.set_aspect('equal')
pyplot.show()