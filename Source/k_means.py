import numpy as np
import pandas as pd
import math
import pprint

# select data file (s1, s2, s3, s4)
# select k: number of means
# select initialization

# init starting centroids
# LABEL A
# for each point, calculate nearest centroid
# calculate average of points in each cluster, make that the new centroid
# GOTO A until no change


data_path = "../data/"


data_points = []
means = []
centroid_sums = []


def euclid_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

def k_means(k, data_file, dist_function):

    with open(data_path+data_file) as f:
        f.readline()
        for line in f:
            x,y = line.split(',')
            data_points.append([[int(x), int(y)], 0, 9999999])

        # select random data points as mean
        clss = 0
        for mean in xrange(k):
            ex = np.random.random_integers(0, len(data_points))
            p = data_points[ex][0]
            means.append([p[0], p[1], clss])
            centroid_sums.append([0, 0, clss])
            clss += 1

        print means

        epocs = 0
        no_change = True
        while no_change:
            epocs += 1
            change_made = False

            # reset collector for centroid sums
            for centro in centroid_sums:
                centro[0] = 0.0
                centro[1] = 0.0

            for rec in data_points:

                for index in xrange(len(means)):
                    dist = dist_function(rec[0], means[index])
                    print index, dist
                    if dist < rec[2]:
                        change_made = True
                        rec[1] = index
                        rec[2] = dist
                    centroid_sums[rec[1]][0] += rec[0][0]
                    centroid_sums[rec[1]][1] += rec[0][1]

                if not change_made:
                    no_change = False
                    break

            if not change_made:
                break

            index = 0
            for centro in centroid_sums:
                centro[0] /= len(data_points)
                centro[1] /= len(data_points)
                print centro
                means[index][0] = centro[0]
                means[index][1] = centro[1]
                index += 1

    out_data = [[], means]
    for rec in data_points:
        del rec[2]
        out_data[0].append([rec[0][0], rec[0][1], rec[1]])

    return out_data







dat = k_means(6, "s1.txt", euclid_dist)
pprint.pprint(dat)


