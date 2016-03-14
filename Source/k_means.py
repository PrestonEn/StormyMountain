import numpy as np
import pandas as pd
import math
import pprint
from  matplotlib import pyplot
import seaborn

data_path = "../data/"



def euclid_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

def k_means(k, data_file, dist_function):
    data_points = []
    means = []
    centroid_sums = []

    with open(data_path+data_file) as f:
        f.readline()
        for line in f:
            x,y = line.split(',')
            data_points.append([[int(x), int(y)], 0, 9999999])

        # select random data points as mean
        clss = 0
        num_in_class = []
        for mean in xrange(k):
            ex = np.random.random_integers(0, len(data_points))
            p = data_points[ex][0]
            means.append([p[0], p[1], clss])
            centroid_sums.append([0, 0, clss])
            num_in_class.append(0)
            clss += 1

        epocs = 0
        no_change = True

        while no_change:
            epocs += 1

            # reset collector for centroid sums
            for centro in centroid_sums:
                centro[0] = 0.0
                centro[1] = 0.0

            for i in xrange(len(num_in_class)):
                num_in_class[i] = 0

            # set flag to test for convergence
            no_change = False

            for rec in data_points:

                for index in xrange(len(means)):
                    dist = dist_function(rec[0], means[index])
                    if dist < rec[2]:
                        no_change = True
                        rec[1] = index
                        rec[2] = dist

                num_in_class[rec[1]] += 1
                centroid_sums[rec[1]][0] += rec[0][0]
                centroid_sums[rec[1]][1] += rec[0][1]



            index = 0
            for centro in centroid_sums:
                centro[0] /= num_in_class[index]
                centro[1] /= num_in_class[index]
                means[index][0] = centro[0]
                means[index][1] = centro[1]
                index += 1



    out_data = [[], means]

    for rec in data_points:
        del rec[2]
        out_data[0].append([rec[0][0], rec[0][1], rec[1]])
    print len(data_points)
    print num_in_class
    return out_data







dat = k_means(5, "s3.txt", euclid_dist)

pprint.pprint(dat[0])
data_points = pd.DataFrame.from_records(dat[0],columns=['x', 'y', 'class'])
fg = seaborn.FacetGrid(data=data_points, hue='class', size=6, aspect=1, )
fg.set(yticklabels=[])
fg.set(xticklabels=[])
fg.despine()
fg.map(pyplot.scatter, 'x', 'y', s=5)
fg.savefig("s3_1.png",  bbox_inches='tight')

