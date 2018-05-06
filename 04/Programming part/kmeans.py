# CSE446 HW04 - Question 04
# Omar Adel AlSughayer (1337255)

from numpy import *
import numpy as np
from random import sample


class KMeans():

    # the data we are to do the kmeans on
    x = None; y = None; 

    def main(self, groups=5,random=False):
        k = groups
        # create the data
        data = self.setup()

        # iterate for <= 20 times, recalculating the means and reassigning points
        clusters = self.converge(data, k,random)

        # find the accuracy
        self.print_accuracy(clusters) 


    # sets up the data 
    def setup(self):
        # get the data from the .txt files
        x = np.genfromtxt('digit.txt')
        y = np.genfromtxt('labels.txt', dtype=int)
        # combine the data from both files as tulips
        data = []
        for i in range(0, len(x)):
            data.append((x[i], y[i]))

        return data

    # creates k clusters as an array of tuplis (c = empirical mean of elements, [elements])
    def create_clusters(self, data, k, random=False):
        # the indeces of the starting centroids we pick, intiallay not random
        centroids = range(0, k)
        # the clusters
        clusters = []

        # change the list of centriods so it is random
        if(random):
            centriods = sample(range(0, len(data)), k)

        # fill clusters with the randomly chosen points
        for i in range(0, k):
            (c, y) = data[centriods[i]]
            clusters.append((np.copy(c), []))

        return clusters

    # finds the index of the nearest centriod and returns it
    def find_nearst_centroid(self, x, clusters):
        # the third argument "0" means the column, and "1" means the line.
        min_dis = np.linalg.norm(x-clusters[0][0], 2, 0) 
        min_i = 0
        # loop over all the centriods to find the nearest one
        for i in range(1, len(clusters)):
            dis = np.linalg.norm(x-clusters[i][0], 2, 0)
            if (dis < min_dis):
                min_dis = dis
                min_i = i

        return min_i

    # iteratively recalculates the means of elements in each cluster to create a centroid, then
    # reassigns points to the cluster with the nearest centroid; either until convergence or for 
    # @iterattions number of times, whichever is smaller. 
    def converge(self, data, k, random, iterattions=20):
        converged = False
        # create the clusters
        clusters = self.create_clusters(data, k, random)
        # assign the elements in data to their initial clustor 
        for i in range(0, len(data)):
            (x, y) = data[i]
            j = self.find_nearst_centroid(x, clusters)
            clusters[j][1].append(data[i])

        # starts looping from 1 because we have already assigned points once 
        for n in range(1, iterattions):
            
            # if no elements have been reassigned in the last iteration, halt
            if(converged):
                return clusters

            # assign converged to True so it remains that way if we did not 
            # move a point
            converged = True
            # recalculate the centroids
            new_clusters = self.calculate_centroids(clusters)
            
            # for every point, attempt to reassign
            for i in range(0, len(clusters)):
               for j in range(0, len(clusters[i][1])):
                    # find the point and remove it from the list of elements
                    (x, y) = (clusters[i][1])[j]
                    # find the nearest newly calculated centroid
                    k = self.find_nearst_centroid(x, new_clusters)
                    # check if the point have been reassigned 
                    if (k != i):
                        converged = False
                    # add the point in its correct cluster position in new_clusters
                    new_clusters[k][1].append((x, y))

            # change clusters to new_clusters
            clusters = new_clusters

            #print "itterated for", (n+1), "times so far"

        # if all itterations are over return the current cluster
        return clusters


    # calculates the centroids of each cluster as the the empirical mean
    # of the observations in it
    def calculate_centroids(self, clusters):
        # create a new set of clusters
        new_clusters = []
        # for every cluster
        for i in range(0, len(clusters)):
            (centroid, elements) = clusters[i]
            new_centroid = np.mean(elements)
            new_clusters.append((new_centroid, []))

        return new_clusters

    # decides the label of each cluster by majority votes, then calculates and 
    # prints the assignments accuracy on that cluster
    def print_accuracy(self, clusters):
        # number of roral misclassifications 
        errors = 0
        # for every cluster
        for c in range(0, len(clusters)):
            # get the elements and initialize array of counts
            elements = clusters[c][1]
            # counts = [#1, #3, #5, #7]
            counts = [0, 0, 0, 0]
            # count points 
            for i in range(0, len(elements)):
                (x, y) = elements[i]
                counts[(y-1)/2] += 1

            # final results
            label = counts.index(max(counts))
            label = label*2 + 1
            acc = 1.0*max(counts)/sum(counts)
            print "cluster", (c+1), "has label", label, "with accuraay =", acc

            # update errors
            errors += sum(counts) - max(counts)
        #print "      total number of mistakes =", errors
        # calculate and print within group sum of squares
        # print "      SS_total =", self.total_squared_sum(clusters)

    # answers the questions in the handout
    def total_squared_sum(self, clusters):
        # total sum
        ss_sum = 0

        # for every cluster
        for i in range(0, len(clusters)):
            (c, elements) = clusters[i]
            # for every point in the cluster
            for j in range(0, len(elements)):
                (x, y) = elements[j]
                # calculate the distance between x and c then square it
                dist = np.linalg.norm(x-c, 2, 0)
                dist = dist**2
                # add to total distance
                ss_sum += dist

        return ss_sum


# answers for questions 4.5 i-iv

# question 4.5.i, sum of squares
'''
for i in range(0, 3):
    model = KMeans()
    print "for k = ", (i+1)*2
    model.main((i+1)*2)
'''
# output:-
# for k = 2, SS_total = 644037168.877
# for k = 4, SS_total = 585896034.666
# for k = 6, SS_total = 562984384.698


# question 4.5.ii, how many iterations for k = 6
'''
model = KMeans()
model.main(6)
'''
# output: the algorithm had to itterate all 20 times then stop, instead of converging

# question 4.5.iii, the within groups sum of squares for k = 1, 2,...,10 with
# random starting centroids
'''
for i in range(0, 10):
    model = KMeans()
    print "for k = ", (i+1)
    model.main((i+1), True)
'''
# output: as you might expect, the sum of within groups sum of squares is decreasing
# linearly, save for a few outskirts caused by the randomness of initial centroids:  
# for k =  1, SS_total = 686394594.732
# for k =  2, SS_total = 644082676.77
# for k =  3, SS_total = 603490643.791
# for k =  4, SS_total = 585813294.696
# for k =  5, SS_total = 571073139.354
# for k =  6, SS_total = 565046908.49
# for k =  7, SS_total = 553893408.572
# for k =  8, SS_total = 541763918.202
# for k =  9, SS_total = 546519400.762
# for k =  10, SS_total = 534652908.343

# question 4.5.iv, the total mistakes (misclassifications) for k = 1, 2,..., 10 with
# random starting centroids
'''
for i in range(0, 10):
    model = KMeans()
    print "for k = ", (i+1)
    model.main((i+1), True)
'''
# output: as you might expect, the number of total mistakes is decreasing linearly, save
# for a few outskirts caused by the randomness of initial centroids: 
# for k =  1, total number of mistakes =  709
# for k =  2, total number of mistakes =  454
# for k =  3, total number of mistakes =  229
# for k =  4, total number of mistakes =  220
# for k =  5, total number of mistakes =  217
# for k =  6, total number of mistakes =  140
# for k =  7, total number of mistakes =  191
# for k =  8, total number of mistakes =  122
# for k =  9, total number of mistakes =  115
# for k =  10, total number of mistakes =  91

# a place holder, just to show off my accuracy, which isn't even that great but come on
# gimme a break I wrote this thing in like ninty minutes
model = KMeans()
model.main(20, True)
