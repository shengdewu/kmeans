import numpy as np
import random

class tool(object):
    def loadData(self, path):
        dataSet = []
        with open(path, 'r') as rf:
            while True:
                line = rf.readline()
                if '' == line:
                    break
                dataSet.append(list(map(float,line.strip('\n').split('\t'))))
        return np.mat(dataSet)

class kmeans(object):
    def randomCenter(self, dataSet, K):
        m, n = np.shape(dataSet)
        center = np.mat(np.zeros((K, n)))
        for i in range(K):
            center[i,:] = dataSet[random.randint(0, m-1), :]
        return center

    def disElua(self, vec1, vec2):
        return np.sqrt(np.sum(np.power(vec1-vec2, 2)))

    def kmeans(self, dataSet, K):
        center = self.randomCenter(dataSet, K) #每个类的质心
        m, n = np.shape(dataSet)
        cluster = np.mat(np.ones((m,2)))  #0：类， 1：误差 , 类 从零开始
        run = True
        while run:
            run = False
            for i in range(m):
                minDis = np.inf
                bestIndex = -1
                for n in range(K):
                    dist = self.disElua(center[n,:], dataSet[i,:])
                    if dist < minDis:
                        minDis = dist
                        bestIndex = n
                if cluster[i,0] != bestIndex:
                    run = True
                cluster[i,0] = bestIndex
                cluster[i,1] = minDis
            for i in range(K):
                data = dataSet[np.nonzero(cluster[:,0] == i)[0]]
                center[i,:] = np.mean(data, axis=0)
        return center, cluster

    def biKmeans(self, dataSet, k):
        m, n = np.shape(dataSet)
        center = np.mean(dataSet, axis=0)
        centerList = [center.tolist()[0]]
        cluster = np.mat(np.zeros((m, 2)))
        for i in range(m):
            cluster[i,1] += self.disElua(center, dataSet[i,0])

        while len(centerList) < k:
            err = np.inf
            bestCenter = None
            bestCluster = None
            bestIndex = np.inf
            for i in range(len(centerList)):
                data = dataSet[np.nonzero(cluster[:,0] == i)[0]]
                splitCenter, splitCluster = self.kmeans(data, 2)
                splitErr = np.sum(splitCenter[:,1])
                noSpitErr = np.sum(cluster[np.nonzero(cluster[:,0] != i)[0], 1])
                if splitErr + noSpitErr < err:
                    err = splitErr + noSpitErr
                    bestCenter = splitCenter
                    bestCluster = splitCluster.copy()
                    bestIndex = i

            bestCluster[np.nonzero(bestCluster[:,0] == 0)[0], 0] = bestIndex
            bestCluster[np.nonzero(bestCluster[:,0] == 1)[0], 0] = len(centerList)
            centerList[bestIndex] = bestCenter[0,:].tolist()[0]
            centerList.append(bestCenter[1,:].tolist()[0])
            cluster[np.nonzero(cluster[:,0] == bestIndex)[0], :] = bestCluster

        return np.mat(centerList), cluster




