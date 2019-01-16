from algorithm import kmeans
import matplotlib.pyplot as plt
import numpy as np

if '__main__' == __name__:
    load = kmeans.tool()
    dataMat = load.loadData('./testData/testSet.txt')

    #print(dataMat)

    alg = kmeans.kmeans()
    center = alg.randomCenter(dataMat, 10)

    k = 2

    center, cluster = alg.kmeans(dataMat, k)

    color = ['ro', 'bo', 'go', 'r+']
    for i in range(k):
        data = dataMat[np.nonzero(cluster[:,0] == i)[0]]
        plt.plot(data[:,0], data[:,1], color[i])
    plt.show()

    print(center)