from sklearn.cluster import KMeans
import numpy as np
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:,]

result = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(x)
    result.append(kmeans.inertia_)
    
plt.plot(range(1,11),result)
plt.title('Elbow Method for selection of optimal "K" clusters')
plt.xlabel('K')
plt.ylabel('Averge Dispersion')
plt.plot(2, 150, linestyle='dashed', marker='$\u25CC$',markersize=25)
plt.annotate("Elbow Point",(2,150),(4,200),arrowprops=dict(arrowstyle="-|>",fc="k", ec="k", linestyle='dashed',connectionstyle="arc3,rad=0.3"))

plt.savefig("elbow.png")

plt.close()