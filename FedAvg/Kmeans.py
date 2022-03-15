import random
import getClientsData as gcd
from sklearn.cluster import KMeans

x, y = gcd.file2matrix('clientsData.txt')
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300)
kmeans.fit(x)

b = [2383, 16630, 1910,  62, 685, 685]
print('b=', b, ',类别：', kmeans.predict([b]))

for i in range(100):
    dataSize = random.randint(500, 5000)
    c = random.randint(0, 1000)
    a = [dataSize, dataSize * random.randint(10, 15), random.randint(300, 6000), random.randint(1, 100),
         int(c * random.random()), c]
    print('a=', a, ',类别：', (kmeans.predict([a])))

for i in x:
    print('i=', i, ',类别：', (kmeans.predict([i])))