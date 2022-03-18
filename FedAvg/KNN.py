import random
from sklearn.neighbors import KNeighborsClassifier
import getClientsData as gcd
import time

x, y, weights = gcd.file2matrixWithLabel("clientsData.txt")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(x, y)
print(x.shape)

cal = 0
for i in range(1000):
    c = random.randint(0, 1000)
    a = [random.randint(500, 5000), random.randint(3000, 12000), random.randint(30, 6000), random.randint(1, 100),
         int(c * random.random()), c]
    start = time.time()
    result = knn.predict([a])
    end = time.time()
    cal += end - start
    print(a, result)
print(cal)
