from sklearn import preprocessing

import getClientsData as gcd
from sklearn.cluster import KMeans

x, y, w = gcd.file2matrixWithLabel('clientsData.txt')
data = []
for a, b in zip(x, y):
    d = [a[1]/a[0], a[2], a[4]/a[5] if b < 2 else (a[4]/a[5])-1]
    data.append(d)
#print(data)

predictData = preprocessing.scale(data)

kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300)
kmeans.fit(predictData)

# b = [2383, 16630, 1910,  62, 685, 685]
# print('b=', b, ',类别：', kmeans.predict([b]))
#
# for i in range(100):
#     dataSize = random.randint(500, 5000)
#     c = random.randint(0, 1000)
#     a = [dataSize, dataSize * random.randint(10, 15), random.randint(300, 6000), random.randint(1, 100),
#          int(c * random.random()), c]
#     print('a=', a, ',类别：', (kmeans.predict([a])))
#
# for i in x:
#     print('i=', i, ',类别：', (kmeans.predict([i])))

result = kmeans.predict(predictData)

count_of_class = {}

for i, k in zip(result, y):
    if i in count_of_class:
        count_of_class[i][k] += 1
    else:
        count_of_class[i] = [0] * 3
        count_of_class[i][k] = 1
print(count_of_class)

s = [0] * 10
count = [0] * 10
for i, j in zip(result, data):
    s[i] += j[2]
    count[i] += 1

reward = [a / b for a, b in zip(s, count)]

print(s)
print(count)
print(reward)
