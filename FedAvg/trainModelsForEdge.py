import argparse
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import QL
import getClientsData as gcd
from sklearn.cluster import KMeans
import joblib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-p', '--path', type=str, default='clientsData.txt',
                    help='read the random data to the file of this path')

parser.add_argument('-rp', '--rewardPath', type=str, default='classReward.txt',
                    help='the path of classReward')


def trainAndSaveQLearning(reward, knn, buckets):
    ql = QL.QLearning(3, 10, reward, knn, buckets)
    ql.fix()
    joblib.dump(ql, 'saved_model/ql.pkl')


def transform2Str(x):
    s = ''
    n = 0
    while n < len(x):
        s += str(x[n])
        n += 1
        s += " " if n != len(x) else "\n"
    return s


def trainAndSaveKmeans(x):
    data = []
    for a, b in zip(x, y):
        d = [a[1] / a[0], a[2], a[4] / a[5] if b < 2 else (a[4] / a[5]) - 1]
        data.append(d)
    # print(data)
    predictData = preprocessing.scale(data)

    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(predictData)

    result = kmeans.predict(predictData)

    print('用于聚类的数据：')
    print(predictData)

    s = [0] * 10
    count = [0] * 10
    for i, j in zip(result, data):
        s[i] += j[2]
        count[i] += 1

    reward = [a / b for a, b in zip(s, count)]

    with open(args['rewardPath'], mode='w') as fileObj:
        fileStr = transform2Str(reward)
        fileObj.write(fileStr)
        print('reward:')
        print(fileStr)

    joblib.dump(kmeans, 'saved_model/kmeans.pkl')

    return kmeans, reward, result


def trainAndSaveKNN(x, y):
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(x, y)

    joblib.dump(knn, 'saved_model/knn.pkl')

    return knn


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__  # 得到参数字典

    x, y, weights = gcd.file2matrixWithLabel(args['path'])
    knn = trainAndSaveKNN(x, y)
    kmeans, reward, result = trainAndSaveKmeans(x)

    buckets = {}
    for a, b, c in zip(x, result, y):
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(a)

    trainAndSaveQLearning(reward, knn, buckets)

    print('训练和保存完毕')
