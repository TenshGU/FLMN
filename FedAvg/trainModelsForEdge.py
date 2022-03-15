import argparse
from sklearn.neighbors import KNeighborsClassifier
import getClientsData as gcd
from sklearn.cluster import KMeans
import joblib

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-p', '--path', type=str, default='clientsData.txt',
                    help='read the random data to the file of this path')


def trainAndSaveKmeans(x):
    kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(x)

    joblib.dump(kmeans, 'saved_model/kmeans.pkl')


def trainAndSaveKNN(x, y):
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(x, y)

    joblib.dump(knn, 'saved_model/knn.pkl')


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__  # 得到参数字典

    x, y, weights = gcd.file2matrix(args['path'])
    trainAndSaveKmeans(x)
    trainAndSaveKNN(x, y)

    print('训练和保存完毕')
