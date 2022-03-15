import argparse
import matplotlib.pyplot as plt
import getClientsData as gcd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-p', '--path', type=str, default='clientsData.txt',
                    help='read the random data to the file of this path')

if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__  # 得到参数字典

    x, y = gcd.file2matrix(args['path'])

    print(type(x), type(y))
    print(x)
    print(y)

    ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='*', c=y)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    plt.show()