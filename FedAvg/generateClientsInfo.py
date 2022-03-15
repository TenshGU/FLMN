import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-p', '--path', type=str, default='clientsData.txt',
                    help='save the random data to the file of this path')


def judgeClass(factors):
    nw_delay_limit1 = random.randint(3500, 4000)
    nw_delay_limit2 = random.randint(1200, 3499)

    off_line_factor_limit1 = random.uniform(0.6, 0.65)
    off_line_factor_limit2 = random.uniform(0.65, 0.92)

    calPower_limit1 = random.randint(15, 18)

    # factor[3]电量，factor[2]离线比例，factor[1]网络延迟，factor[0]算力因子
    if factors[2] < off_line_factor_limit1:
        return 2

    if factors[1] > nw_delay_limit1:
        return 2

    if factors[0] > calPower_limit1:
        return 2

    if off_line_factor_limit1 <= factors[2] < off_line_factor_limit2:
        return 1

    if nw_delay_limit2 < factors[1] <= nw_delay_limit1:
        return 1

    if 10 <= factors[0] < calPower_limit1:
        return 1

    if factors[3] <= 20:
        return 1
    return 0


def transform2Str(x, y):
    s = ''
    n = 0
    while n < len(x):
        s += str(x[n])
        n += 1
        s += " "
    s += str(y)
    s += "\n"
    return s


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__  # 得到参数字典

    x = []
    y = []
    a, b, c = 0, 0, 0

    with open(args['path'], mode='w') as fileObj:
        for i in range(2000):
            data_size = random.randint(200, 3000)
            # 500 -> 7s，假定平均1s处理100数据
            calPower = random.randint(data_size * 4, data_size * 20)

            nw_delay = random.randint(20, 6000)

            power = random.randint(1, 100)

            chosen_times = random.randint(0, 1000)

            off_line_factor = 1
            hardWork = random.randint(1, 100)
            if hardWork <= 55:
                # 全勤
                train_times = chosen_times
            elif 55 < hardWork <= 80:
                off_line_factor = random.uniform(0.65, 0.99)
                train_times = int(chosen_times * off_line_factor)
            else:
                off_line_factor = random.uniform(0, 0.65)
                train_times = int(chosen_times * off_line_factor)
            data = [data_size, calPower, nw_delay, power, train_times, chosen_times]

            factors = [int(data_size / calPower), nw_delay, off_line_factor, power]
            classification = judgeClass(factors)
            y.append(classification)

            if classification == 0:
                a += 1
            elif classification == 1:
                b += 1
            else:
                c += 1

            x.append(data)
            #print(transform2Str(data, classification))
            s = transform2Str(data, classification)
            fileObj.write(s)

    # x = np.array(x)
    print('A共有{}个'.format(a), 'B共有{}个'.format(b), 'C共有{}个'.format(c))
    print('数据量——算力——网络延迟——电量——训练次数——选中次数')
    print(x)
    print(y)
    print('保存数据在{}下'.format(args['path']))