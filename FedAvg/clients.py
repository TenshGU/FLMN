import random
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet

class Client(object):
    def __init__(self, off_line_factor, trainDataSet, dev, serialNum, power, dataSize):
        self.off_line_factor = off_line_factor
        self.train_ds = trainDataSet
        self.dev = dev
        self.clientInfo = {'serialNum': serialNum, 'calPower': 0,
                           'power': power, 'nwDelay': 6000,
                           'dataSize': dataSize}
        self.train_dl = None
        self.local_parameters = None

    # 模拟设备训练
    def update(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回字典，包括有客户端参数，当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载当前通信中最新全局参数
        # 传入网络模型，并加载global_parameters参数的
        Net.load_state_dict(global_parameters, strict=True)
        # 载入Client自有数据集
        # 加载本地数据
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        # 设置迭代次数
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
                loss = lossFun(preds, label)
                # 反向传播
                loss.backward()
                # 计算梯度，并更新梯度
                opti.step()
                # 将梯度归零，初始化梯度
                opti.zero_grad()
        # 返回当前Client基于自己的数据训练得到的新的模型参数
        return Net.state_dict()

    # 本地更新，涉及到设备参数以及实际因素影响
    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        # 模拟训练中突发离线/电量耗尽关机等情况：延时0~10秒
        if random.randint(1, self.off_line_factor) == 5:
            time.sleep(random.randint(0, 10))
            return None
        else:
            start_time = time.time()
            state_dict = self.update(localEpoch, localBatchSize, Net, lossFun, opti, global_parameters)
            end_time = time.time()

            # 电量衰减 每次训练导致电量降低0.5~3.5电量
            self.clientInfo['power'] -= random.uniform(0.5, 3.5)
            # 模拟网络延迟：20ms ~ 6000ms延迟
            nwDelay = random.randint(20, 6000)
            self.clientInfo['nwDelay'] = nwDelay

            # 如果设备电量衰减到低电量，模拟算力下降（延迟0.5-1s）
            cal_reduce = random.randint(5, 10) * 100
            # 算力根据实际代码运行时间，越小越强
            self.clientInfo['calPower'] = int((end_time - start_time) * 1000) + cal_reduce

            # 模拟通信延迟
            if self.clientInfo['power'] < 20:
                time.sleep((nwDelay + cal_reduce) / 1000)
            else:
                time.sleep(nwDelay / 1000)

            info = [self.clientInfo['dataSize'], self.clientInfo['calPower'], self.clientInfo['nwDelay'], self.clientInfo['power']]
            return {'clientInfo': info, 'stateDict': state_dict}

    def local_val(self):
        pass

class ClientsGroup(object):
    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端

    '''

    def __init__(self, off_line_factor, dataSetName, isIID, numOfClients, dev):
        self.off_line_factor = off_line_factor
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        # 得到已经被重新分配的数据
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        # 加载测试数据
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        '''
            然后将其划分为200组大小为300的数据切片,然后分给每个Client两个切片
        '''

        # 60000 /100 = 600/2 = 300
        shard_size = mnistDataSet.train_data_size / self.num_of_clients / 2
        # print("shard_size:"+str(shard_size))

        # np.random.permutation 将序列进行随机排序
        # np.random.permutation(60000/300=200)
        shards_id = np.random.permutation(mnistDataSet.train_data_size / shard_size)
        # 一共200个
        print("*" * 100)
        print(shards_id)
        print(shards_id.shape)
        print("*" * 100)
        for i in range(self.num_of_clients):
            ## shards_id1
            ## shards_id2
            ## 是所有被分得的两块数据切片
            # 0 2 4 6...... 偶数
            shards_id1 = shards_id[i * 2]
            # 0+1 = 1 2+1 = 3 .... 奇数
            shards_id2 = shards_id[i * 2 + 1]
            #
            # 例如shard_id1 = 10
            # 10* 300 : 10*300+300
            # 将数据以及的标签分配给该客户端
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            #
            # np.vstack 是按照垂直方向堆叠
            # np.hstack: 按水平方向（列顺序）堆叠数组构成一个新的数组
            '''
                In[4]:
                a = np.array([[1,2,3]])
                a.shape
                # (1, 3)
                
                In [5]:
                b = np.array([[4,5,6]])
                b.shape             
                # (1, 3)
                
                In [6]:
                c = np.vstack((a,b)) # 将两个（1,3）形状的数组按垂直方向叠加
                print(c)
                c.shape # 输出形状为（2,3）
                [[1 2 3]
                 [4 5 6]]
                # (2, 3)
            
            '''

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            # 创建一个客户端
            someone = Client(self.off_line_factor, TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev, i,
                             random.randint(15, 100))
            # 为每一个clients 设置一个名字
            # client10
            self.clients_set['client{}'.format(i)] = someone


if __name__ == "__main__":
    MyClients = ClientsGroup('mnist', True, 100, 0)
    print("finish building ClientsGroup:")
    train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    print(train_ids)
    i = 0
    for x_train in train_ids[0]:
        print("client10 数据:" + str(i))
        print(x_train)
        i = i + 1
    print(MyClients.clients_set['client11'].train_ds[400:500])
