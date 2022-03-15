import random
import math

import joblib
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from model.WideResNet import WideResNet
from clients import ClientsGroup

# 每一个边
class Edge(object):
    def __init__(self, args, dataSetName, isIID, dev, com_round):
        self.args = args
        self.com_round = com_round
        #决定使用什么模型及其参数
        self.net = None
        #设备
        self.dev = dev
        self.loss_func = None
        self.opti = None

        #加载模型
        self.kmeans = None
        self.knn = None

        #创建缓存，缓存客户端提前训练的
        self.paramCache = {}
        #状态记录字典
        self.stateDict = {'A':[], 'B':[], 'C':[]}
        #顺序数组
        #创建bitmap
        self.blackList = None
        #创建离线记录数组
        self.recordArr = np.zeros((self.args['num_of_clients_edge'], 2), dtype=np.int, order='C')

        #Q-table
        self.q_table = np.matrix()
        #reward-table
        self.reward

        #智能协同设备群
        self.noc = 0 #num of coordClients
        self.coordClients = {}

        self.readyNN()
        # 创建端群
        self.clientsGroup = ClientsGroup(self.args['off_line_factor'], dataSetName, isIID, self.args['num_of_clients_edge'], self.dev)
        self.init()
        self.preheat()

    #硬件准备
    def readyNN(self):
        # 初始化模型
        net = None
        # mnist_2nn
        if self.args['model_name'] == 'mnist_2nn':
            net = Mnist_2NN()
        # mnist_cnn
        elif self.args['model_name'] == 'mnist_cnn':
            net = Mnist_CNN()
        # ResNet网络
        elif self.args['model_name'] == 'wideResNet':
            net = WideResNet(depth=28, num_classes=10).to(self.dev)

        ## 如果有多个GPU就使用并行
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = torch.nn.DataParallel(net)

        # 将Tenor 张量 放在 GPU/CPU上
        self.net = net.to(self.dev)

        '''
            回头直接放在模型内部
        '''
        # 定义损失函数
        self.loss_func = F.cross_entropy
        # 优化算法的，随机梯度下降法
        # 使用Adam下降法
        self.opti = optim.Adam(net.parameters(), lr=self.args['learning_rate'])

    # 预热，探测参与意愿，仅仅限制10的3次方数量级
    def preheat(self):
        # 全员训练，摸底
        num_in_comm = self.args['num_of_clients_edge']

        # 预热
        print("before FL trainning, edge server preheating all client")

        globalParam = {}

        clients_in_comm = ['client{}'.format(i) for i in num_in_comm]

        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数
        # 这里的clients
        for clientInGroup in tqdm(clients_in_comm):
            # 获取当前Client训练得到的参数，但是模拟设备退出(离线)的情况，概率1/off_line_factor
            # 这一行代码表示Client端的训练函数，我们详细展开：
            # local_parameters 得到客户端的局部变量
            result_dict = self.clientsGroup.clients_set[clientInGroup].localUpdate(self.args['epoch'],
                                                                                   self.args['batchsize'], self.net,
                                                                                   self.loss_func,
                                                                                   self.opti, globalParam)
            if result_dict is None:
                # 离线
                self.stateDict['C'].append()
            else:
                local_parameters = result_dict['stateDict']
                self.saveClientState(result_dict['clientInfo'])


    # 初始化
    def init(self):
        self.noc = random.randint(1, self.args['num_of_coordination'])
        buffer = []
        # 随机选取连续的客户端
        for i in range(self.noc):
            soc = random.randint(3,self.args['size_of_coordination'])
            n = random.randint(1, self.args['num_of_clients_edge'])

            coordi = []
            j = 0
            while j < soc:
                if n not in buffer:
                    buffer.append(n)
                    coordi.append(n)
                    j += 1
                n = (n + 1) % self.args['num_of_clients_edge']
            self.coordClients['coord{}'.format(i)] = coordi

        print('智能协同设备群序列为：{}'.format(self.coordClients))

        #加载模型
        self.kmeans = joblib.load('saved_model/kmeans.pkl')
        self.knn = joblib.load('saved_model/knn.pkl')

    #q-leanring
    def qLearning(self):
        self.kmeans

    # 启用缓存
    def spareFill(self):


    # 收集数据记录客户端设备状态
    def saveClientState(self, clientState):


    # 可用客户端
    def activeClient(self):


    # 工作窃取：定时器：清除那些长期离开的客户端
    def edgeWork(self):


    # 开始训练，返回层聚合的模型
    def trainClientsGroup(self, globalParam):
        # 每次随机选取10个Clients + 3个备用Clients
        num_in_comm = int(max(math.ceil(self.args['num_of_clients_edge'] * self.args['cfraction']), 1))

        #获得全局参数继续训练
        #self.net.load_state_dict(globalParam, strict=True)

        # 得到边的参数
        edgeParameters = {}
        # net.state_dict()  # 获取模型参数以共享

        # 得到每一层中全连接层中的名称fc1.weight
        # 以及权重weights(tenor)
        # 得到网络每一层上
        for key, var in self.net.state_dict().items():
            # print("key:"+str(key)+",var:"+str(var))
            print("张量的维度:" + str(var.shape))
            print("张量的Size" + str(var.size()))
            edgeParameters[key] = var.clone()

        # 第i轮通讯
        print("communicate round {}".format(self.com_round))

        # 对随机选的将100个客户端进行随机排序
        order = np.random.permutation(self.args['num_of_clients_edge'])
        print("order:")
        print(len(order))
        print(order)
        clientsNum = order[0:num_in_comm]
        # 生成当前选择的客户端
        clients_in_comm = ['client{}'.format(i) for i in clientsNum]

        print("客户端" + str(clients_in_comm))
        print(type(clients_in_comm))  # <class 'list'>

        sum_parameters = None
        # 每个Client基于当前模型参数和自己的数据训练并更新模型
        # 返回每个Client更新后的参数
        '''
            进度条
            import time
            import tqdm
            # 方法1
            # tqdm(list)方法可以传入任意list，如数组
            for i in tqdm.tqdm(range(100)):
                time.sleep(0.5)
                pass
            # 或 string的数组
            for char in tqdm.tqdm(['a','n','c','d']):
                time.sleep(0.5)
                pass
        '''
        # 这里的clients
        for clientNum in tqdm(clientsNum):
            clientInGroup = 'client{}'.format(clientNum)
            client = self.clientsGroup.clients_set[clientInGroup]
            # 记录抽取到的次数加1
            self.recordArr[clientNum-1][1] += 1
            # 获取当前Client训练得到的参数，但是模拟设备退出(离线)的情况，概率1/off_line_factor
            # 这一行代码表示Client端的训练函数，我们详细展开：
            # local_parameters 得到客户端的局部变量
            result_dict = client.localUpdate(self.args['epoch'], self.args['batchsize'],self.net,self.loss_func, self.opti,globalParam)
            if result_dict is None:
                #离线
                #1.记录该客户端离线次数,更新reward
                #2.快速启用缓存，若本次训练达到缓存不足阈值，加入客户端进行训练
                #3.记录该客户端离线次数，和已经选择训练的次数
                self.recordArr[clientNum-1][0] += 1
                local_parameters = self.spareFill()
                #本次通信缓存用完，启动活跃的客户端
                if local_parameters is None:
                    clients_in_comm.append(self.activeClient())
                    continue
            else:
                local_parameters = result_dict['stateDict']
                self.saveClientState(result_dict['clientInfo'])

            # 聚合对所有的Client返回的参数累加（最后取平均值）
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 取平均值，得到本次通信中边节点Edge得到的更新后的模型参数
        for var in edgeParameters:
            edgeParameters[var] = (sum_parameters[var] / num_in_comm)
        return edgeParameters


#所有边
class EdgesGroup(object):
    def __init__(self, args, dataSetName, isIID, dev, com_round):
        self.args = args
        self.dataSetName = dataSetName
        self.isIID = isIID
        self.dev = dev
        self.com_round = com_round
        self.edges_set = {}

        self.createEdge()

    def createEdge(self):
        for i in range(self.args['num_of_edge']):
            self.edges_set['edge{}'.format(i)] = Edge(self.args, self.dataSetName, self.isIID, self.dev, self.com_round)