import numpy as np

# Q 矩阵初始化为0
q = np.matrix(np.zeros([3, 1000]))

# Reward 矩阵为提早定义好的。 相似与HMM的生成矩阵。-1表示无相链接的边
r = np.matrix()

# hyperparameter
#折扣因子
gamma = 0.8
#是否选择最后策略的几率
epsilon = 0.4
# the main training loop
for episode in range(101):
    # random initial state
    state = np.random.randint(0, 6)
    # 若是不是最终转态
    for i in range(1000):
        # 选择可能的动做
        # Even in random case, we cannot choose actions whose r[state, action] = -1.
        possible_actions = []
        possible_q = []
        for action in range(1000):
            if r[state, action] >= 0:
                possible_actions.append(action)
                possible_q.append(q[state, action])

        # Step next state, here we use epsilon-greedy algorithm.
        action = -1
        if np.random.random() < epsilon:
            # choose random action
            action = possible_actions[np.random.randint(0, len(possible_actions))]
        else:
            # greedy
            action = possible_actions[np.argmax(possible_q)]

        # Update Q value
        q[state, action] = r[state, action] + gamma * q[action].max()

        # Go to the next state
        state = action

    # Display training progress
    if episode % 10 == 0:
        print("------------------------------------------------")
        print("Training episode: %d" % episode)
        print(q)