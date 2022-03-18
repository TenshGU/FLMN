import random
import joblib
import numpy as np
from sklearn import preprocessing
import getClientsData as gcd


def judgeConvergence(q, error):
    for a in range(q.shape[0]):
        if a + 1 < q.shape[0]:
            for b in range(q[a].shape[1]):
                if abs(q[a, b] - q[a + 1, b]) >= error:
                    return False
    return True


class QLearning:
    def __init__(self, stateNum, actionNum, reward, knn, buckets, startState=0, learning_rate=0.5, gamma=0.6, epsilon=0.6,
                 episode=1000, error=0.01, maxProb=0.6, topk=6):
        self.q = np.matrix(np.zeros([stateNum, actionNum]))
        self.reward = reward
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode = episode
        self.error = error
        self.maxProb = maxProb
        self.topk = topk

        self.knn = knn
        # 数据源，装有原始客户端数据的n个桶、这些客户端的标签y
        self.buckets = buckets
        self.startState = startState

    def fix(self):
        print('begin to fix,the reward array is:')
        print(self.reward)

        # random y for start state
        state = self.startState

        rounds = 0

        # the main training loop
        for episode in range(self.episode):
            # Choose an action a in the current world state (s)
            # First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)

            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > self.epsilon:
                action = np.argmax(self.q[state, :])
            # Else doing a random choice --> exploration
            else:
                action = random.randint(0, len(self.reward) - 1)

            # Take the action (a) and observe the outcome state(s') and reward (r)
            b = self.buckets[action]
            nextClient = b[random.randint(0, len(b) - 1)]

            new_state = self.knn.predict([nextClient])
            reward = self.reward[action]

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            self.q[state, action] = self.q[state, action] + self.learning_rate * \
                                    (reward + self.gamma * np.max(self.q[new_state, :]) - self.q[state, action])

            # Our new state is state
            state = new_state

            rounds += 1

            if judgeConvergence(self.q, self.error):
                break

            # Display training progress
            # if episode % 10 == 0 or episode == self.episode-1:
            #     print("------------------------------------------------")
            #     print('the next state is state', state)
            #     print("Training episode: %d" % episode)
            #     print(self.q)
        print('>>> finished train!', rounds, 'rounds')
        print("-----------the final q-table as follow--------------")
        print(self.q)

    def nextSelection(self, state):
        rank = np.argsort(-self.q[state, :])
        e = random.uniform(0, 1)
        if e > self.maxProb:
            action = np.argmax(self.q[state, :])
        # Else doing a random choice --> exploration
        else:
            index = random.randint(0, self.topk-1)
            while self.q[0, index] < 0:
                index = random.randint(0, self.topk - 1)
            action = rank[0, index]
        return action


if __name__ == "__main__":
    x, y, weights = gcd.file2matrixWithLabel("clientsData.txt")
    knn = joblib.load('saved_model/knn.pkl')
    kmeans = joblib.load('saved_model/kmeans.pkl')

    reward = gcd.file2matrix('classReward.txt', 10)[0]

    data = []
    for a, b in zip(x, y):
        d = [a[1] / a[0], a[2], a[4] / a[5] if b < 2 else (a[4] / a[5]) - 1]
        data.append(d)
    data = preprocessing.scale(data)
    result = kmeans.predict(data)

    buckets = {}
    for a, b in zip(x, result):
        if b not in buckets:
            buckets[b] = []
        buckets[b].append(a)

    count_of_class = {}

    for i, k in zip(result, y):
        if i in count_of_class:
            count_of_class[i][k] += 1
        else:
            count_of_class[i] = [0] * 3
            count_of_class[i][k] = 1

    ql = QLearning(3, 10, reward, knn, buckets)
    ql.fix()

    #
    # data = []
    # for a, b in zip(x, y):
    #     d = [a[1] / a[0], a[2], a[4] / a[5] if b < 2 else (a[4] / a[5]) - 1]
    #     data.append(d)
    # result = kmeans.predict(data)
    #
    # buckets = {}
    # for a, b in zip(x, result):
    #   if b not in buckets:
    #       buckets[b] = []
    #   buckets[b].append(a)
    #
    index = random.randint(0, len(x) - 1)
    nextState = y[index]
    nextChosen = 0
    a2, b2, c2 = 0, 0, 0
    for i in range(1000):
        nextChosen = ql.nextSelection(nextState)
        b = buckets[nextChosen]
        nextClient = b[random.randint(0, len(b) - 1)]
        nextState = knn.predict([nextClient])
        if nextState == 0:
            a2 += 1
        elif nextState == 1:
            b2 += 1
        else:
            c2 += 1
    print(count_of_class[nextChosen])
    print(a2, b2, c2)
