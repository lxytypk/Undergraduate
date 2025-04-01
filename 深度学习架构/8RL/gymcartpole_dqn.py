# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, alpha=0.01, alpha_decay=0.01, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000) #记忆体使用队列实现，队列满后根据插入顺序自动删除老数据
        self.env = gym.make('CartPole-v0') #环境
        if monitor: self.env = gym.wrappers.Monitor(self.env, './data/cartpole-1', force=True)
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon #探索率
        self.epsilon_min = epsilon_min #探索率衰减到此值以下时停止
        self.epsilon_decay = epsilon_log_decay #探索率衰减因子
        self.alpha = alpha
        self.alpha_decay = alpha_decay #学习率衰减因子
        self.n_episodes = n_episodes #训练的总次数
        self.n_win_ticks = n_win_ticks #连续胜利的次数
        self.batch_size = batch_size #每次训练时的样本批大小
        self.quiet = quiet
        self.filepath="model-epoch-{}.hdf5" #模型保存的文件路径
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        #Neural Net for Deep-Q learning Model
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=4, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha, decay=self.alpha_decay)) # 指定损失函数以及优化器

    # 在记忆体（经验回放池）中保存具体某一时刻的当前状态信息
    #存储在memory
    def remember(self, state, action, reward, next_state, done):
        #当前状态、动作、奖励、下一个状态、是否结束
        self.memory.append((state, action, reward, next_state, done))

    #根据当前状态选择一个动作
    #在训练初期，探索率较高，agent更多地进行探索；而随着训练的进行，探索率逐渐减小，agent更多地利用已知信息
    def choose_action(self, state, epsilon):
        '''贪婪策略'''
        #如果一个随机数小于等于探索率epsilon，则随机选择一个动作，即从动作空间中随机采样一个动作。
        #否则，使用当前神经网络模型（self.model）对当前状态进行预测，然后选择预测值最大的动作作为输出。这是一种基于贪婪策略（greedy policy）的行为，即选择对应预测值最大的动作，以最大化预期奖励
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    #调整探索率
    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    #对状态进行预处理
    #将原始状态转换成神经网络能够接受的输入格式
    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    #执行经验回放
    #batch_size是每次从经验池中抽取的样本数量
    def replay(self, batch_size):
        #存储输入和目标输出
        x_batch, y_batch = [], []
        #从经验池（self.memory）中随机抽取的一批样本，数量为batch_size和经验池大小的较小值
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        #对于抽取的每一个样本操作
        for state, action, reward, next_state, done in minibatch:
            #神经网络模型对当前状态（state）进行预测，得到预测值
            y_target = self.model.predict(state)
            #根据当前的奖励（reward）、是否终止（done）以及下一个状态（next_state）的最大预测值，更新目标值中对应动作的预测值
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        #训练，更新神经网络模型的参数
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        #当前的探索率epsilon大于最小探索率epsilon_min，则将探索率进行衰减，以控制探索率随训练进行逐渐减小
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            #将环境重置为初始状态，并对状态进行预处理
            state = self.preprocess_state(self.env.reset())
            done = False
            i = 0
            while not done:
                #根据当前状态和一个随训练次数变化的ε值选择一个动作
                action = self.choose_action(state, self.get_epsilon(e))
                #执行选定的动作，并获取下一个状态、奖励以及是否完成的标志
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                #将经验存储到记忆缓冲区中
                self.remember(state, action, reward, next_state, done)
                #更新当前状态
                state = next_state
                #该轮次的步数
                i += 1

            scores.append(i)
            #计算最近100个轮次的平均得分
            mean_score = np.mean(scores)
            #如果平均得分超过指定的阈值，并且已经运行了足够多的轮次，则任务被视为已解决
            if mean_score >= self.n_win_ticks and e >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return e - 100
            if e % 100 == 0 and not self.quiet:
                self.model.save(self.filepath.format(e))
                print('[Episode {}] - Mean survival time {}'.format(e, mean_score))

            #使用记忆缓冲区中的经验进行训练
            self.replay(self.batch_size)

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e

if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()