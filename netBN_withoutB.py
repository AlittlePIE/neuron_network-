from sklearn.model_selection import train_test_split
import time
import numpy as np

np.warnings.filterwarnings('ignore')

class NeuronNet:
    def __init__(self, input_nodes=784,
                 hidden_nodes=200,
                 output_nodes=10):
        self.w_i_h = np.random.randn(input_nodes * hidden_nodes).reshape(input_nodes, hidden_nodes)
        self.w_h_o = np.random.randn(hidden_nodes * output_nodes).reshape(hidden_nodes, output_nodes)
        self.gamma = 1
        self.beta = 0
        self.bn_params = {'bn_mean': 0, 'bn_var': 0}

    def Relu(self, inputs):
        inputs[inputs < 0] = 0
        return inputs

    def input_op(self, inputs):
        """
        输入层的操作
        :param inputs: 输入层的输入
        :return: 输入层的输出
        """
        return inputs

    def hidden_train_op(self, output_i):
        '''
        隐藏层的操作
        :param inputs: 隐藏层的输入
        :return: 隐藏层的输出
        '''
        inputs_h = output_i.dot(self.w_i_h)
        inputs_h, bn_cache, mu, var = self.batchnorm_forward(inputs_h, self.gamma, self.beta)
        hidden_output = self.Relu(inputs_h)
        return hidden_output, bn_cache, mu, var

    def hidden_op(self, output_i):
        inputs_h = output_i.dot(self.w_i_h)
        output_i = (inputs_h - self.bn_params['bn_mean']) / np.sqrt(self.bn_params['bn_var'] + 1e-8)
        input_h = self.gamma * output_i + self.beta
        hidden_output = self.Relu(input_h)
        return hidden_output

    # 输出层的操作
    def output_op(self, output_h):
        '''
        输出层的操作
        :param inputs: 输出层的输入
        :return: 输出层的输出
        '''
        input_o = np.dot(output_h, self.w_h_o)
        output = self.softmax(input_o)
        return output

    def get_io_data(self, inputs):
        output_i = self.input_op(inputs)
        output_h, bn_cache, mu, var = self.hidden_train_op(output_i)
        output_o = self.output_op(output_h)
        return output_i, output_h, bn_cache, mu, var, output_o

    def get_io_data_test(self, inputs):
        output_i = self.input_op(inputs)
        output_h = self.hidden_op(output_i)
        output_o = self.output_op(output_h)
        return output_i, output_h, output_o

    # 计算梯度
    def gd(self, inputs, y_labels, eta, output_i, output_h, output_o, cache):
        # 求出每个样本标签的one hot编码
        y_one_hot = np.array([[ele == y_label for ele in range(10)] for y_label in y_labels], dtype=int)
        # 获取之前的值，方便之后偏导的计算
        dX, dgamma, dbeta = self.batchnorm_backward(((output_o - y_one_hot).dot(self.w_h_o.T) * (output_i.dot(self.w_i_h) > 0)), cache)
        dw_h_o = (output_o - y_one_hot).T.dot(output_h).T
        dw_i_h = dX.T.dot(output_i).T

        self.w_h_o -= eta * dw_h_o
        self.w_i_h -= eta * dw_i_h
        self.gamma -= eta * dgamma
        self.beta -= eta * dbeta

    def batchnorm_backward(self, dout, cache):
        X, X_norm, mu, var, gamma, beta = cache

        N, D = X.shape

        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + 1e-8)

        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv ** 3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)

        return dX, dgamma, dbeta

    def batchnorm_forward(self, X, gamma, beta):
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        return out, cache, mu, var

    def softmax(self, x):
        x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return x

    def mini_train(self, inputs, y_labels, eta, n_iters, batch_num):
        batch_times = len(inputs) // batch_num + 1
        step = 1
        for _ in range(n_iters):
            self.shuffle(inputs, y_labels)
            for i in range(batch_times):
                if (i + 1) == batch_times:
                    mini_batch = inputs[self.index(i, batch_num):]
                    mini_labels = y_labels[self.index(i, batch_num):]
                else:
                    mini_batch = inputs[self.index(i, batch_num):self.index(i+1, batch_num)]
                    mini_labels = y_labels[self.index(i, batch_num):self.index(i+1, batch_num)]
                y_one_hot = np.array([[ele == y_label for ele in range(10)] for y_label in mini_labels])
                output_i, output_h, bn_cache, mu, var, output_o = self.get_io_data(mini_batch)
                score_mini = self.score(mini_batch, mini_labels)
                self.bn_params['bn_mean'] = .9 * self.bn_params['bn_mean'] + .1 * mu
                self.bn_params['bn_var'] = .9 * self.bn_params['bn_var'] + .1 * var
                loss_mini = -np.sum(np.sum(y_one_hot * np.log(output_o), axis=1), axis=0) / len(mini_labels)
                self.gd(mini_batch, mini_labels, eta, output_i, output_h, output_o, bn_cache)
                print("step: {0}, score: {1}, loss_mini: {2}".format(step, score_mini, loss_mini))
                step += 1

    def index(self, x, batch_num):
        assert x >= 0
        return batch_num * x

    def predict(self, inputs):
        res = []
        output_i, output_h, output_o = self.get_io_data_test(inputs)
        for output in output_o:
            res.append(np.argmax(output))
        return res

    def score(self, inputs, y_labels):
        y_outputs = self.predict(inputs)
        comparison = [y_output == y_label for y_output, y_label in zip(y_outputs, y_labels)]
        return sum(comparison) / len(comparison)

    def shuffle(self, inputs, y_labels):
        index = np.arange(len(inputs))
        np.random.shuffle(index)
        return inputs[index], y_labels[index]


nn = NeuronNet()
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../dataset/fashion')
(inputs, y) = data.train.next_batch(1000000)
# y = mnist['target']
inputs = (inputs / 255 * 0.99) + 0.01
# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(inputs, y, test_size=0.3)
start_time = time.clock()
nn.mini_train(X_train, y_train, 0.005, 8, 256)
score = nn.score(X_test, y_test)
print("测试集的分数为：{0}".format(score))
print("time={0}".format(time.clock() - start_time))


