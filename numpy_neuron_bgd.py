from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import time
import numpy as np

mnist = fetch_mldata("MNIST original", data_home="./")
# (1, 784)
X = np.array(mnist['data'])
y = np.array(mnist['target'])
np.warnings.filterwarnings('ignore')

class NeuronNet:
    def __init__(self, input_nodes=784,
                 hidden_nodes=200,
                 output_nodes=10):
        # (784, 500)
        self.w_i_h = np.random.normal(0, 1, size=(input_nodes, hidden_nodes))
        # (100, 10)
        self.w_h_o = np.random.normal(0, 1, size=(hidden_nodes, output_nodes))

    def Relu(self, inputs):
        inputs[inputs < 0] = 0
        return inputs

    def Sigmoid(self, inputs):
        return 1 / (1 + np.exp(-1 * inputs))

    def input_op(self, inputs):
        """
        输入层的操作
        :param inputs: 输入层的输入
        :return: 输入层的输出
        """
        return inputs

    def hidden_op(self, output_i):
        '''
        隐藏层的操作
        :param inputs: 隐藏层的输入
        :return: 隐藏层的输出
        '''
        inputs_h = output_i.dot(self.w_i_h)
        hidden_output = self.Relu(inputs_h)
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
        output_h = self.hidden_op(output_i)
        output_o = self.output_op(output_h)
        return output_i, output_h, output_o

    # 计算梯度
    def gd(self, inputs, y_labels, eta, output_i, output_h, output_o):
        # 求出每个样本标签的one hot编码
        y_one_hot = np.array([[ele == y_label for ele in range(10)] for y_label in y_labels], dtype=int)
        # 获取之前的值，方便之后偏导的计算
        dw_h_o = (output_o - y_one_hot).T.dot(output_h).T
        dw_i_h = ((output_o - y_one_hot).dot(self.w_h_o.T) * (output_i.dot(self.w_i_h) > 0)).T.dot(output_i).T

        self.w_h_o -= eta * dw_h_o
        self.w_i_h -= eta * dw_i_h

    def softmax(self, x):
        """
        对输入x的每一行计算softmax。

        该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

        代码利用softmax函数的性质: softmax(x) = softmax(x + c)

        参数:
        x -- 一个N维向量，或者M x N维numpy矩阵.

        返回值:
        x -- 在函数内部处理后的x
        """
        x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return x

    def train(self, inputs, y_labels, eta):
        start_time_origin = time.clock()
        input_origin = inputs
        output_i, output_h, output_o = self.get_io_data(inputs)
        step = 1
        # 通过对训练前和训练后的损失函数进行比较，当差值小于一定值时，完成梯度下降
        while True:
            start_time = time.clock()
            self.gd(inputs, y_labels, eta, output_i, output_h, output_o)
            print("gd time:{0}".format(time.clock() - start_time))
            output_i, output_h, output_o = self.get_io_data(inputs)
            step += 1
            print("train one time: {0}".format(time.clock() - start_time_origin))
            print(output_o)
            if step == 2000:
                # print("step = {0}, loss={1}, loss_diff={2}".format(step, loss_after, loss_after - loss_origin))
                print("训练结束")
                break
            # loss_origin = loss_after

    def predict(self, inputs):
        res = []
        output_i, output_h, output_o = self.get_io_data(inputs)
        for output in output_o:
            res.append(np.argmax(output))

        return res

    def score(self, inputs, y_labels):
        y_outputs = self.predict(inputs)
        comparison = [y_output == y_label for y_output, y_label in zip(y_outputs, y_labels)]
        return sum(comparison) / len(comparison)


# def train_test_split(inputs, y_labels, test_size=0.3):
#     index = np.arange(len(inputs))
#     np.random.shuffle(index)
#     X_train = inputs[:int(len(inputs) * test_size)]
#     X_test = inputs[int(len(inputs) * test_size) + 1:]
#     y_train = y_labels[:int(len(inputs) * test_size)]
#     y_test = y_labels[int(len(inputs) * test_size) + 1:]
#     return X_train, X_test, y_train, y_test

nn = NeuronNet()
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('../dataset/fashion')
(inputs, y) = data.train.next_batch(1000000)
# y = mnist['target']
inputs = (inputs / 255 * 0.99) + 0.01
# 将数据集分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(inputs, y, test_size=0.3)
start_time = time.clock()
nn.train(X_train, y_train, 0.00005)
score = nn.score(X_test, y_test)
print(score)
print("time={0}".format(time.clock() - start_time))


