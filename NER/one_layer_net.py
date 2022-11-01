import numpy as np


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
    y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return y


def cross_entropy_error(y, label):
    size = y.shape[0]
    delta = 1e-7
    loss = -np.sum(np.log(y[np.arange(size), label.astype(int)] + delta)) / size  # 防止负无穷大
    return loss


# 找到所有实体，标出开始和结束的位置，进行对比
def find_entity(y):
    entity_start = 0
    entity_end = 0
    entity_label = list()
    for i in range(y.shape[0]):
        if y[i] != 6:
            if y[i] == 0 or y[i] == 2 or y[i] == 4:
                if entity_end != 0:
                    entity_label.append([entity_start, entity_end])
                    # print(":",t[entity_start],t[entity_end],":")
                entity_start = i
                entity_end = i
            elif (y[i] == 1 and (y[i - 1] == 0 or y[i - 1] == 1)) \
                    or (y[i] == 3 and (y[i - 1] == 2 or y[i - 1] == 3)) \
                    or (y[i] == 5 and (y[i - 1] == 4 or y[i - 1] == 5)):
                entity_end = i
            elif y[i] == 1 or y[i] == 3 or y[i] == 5:  # 对于预测错的结果中单独出现的I
                if entity_end != 0:
                    entity_label.append([entity_start, entity_end])
                entity_start = i
                entity_end = i
    entity_label.append([entity_start, entity_end])
    # print("entity_y:",len(entity_label))
    return entity_label

#检查找到的实体，计算实际的precision/recall
def check_accuracy(entity_label, y, t):
    current = 0
    entity_total = len(entity_label)
    entity_TP = 0
    i = 0
    while i < t.shape[0]:
        if i == entity_label[current][0]:
            flag = True
            while i <= entity_label[current][1]:
                if y[i] != t[i]:
                    flag = False
                    break
                i += 1
            if flag:
                i = entity_label[current][1]
                if i == t.shape[0] - 1 or t[i + 1] == 6 or (((t[i] == 0 or t[i] == 1) and t[i + 1] != 1) or
                                                            ((t[i] == 2 or t[i] == 3) and t[i + 1] != 3) or
                                                            ((t[i] == 4 or t[i] == 5) and t[i + 1] != 5)):
                    entity_TP += 1
            i = entity_label[current][1] + 1
            if current < len(entity_label) - 1:
                current += 1
            else:
                i += 1  # 如果不加的话，会一直陷入死循环
        else:
            i += 1
    # print(entity_TP)
    precision = entity_TP / float(entity_total)
    return precision


class SoftmaxWithLoss:
    def __init__(self, W):
        self.W = W  # (1500, 7)
        self.x = None
        self.y = None  # softmax输出
        self.t = None  # 标签
        self.loss = None  # 损失
        self.dW = None
        self.best_accuracy = 0

    def forward(self):
        # Affine
        out = np.dot(self.x, self.W)  # (size, 7)
        # Softmax
        self.y = softmax(out)  # (size, 7)
        # loss
        self.loss = cross_entropy_error(self.y, self.t)
        # return self.loss

    def backward(self):
        # 手动求导
        size = self.t.shape[0]
        dOut = self.y.copy()
        dOut[np.arange(size), self.t.astype(int)] -= 1
        self.dW = np.dot(self.x.T, dOut) / size
        # return self.dW

    def gradient(self, x, t):
        self.x = x  # (size,1500)
        self.t = t  # (size, 7)
        self.forward()
        self.backward()
        return self.dW

    def accuracy(self, x, t):
        self.gradient(x, t)
        y = np.argmax(self.y, axis=1).astype(int)
        t = t.astype(int)
        # print("y:", y.shape, "t:", self.t.shape)
        # 求查准率，首先找出y中预测的实体
        entity_label = find_entity(y)
        # 查看y的预测在t中是否正确
        precision = check_accuracy(entity_label, y, t)
        print("Precision:", precision)
        # 查全率，首先找出所有实际的实体
        entity_label = find_entity(t)
        # 查看t的实体在y中是否正确
        recall = check_accuracy(entity_label, t, y)
        print("Recall:", recall)
        if precision == 0 and recall == 0:
            accuracy = 0
        else:
            accuracy = 2 * precision * recall / float(precision + recall)
        print("F1-measure:", accuracy)
        return accuracy
