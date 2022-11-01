import numpy as np
import one_layer_net

# 读文件
x_train = np.load('data/x_train.npy', allow_pickle=True)
t_train = np.load('data/t_train.npy', allow_pickle=True)
x_dev = np.load('data/x_test.npy', allow_pickle=True)
t_dev = np.load('data/t_test.npy', allow_pickle=True)
x_test = np.load('data/x_final.npy', allow_pickle=True)
t_test = np.load('data/t_final.npy', allow_pickle=True)
print("Input size:",t_train.shape,t_dev.shape,t_test.shape)

# 初始化
shape_input = 1000 * 3
weight_init = 0.01
W = weight_init * np.random.randn(shape_input, 7)
layer = one_layer_net.SoftmaxWithLoss(W)
iter_num = 40000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 2
iter_epoch = 100

train_loss = list()
train_acc = list()  
dev_acc = list()
test_acc = list()

w_best = np.zeros((shape_input,7))
for i in range(iter_num):
    batch_position = np.random.choice(train_size,batch_size)
    x_input = x_train[batch_position]
    t_input = t_train[batch_position]
    grad = layer.gradient(x_input, t_input) # 计算损失函数的导数
    if i % iter_epoch == 0: 
        train_loss.append(layer.loss) #存储损失函数的大小

        # 输出准确率
        print("##############")
        print("Training set:")
        train = layer.accuracy(x_train, t_train)
        train_acc.append(train)
        print("##############")
        print("Dev set:")
        dev = layer.accuracy(x_dev, t_dev)
        dev_acc.append(dev)
        print("##############")
        print("F1-measure:","[",train,",", dev,"]")
        current_num = float(i/iter_num)
        print("Rate of process:",current_num*100,"%","loss:",layer.loss)

        #保存验证集中准确率最高的参数
        if layer.best_accuracy < dev:
            w_best = W
            layer.best_accuracy = dev
            # print(layer_best.W)
    W -= learning_rate * grad  # 对参数进行更新
    
#对结果进行存储
train_output = np.array(train_acc)
dev_output = np.array(dev_acc)
loss_output = np.array(train_loss)
np.savetxt("result/result_train.txt",train_output,fmt='%f', delimiter=' ')
np.savetxt("result/result_dev.txt",dev_output,fmt='%f', delimiter=' ')
np.savetxt("result/loss.txt",loss_output,fmt='%f', delimiter=' ')

#将迭代过程中存储的W用于测试集准确率计算
layer_best = one_layer_net.SoftmaxWithLoss(w_best)
print("##############")
print("Best Model:")
print("##############")
print("Training set:")
train = layer_best.accuracy(x_train, t_train)
print("##############")
print("Dev set:")
dev = layer_best.accuracy(x_dev, t_dev)
print("##############")
print("Test set:")
test = layer_best.accuracy(x_test, t_test)
print("##############")
print("F1-measure:")
print("[",train,",", dev,",", test,"]")
best = [train,test,dev]
np.savetxt("result/best_F1.txt",best,fmt='%f', delimiter=',')
np.savetxt("result/result_W_best.txt", layer_best.W, fmt='%f', delimiter=' ')