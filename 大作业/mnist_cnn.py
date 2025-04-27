import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import pickle
import struct
import time

def load_mnist_data():
    npz_path = './data/mnist/mnist.npz'
    
    if not os.path.exists(npz_path):
        print(f"未找到文件: {npz_path}")
        return None

    print("从npz文件加载MNIST数据集...")
    with np.load(npz_path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    def _to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((labels.shape[0], num_classes))
        for i in range(labels.shape[0]):
            one_hot[i, labels[i]] = 1
        return one_hot
    
    y_train_one_hot = _to_one_hot(y_train)
    y_test_one_hot = _to_one_hot(y_test)
    
    print(f"数据加载完成: 训练集 {x_train.shape[0]} 样本, 测试集 {x_test.shape[0]} 样本")
    return (x_train, y_train_one_hot, y_train), (x_test, y_test_one_hot, y_test)


def generate_synthetic_digits(num_samples=1000):
    """生成简单的合成数字数据用于测试"""
    print("生成合成数字数据用于测试...")
    X = np.zeros((num_samples, 28, 28, 1))
    y = np.zeros(num_samples, dtype=np.int)
    
    for i in range(num_samples):
        digit = i % 10
        y[i] = digit

        if digit == 0:
            rr, cc = np.mgrid[0:28, 0:28]
            center = (14, 14)
            radius = 10
            circle = ((rr - center[0])**2 + (cc - center[1])**2) < radius**2
            X[i, :, :, 0] = circle * 1.0
        elif digit == 1:
            X[i, 5:23, 13:16, 0] = 1.0
        elif digit == 2:
            for j in range(5, 23):
                pos = 13 + int(5 * np.sin((j - 5) / 18 * np.pi))
                X[i, j, pos:pos+3, 0] = 1.0
        elif digit == 3:
            rr, cc = np.mgrid[0:28, 0:28]
            center1 = (8, 14)
            center2 = (20, 14)
            radius = 6
            circle1 = ((rr - center1[0])**2 + (cc - center1[1])**2) < radius**2
            circle2 = ((rr - center2[0])**2 + (cc - center2[1])**2) < radius**2
            X[i, :, :, 0] = (circle1 | circle2) * 1.0
        elif digit == 4:
            X[i, 5:23, 13:16, 0] = 1.0
            X[i, 13:16, 5:23, 0] = 1.0
        elif digit == 5:
            X[i, 5:23, 5:23, 0] = 1.0
            X[i, 8:20, 8:20, 0] = 0.0
        elif digit == 6:
            for j in range(5, 23):
                width = int((j - 5) / 18 * 18)
                start = 14 - width // 2
                X[i, j, start:start+width, 0] = 1.0
        elif digit == 7:
            for j in range(5, 23):
                X[i, j, j, 0] = 1.0
                X[i, j, j+1, 0] = 1.0
                X[i, j, j-1, 0] = 1.0
        elif digit == 8:
            X[i, 5:23, 13:16, 0] = 1.0
            X[i, 13:16, 5:23, 0] = 1.0
            for j in range(5, 23):
                X[i, j, j, 0] = 1.0
                X[i, j, 28-j, 0] = 1.0
        elif digit == 9:
            rr, cc = np.mgrid[0:28, 0:28]
            center = (14, 14)
            radius1 = 10
            radius2 = 5
            ring = (((rr - center[0])**2 + (cc - center[1])**2) < radius1**2) & \
                  (((rr - center[0])**2 + (cc - center[1])**2) > radius2**2)
            X[i, :, :, 0] = ring * 1.0

        X[i, :, :, 0] += np.random.normal(0, 0.1, (28, 28))
        X[i, :, :, 0] = np.clip(X[i, :, :, 0], 0, 1)

    y_one_hot = np.zeros((num_samples, 10))
    for i in range(num_samples):
        y_one_hot[i, y[i]] = 1

    train_size = int(0.8 * num_samples)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    y_train_one_hot = y_one_hot[:train_size]
    y_test_one_hot = y_one_hot[train_size:]
    
    print(f"合成数据生成完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
    return (X_train, y_train_one_hot, y_train), (X_test, y_test_one_hot, y_test)

# CNN基本组件的实现
class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重和偏置
        self.W = None
        self.b = None
        
        # 缓存输入和中间结果，用于反向传播
        self.X_col = None
        self.W_col = None
        self.X = None
        
    def initialize_params(self, input_channels):
        # He初始化
        scale = np.sqrt(2.0 / (input_channels * self.filter_size * self.filter_size))
        self.W = np.random.normal(0, scale, (self.num_filters, input_channels, self.filter_size, self.filter_size))
        self.b = np.zeros(self.num_filters)
        
    def im2col(self, X, filter_h, filter_w, stride, pad):
        """将输入数据展开为列，以便进行卷积运算"""
        N, H, W, C = X.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        
        # 添加padding
        img = np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')
        
        # 初始化输出
        col = np.zeros((N, out_h, out_w, C, filter_h, filter_w))
        
        # 填充输出
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, :, :, y, x] = img[:, y:y_max:stride, x:x_max:stride, :]
        
        # 重塑为(N*out_h*out_w, C*filter_h*filter_w)
        col = col.reshape(N * out_h * out_w, -1)
        return col
    
    def col2im(self, col, X_shape, filter_h, filter_w, stride, pad):
        """将列数据重塑回原始输入形状"""
        N, H, W, C = X_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        
        # 初始化结果
        img = np.zeros((N, H + 2 * pad, W + 2 * pad, C))
        
        # 重塑col
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
        
        # 累加到img
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, y:y_max:stride, x:x_max:stride, :] += col[:, :, :, :, y, x]
        
        # 移除padding
        return img[:, pad:H + pad, pad:W + pad, :]
    
    def forward(self, X):
        """前向传播"""
        N, H, W, C = X.shape
        
        # 如果是第一次运行，初始化参数
        if self.W is None:
            self.initialize_params(C)
        
        # 计算输出尺寸
        out_h = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # im2col处理输入
        self.X = X
        X_col = self.im2col(X, self.filter_size, self.filter_size, self.stride, self.padding)
        self.X_col = X_col
        
        # 重塑卷积核参数
        W_col = self.W.reshape(self.num_filters, -1).T
        self.W_col = W_col
        
        # 卷积操作：矩阵乘法
        out = X_col.dot(W_col) + self.b
        
        # 重塑输出
        out = out.reshape(N, out_h, out_w, self.num_filters)
        
        # 返回(N, H', W', C')格式的输出
        return out
    
    def backward(self, dout, learning_rate):
        """反向传播"""
        N, H, W, C = self.X.shape
        
        # 重塑梯度
        dout_flat = dout.reshape(-1, self.num_filters)
        
        # 计算偏置梯度
        db = np.sum(dout_flat, axis=0)
        
        # 计算权重梯度
        dW_col = self.X_col.T.dot(dout_flat)
        dW = dW_col.T.reshape(self.W.shape)
        
        # 计算输入梯度
        dX_col = dout_flat.dot(self.W_col.T)
        
        # 重塑为原始输入形状
        dX = self.col2im(dX_col, self.X.shape, self.filter_size, self.filter_size, self.stride, self.padding)
        
        # 更新参数
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX

class MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.max_idx = None
    
    def forward(self, X):
        """前向传播"""
        N, H, W, C = X.shape
        
        # 计算输出尺寸
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # 初始化输出和最大值索引
        out = np.zeros((N, out_h, out_w, C))
        self.max_idx = np.zeros((N, out_h, out_w, C, 2), dtype=int)
        
        # 池化操作
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # 提取当前区域
                X_pool = X[:, h_start:h_end, w_start:w_end, :]
                
                # 找出最大值
                for n in range(N):
                    for c in range(C):
                        # 找到该区域最大值位置
                        max_val = np.max(X_pool[n, :, :, c])
                        max_pos = np.where(X_pool[n, :, :, c] == max_val)
                        
                        # 由于可能有多个最大值，取第一个
                        if len(max_pos[0]) > 0:
                            self.max_idx[n, i, j, c] = [max_pos[0][0], max_pos[1][0]]
                        
                        # 记录最大值
                        out[n, i, j, c] = max_val
        
        self.X = X
        return out
    
    def backward(self, dout):
        """反向传播"""
        N, H, W, C = self.X.shape
        dX = np.zeros_like(self.X)
        
        # 输出梯度尺寸
        _, out_h, out_w, _ = dout.shape
        
        # 分配梯度
        for n in range(N):
            for i in range(out_h):
                for j in range(out_w):
                    for c in range(C):
                        # 获取最大值索引
                        h_idx, w_idx = self.max_idx[n, i, j, c]
                        
                        # 计算原始数据中的位置
                        h_pos = i * self.stride + h_idx
                        w_pos = j * self.stride + w_idx
                        
                        # 将梯度添加到最大值位置
                        dX[n, h_pos, w_pos, c] += dout[n, i, j, c]
        
        return dX

class Flatten:
    def __init__(self):
        self.X_shape = None
    
    def forward(self, X):
        """前向传播"""
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, dout):
        """反向传播"""
        return dout.reshape(self.X_shape)

class FullyConnected:
    def __init__(self, output_size):
        self.output_size = output_size
        self.W = None
        self.b = None
        self.X = None
    
    def initialize_params(self, input_size):
        # He初始化
        scale = np.sqrt(2.0 / input_size)
        self.W = np.random.normal(0, scale, (input_size, self.output_size))
        self.b = np.zeros(self.output_size)
    
    def forward(self, X):
        """前向传播"""
        # 如果第一次运行，初始化参数
        if self.W is None:
            self.initialize_params(X.shape[1])
        
        self.X = X
        return X.dot(self.W) + self.b
    
    def backward(self, dout, learning_rate):
        """反向传播"""
        # 计算梯度
        dW = self.X.T.dot(dout)
        db = np.sum(dout, axis=0)
        dX = dout.dot(self.W.T)
        
        # 更新参数
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX

class ReLU:
    def __init__(self):
        self.X = None
    
    def forward(self, X):
        """前向传播"""
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, dout):
        """反向传播"""
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX

class Softmax:
    def __init__(self):
        self.probs = None
    
    def forward(self, X):
        """前向传播"""
        # 数值稳定性处理
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_scores = np.exp(X_shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, dout):
        """反向传播"""
        # Softmax + CrossEntropy的梯度
        return dout

class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None
    
    def forward(self, probs, labels):
        self.probs = probs
        self.labels = labels
        
        N = probs.shape[0]
        log_likelihood = -np.log(probs[range(N), np.argmax(labels, axis=1)])
        loss = np.sum(log_likelihood) / N
        return loss
    
    def backward(self):
        N = self.probs.shape[0]
        dprobs = self.probs.copy()
        dprobs[range(N), np.argmax(self.labels, axis=1)] -= 1
        dprobs /= N
        return dprobs

class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(num_filters=6, filter_size=5, padding=2)
        self.relu1 = ReLU()
        self.pool1 = MaxPooling(pool_size=2, stride=2)
        
        self.conv2 = Conv2D(num_filters=16, filter_size=5)
        self.relu2 = ReLU()
        self.pool2 = MaxPooling(pool_size=2, stride=2)
        
        self.flatten = Flatten()
        
        self.fc1 = FullyConnected(output_size=120)
        self.relu3 = ReLU()
        
        self.fc2 = FullyConnected(output_size=84)
        self.relu4 = ReLU()
        
        self.fc3 = FullyConnected(output_size=10)
        
        self.softmax = Softmax()
        self.cross_entropy = CrossEntropyLoss()
    
    def forward(self, X, y=None):
        out = self.conv1.forward(X)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.fc2.forward(out)
        out = self.relu4.forward(out)
        out = self.fc3.forward(out)
        probs = self.softmax.forward(out)
        if y is None:
            return probs

        loss = self.cross_entropy.forward(probs, y)
        return probs, loss
    
    def backward(self, learning_rate=0.01):
        dout = self.cross_entropy.backward()
        dout = self.fc3.backward(dout, learning_rate)
        dout = self.relu4.backward(dout)
        dout = self.fc2.backward(dout, learning_rate)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout, learning_rate)
        dout = self.flatten.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout, learning_rate)
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout, learning_rate)

def train_lenet5(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=10, learning_rate=0.01):
    num_train = X_train.shape[0]
    
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"开始第 {epoch+1}/{epochs} 轮训练...")
        
        indices = np.random.permutation(num_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        num_batches = int(np.ceil(num_train / batch_size))
        epoch_loss = 0
        
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, num_train)
            
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            _, loss = model.forward(X_batch, y_batch)
            epoch_loss += loss

            model.backward(learning_rate)

            if batch % 20 == 0 or batch == num_batches - 1:
                print(f"  批次 {batch+1}/{num_batches}, 损失: {loss:.4f}")

        avg_loss = epoch_loss / num_batches
        train_loss_history.append(avg_loss)

        print("计算训练集准确率...")
        train_acc = evaluate(model, X_train[:1000], y_train[:1000])
        train_acc_history.append(train_acc)

        print("计算验证集准确率...")
        val_acc = evaluate(model, X_val, y_val)
        val_acc_history.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"第 {epoch+1} 轮完成! 耗时: {epoch_time:.2f}秒, 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 验证准确率: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time:.2f}秒")
    
    return train_loss_history, train_acc_history, val_acc_history

def evaluate(model, X, y):
    batch_size = 32
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    correct = 0
    
    for batch in range(num_batches):
        start = batch * batch_size
        end = min(start + batch_size, num_samples)
        
        X_batch = X[start:end]
        y_batch = y[start:end]

        probs = model.forward(X_batch)

        pred_labels = np.argmax(probs, axis=1)
        true_labels = np.argmax(y_batch, axis=1)

        correct += np.sum(pred_labels == true_labels)

    accuracy = correct / num_samples * 100
    return accuracy

def plot_history(train_loss_history, train_acc_history, val_acc_history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """可视化预测结果"""
    # 随机选择样本
    indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
    X_samples = X_test[indices]
    y_samples = y_test[indices]
    
    # 获取预测
    probs = model.forward(X_samples)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_samples, axis=1)
    
    # 绘制结果
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_samples[i, :, :, 0], cmap='gray')
        plt.title(f'Pred: {pred_labels[i]}\nTrue: {true_labels[i]}')
        plt.axis('off')
    plt.savefig('predictions.png')
    plt.show()

def confusion_matrix(y_true, y_pred, num_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def plot_confusion_matrix(cm):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加刻度标签
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存为 '{filename}'")

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 尝试加载MNIST数据集
    print("尝试加载MNIST数据集...")
    data = load_mnist_data()
    
    # 如果加载失败，使用合成数据
    if data is None:
        print("无法加载MNIST数据集，将使用合成数据...")
        data = generate_synthetic_digits(num_samples=2000)
    
    (X_train, y_train_one_hot, y_train), (X_test, y_test_one_hot, y_test) = data
    
    # 仅使用部分数据进行训练（加快训练速度）
    train_size = min(10000, len(X_train))  # 使用最多10000个样本训练
    test_size = min(1000, len(X_test))     # 使用最多1000个样本测试
    
    X_train_small = X_train[:train_size]
    y_train_small = y_train_one_hot[:train_size]
    X_test_small = X_test[:test_size]
    y_test_small = y_test_one_hot[:test_size]
    y_test_labels = y_test[:test_size]
    
    print(f"训练集大小: {X_train_small.shape}")
    print(f"测试集大小: {X_test_small.shape}")
    
    # 创建LeNet-5模型
    model = LeNet5()
    
    # 训练模型
    print("\n开始训练模型...")
    train_loss_history, train_acc_history, val_acc_history = train_lenet5(
        model, X_train_small, y_train_small, X_test_small, y_test_small,
        batch_size=32, epochs=10, learning_rate=0.01
    )
    
    # 绘制训练历史
    plot_history(train_loss_history, train_acc_history, val_acc_history)
    
    # 在测试集上评估
    print("\n在完整测试集上评估模型...")
    test_acc = evaluate(model, X_test_small, y_test_small)
    print(f"测试集准确率: {test_acc:.2f}%")
    
    # 计算混淆矩阵
    probs = model.forward(X_test_small)
    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_test_small, axis=1)
    cm = confusion_matrix(y_test_labels, pred_labels)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm)
    
    # 可视化预测结果
    visualize_predictions(model, X_test_small, y_test_small)
    
    # 保存模型
    save_model(model, 'lenet5_model.pkl')

if __name__ == '__main__':
    main() 