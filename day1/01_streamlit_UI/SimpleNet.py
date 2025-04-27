import numpy as np

# **ソフトマックス関数**
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 数値の安定性を考慮
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# **クロスエントロピー損失関数**
def cross_entropy_loss(y_pred, y_true):
    delta = 1e-7  # 数値の安定性
    return -np.sum(y_true * np.log(y_pred + delta)) / y_true.shape[0]

# **MNISTデータの読み込み**
def load_mnist():
    x_train = np.load(r"C:\\Users\\ibmkt\\lecture-ai-engineering\\day1\\01_streamlit_UI\\x_train.npy")
    y_train= np.load(r"C:\\Users\\ibmkt\\lecture-ai-engineering\\day1\\01_streamlit_UI\\y_train.npy")
    
    x_train = x_train.reshape(-1, 784).astype("float32") / 255  # 正規化
    y_train = np.eye(10)[y_train.astype("int32")]  # One-hotエンコーディング
    
    return x_train, y_train

# **モデルクラス**
class SimpleNet:
    def __init__(self, input_dim=784, output_dim=10, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)  # 重みの初期化
        self.b = np.zeros(output_dim)  # バイアスの初期化
        self.lr = lr  # 学習率

        # Adam のパラメータ
        self.m_W, self.v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        self.m_b, self.v_b = np.zeros_like(self.b), np.zeros_like(self.b)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.t = 1  # 更新回数

    def forward(self, x):
        return softmax(np.dot(x, self.W) + self.b)  # 順伝播

    def backward(self, x, y_true, y_pred):
        batch_size = x.shape[0]
        grad_W = np.dot(x.T, (y_pred - y_true)) / batch_size
        grad_b = np.sum(y_pred - y_true, axis=0) / batch_size

        # **Adam更新**
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_W
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_W ** 2)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        self.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        self.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        self.t += 1  # 更新回数を増やす

    def train(self, x_train, y_train, epochs=1000, batch_size=1000):
        for epoch in range(epochs):
            batch_mask = np.random.choice(len(x_train), batch_size)
            batch_x = x_train[batch_mask]
            batch_t = y_train[batch_mask]

            y_pred = self.forward(batch_x)
            loss = cross_entropy_loss(y_pred, batch_t)
            self.backward(batch_x, batch_t, y_pred)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        print("学習完了")

    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

    def save_model(self, filename="model/mnist_model_adam.npz"):
        np.savez(filename, weights=self.W, bias=self.b)

    @staticmethod
    def load_model(filename="model/mnist_model_adam.npz"):
        data = np.load(filename)
        model = SimpleNet()
        model.W = data["weights"]
        model.b = data["bias"]
        return model

# **モデルの学習**
x_train, y_train = load_mnist()
model = SimpleNet()
model.train(x_train, y_train, epochs=1000, batch_size=1000)
model.save_model("mnist_model.npz")