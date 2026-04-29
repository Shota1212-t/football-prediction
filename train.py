import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. データの読み込み
df = pd.read_csv('final_training_data.csv')

# 入力(X)と正解(y)に分ける
X = df.drop('result', axis=1).values
y = df['result'].values

# 2. データの正規化（NNの学習を安定させるために必須）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 学習用とテスト用に分割 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorchのTensor型に変換
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 3. ニューラルネットワークのモデル定義
class SoccerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SoccerPredictor, self).__init__()
        # 層の積み上げ
        self.fc1 = nn.Linear(input_size, 128) # 第1層
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)        # 過学習防止
        self.fc2 = nn.Linear(128, 64)         # 第2層
        self.fc3 = nn.Linear(64, 3)           # 出力層（勝・分・負の3択）
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SoccerPredictor(X_train.shape[1])

# 4. 損失関数と最適化アルゴリズムの設定
criterion = nn.CrossEntropyLoss() # 分類問題の定番
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 学習ループ
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. 精度の確認
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'テストデータの正解率: {accuracy * 100:.2f}%')

# 7. モデルの保存（後で予測に使うため）
torch.save(model.state_dict(), 'soccer_model.pth')
print("モデルを 'soccer_model.pth' として保存しました。")