import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # 追加：物差しを保存するため

# 1. データの読み込み
df = pd.read_csv('final_training_data.csv')

X = df.drop('result', axis=1).values
y = df['result'].values

# 2. データの正規化（スケーリング）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 3. モデル定義（極端な予測を防ぐため、深くしてDropoutを強化）
class SoccerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SoccerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4) # 強化
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4) # 追加
        self.fc3 = nn.Linear(64, 32)    # 追加：層を深くする
        self.fc4 = nn.Linear(32, 3)     # 出力層
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SoccerPredictor(X_train.shape[1])

# 4. 損失関数と最適化
criterion = nn.CrossEntropyLoss()
# weight_decay (L2正則化) を追加して、自信過剰（極端な確率）を抑制
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

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

# 7. 保存（モデルと物差し）
torch.save(model.state_dict(), 'soccer_model.pth')
joblib.dump(scaler, 'scaler.pkl')
print("モデル('soccer_model.pth')と正規化スケール('scaler.pkl')を保存しました。")