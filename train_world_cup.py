import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Wカップ学習用データの読み込み
try:
    df = pd.read_csv('world_cup_training_data.csv')
    
    if len(df) < 10:
        print("データ少ないですが続行します")
        
except FileNotFoundError:
    print("エラー: world_cup_training_data.csv が見つかりません")
    print("world_cup_data.py を実行してください: python world_cup_data.py")
    exit(1)

print(f"📊 読み込んだデータ: {len(df)} 件")
print(df.head())

X = df.drop('result', axis=1).values
y = df['result'].values

# 2. データの正規化（スケーリング）
scaler_wc = StandardScaler()
X = scaler_wc.fit_transform(X)

# 学習用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(f"\n📈 学習データ: {len(X_train)}, テストデータ: {len(X_test)}")

# 3. Wカップ用モデル定義
class WorldCupPredictor(nn.Module):
    def __init__(self, input_size):
        super(WorldCupPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 3)  # 3クラス: ホーム勝, 引き分け, アウェイ勝
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model_wc = WorldCupPredictor(X_train.shape[1])

# 4. 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_wc.parameters(), lr=0.001, weight_decay=1e-4)

# 5. 学習ループ
epochs = 200
print("\n🚀 モデル学習開始...")

for epoch in range(epochs):
    model_wc.train()
    optimizer.zero_grad()
    outputs = model_wc(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. 精度の確認
model_wc.eval()
with torch.no_grad():
    test_outputs = model_wc(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'\n✅ テストデータの正解率: {accuracy * 100:.2f}%')

# 7. 保存（モデルと正規化スケール）
torch.save(model_wc.state_dict(), 'world_cup_model.pth')
joblib.dump(scaler_wc, 'world_cup_scaler.pkl')

print("\n💾 モデルを保存しました:")
print("   - world_cup_model.pth")
print("   - world_cup_scaler.pkl")
