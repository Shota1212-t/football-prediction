# football-prediction

サッカーの試合結果（勝ち・引き分け・負け）を予測する機械学習プロジェクトです。  
試合データをもとに特徴量を作成し、ニューラルネットワークを用いて **3クラス分類（勝 / 分 / 負）** を行います。

### ライブデモ
**[アプリをブラウザで開く](https://football-prediction-kluuahgskkiv773eiqhdvy.streamlit.app/)**

## Overview

このプロジェクトでは、サッカーの試合データをもとに試合結果を予測するモデルを構築しています。  
学習済みモデルを用いることで、入力された試合データから結果を推定できます。

主な流れは以下の通りです。

1. 学習データの読み込み
2. 特徴量の標準化
3. モデルの学習
4. テストデータで精度確認
5. 学習済みモデルの保存
6. 保存したモデルを用いた予測

## Features

- サッカー試合結果の **3クラス分類**
- PyTorch を用いたニューラルネットワーク学習
- `StandardScaler` による特徴量の標準化
- 学習済みモデル (`soccer_model.pth`) の保存
- 学習処理と予測処理を分離した構成

## Project Structure

```bash
football-prediction/
├── app.py                    # アプリ実行用ファイル（UI / エントリーポイント想定）
├── predict.py                # 学習済みモデルを用いた予測処理
├── train.py                  # モデル学習用スクリプト
├── final_training_data.csv   # 学習データ
└── soccer_model.pth          # 学習後に保存されるモデル
```

## Model

`train.py` では、PyTorch を使ってニューラルネットワークを構築しています。

- 入力層
- 全結合層（128ユニット）
- ReLU
- Dropout（0.2）
- 全結合層（64ユニット）
- 出力層（3クラス）

損失関数には `CrossEntropyLoss`、最適化手法には `Adam` を使用しています。

## Environment

想定環境：

- Python 3.x
- PyTorch
- pandas
- scikit-learn

## Installation

必要なライブラリをインストールします。

```bash
pip install torch pandas scikit-learn
```

## Usage

### 1. モデルの学習

まず、学習データ `final_training_data.csv` を用意したうえで、以下を実行します。

```bash
python train.py
```

実行すると、モデルの学習が行われ、テストデータに対する精度が表示されます。  
その後、学習済みモデルが `soccer_model.pth` として保存されます。

### 2. 予測の実行

学習済みモデルを用いて予測を行う場合は、以下を実行します。

```bash
python predict.py
```

### 3. アプリの起動

アプリとして実行する場合は、以下を実行します。

```bash
python app.py
```

> ※ `app.py` の実装によって起動方法が異なる場合があります。

## Data

このプロジェクトでは、試合結果を予測するための特徴量を含む CSV データを使用します。  
目的変数は `result` 列で、以下の3分類を想定しています。

- 勝ち
- 引き分け
- 負け

入力特徴量には、`result` 以外の列を使用します。

## Training Flow

`train.py` の処理概要：

1. `final_training_data.csv` を読み込む
2. `result` を目的変数、それ以外を特徴量として分離
3. `StandardScaler` で特徴量を標準化
4. 学習データとテストデータに分割
5. ニューラルネットワークを学習
6. テストデータで精度を算出
7. 学習済みモデルを保存

## Future Improvements

- 特徴量エンジニアリングの改善
- ハイパーパラメータ調整
- 他モデルとの比較（XGBoost / LightGBM / RandomForest など）
- データ量の拡張
- Webアプリ化による UI 改善
- 推論結果の可視化

## Author

Shota Ishizaki

## Notes

この README は、現在のリポジトリ構成をもとに整理したものです。  
今後、データ取得方法・特徴量設計・モデル改善内容などを追記することで、  
プロジェクトの内容がより伝わりやすくなります。
