# Premier League Standings & Squad Analyzer
**プレミアリーグの最新順位とチーム詳細を可視化するダッシュボード**

Football-Data.org APIと連携し、プレミアリーグの順位表表示、および各チームのスカッド情報を瞬時に取得・表示するWebアプリケーションです。

### ライブデモ
**[アプリをブラウザで開く](https://football-prediction-kluuahgskkiv773eiqhdvy.streamlit.app/)**

## Features
*   **リアルタイム順位表**: 最新の勝ち点、勝敗数、得失点差を一覧表示。
*   **インタラクティブな詳細表示**: 順位表からチームを選択すると、そのチームの監督や所属選手一覧を動的にロード。
*   **クイックサーチ機能**: 選択したチームをGoogleで即座に調べるためのダイレクトリンクを搭載。
*   **キャッシュ最適化**: Streamlitのキャッシュ機能を利用し、APIのレート制限を守りつつ高速な動作を実現。

## Tech Stack
*   **Language**: Python 3.12+
*   **Framework**: Streamlit
*   **Data Library**: Pandas
*   **External API**: Football-Data.org API
*   **Deployment**: Streamlit Cloud

## Setup & Installation
1.  **リポジトリのクローン**
    ```bash
    git clone https://github.com/Shota1212-t/football-prediction.git
    cd football-prediction
    ```

2.  **ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```

3.  **環境変数の設定**
    `.env` ファイルを作成し、取得したAPIキーを記述してください。
    ```env
    FOOTBALL_API_KEY='Your_API_Key_Here'
    ```

4.  **実行**
    ```bash
    streamlit run app.py
    ```

## Note
*   Football-Data.org の無料プランを使用しているため、APIの呼び出し制限（1分間に10回まで）があります。連続してデータの更新を行うとエラーが出る場合があります。

## Author
*   **Shota**
*   **GitHub**: [Shota1212-t](https://github.com/Shota1212-t)
