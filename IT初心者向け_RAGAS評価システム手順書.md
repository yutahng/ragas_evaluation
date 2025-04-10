# IT初心者向け RAGAS評価システム手順書

このドキュメントは、Google Cloud Consoleを使用してRAG（Retrieval Augmented Generation）システムを評価するための手順を説明します。

## 目次

1. [前提条件](#前提条件)
2. [Google Cloud Consoleの設定](#google-cloud-consoleの設定)
3. [APIキーの取得](#apiキーの取得)
4. [プロジェクトのセットアップ](#プロジェクトのセットアップ)
5. [評価スクリプトの実行](#評価スクリプトの実行)
6. [結果の確認](#結果の確認)
7. [トラブルシューティング](#トラブルシューティング)

## 前提条件

- Googleアカウント
- インターネット接続
- 基本的なコマンドラインの知識（コピー＆ペーストができれば大丈夫です）

## Google Cloud Consoleの設定

1. **Google Cloud Consoleにアクセス**
   - ブラウザで [Google Cloud Console](https://console.cloud.google.com/) を開きます。
   - Googleアカウントでログインします。

2. **新しいプロジェクトの作成**
   - 画面上部のプロジェクト選択ドロップダウンをクリックします。
   - 「新しいプロジェクト」をクリックします。
   - プロジェクト名を入力します（例：「rag-evaluation」）。
   - 「作成」ボタンをクリックします。

3. **Cloud Shellの起動**
   - 画面右上の「Cloud Shell」アイコン（>_）をクリックします。
   - Cloud Shellが画面下部に表示されるまで待ちます。

## APIキーの取得

### OpenAI APIキーの取得（必須）

1. ブラウザの新しいタブで [OpenAI API](https://platform.openai.com/) にアクセスします。
2. アカウントを作成またはログインします。
3. 右上のプロフィールアイコンをクリックし、「View API keys」を選択します。
4. 「Create new secret key」ボタンをクリックします。
5. キーの名前を入力し（例：「RAG Evaluation」）、「Create secret key」をクリックします。
6. 表示されたAPIキーをコピーして安全な場所に保存します（このキーは再表示できません）。

### Google AI (Gemini) APIキーの取得（オプション）

1. Google Cloud Consoleで、左側のメニューから「APIとサービス」→「ライブラリ」を選択します。
2. 検索ボックスに「Generative Language API」と入力し、検索結果をクリックします。
3. 「有効にする」ボタンをクリックします。
4. 左側のメニューから「APIとサービス」→「認証情報」を選択します。
5. 「認証情報を作成」→「APIキー」をクリックします。
6. 作成されたAPIキーをコピーして安全な場所に保存します。
7. （推奨）APIキーの制限を設定するために「APIキーを制限」をクリックし、「Generative Language API」のみにアクセスを制限します。

## プロジェクトのセットアップ

1. **リポジトリのクローン**
   - Cloud Shellで以下のコマンドを実行します：

   ```bash
   git clone https://github.com/yourusername/ragas_evaluation.git
   cd ragas_evaluation
   ```

2. **依存関係のインストール**
   - 以下のコマンドを実行して必要なライブラリをインストールします：

   ```bash
   pip install -r requirements.txt
   ```

3. **環境変数の設定**
   - 以下のコマンドを実行して環境変数ファイルを作成します：

   ```bash
   cp .env.example .env
   ```

   - Cloud Shellのエディタで.envファイルを開きます：

   ```bash
   nano .env
   ```

   - 以下のように編集します（先ほど取得したAPIキーを使用）：

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   - Ctrl+O を押して保存し、Ctrl+X を押して終了します。

## 評価スクリプトの実行

### 基本的な実行方法

1. 以下のコマンドを実行して評価を開始します：

   ```bash
   python evaluate_rag.py --data sample_data.json
   ```

   このコマンドは、サンプルデータを使用してRAGシステムの評価を実行します。

### カスタムデータを使用する場合

1. 自分のデータを評価する場合は、sample_data.jsonと同じ形式でJSONLファイルを作成します：

   ```json
   {"question": "質問文", "answer": "回答文", "contexts": ["コンテキスト1", "コンテキスト2", "コンテキスト3"], "reference": "参照テキスト"}
   ```

2. 以下のコマンドを実行して評価を開始します：

   ```bash
   python evaluate_rag.py --data your_data.jsonl
   ```

### Geminiモデルを使用する場合

1. Geminiモデルを使用する場合は、以下のコマンドを実行します：

   ```bash
   python evaluate_rag.py --data sample_data.json --llm gemini
   ```

### その他のオプション

- カスタムカラム名を指定する場合：

   ```bash
   python evaluate_rag.py --data your_data.csv --question_key "user_query" --answer_key "generated_response" --contexts_key "retrieved_documents"
   ```

- 出力ディレクトリを指定する場合：

   ```bash
   python evaluate_rag.py --data sample_data.json --output_dir "my_results"
   ```

## 結果の確認

1. 評価が完了すると、結果は「results」ディレクトリ（または指定した出力ディレクトリ）に保存されます。

2. **結果ファイルの確認**
   - `evaluation_results.csv`: 各サンプルの評価スコア
   - `evaluation_results.png`: 評価指標の平均スコアを示す棒グラフ

3. **結果ファイルのダウンロード**
   - Cloud Shellの「その他」メニュー（⋮）をクリックします。
   - 「ダウンロード」を選択します。
   - ダウンロードしたいファイルのパスを入力します（例：`ragas_evaluation/results/evaluation_results.csv`）。
   - 「ダウンロード」ボタンをクリックします。

## トラブルシューティング

### APIキーのエラー

- エラーメッセージ: `OPENAI_API_KEYが設定されていません`
  - 解決策: `.env`ファイルにAPIキーが正しく設定されているか確認してください。

### ライブラリのインストールエラー

- エラーメッセージ: `ModuleNotFoundError: No module named 'ragas'`
  - 解決策: `pip install -r requirements.txt`コマンドを再実行してください。

### データ形式のエラー

- エラーメッセージ: `データセットに必要なカラムがありません`
  - 解決策: データファイルが正しい形式（question, answer, contexts, referenceフィールドを含む）であることを確認してください。

### メモリエラー

- エラーメッセージ: `MemoryError`
  - 解決策: データセットのサイズを小さくするか、より大きなメモリを持つマシンタイプに変更してください。

### その他のエラー

- 問題が解決しない場合は、エラーメッセージをコピーしてGoogle検索するか、プロジェクト管理者に連絡してください。
