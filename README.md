# RAGAS評価フレームワーク

このリポジトリは、RAG（Retrieval Augmented Generation）システムを評価するためのRAGAS（Retrieval Augmented Generation Assessment）フレームワークを使用したスクリプトを提供します。

## 概要

RAGASは、RAGシステムの品質を評価するための包括的なフレームワークです。以下の評価指標を提供します：

### 基本的な評価指標（ground truthなしで使用可能）

- **Faithfulness（忠実性）**: 生成された回答が検索されたコンテキストに忠実かどうかを評価します。
- **AnswerRelevancy（回答の関連性）**: 生成された回答が質問に関連しているかどうかを評価します。
- **ContextRelevance（コンテキストの関連性）**: 検索されたコンテキストが質問に関連しているかどうかを評価します。
- **ContextPrecision（コンテキストの精度）**: 検索されたコンテキストが質問に対して過不足なく適切かどうかを評価します。
- **Critique（側面批評）**: 回答の様々な側面（明確さ、簡潔さ、有用性など）を評価します。

### 追加の評価指標（ground truthが必要）

- **ContextRecall（コンテキストの再現性）**: 検索されたコンテキストが質問に対して必要な情報をすべて含んでいるかどうかを評価します。
- **ContextEntityRecall（コンテキストのエンティティ再現性）**: 検索されたコンテキストが正解に含まれるエンティティをどれだけカバーしているかを評価します。
- **AnswerSimilarity（回答の意味的類似性）**: 生成された回答と正解の意味的な類似性を評価します。
- **AnswerCorrectness（回答の正確性）**: 生成された回答が正解と一致しているかどうかを評価します。
- **SummarizationScore（要約スコア）**: 回答が検索されたコンテキストの適切な要約になっているかを評価します。

## セットアップ

### 前提条件

- Python 3.8以上
- OpenAI APIキー（または他のLLM APIキー）

### インストール

1. リポジトリをクローンまたはダウンロードします。

2. 依存関係をインストールします：

```bash
cd ragas_evaluation
pip install -r requirements.txt
```

3. `.env.example`ファイルを`.env`にコピーし、必要なAPIキーを設定します：

```bash
cp .env.example .env
```

`.env`ファイルを編集して、OpenAI APIキーを設定します：

```
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### データ形式

評価用データセットは、以下のフィールドを含むJSONLまたはCSV形式で準備する必要があります：

- `question`: ユーザーの質問
- `answer`: RAGシステムによって生成された回答
- `contexts`: 検索されたコンテキスト（文書の配列）
- `reference`: 質問に対する参照テキスト（ContextPrecisionなどの評価指標に必要）
- `ground_truth`（オプション）: 質問に対する正解（ContextRecallなどの評価指標に必要）

サンプルデータ形式（JSONL）：

```json
{
  "question": "質問文", 
  "answer": "回答文", 
  "contexts": ["コンテキスト1", "コンテキスト2", "コンテキスト3"],
  "reference": "参照テキスト（質問に対する正確な回答を含むテキスト）"
}
```

**注意**: 最新のRAGASライブラリでは、データセットが特定の構造（SINGLE_TURN）を持つ必要があります。スクリプトは自動的にデータセットを適切な形式に変換します。具体的には、以下のような構造に変換されます：

```json
{
  "SINGLE_TURN": {
    "user_input": "質問文",
    "response": "回答文",
    "retrieved_contexts": ["コンテキスト1", "コンテキスト2", "コンテキスト3"],
    "reference": "参照テキスト"
  }
}
```

各評価指標は特定のカラムを必要とします。スクリプトは利用可能な評価指標とその必要なカラムを自動的に確認し、データセットに含まれるカラムに基づいて使用可能な評価指標のみを使用します。

### スクリプトの実行

基本的な使用方法：

```bash
python evaluate_rag.py --data sample_data.jsonl
```

カスタムカラム名を指定する場合：

```bash
python evaluate_rag.py --data your_data.csv --question_key "user_query" --answer_key "generated_response" --contexts_key "retrieved_documents"
```

ground truthを使用する場合：

```bash
python evaluate_rag.py --data your_data.jsonl --ground_truth_key "correct_answer"
```

出力ディレクトリを指定する場合：

```bash
python evaluate_rag.py --data sample_data.jsonl --output_dir "evaluation_results"
```

### 出力

評価結果は以下の形式で出力されます：

1. CSVファイル（`evaluation_results.csv`）: 各サンプルの評価スコア
2. 可視化グラフ（`evaluation_results.png`）: 評価指標の平均スコアを示す棒グラフ

## 評価指標の詳細

### 基本的な評価指標

#### Faithfulness（忠実性）

生成された回答が検索されたコンテキストに忠実かどうかを評価します。高いスコアは、回答がコンテキストの情報に基づいており、幻覚（hallucination）が少ないことを示します。

#### AnswerRelevancy（回答の関連性）

生成された回答が質問に関連しているかどうかを評価します。高いスコアは、回答が質問に直接対応していることを示します。

#### ContextRelevance（コンテキストの関連性）

検索されたコンテキストが質問に関連しているかどうかを評価します。高いスコアは、検索システムが質問に関連する文書を適切に取得できていることを示します。

#### ContextPrecision（コンテキストの精度）

検索されたコンテキストが質問に対して過不足なく適切かどうかを評価します。高いスコアは、検索されたコンテキストが質問に関連する情報を効率的に含んでいることを示します。

#### Critique（側面批評）

回答の様々な側面（明確さ、簡潔さ、有用性、完全性など）を評価します。この指標は、回答の質を多角的に評価するのに役立ちます。

### 追加の評価指標（ground truthが必要）

#### ContextRecall（コンテキストの再現性）

検索されたコンテキストが質問に対して必要な情報をすべて含んでいるかどうかを評価します。高いスコアは、検索されたコンテキストが質問に答えるために必要なすべての情報を含んでいることを示します。

#### ContextEntityRecall（コンテキストのエンティティ再現性）

検索されたコンテキストが正解に含まれるエンティティ（人物、組織、場所、日付など）をどれだけカバーしているかを評価します。高いスコアは、重要なエンティティが検索結果に含まれていることを示します。

#### AnswerSimilarity（回答の意味的類似性）

生成された回答と正解の意味的な類似性を評価します。高いスコアは、生成された回答が正解と意味的に近いことを示します。

#### AnswerCorrectness（回答の正確性）

生成された回答が正解と一致しているかどうかを評価します。高いスコアは、回答が事実的に正確であることを示します。

#### SummarizationScore（要約スコア）

回答が検索されたコンテキストの適切な要約になっているかを評価します。高いスコアは、回答がコンテキストの重要な情報を過不足なく含んでいることを示します。

## 注意事項

- 評価には、OpenAI APIを使用するため、APIキーが必要です。
- 大量のデータを評価する場合は、APIの使用量と料金に注意してください。
- 評価結果は使用するLLMによって異なる場合があります。

## 参考資料

- [RAGAS公式ドキュメント](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAGASに関する論文](https://arxiv.org/abs/2309.15217)
