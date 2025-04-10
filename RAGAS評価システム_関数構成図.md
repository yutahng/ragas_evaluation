# RAGAS評価システム 関数構成図

このドキュメントでは、RAGAS評価システムの関数構成を図で説明します。

## 全体の処理フロー

```mermaid
graph TD
    A[main] --> B[load_data]
    A --> C[prepare_evaluation_data]
    A --> D[run_evaluation]
    D --> E[get_available_metrics]
    D --> F[evaluate]
    A --> G[save_results]
    
    style A fill:#f9d77e,stroke:#333,stroke-width:2px
    style D fill:#a8d5ba,stroke:#333,stroke-width:2px
    style G fill:#a8d5ba,stroke:#333,stroke-width:2px
```

## 関数の詳細説明

### main関数

```mermaid
graph TD
    A[main] --> B[コマンドライン引数の解析]
    B --> C[データの読み込み\nload_data]
    C --> D[評価用データの準備\nprepare_evaluation_data]
    D --> E[評価の実行\nrun_evaluation]
    E --> F[結果の保存\nsave_results]
    
    style A fill:#f9d77e,stroke:#333,stroke-width:2px
```

### load_data関数

```mermaid
graph TD
    A[load_data] --> B{ファイル形式の判定}
    B -->|JSONL| C[JSONLファイルの読み込み]
    B -->|JSON| D[JSONファイルの読み込み]
    B -->|CSV| E[CSVファイルの読み込み]
    C --> F[Dataset.from_list]
    D --> F
    E --> G[Dataset.from_pandas]
    F --> H[Datasetオブジェクト]
    G --> H
    
    style A fill:#a8d5ba,stroke:#333,stroke-width:2px
    style H fill:#f9d77e,stroke:#333,stroke-width:2px
```

### prepare_evaluation_data関数

```mermaid
graph TD
    A[prepare_evaluation_data] --> B[必要なカラムの確認]
    B --> C[SINGLE_TURNフォーマットへの変換]
    C --> D[変換後のデータセット]
    
    style A fill:#a8d5ba,stroke:#333,stroke-width:2px
    style D fill:#f9d77e,stroke:#333,stroke-width:2px
```

### run_evaluation関数

```mermaid
graph TD
    A[run_evaluation] --> B[利用可能な評価指標の取得\nget_available_metrics]
    B --> C[データセットの確認]
    C --> D[評価指標の設定]
    D --> E{LLMタイプの選択}
    E -->|OpenAI| F[OpenAI LLMの設定]
    E -->|Gemini| G[Gemini LLMの設定]
    F --> H[評価の実行\nevaluate]
    G --> H
    H --> I[評価結果]
    
    style A fill:#a8d5ba,stroke:#333,stroke-width:2px
    style I fill:#f9d77e,stroke:#333,stroke-width:2px
```

### get_available_metrics関数

```mermaid
graph TD
    A[get_available_metrics] --> B[評価指標クラスの確認]
    B --> C[各評価指標のインスタンス化]
    C --> D[利用可能な評価指標の辞書]
    
    style A fill:#a8d5ba,stroke:#333,stroke-width:2px
    style D fill:#f9d77e,stroke:#333,stroke-width:2px
```

### save_results関数

```mermaid
graph TD
    A[save_results] --> B[出力ディレクトリの作成]
    B --> C[結果の型の確認]
    C --> D[DataFrameへの変換]
    D --> E[CSVファイルとして保存]
    D --> F[グラフの作成と保存]
    
    style A fill:#a8d5ba,stroke:#333,stroke-width:2px
```

## 評価指標の関係

```mermaid
graph TD
    A[RAGAS評価指標] --> B[基本的な評価指標\nground truthなし]
    A --> C[追加の評価指標\nground truth必要]
    
    B --> D[Faithfulness\n忠実性]
    B --> E[AnswerRelevancy\n回答の関連性]
    B --> F[ContextRelevance\nコンテキストの関連性]
    B --> G[ContextPrecision\nコンテキストの精度]
    B --> H[Critique\n側面批評]
    
    C --> I[ContextRecall\nコンテキストの再現性]
    C --> J[ContextEntityRecall\nコンテキストのエンティティ再現性]
    C --> K[AnswerSimilarity\n回答の意味的類似性]
    C --> L[AnswerCorrectness\n回答の正確性]
    C --> M[SummarizationScore\n要約スコア]
    
    style A fill:#f9d77e,stroke:#333,stroke-width:2px
    style B fill:#a8d5ba,stroke:#333,stroke-width:2px
    style C fill:#f8b9b2,stroke:#333,stroke-width:2px
```

## データの流れ

```mermaid
flowchart LR
    A[入力データ\nJSONL/JSON/CSV] --> B[load_data]
    B --> C[Dataset]
    C --> D[prepare_evaluation_data]
    D --> E[SINGLE_TURN形式\nDataset]
    E --> F[run_evaluation]
    F --> G[評価結果]
    G --> H[save_results]
    H --> I[CSV結果ファイル]
    H --> J[グラフ画像]
    
    style A fill:#f8b9b2,stroke:#333,stroke-width:2px
    style C fill:#a8d5ba,stroke:#333,stroke-width:2px
    style E fill:#a8d5ba,stroke:#333,stroke-width:2px
    style G fill:#a8d5ba,stroke:#333,stroke-width:2px
    style I fill:#f9d77e,stroke:#333,stroke-width:2px
    style J fill:#f9d77e,stroke:#333,stroke-width:2px
```
