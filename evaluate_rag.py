#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAGASを使用してRAGシステムを評価するスクリプト
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Literal
from datasets import Dataset

# RAGASのインポート
try:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRelevance,
        ContextPrecision,
        ContextRecall,
        ContextEntityRecall,
        AnswerSimilarity,
        AnswerCorrectness
    )
    # 追加の評価指標を試行
    try:
        from ragas.metrics import Critique, SummarizationScore
    except ImportError:
        Critique = None
        SummarizationScore = None
        print("警告: Critique と SummarizationScore をインポートできませんでした。これらの指標は使用できません。")
except ImportError as e:
    print(f"RAGASライブラリのインポートエラー: {e}")
    print("RAGASライブラリがインストールされているか確認してください。")
    exit(1)

from ragas import evaluate

# LangChainのインポート（最新バージョン用）
try:
    from langchain_community.chat_models import ChatOpenAI
except ImportError:
    # 古いバージョン用
    try:
        from langchain.chat_models import ChatOpenAI
    except ImportError:
        print("langchainライブラリがインストールされているか確認してください。")
        exit(1)

from langchain_openai import ChatOpenAI as LangchainOpenAI

# Google AI (Gemini) のインポート
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    print("警告: Google Generative AI (Gemini) ライブラリがインストールされていません。")
    print("Geminiモデルを使用するには、以下のコマンドを実行してください:")
    print("pip install google-generativeai langchain-google-genai")
    GEMINI_AVAILABLE = False

# 環境変数の読み込み
load_dotenv()

# Google AI APIキーの設定（存在する場合）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GOOGLE_API_KEY)

def load_data(file_path: str) -> Dataset:
    """
    評価用データセットを読み込む関数
    
    Args:
        file_path: データセットのファイルパス（JSON、JSONLまたはCSV）
        
    Returns:
        Dataset: Hugging Face Datasetsフォーマットのデータセット
    """
    if file_path.endswith('.jsonl'):
        # JSONLファイルの読み込み
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data)
    
    elif file_path.endswith('.json'):
        # JSONファイルの読み込み（JSONL形式として処理）
        try:
            # まずJSONL形式として読み込みを試みる
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            return Dataset.from_list(data)
        except json.JSONDecodeError:
            # 通常のJSONとして読み込みを試みる
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # データが辞書の場合はリストに変換
            if isinstance(data, dict):
                data = [data]
            return Dataset.from_list(data)
    
    elif file_path.endswith('.csv'):
        # CSVファイルの読み込み
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)
    
    else:
        raise ValueError(f"サポートされていないファイル形式です: {file_path}")

def prepare_evaluation_data(
    dataset: Dataset,
    question_key: str = "question",
    answer_key: str = "answer",
    contexts_key: str = "contexts",
    reference_key: str = "reference",
    ground_truth_key: Optional[str] = None
) -> Dataset:
    """
    RAGASの評価用にデータセットを準備する関数
    
    Args:
        dataset: 元のデータセット
        question_key: 質問が格納されているカラム名
        answer_key: 回答が格納されているカラム名
        contexts_key: コンテキスト（検索結果）が格納されているカラム名
        reference_key: 参照テキストが格納されているカラム名
        ground_truth_key: 正解が格納されているカラム名（オプション）
        
    Returns:
        Dataset: RAGAS評価用に整形されたデータセット
    """
    # 必要なカラムの確認
    required_columns = {question_key, answer_key, contexts_key, reference_key}
    if not all(col in dataset.column_names for col in required_columns):
        missing = required_columns - set(dataset.column_names)
        raise ValueError(f"データセットに必要なカラムがありません: {missing}")
    
    # 最新のRAGASライブラリでは、SINGLE_TURNという構造が必要
    def create_single_turn_format(example):
        # SINGLE_TURNフォーマットを作成
        single_turn = {
            "user_input": example[question_key],
            "response": example[answer_key],
            "retrieved_contexts": example[contexts_key],
            "reference": example[reference_key]
        }
        
        # ground_truthが存在する場合は追加
        if ground_truth_key and ground_truth_key in example:
            single_turn["ground_truth"] = example[ground_truth_key]
            
        return {"SINGLE_TURN": single_turn}
    
    # データセットを変換
    transformed_dataset = dataset.map(create_single_turn_format)
    print(f"変換後のデータセット構造: {transformed_dataset[0]}")
    
    return transformed_dataset

def get_available_metrics() -> Dict[str, Any]:
    """
    利用可能な評価指標とその必要なカラムを確認する関数
    
    Returns:
        Dict: 評価指標名とそのインスタンスのマッピング
    """
    # 利用可能な評価指標を格納する辞書
    available_metrics = {}
    
    # 基本的な評価指標を確認
    metrics_to_check = {
        "Faithfulness": Faithfulness,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextRelevance": ContextRelevance,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "ContextEntityRecall": ContextEntityRecall,
        "AnswerSimilarity": AnswerSimilarity,
        "AnswerCorrectness": AnswerCorrectness
    }
    
    # Critiqueが利用可能な場合は追加
    if Critique is not None:
        metrics_to_check["Critique"] = Critique
    
    # SummarizationScoreが利用可能な場合は追加
    if SummarizationScore is not None:
        metrics_to_check["SummarizationScore"] = SummarizationScore
    
    # 各評価指標をインスタンス化し、必要なカラムを確認
    for name, metric_class in metrics_to_check.items():
        try:
            metric_instance = metric_class()
            # 必要なカラムを取得（可能な場合）
            required_columns = getattr(metric_instance, "required_columns", [])
            print(f"評価指標 {name} が利用可能です。必要なカラム: {required_columns}")
            available_metrics[name] = metric_instance
        except Exception as e:
            print(f"評価指標 {name} は利用できません: {str(e)}")
    
    return available_metrics

def run_evaluation(
    dataset: Dataset, 
    use_ground_truth: bool = False, 
    llm_type: Literal["openai", "gemini"] = "openai",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    RAGASを使用して評価を実行する関数
    
    Args:
        dataset: 評価用データセット
        use_ground_truth: ground_truthを使用するかどうか
        llm_type: 使用するLLMの種類 ("openai" または "gemini")
        model_name: 使用するモデル名（指定しない場合はデフォルト値を使用）
        
    Returns:
        Dict: 評価結果
    """
    # 利用可能な評価指標を取得
    print("利用可能な評価指標を確認しています...")
    available_metrics = get_available_metrics()
    
    # データセットに含まれるカラムを表示
    print(f"データセットのカラム: {dataset.column_names}")
    print(f"データセットのサンプル数: {len(dataset)}")
    
    # データセットの最初の数サンプルを表示
    for i in range(min(3, len(dataset))):
        print(f"サンプル {i+1}:")
        for key, value in dataset[i].items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, (list, tuple)) and len(v) > 3:
                        print(f"    {k}: [{v[0]}, {v[1]}, {v[2]}, ...]")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
    
    # 基本的な評価指標の設定
    metrics = []
    
    # 基本的な評価指標を追加
    metrics.extend([
        Faithfulness(),
        AnswerRelevancy(),
        ContextRelevance(),
        ContextPrecision()
    ])
    
    # Critiqueが利用可能な場合は追加
    if Critique is not None:
        metrics.append(Critique())
    
    # ground_truthが必要な指標（use_groundがTrueの場合のみ追加）
    if use_ground_truth:
        metrics.extend([
            ContextRecall(),
            ContextEntityRecall(),
            AnswerSimilarity(),
            AnswerCorrectness()
        ])
        
        # SummarizationScoreが利用可能な場合は追加
        if SummarizationScore is not None:
            metrics.append(SummarizationScore())
    
    # LLMの設定
    if llm_type == "openai":
        # OpenAIのAPIキーが設定されているか確認
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEYが設定されていません。.envファイルを確認してください。")
        
        # モデル名の設定（デフォルトはgpt-4o）
        openai_model = model_name or "gpt-4o"
        print(f"OpenAIモデルを使用します: {openai_model}")
        llm = LangchainOpenAI(model=openai_model, temperature=0)
    
    elif llm_type == "gemini":
        # Geminiが利用可能か確認
        if not GEMINI_AVAILABLE:
            raise ValueError("Geminiモデルを使用するには、Google Generative AIライブラリをインストールしてください。")
        
        # Google AIのAPIキーが設定されているか確認
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEYが設定されていません。.envファイルを確認してください。")
        
        # モデル名の設定（デフォルトはgemini-pro）
        gemini_model = model_name or "gemini-pro"
        print(f"Geminiモデルを使用します: {gemini_model}")
        llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
    
    else:
        raise ValueError(f"サポートされていないLLMタイプです: {llm_type}")
    
    print(f"使用する評価指標: {[type(m).__name__ for m in metrics]}")
    print(f"評価するサンプル数: {len(dataset)}")
    
    # 評価の実行
    try:
        # 各サンプルが評価されることを確認するためのカスタム評価関数
        def custom_evaluate():
            print("RAGASの評価を開始します...")
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                raise_exceptions=True
            )
            print(f"評価が完了しました。結果の型: {type(result)}")
            
            # 結果の詳細を表示
            if isinstance(result, dict):
                print(f"結果のキー: {list(result.keys())}")
                for key, value in result.items():
                    if isinstance(value, list):
                        print(f"  {key}: {len(value)}件の結果")
                        if len(value) > 0:
                            print(f"    最初の結果: {value[0]}")
            
            return result
        
        result = custom_evaluate()
        return result
    except Exception as e:
        print(f"評価中にエラーが発生しました: {str(e)}")
        # エラーの詳細を表示
        import traceback
        traceback.print_exc()
        return {}

def save_results(result: Any, output_dir: str = "results", dataset: Optional[Dataset] = None) -> None:
    """
    評価結果を保存する関数
    
    Args:
        result: 評価結果
        output_dir: 結果を保存するディレクトリ
        dataset: 評価に使用したデータセット（オプション）
    """
    # 結果が空の場合は何もしない
    if not result:
        print("保存する結果がありません。")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 結果の型を確認して適切に処理
    print(f"評価結果の型: {type(result)}")
    
    try:
        # EvaluationResult型の処理を追加
        if 'ragas.dataset_schema' in str(type(result)) and 'EvaluationResult' in str(type(result)):
            print("EvaluationResult型の結果を処理しています...")
            
            # EvaluationResultからデータを抽出
            try:
                # 結果をDataFrameに変換
                if hasattr(result, 'to_pandas'):
                    # to_pandasメソッドがある場合はそれを使用
                    result_df = result.to_pandas()
                    print(f"to_pandasメソッドを使用してDataFrameに変換しました。")
                elif hasattr(result, 'scores'):
                    # scoresプロパティがある場合はそれを使用
                    scores = result.scores
                    if isinstance(scores, dict):
                        result_df = pd.DataFrame(scores)
                        print(f"scoresプロパティからDataFrameを作成しました。")
                    else:
                        result_df = pd.DataFrame([{'scores': scores}])
                elif hasattr(result, '__dict__'):
                    # __dict__属性を使用
                    result_dict = result.__dict__
                    # 辞書からDataFrameを作成
                    result_df = pd.DataFrame([result_dict])
                    print(f"__dict__属性からDataFrameを作成しました。")
                else:
                    # 文字列表現から情報を抽出
                    result_str = str(result)
                    print(f"結果の文字列表現: {result_str}")
                    # 文字列からスコアを抽出する試み
                    import re
                    scores = {}
                    # スコアのパターンを検索 (例: "metric_name: 0.123")
                    matches = re.findall(r'(\w+):\s+([0-9.]+)', result_str)
                    for metric, score in matches:
                        try:
                            scores[metric] = float(score)
                        except ValueError:
                            scores[metric] = score
                    
                    if scores:
                        result_df = pd.DataFrame([scores])
                        print(f"文字列表現から抽出したスコア: {scores}")
                    else:
                        # 最後の手段として単純な文字列として保存
                        with open(f"{output_dir}/evaluation_results.txt", "w") as f:
                            f.write(result_str)
                        print(f"結果を文字列として保存しました。")
                        return
            except Exception as e:
                print(f"EvaluationResult型の処理中にエラーが発生しました: {str(e)}")
                # 文字列として保存
                with open(f"{output_dir}/evaluation_results.txt", "w") as f:
                    f.write(str(result))
                return
        # 既存の型の処理
        elif isinstance(result, pd.DataFrame):
            # すでにDataFrameの場合
            result_df = result
        elif isinstance(result, dict):
            # 辞書の場合
            if all(isinstance(v, (list, tuple)) for v in result.values()):
                # 各キーに対応する値がリストの場合
                result_df = pd.DataFrame(result)
                
                # 各質問に対する個別の評価結果を表示
                print("\n各質問に対する評価結果:")
                if dataset is not None and len(result_df) == len(dataset):
                    for i in range(len(dataset)):
                        question = dataset[i].get('SINGLE_TURN', {}).get('user_input', f'質問 {i+1}')
                        print(f"\n質問 {i+1}: {question}")
                        for metric_name, values in result.items():
                            if i < len(values):
                                print(f"  {metric_name}: {values[i]}")
                else:
                    # データセットがない場合や長さが一致しない場合は、行ごとに表示
                    for i in range(len(result_df)):
                        print(f"\n行 {i+1}:")
                        for col in result_df.columns:
                            print(f"  {col}: {result_df.iloc[i][col]}")
            else:
                # ネストされた辞書の場合
                result_df = pd.DataFrame([result])
                print("\n評価結果の平均値:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
        elif isinstance(result, list):
            # リストの場合
            result_df = pd.DataFrame(result)
        else:
            # その他の型の場合
            print(f"未対応の結果型: {type(result)}")
            # 文字列として保存
            with open(f"{output_dir}/evaluation_results.txt", "w") as f:
                f.write(str(result))
            return
        
        # DataFrameをCSVとして保存
        result_df.to_csv(f"{output_dir}/evaluation_results.csv", index=False)
        
        # 結果の可視化（数値データがある場合のみ）
        if not result_df.empty and result_df.select_dtypes(include=['number']).columns.any():
            plt.figure(figsize=(10, 6))
            # 数値データのみを選択
            numeric_df = result_df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                numeric_df.mean().plot(kind='bar')
                plt.title('RAG System Evaluation Results')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/evaluation_results.png")
        
        print(f"\n評価結果を {output_dir} に保存しました。")
    except Exception as e:
        print(f"結果の保存中にエラーが発生しました: {str(e)}")
        # エラーの詳細を表示
        import traceback
        traceback.print_exc()

def main():
    """
    メイン関数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGASを使用してRAGシステムを評価するスクリプト')
    parser.add_argument('--data', type=str, required=True, help='評価用データセットのパス（JSON、JSONLまたはCSV）')
    parser.add_argument('--question_key', type=str, default='question', help='質問が格納されているカラム名')
    parser.add_argument('--answer_key', type=str, default='answer', help='回答が格納されているカラム名')
    parser.add_argument('--contexts_key', type=str, default='contexts', help='コンテキストが格納されているカラム名')
    parser.add_argument('--reference_key', type=str, default='reference', help='参照テキストが格納されているカラム名')
    parser.add_argument('--ground_truth_key', type=str, default=None, help='正解が格納されているカラム名（オプション）')
    parser.add_argument('--output_dir', type=str, default='results', help='結果を保存するディレクトリ')
    parser.add_argument('--llm', type=str, choices=['openai', 'gemini'], default='openai', 
                        help='使用するLLMの種類（openai または gemini）')
    parser.add_argument('--model', type=str, default=None, 
                        help='使用するモデル名（指定しない場合はデフォルト値を使用。openaiの場合はgpt-4o、geminiの場合はgemini-pro）')
    
    args = parser.parse_args()
    
    # データの読み込み
    print(f"データセットを読み込んでいます: {args.data}")
    dataset = load_data(args.data)
    
    # 評価用データの準備
    print("評価用データを準備しています...")
    eval_dataset = prepare_evaluation_data(
        dataset,
        question_key=args.question_key,
        answer_key=args.answer_key,
        contexts_key=args.contexts_key,
        reference_key=args.reference_key,
        ground_truth_key=args.ground_truth_key
    )
    
    # 評価の実行
    print("評価を実行しています...")
    use_ground_truth = args.ground_truth_key is not None
    
    # Geminiが選択されたが利用できない場合の処理
    if args.llm == 'gemini' and not GEMINI_AVAILABLE:
        print("警告: Geminiが選択されましたが、必要なライブラリがインストールされていません。")
        print("OpenAIにフォールバックします。")
        llm_type = 'openai'
    else:
        llm_type = args.llm
    
    result = run_evaluation(
        eval_dataset, 
        use_ground_truth=use_ground_truth,
        llm_type=llm_type,
        model_name=args.model
    )
    
    # 結果の保存（データセットも渡す）
    save_results(result, output_dir=args.output_dir, dataset=eval_dataset)
    
    print("評価が完了しました。")

if __name__ == "__main__":
    main()
