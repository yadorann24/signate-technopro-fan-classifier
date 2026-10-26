# コードの読み方

このリポジトリの入口は `fan-classifier` コマンドです。モデルを選ぶ設定、学習画像の分割、学習、テスト画像の推論を独立したモジュールに分けています。以下のリンク先はすべて、このリポジトリに含まれるファイルです。

## 最初に読むファイル

1. [README](../README.md)：課題、コンペティションの成果、設計判断、実装の概要。[再現手順](reproduction.md)に環境とコマンド。
2. [設計と手法](solution.md)：画像の縦横比を保つ理由、Triple Stratified分割、モデルと確率平均。
3. [registry.py](../src/fan_classifier/registry.py)：6モデルの識別子・入力サイズ・学習設定と `build_model()`。
4. [cli.py](../src/fan_classifier/cli.py)：コマンドの引数が各処理に渡る流れ。
5. 関心のある処理を、下の対応表からたどる。

## コマンドと実装の対応

| コマンド | 入力と出力 | 主な実装 |
| --- | --- | --- |
| `list-models` / `describe-model` | 登録モデルの一覧・設定を表示 | [registry.py](../src/fan_classifier/registry.py)、[cli.py](../src/fan_classifier/cli.py) |
| `prepare-strata` | 学習画像 → クラスタ・明るさ区分・foldを含むCSV | [preparation.py](../src/fan_classifier/preparation.py)、[features.py](../src/fan_classifier/features.py)、[stratification.py](../src/fan_classifier/stratification.py) |
| `train` | 学習画像・分割CSV・モデル名 → foldごとのcheckpointと評価記録 | [config.py](../src/fan_classifier/config.py)、[data.py](../src/fan_classifier/data.py)、[transforms.py](../src/fan_classifier/transforms.py)、[engine.py](../src/fan_classifier/engine.py) |
| `predict` | テスト画像・学習済みcheckpoint → fold平均の確率CSVと判定CSV | [prediction.py](../src/fan_classifier/prediction.py)、[data.py](../src/fan_classifier/data.py)、[transforms.py](../src/fan_classifier/transforms.py) |
| `ensemble` | モデルごとの確率CSV → モデル平均の確率CSVと判定CSV | [prediction.py](../src/fan_classifier/prediction.py) |
| `evaluate-oof` / `compare-oof` | 学習runの検証fold予測 → 単体・平均の指標と誤分類 | [oof.py](../src/fan_classifier/oof.py) |

`prepare-strata` と `train` は学習画像を使い、`predict` が提出用テスト画像を読み込みます。`ensemble` は予測済みの数値を平均し、画像やモデルを再学習しません。

## 各モジュールの責務

| ファイル | 主な関数・クラスと判断 |
| --- | --- |
| [registry.py](../src/fan_classifier/registry.py) | `ModelSpec` でモデル名と設定を管理。`build_model()` が対応するtimmモデルを2出力で作成 |
| [config.py](../src/fan_classifier/config.py) | `resolve_training_config()` が既定値・JSON・CLI引数を統合し、不正な設定を検出 |
| [normalization.py](../src/fan_classifier/normalization.py) | timmのモデル別mean/stdを解決し、checkpointの正規化設定を復元 |
| [features.py](../src/fan_classifier/features.py) | `extract_vgg16_features()` がRGB変換とVGG16前処理を行い、画像ごとの25,088次元特徴をバッチ単位で保存 |
| [stratification.py](../src/fan_classifier/stratification.py) | `cluster_scene_features()` でPCA・K-means、`quantile_brightness_thresholds()` で明るさの分位点、`build_triple_stratification()` で複合カテゴリとfoldを作成 |
| [preparation.py](../src/fan_classifier/preparation.py) | `prepare_stratification()` が学習画像の走査、特徴抽出またはクラスタ再利用、分割、CSV・集計保存をつなぐ |
| [data.py](../src/fan_classifier/data.py) | 画像ID・ラベル・パスを検証。`split_records()` は保存済みfoldの設定と件数を確認。`FanHoldingDataset` が画像を読み込む |
| [transforms.py](../src/fan_classifier/transforms.py) | `build_transforms()` がリサイズ・パディング・正規化を定義し、学習時だけランダムな拡張を追加 |
| [engine.py](../src/fan_classifier/engine.py) | `fit_model()` が学習と検証を行い、指定した検証指標で最良checkpointを保存 |
| [prediction.py](../src/fan_classifier/prediction.py) | `predict_checkpoints()` がモデルを復元してfold予測を平均。`ensemble_probability_files()` が画像IDを照合してモデル間の確率を平均 |
| [oof.py](../src/fan_classifier/oof.py) | 検証fold予測の保存・照合、画像IDをそろえたF1・誤分類・平均の比較 |
| [cli.py](../src/fan_classifier/cli.py) | 入出力先・モデル・設定の整合性を確認して各処理を呼び出す |

## 設定・環境・検証

| ファイル | 用途 |
| --- | --- |
| [small_batch.example.json](../configs/small_batch.example.json) | 学習設定をJSONで上書きする例 |
| [pyproject.toml](../pyproject.toml) | パッケージ情報、直接依存、VGG16用の追加依存、CLI登録 |
| [requirements-cpu.lock](../requirements-cpu.lock) | 学習・推論の依存を固定したLinux・Python 3.12向けCPU環境 |
| [requirements-stratification-cpu.lock](../requirements-stratification-cpu.lock) | TensorFlowによる特徴抽出を含むCPU環境 |
| [test_preparation.py](../tests/test_preparation.py) | 学習画像の分位点、テスト画像を集計しないこと、分割CSVから学習への接続を合成画像で確認 |
| [test_stratification.py](../tests/test_stratification.py) | 分位点の境界・同値、各層の件数、foldの排他性、クラスタリングを確認 |
| [test_cli.py](../tests/test_cli.py) | 小型モデルで学習・保存・推論まで実行し、欠損foldや上書きを検出 |
| [test_model_integration.py](../tests/test_model_integration.py) | 6モデルをmetaデバイスで構築し、入力サイズと出力形状を確認 |
| [check_public_files.py](../scripts/check_public_files.py) | 公開フォルダの許可ファイルと代表的な秘密情報パターンを検査 |
| [ci.yml](../.github/workflows/ci.yml) | GitHub Actionsで依存導入、静的検査、公開内容検査、合成入力によるテストを実行 |

自動テストの項目は[実験と学習設定](experiments.md#自動テスト)に記載しています。
