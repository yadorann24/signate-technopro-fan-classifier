# 再現手順と実行環境

[プロジェクト概要](../README.md)から、データ準備・学習・推論へ進むための手順です。競技配布データ・CSV・学習済み重みは含めません。利用権限のある画像をリポジトリ外に配置してください。

## 環境構築

以下は、リポジトリのルートディレクトリで実行します。Python 3.10〜3.12を対象とし、直接依存を[pyproject.toml](../pyproject.toml)に固定しています。

事前学習重みの取得には `huggingface-hub==0.36.0` を使用します。`timm==1.0.11` が利用するAPIとの互換性を保つため、通常のインストールと両方のCPU lockで同じバージョンを指定しています。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
fan-classifier list-models
fan-classifier describe-model --model eva_large_336
```

画像からTriple Stratified分割を作成する場合は、Keras VGG16用の追加依存も入れます。既成の学習CSVを使うだけなら不要です。

```bash
python -m pip install -e '.[stratification]'
```

この追加依存は `tensorflow-cpu==2.18.0` です。特徴抽出はCPUで実行し、初回はKerasのImageNet事前学習重みを取得します。EVA/EVA02の学習・推論はPyTorchで行います。

Linux x86_64・Python 3.12で検証したCPU環境を再構築する場合は、lockを使います。推移的な依存もバージョンとハッシュで固定しています。学習・推論のみの環境は次の手順です。

```bash
python -m pip install --require-hashes -r requirements-cpu.lock
python -m pip install --no-deps -e .
```

VGG16特徴抽出まで含む環境には、上の2行の代わりに次を使います。既存のCPU lockのバージョンを保ち、TensorFlowとその依存関係を追加した完全なlockです。

```bash
python -m pip install --require-hashes -r requirements-stratification-cpu.lock
python -m pip install --no-deps -e '.[stratification]'
```

学習用の主な依存はPyTorch、torchvision、timm、Albumentations、NumPy、OpenCV、scikit-learnです。特にGiantモデルの学習は大きなメモリを使用するため、利用するGPUに合わせてbatch sizeを調整してください。掲載した既定値は、任意のGPUでの実行を保証する値ではありません。

## データの用意

データはリポジトリ外へ配置し、パスを引数で渡します。`train/hold/` は扇子を持つ画像、`train/not-hold/` は持たない画像の配置先です。画像IDは各クラスフォルダ内の相対パスで、両クラスを通して一意である必要があります。

```text
private-data/
├── test.csv
├── train/
│   ├── hold/
│   └── not-hold/
├── test/
└── prepared/                # prepare-strataの出力先
```

学習画像を分割して作る検証データを、モデルの評価とcheckpointの選択に使います。提出用の `test/` は学習済みモデルによる推論にのみ使い、明るさの境界計算・クラスタリング・学習には使用しません。

## Triple Stratified分割の作成

正解ラベルだけで分割すると、画像の見た目や明るさの分布がfold間で偏る可能性があります。そこで、VGG16で画像特徴を抽出し、PCAで次元を減らしてからK-meansで似た画像をまとめます。この `scene_cluster` と明るさの区分を正解ラベルに組み合わせて層化します。`scene_cluster` は見た目によるグループであり、実際の撮影場所のラベルではありません。

```text
hold / not-hold画像
  ├─ VGG16特徴量 → PCA 50次元 → K-means 3クラスタ → scene_cluster
  ├─ 元画像のRGB平均画素値 → 学習画像の分位点で分類 → brightness_bin
  └─ 親フォルダから正解ラベルを取得 → holding_label
       ↓ 3項目を組み合わせる
  stratification_label → StratifiedKFold → fold_index
```

`hold` / `not-hold` の学習画像全体から明るさを計算し、**各区分の画像数がほぼ均等になる分位点を境界にします。** `--brightness-bins` で分類数を指定し、既定の3分類なら学習画像の明るさの1/3・2/3分位点を使います。境界の値は入力画像から決まり、テスト画像は使いません。

```bash
fan-classifier prepare-strata \
  --model eva_large_336 \
  --image-root /path/to/private-data/train \
  --output-dir /path/to/private-data/prepared/eva_large_336 \
  --brightness-bins 3
```

明るさはリサイズ前のRGB画像の全画素・全チャンネルの平均（0〜255）です。分位点は線形補間で求め、区切り値と同じ明るさは上側のbinへ入れます。同じ明るさの画像は同じbinに保つため、画像数が厳密に同数になるとは限りません。区切り値の重複や空のbinが生じる場合は停止するので、分類数を明示的に調整してください。求めた区切り値とbinごとの件数は `summary.json` に保存されます。

区切り値を直接指定する場合は `--brightness-bins` の代わりに `--brightness-thresholds 100 150` のように渡します。これは任意の値を使う例です。層化カテゴリの画像数がfold数を下回る場合も停止し、分類の統合やラベルのみの分割へ自動変更しません。

既定ではPCAは50次元、クラスタ数は3、特徴抽出のbatch sizeは16です。変更は `--pca-components`、`--num-clusters`、`--batch-size` で明示します。PCAの次元数に対して画像が足りない場合も自動縮小せず停止します。クラスタリングのseedは既定42で、`--cluster-seed` により変更できます。学習モデルの分割用seedとは独立しています。

VGG16特徴抽出は画像をバッチ単位で処理します。一方、PCAには全特徴行列と作業用メモリが必要です。特徴行列だけで画像数×25,088×4バイトとなるため、`--batch-size` を下げてもPCAのメモリ使用量は減りません。

| 出力 | 内容 |
| --- | --- |
| `train_strata.csv` | 画像ID・正解ラベル・画像クラスタ・明るさ・層化ラベル・fold割当・分割seedとfold数 |
| `summary.json` | 特徴抽出・クラスタリング・明るさの分類数と区切り値・分位点の計算対象と方法・binごとの件数・分割条件と集計 |
| `features.npy` | VGG16で抽出した画像特徴量。新規に特徴抽出する場合のみ |
| `clustering.npz` | PCAとK-meansの状態。新規に特徴抽出する場合のみ |

これらは配布データから生成する非公開の成果物です。リポジトリ外に保存します。学習では保存された `fold_index` を使い、seed・fold数との一致を確認します。`--model` から学習と同じ分割設定を選ぶため、モデルを変える場合はそのモデル名で分割CSVも準備してください。同じ分割で比較する場合は、準備と学習の両方に同じ `--seed` と `--num-folds` を指定します。

同じ画像に対するクラスタは再利用できます。次は前の結果を使い、EVA Large 196のseed・fold数で分割し直す例です。特徴抽出・PCA・K-meansを省略でき、この処理にTensorFlowは不要です。

```bash
fan-classifier prepare-strata \
  --model eva_large_196 \
  --image-root /path/to/private-data/train \
  --clusters-csv /path/to/private-data/prepared/eva_large_336/train_strata.csv \
  --output-dir /path/to/private-data/prepared/eva_large_196 \
  --brightness-bins 3
```

クラスタを別途計算している場合は、`image_filename` と整数の `scene_cluster`（別名 `cluster` も可）を列に持つCSVを指定できます。CSVと画像ディレクトリの画像IDが完全に一致することを検証し、正解ラベルは親フォルダから取得、明るさは元画像から再計算します。

既成のCSVを利用する場合は、次の列を読み込めます。

| CSV | 必須列 | 意味 |
| --- | --- | --- |
| 学習 | `image_filename` | クラスディレクトリ内の画像名 |
| 学習 | `holding_label` または既存の `check` | `1`: hold、`0`: not-hold |
| 学習 | `stratification_label` または既存の `new_fold` | 分割時に比率を保つカテゴリ。既成のfold番号ではない |
| 推論 | `image_filename` | `test/` 内の画像名。CSVの順序を保持 |

CSVにはヘッダーが必要です。画像IDの重複、画像の欠損、ラベルの不整合、少なすぎる層化カテゴリを実行前に検証します。各層化カテゴリにはfold数以上の画像が必要です。

`new_fold` は `stratification_label` の別名で、層化カテゴリとして扱います。検証用の組の番号を記録する `fold_index` とは役割が異なります。保存済みの分割を使うCSVには、`fold_index`・`split_seed`・`split_count` の3列をそろえて指定します。

## 学習と推論

例では `eva_large_336` を選択しています。別モデルを試す場合は、`--model` と出力先を変更します。

```bash
fan-classifier train \
  --model eva_large_336 \
  --train-csv /path/to/private-data/prepared/eva_large_336/train_strata.csv \
  --image-root /path/to/private-data/train \
  --output-dir runs/eva_large_336

fan-classifier predict \
  --model eva_large_336 \
  --test-csv /path/to/private-data/test.csv \
  --image-root /path/to/private-data/test \
  --run-dir runs/eva_large_336 \
  --output-dir outputs/eva_large_336
```

既定では全foldを対象とします。一部を実行する場合は、学習と推論の両方に `--folds 0` のように明示します。予測時に必要なfoldのcheckpointがなければ停止します。

batch sizeなどを変更するには、CLI引数またはJSON設定を使います。優先順位は「モデルの既定値 → JSON → CLI引数」です。

```bash
fan-classifier train \
  --model eva_large_196 \
  --train-csv /path/to/private-data/prepared/eva_large_196/train_strata.csv \
  --image-root /path/to/private-data/train \
  --output-dir runs/eva_large_196_small_batch \
  --config configs/small_batch.example.json \
  --folds 0
```

学習中はCrossEntropyLoss、AdamW、CosineAnnealingLRを使い、CUDAではAMPを有効にします。モデルごとに指定した検証指標で最良checkpointを選びます。検証・推論は評価モードで実行します。詳細な既定値は `describe-model` と[実験記録](experiments.md)で確認できます。

正規化は、固定timm版のモデル別mean/stdを既定とします。`--normalization imagenet` を指定するとImageNet共通値を使います。正規化の記録がないcheckpointの推論にもImageNet値を適用します。[設定と比較方法](evaluation.md)

学習結果はfoldごとの `best.pt`、`metrics.csv`、`training.json`、`fold_manifest.json` に保存します。推論結果は次の3ファイルです。

新規学習では検証foldのみの `oof.csv` と出自を記録する `oof.json` も保存します。全foldの完了後、`evaluate-oof` / `compare-oof` で評価できます。ここで保存するOOFと、下表の提出用fold平均は別の出力です。[OOF評価の手順](evaluation.md)

実行全体の `run.json` には設定・学習CSVのSHA-256・環境を保存します。同じrunへ別foldを追加する場合は、それらの一致を確認します。推論時にもcheckpointとrunのモデル・設定・CSVハッシュを照合し、学習が完了していないfoldを拒否します。CSVハッシュは画像ファイルの内容変更までは検出しないため、同じrunの途中で入力画像を変更しないでください。

| 出力 | 内容 |
| --- | --- |
| `probabilities.csv` | 画像IDと、fold平均した `holding_probability` |
| `submission.csv` | 画像IDと `holding_probability > 0.5` の判定値。ヘッダーなし |
| `prediction.json` | モデル・対象fold・集約方法などの記録 |

`submission.csv` は入力の `test.csv` の順序で、画像IDと判定値の2列を出力します。実際に提出する場合は、対象の提出仕様と一致するか確認してください。既存の実行結果を上書きしないため、再実験には新しい出力先を使います。

## モデル間の確率平均

各モデルの `predict` でfold平均を作成した後、モデル単位のCSVを集約します。以下は2モデルの例です。

```bash
fan-classifier ensemble \
  --inputs outputs/eva_large_336/probabilities.csv \
           outputs/eva_large_196/probabilities.csv \
  --output-dir outputs/ensemble
```

6モデルを単純平均する場合は、上のモデル表の6種類について学習・推論を実行し、それぞれの `probabilities.csv` を1つずつ渡します。fold数が異なるため、「fold平均 → モデル平均」の順序で集約し、各モデルが同じ重みになるようにします。

## 構成と検証

```text
.
├── README.md
├── configs/                 # 学習設定の上書き例
├── docs/                    # 設計・実験・コードの案内・利用技術
├── src/fan_classifier/
│   ├── registry.py          # モデル名・入力サイズ・既定値・factory
│   ├── config.py            # 設定の統合と検証
│   ├── features.py          # VGG16特徴抽出と元画像の明るさ
│   ├── stratification.py    # PCA・K-means・Triple Stratified分割
│   ├── preparation.py       # 特徴抽出・クラスタ再利用・非公開CSV保存
│   ├── data.py              # CSV・画像・層化分割
│   ├── transforms.py        # リサイズ・パディング・augmentation
│   ├── engine.py            # 学習・検証・checkpoint
│   ├── prediction.py        # 推論と画像IDを照合する平均
│   └── cli.py               # 共通の実行入口
├── tests/                   # 合成入力での検証
├── scripts/                 # 公開候補ファイルの検査
├── requirements-cpu.lock    # 検証したPython 3.12 / Linux CPU環境
├── requirements-stratification-cpu.lock  # TensorFlow特徴抽出を含むCPU環境
└── pyproject.toml
```

```bash
python -m unittest discover -s tests -v
python scripts/check_public_files.py
```

学習・checkpoint保存・推論は合成画像と小型モデルを使い、6種類の実モデルはmetaデバイスで生成と出力形状を検証します。Triple Stratified分割では明るさの分位点とbinごとの件数、同じ明るさの扱い、少数カテゴリの拒否、分割の整合性を確認します。TensorFlow環境では、重みをダウンロードせず、初期化したVGG16の実際のforward処理と特徴形状も検証します。テストで配布画像は使用しません。GitHub ActionsはTensorFlowを含むCPU lockを導入し、これらを実行します。

6モデルの既定の `pretrained=True` についても、timmから対応するHugging Faceリポジトリの取得処理に到達することを確認します。ダウンロード処理はテスト用に置き換えるため、重みの実取得は行いません。また、checkpoint保存用のCPUコピーが次のepochまで残らないことと、保存に失敗しても直前の正常なcheckpointを読み込めることを確認します。

metaデバイスではパラメータや中間特徴の実メモリを確保せず、モデル構造とテンソルの形状を確認できます。前処理・分割・モデル・予測の集約を選んだ理由は[設計と手法](solution.md)に記載しています。

実装を読む順序と各ファイルの役割は[コードの読み方](inventory.md)、ライブラリに任せている部分とこのプロジェクトで実装した部分は[実装範囲と利用技術](provenance.md)にまとめています。
