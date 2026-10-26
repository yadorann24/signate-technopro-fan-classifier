# 正規化・OOF評価・比較実験

このページでは、正規化を設定し、OOF予測からモデルや前処理を比較する手順を説明します。実行先の `runs/`、評価出力、画像IDを含むCSVは非公開で管理します。

## モデル別の正規化

新規学習では `timm==1.0.11` の登録済み事前学習設定からmean/stdを取得します。設定の取得だけでは、モデルの生成や重みのダウンロードは行いません。画像をRGBの0〜1へ変換した後、チャンネルごとに `(value - mean) / std` を適用します。

| モデル | 新規学習の既定正規化 |
| --- | --- |
| `eva02_large_448` | OpenAI CLIP |
| `eva_giant_560` | ImageNet |
| `eva_giant_336` | ImageNet |
| `eva_large_336` | OpenAI CLIP |
| `eva_giant_clip_224` | OpenAI CLIP |
| `eva_large_196` | OpenAI CLIP |

| 設定 | mean（RGB） | std（RGB） |
| --- | --- | --- |
| ImageNet | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| OpenAI CLIP | 0.48145466, 0.4578275, 0.40821073 | 0.26862954, 0.26130258, 0.27577711 |

対応するtimm識別子は[モデル一覧](solution.md)にあります。値の出典は[固定版のEVAモデル設定](https://github.com/huggingface/pytorch-image-models/blob/v1.0.11/timm/models/eva.py)です。

`--normalization` でモデルの事前学習設定とImageNet共通値を選べます。比較するときは、正規化以外の条件をそろえて別々のrunで学習します。

```bash
# 新しいrunでモデルの事前学習設定を使う
fan-classifier train --model eva_large_336 \
  --train-csv /path/private/prepared/train_strata.csv \
  --image-root /path/private/train --output-dir runs/eva336_pretrained \
  --normalization pretrained

# 同じCSVと学習条件で、ImageNet共通値を比較する
fan-classifier train --model eva_large_336 \
  --train-csv /path/private/prepared/train_strata.csv \
  --image-root /path/private/train --output-dir runs/eva336_imagenet \
  --normalization imagenet
```

mean/std・設定モード・取得元は `run.json` とcheckpointに保存され、推論では保存値を復元します。**正規化の記録がないcheckpointにはImageNet値を使います。** 設定を変える場合は、新しい出力先で学習してください。

独自の値を比較する場合は、JSONの `normalization` に `{"mode": "custom", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}` のように指定します。これは書式を示すための例です。

## 検証foldだけからOOFを保存する

新しく `train` を実行すると、各foldの学習終了後に **best.ptを読み直し、そのfoldの検証画像だけ**を推論します。最終epochの重みをそのまま使うのではなく、設定した指標で選んだcheckpointを使用します。

| fold内のファイル | 内容 |
| --- | --- |
| `oof.csv` | `image_filename`, `holding_label`, `fold_index`, `holding_probability` |
| `oof.json` | モデル、fold、選択epoch、設定、学習CSV・checkpoint・OOFのハッシュ |
| `fold_manifest.json` | 学習画像ID、検証画像IDと正解ラベル、分割条件 |

各画像について、その画像を学習に使っていないfoldから得た予測をOOF（out-of-fold）と呼びます。全foldが完成すると、学習画像全体を1画像1予測で評価できます。

```bash
fan-classifier evaluate-oof \
  --run-dir runs/eva336_pretrained \
  --output-dir outputs/eva336_oof

fan-classifier compare-oof \
  --run-dirs runs/eva336_pretrained runs/eva336_imagenet \
  --output-dir outputs/normalization_comparison
```

評価には全foldの `best.pt` とOOFの記録を残しておく必要があります。未完了fold、検証IDの重複・欠落、学習IDとの重複、ラベル不一致、checkpointやOOFの内容変更は拒否します。学習画像を通常の `predict` で全fold推論したCSVはOOFとして受け付けません。

| 評価出力 | 内容 |
| --- | --- |
| `evaluation.json` | 単体の全体・fold別accuracy / F1 / log-loss / 混同行列、設定と出自。複数runなら等重みのOOF確率平均も評価 |
| `errors.csv` | 誤分類した画像ID・正解・確率・fold。`member_index` は入力runの0始まり順、平均は `ensemble` |
| `probabilities.csv` | 単体のOOF確率、または各runのOOF確率を平均した値 |

判定は提出用推論と同じ `holding_probability > 0.5`、F1の正例は `hold=1` です。比較するrunは画像IDと正解ラベルの一致が必須です。foldが異なっても各画像を除外して学習した予測同士なら比較できますが、`same_fold_assignments=false` と記録します。条件を一つだけ変える実験では、同じfold割当を使用してください。

## この評価で分かること・残る限界

OOFで平均するのは、各画像について各runの検証担当foldが出した確率です。**提出時の全fold平均そのものの精度測定ではありません。** また、同じ検証データでbest epochを選んでいるため、独立したテストのスコアでもありません。OOFで重みや閾値を何度も調整する場合は、別のholdoutや外側の交差検証で選択後の性能を確認します。

Triple Stratifiedはラベル・見た目・明るさの構成をそろえますが、同一人物・動画・撮影セッションの混在を防ぎません。未知の人物や撮影条件への汎化を評価する場合は、その単位でgroupを分ける設計が必要です。

`errors.csv` から、false positive（持っていないのにholdと判定）とfalse negativeを分け、背景の扇子、見切れ、明るさなどとの関係を**学習画像で**確認します。画像クラスタや明るさ別の精度を調べる場合は、非公開の分割CSVと画像IDで結合します。集計値を公開する場合も、入力画像や個別IDを含めない形にします。

## 比較実験の設計

| 比較 | 変える条件 | 共通にする条件 | 記録する結果 |
| --- | --- | --- | --- |
| 正規化 | ImageNet統一 / モデル標準 | モデル、fold、seed、epoch、拡張、選択指標 | OOF F1、FP/FN、選択epoch |
| モデル選定 | モデル構成 | 画像とfold、比較可能な学習予算、選択指標 | OOF F1、推論時間、重み容量、メモリ |
| 平均の有効性 | 単体 / 少数モデル / 6モデル | 同じ画像IDのOOF、判定閾値 | F1差、誤りの重なり、推論モデル数 |
| 背景対策 | 正解を変えない拡張の有無 | その他の学習・分割条件 | 対象例でのFP/FNと全体F1 |

モデルごとの既定値にはfold数・seed・checkpoint選択指標の違いがあります。モデル固有の効果を説明する比較では、`--seed`、`--num-folds` とJSON設定をそろえます。
