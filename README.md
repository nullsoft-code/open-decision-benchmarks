# Open Decision Models Benchmark Suite 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

オープンソース判断特化モデル（Decision Models）および汎用LLM（Direct Logit方式）の実機性能・境界識別力・確率較正を横断検証するためのベンチマークスイートです。

TypeSafe AI によるプロプライエタリな判断モデル **Jev** と、オープンな代替モデル（**Kev-4B**、**Qwen3.5-4B Direct Logit** など）を同一のテストデータ・条件で比較検証できます。

---

## 🎯 主な特徴

1. **データとコードの完全分離（Single Source of Truth）**:
   テストケース（童話比喩、虚偽 vs 変装、物理遮蔽、サイコロ較正）やプロンプト定義を `data/*.json` で一元管理。コード内にテスト文章をハードコードしません。
2. **共通のRunner抽象化**:
   `BaseDecisionRunner` 基底クラスにより、Cloud API（本家Jev）、ローカルPyTorch / DeltaNet（Kev-4B）、およびローカル推論サーバー（llama-server / SGLang）を同一インターフェースで統一評価。
3. **命題設計（Re-framing）の比較評価**:
   疑問形フラグ（旧命題）と、対称な事象空間を定義した新命題（Re-framed）の感度差を自動測定。

---

## 📂 ディレクトリ構成

```text
open-decision-benchmarks/
├── README.md                    # 本ドキュメント
├── requirements.txt             # 依存ライブラリ
├── run_benchmark.py             # 統合ベンチマーク実行 CLI
│
├── data/                        # ベンチマークデータセット（ハードコード全廃）
│   ├── prompts.json             # 命題定義（旧命題 / 新命題）
│   ├── boundary_cases.json      # 境界識別力テストケース（13問）
│   └── dice_cases.json          # 6面・100面サイコロ確率較正ケース
│
├── models/                      # モデル実行アダプター
│   ├── base.py                  # BaseDecisionRunner 基底クラス
│   ├── jev_runner.py            # 本家 Jev (TypeSafe AI API)
│   ├── kev_runner.py            # Kev-4B (DeltaNet / Pointer Head)
│   └── qwen_runner.py           # Qwen3.5-4B (llama-server Direct Logit)
│
└── results/                     # 実機検証の生データ（JSON）と集計レポート
    ├── summary_table.md         # マークダウン形式の集計比較表
    ├── jev_results.json         # 本家 Jev の実測生データ
    ├── kev_results.json         # Kev-4B の実測生データ
    └── qwen_results.json        # Qwen3.5-4B の実測生データ
```

---

## 🚀 クイックスタート

### 1. インストール
```bash
git clone git@github.com:nullsoft-code/open-decision-benchmarks.git
cd open-decision-benchmarks
pip install -r requirements.txt
```

### 2. 環境変数の設定（Jev API を利用する場合）
`.env` ファイルを作成し、API キーを設定します：
```bash
TYPESAFE_API_KEY=your_typesafe_api_key_here
```

### 3. ベンチマークの実行

#### Qwen3.5-4B (llama-server) のみ実行:
```bash
# llama-server を起動（ポート 8089）後：
python run_benchmark.py --model qwen --task boundary
```

#### 本家 Jev のみ実行:
```bash
python run_benchmark.py --model jev --task boundary
```

#### 利用可能な全モデルを一括実行:
```bash
python run_benchmark.py --model all --task boundary
```

結果は `results/boundary_benchmark_results.json` および `results/summary_table.md` に自動出力されます。

---

## 📊 実機検証サマリ（RTX 3090 / Cloud API）

詳細な測定結果および考察は [`results/summary_table.md`](results/summary_table.md) をご覧ください。

### 主なハイライト
- **本家 Jev の盲点**:  
  「鶴の恩返し（覗かないでください）」において、物理的な目隠しを正体隠蔽と誤認し **95.0%** で誤判定。「スパイの身分提示（警察手帳の開示）」でも「潜入捜査官」という過去の語彙に釣られて **65.0%〜97.0%** で誤判定を起こす現象を確認。
- **Kev-4B の得意領域**:  
  正体開示（かぐや姫: **3.3%**、スパイ身分提示: **2.4%**）や演技（名優: **2.1%**）において、本家 Jev よりも鋭敏に正体隠蔽を否定・除外できる高い解像度を実証。
- **Qwen3.5-4B (llama-server) の安定性**:  
  旧命題・新命題の全 26 試行において **100% 正解（誤検知ゼロ）** を達成。実務での意味grep（全行ダイレクト走査）における強固な安定性を裏付け。
- **命題設計（Re-framing）の威力**:  
  花粉症マスクや着替えカーテンなどの物理遮蔽において、対称な命題（身分や正体を意図的に偽り隠蔽しているか？）を再定義することで、全モデルが一斉に非該当（1〜5%）へとシャープに較正されることを実証。

---

## 📜 関連技術記事（Zenn 三部作）

本リポジトリの検証データおよび理論的背景は、以下の記事シリーズに基づいています：

1. **第1弾（ローカル再現・エンジン比較）**:  
   [Jevをローカルでどこまで再現できるか検証しました｜オープンDecision Model比較と実戦配備](https://zenn.dev/null_teck/articles/local-jev-reproduce)
2. **第2弾（確率較正・サイコロ問題）**:  
   [Jevはサイコロを振らないがローカルLLMは振れるのか？｜「較正された確率」の検証と命題設計のブレークスルー](https://zenn.dev/null_teck/articles/jev-dice-calibration)
3. **第3弾（実務使い分け・未知知識）**:  
   [JevとローカルLLMは何が違うのか？｜未知知識の挙動、ルール判定、そして「Jevでしか無理な要件」の検証](https://zenn.dev/null_teck/articles/jev-vs-local-usecases)

---

## 🙏 先行研究・謝辞
- **TypeSafe AI**: Jev（Typed Decision Model）の提唱
- **uehaj 氏**: 『[Jevのキラーアプリ、「意味で探す grep」を作った](https://zenn.dev/uehaj/articles/jev-semgrep-grep-by-meaning)』
- **林寛太 氏**: 『[Jevはサイコロを振らない｜「較正された確率」の意外な落とし穴](https://note.com/kantahayashiai/n/n4c54eed30787)』
- **Jared Palmer 氏**: `jaredpalmer/kev-4b`（オープン Decision Model 実装）

---

## 📄 ライセンス
MIT License
