# Open Decision Models Benchmark Suite 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

オープンソース判断特化モデル（Decision Models）および汎用LLM（Direct Logit方式）の実機性能・境界識別力・確率較正を横断検証するためのベンチマークスイートです。

TypeSafe AI によるプロプライエタリな判断モデル **Jev** と、オープンな代替モデル（**Kev-4B**、**Qwen3.5-4B Direct Logit**、**CLM-8B Contrastive LM** など）を同一のテストデータ・条件で比較検証できます。

---

## 🎯 主な特徴

1. **データとコードの完全分離（Single Source of Truth）**:
   テストケース（童話比喩、虚偽 vs 変装、物理遮蔽、サイコロ較正）やプロンプト定義を `data/*.json` で一元管理。コード内にテスト文章をハードコードしません。
2. **共通のRunner抽象化**:
   `BaseDecisionRunner` 基底クラスにより、Cloud API（本家Jev）、ローカルPyTorch / DeltaNet（Kev-4B）、ローカル推論サーバー（llama-server / SGLang）、および Contrastive LM（CLM-8B / Qwen3-8B + 射影ヘッド）を同一インターフェースで統一評価。
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
│   ├── dice_cases.json          # 6面・100面サイコロ確率較正ケース
│   ├── ood_cases.json           # 未知・架空・未観測知識テストケース（13問）
│   ├── route_rule_cases.json    # 条件付きルール判定ケース（3問）
│   └── trivalent_cases.json     # 証拠グラデーション・三値論理ケース（6問）
│
├── models/                      # モデル実行アダプター
│   ├── base.py                  # BaseDecisionRunner 基底クラス
│   ├── jev_runner.py            # 本家 Jev (TypeSafe AI API)
│   ├── kev_runner.py            # Kev-4B (DeltaNet / Pointer Head)
│   ├── qwen_runner.py           # Qwen3.5-4B (llama-server Direct Logit)
│   └── clm_runner.py            # CLM-8B (Qwen3-8B + Contrastive Head / Hazy)
│
└── results/                     # 実機検証の生データ（JSON）と集計レポート
    ├── summary_table.md         # マークダウン形式の集計比較表（4モデル対応）
    ├── jev_results.json         # 本家 Jev の実測生データ
    ├── kev_results.json         # Kev-4B の実測生データ
    ├── qwen_results.json        # Qwen3.5-4B の実測生データ
    ├── clm_results.json         # CLM-8B の実測生データ
    ├── triad_ood_results.json   # 未知知識・ランダム事象の4モデル比較生データ
    ├── triad_route_results.json # 条件付きルール判定の比較生データ
    └── triad_trivalent_results.json # 三値論理・証拠グラデーション比較生データ
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

#### CLM-8B (Contrastive LM) のみ実行:
```bash
python run_benchmark.py --model clm --task boundary
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
  正体開示（かぐや姫: **3.3%**、スパイ身分提示: **2.4%**）や演技（名優: **2.1%**）において、本家 Jev よりも鋭敏に正体隠蔽を否定・除外できる高い解像度を実証。マルチ質問（4問同時評価）でも ~60ms の超低遅延を達成。
- **Qwen3.5-4B (llama-server) の安定性**:  
  旧命題・新命題の全 26 試行において **100% 正解（誤検知ゼロ）** を達成。実務での意味grep（全行ダイレクト走査）における強固な安定性を裏付け。
- **CLM-8B (Contrastive LM) の二面性（新発見）**:  
  - **三値論理の崩壊**: scaled cosine softmax（scale=100）の幾何学的特性により、曖昧なグレーゾーンにおける「保留（どちらとも言えない）」の確率が **1%未満（1.1%〜1.7%）** に潰れ、白黒両極端への強制二極分化（コイン投げ）を起こす弱点を特定。
  - **語彙トラップ**: 童話境界テスト（正解率 **30.8%〜38.5%**）において、「秘密」「嘘」「演技」「マスク」といった単語に埋め込みが過剰に引きずられ、警察手帳の開示や告白といった文脈論理を誤爆。
  - **架空事実の強力な棄却（95%〜96%）**: 一方で、架空鉱石や未確定の未来予測に対しては 95%〜96% の確信度でスパッと No と切り捨てる極めて強固な保守的ネガティブフィルター性能を実証。
- **命題設計（Re-framing）の威力**:  
  花粉症マスクや着替えカーテンなどの物理遮蔽において、対称な命題（身分や正体を意図的に偽り隠蔽しているか？）を再定義することで、全モデルが一斉に非該当（1〜5%）へとシャープに較正されることを実証。
- **未知知識と閉世界仮説（OOD検証）**:  
  Jevは架空命題やコインの表裏（理論値50%）すら **No 79%〜94%** で冷酷に切り捨てる極端な閉世界証拠主義を示す一方、Kev-4Bはコイン表裏で **49.7% vs 50.3%** と数学的対称性を忠実に再現。
- **三値論理（Trivalent Logic）のブレークスルー**:  
  Yes/No二値判定ではグレーゾーンが「No 100%」に潰れて見逃されるが、Choice 3択（ほぼ確定 / どちらとも言えない / 確定不可能）を与えると、Jev・Kev・Qwen全モデルが **70%〜96% で「保留」を完璧に分離・識別** することを発見（CLM-8B のみ崩壊）。

---

## 📜 関連技術記事（Zenn シリーズ）

本リポジトリの検証データおよび理論的背景は、以下の記事シリーズに基づいています：

1. **第1弾（ローカル再現・エンジン比較）**:  
   [Jevをローカルでどこまで再現できるか検証しました｜オープンDecision Model比較と実戦配備](https://zenn.dev/null_teck/articles/local-jev-reproduce)
2. **第2弾（確率較正・サイコロ問題）**:  
   [Jevはサイコロを振らないがローカルLLMは振れるのか？｜「較正された確率」の検証と命題設計のブレークスルー](https://zenn.dev/null_teck/articles/jev-dice-calibration)
3. **第3弾（実務使い分け・未知知識）**:  
   [JevとローカルLLMは何が違うのか？｜未知知識の挙動、ルール判定、そして「Jevでしか無理な要件」の検証](https://zenn.dev/null_teck/articles/jev-vs-local-usecases)
4. **第4弾（英語グローバル発信・CLM比較）**:  
   [Evaluating CLM-8B as a System 1 Decision Engine vs Jev and Kev](https://zenn.dev/null_teck/articles/clm-vs-decision-models)

---

## 🙏 先行研究・謝辞
- **TypeSafe AI**: Jev（Typed Decision Model）の提唱
- **uehaj 氏**: 『[Jevのキラーアプリ、「意味で探す grep」を作った](https://zenn.dev/uehaj/articles/jev-semgrep-grep-by-meaning)』
- **林寛太 氏**: 『[Jevはサイコロを振らない｜「較正された確率」の意外な落とし穴](https://note.com/kantahayashiai/n/n4c54eed30787)』
- **Jared Palmer 氏**: `jaredpalmer/kev-4b`（オープン Decision Model 実装）
- **Jacky Kwok 氏 & Hazy Research**: `Contrastive-LM/CLM`（CLM-8B モデルおよびサービング実装）

---

## 📄 ライセンス
MIT License
