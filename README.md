# Open Decision Benchmarks Suite 🔬

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

オープンソース判断特化モデル（System 1 Decision Models）およびオープン基盤モデルの実機性能、境界識別力、確率較正（Calibration）、長文契約書耐性、語用論的論理トラップ耐性を多角的に検証するための統合ベンチマークスイートです。

TypeSafe AI によるプロプライエタリな判断モデル **Jev**、オープンモデル **Lev-8B**、軽量モデル **Jeff-0.8B**、線形アテンションモデル **Kev-4B**、プレフィックスキャッシュ基盤 **Qwen3.5-4B (SGLang)**、および対照学習モデル **CLM-8B** を同一の検証条件下で比較評価できます。

---

## 🎯 主な検証スコープ

1. **S1Bench（300問）統一評価**:
   - `levbench`（TypeSafe ADR-010 `/v1/systemone` 統一プロトコル）公式ハーネスおよびスタンドアロン実行による厳密測定。
   - Macro Accuracy、5タスク分類正解率、Log-Loss、Brier Score、ECE（較正誤差）、p50レイテンシ、VRAM消費量の網羅的計測。
2. **Phase 2: 長文契約書・仕様書ベンチマーク（15問 / 2,000〜4,500+ トークン）**:
   - 個別業務合意書およびNDA・API仕様書から、局所例外規定や発効条件を判定する実務文書タスク。
   - Cross-Attention（Lev / Jev / Jeff）、RadixAttention（Qwen3.5 SGLang）、線形状態圧縮（Kev-4B）、Bi-Encoder対照埋め込み（CLM-8B）の挙動差を比較。
3. **Phase 3: 語用論・文法論理トラップベンチマーク（20問 / 日本語11問・英語9問）**:
   - 否定疑問文のYes/No反転、障害・クラッシュ等の表層感情バイアス、入れ子例外規定、会話の公準（裏読み）に対する耐性テスト。
4. **CLM-8B Action Caching 実機検証（5シナリオ）**:
   - 選択肢埋め込みの事前計算・キャッシュ機構（Action Caching）が有効に機能する領域（固定選択肢60意図分類、リアルタイムTick制御、言明型リフレーミング）と、構造的制約（三値論理の確信度偏向、長文局所情報の希釈）を実測。

---

## 📂 ディレクトリ構成

```text
open-decision-benchmarks/
├── README.md                           # 本ドキュメント
├── requirements.txt                    # 依存ライブラリ
├── run_benchmark.py                    # 境界テスト統合実行 CLI
│
├── data/                               # ベンチマークデータセット
│   ├── s1bench/                        # S1Bench 全6タスク（300問）データセット
│   │   ├── aegis2.json                 # 安全性ガードレール判定（50問）
│   │   ├── boolq.json                  # 二値QA判定（50問）
│   │   ├── helpsteer2.json             # 5段階評価スコアリング（50問）
│   │   ├── massive-en-US.json          # 60意図分類タスク（50問）
│   │   ├── paws.json                   # 言い換え・語順反転判定（50問）
│   │   └── vitaminc-dev.json           # 事実検証・論理矛盾判定（50問）
│   ├── levbench_tasks/                 # levbench 公式フォーマットタスクデータ
│   ├── phase2_long_context/            # 長文契約書ベンチマーク（15問）
│   │   ├── build_dataset.py            # 長文データセット生成スクリプト
│   │   └── long_context_benchmark.json # 実務契約書・NDAテストケース
│   ├── phase3_pragmatic_logic/         # 語用論・文法論理トラップ（20問）
│   │   └── test_cases.json             # 否定疑問文・入れ子例外テストケース
│   ├── prompts.json                    # 命題定義（旧命題 / 新命題）
│   ├── boundary_cases.json             # 童話境界テストケース（13問）
│   ├── dice_cases.json                 # 確率較正サイコロケース
│   ├── ood_cases.json                  # 未知知識・ランダム事象ケース
│   ├── route_rule_cases.json           # ルール判定ケース
│   └── trivalent_cases.json            # 三値論理・保留ケース
│
├── servers/                            # ADR-010 /v1/systemone 準拠サーバー群
│   ├── jeff_server.py                  # Jeff-0.8B 専用サービングサーバー (:8092)
│   ├── kev_server.py                   # Kev-4B Gated DeltaNet サービングサーバー (:8090)
│   └── sglang_systemone_adapter.py     # Qwen3.5 SGLang 用 SystemOne アダプター (:8002)
│
├── scripts/                            # 各ベンチマーク評価スクリプト
│   ├── run_s1bench_jeff_clm.py         # Jeff-0.8B / CLM-8B S1Bench 評価
│   ├── run_full_eval_jeff_clm.py       # Jeff / CLM 全フェーズ一括実行
│   ├── run_phase2_long_context.py      # Phase 2 長文契約書ベンチマーク評価
│   ├── run_phase3_pragmatic_logic.py   # Phase 3 語用論ロジック評価
│   ├── verify_action_caching_clm.py    # CLM-8B Action Caching 5シナリオ実機検証
│   ├── run_lev_eval.py                 # Lev-8B ローカル評価スクリプト
│   ├── run_jev_eval.py                 # Jev Cloud API 評価スクリプト
│   └── run_qwen_sglang_eval.py         # Qwen3.5 SGLang 評価スクリプト
│
├── models/                             # モデル実行基底・個別アダプター
│   ├── base.py                         # BaseDecisionRunner 基底クラス
│   ├── jev_runner.py                   # Jev Cloud API ランナー
│   ├── kev_runner.py                   # Kev-4B ランナー
│   ├── qwen_runner.py                  # Qwen3.5 Direct Logit ランナー
│   └── clm_runner.py                   # CLM-8B Contrastive LM ランナー
│
└── results/                            # 実機検証結果（JSON）と集計テーブル
    ├── summary_table.md                # 総合ベンチマーク集計マークダウン表
    ├── results_levbench_official.json  # levbench 公式統一ハーネス測定生データ
    ├── results_phase2_long_context.json# Phase 2 長文契約書測定生データ
    ├── results_phase3_pragmatic_logic.json # Phase 3 語用論測定生データ
    ├── clm_action_caching_verification.json # CLM Action Caching 実機検証データ
    ├── results_s1bench_jeff_clm.json   # Jeff / CLM S1Bench スタンドアロン結果
    ├── results_jev.json                # Jev S1Bench 結果
    ├── results_lev.json                # Lev-8B S1Bench 結果
    ├── results_kev.json                # Kev-4B S1Bench 結果
    └── results_qwen_sglang.json        # Qwen3.5 SGLang S1Bench 結果
```

---

## 📊 ベンチマーク総合結果サマリー

### 1. S1Bench（levbench 公式統一ハーネス・全300問）

| モデル | Macro Acc | 分類5タスク | Log-Loss | Brier Score | ECE | p50 レイテンシ | VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Jev (Cloud API)** | **77.33%** | 86.4% | 1.211 | 0.342 | 0.114 | **186 ms** | 0 MB |
| **Lev (Local 8B)** | 75.00% | 81.2% | 0.797 | 0.370 | 0.116 | 573 ms | 約 11 GB |
| **Jeff-0.8B (Local)** | **76.67%** | **88.4% (最優秀)** | **0.545 (最優秀)** | **0.283 (最優秀)** | **0.080 (最優秀)** | 327 ms | **1.65 GB** |
| **Kev-4B (Local)** | 70.00% | 77.6% | 0.852 | 0.413 | 0.120 | **169 ms** | 7.94 GB |
| **Qwen3.5-4B (SGLang)** | 60.00% | 65.2% | 0.906 | 0.521 | 0.179 | 2,128 ms | 約 20.4 GB |
| **CLM-8B (Local)** | *(※未適用)* | - | - | - | - | - | 約 16 GB |

*(※注: CLM-8B は動的HTTP直列リクエストにおいて Action Caching が利用できず、毎問 Cold 推論を強いられるため、本来の性能特性を反映しないことから `levbench` ハーネス測定からは除外し、専用検証スクリプトで測定しました)*

### 2. Phase 2 長文契約書（15問 / 2,000〜4,500+ トークン）

| モデル | 正解率 | 正解数 | 1問平均時間 | 処理特性 |
| :--- | :---: | :---: | :---: | :--- |
| **Jev (Cloud API)** | **100.0%** | 15/15 | **0.22 秒** | クラウド側の専用最適化により低遅延と100%を維持 |
| **Lev (Local 8B)** | **100.0%** | 15/15 | 23.88 秒 | 精度は100%だが、毎回ゼロからのFull Attention再計算で遅延増大 |
| **Jeff-0.8B (Local)** | **100.0%** | 15/15 | **0.56 秒** | 0.8Bの軽量アテンションで高速性と100%を両立 |
| **Qwen3.5-4B (SGLang)**| **100.0%** | 15/15 | 2.85 秒 | RadixAttentionによるKVキャッシュ再利用でLevの8倍以上高速 |
| **Kev-4B (Local)** | 60.0% | 9/15 | **0.25 秒** | 線形時間推論（0.25s/問）だが、固定状態圧縮で局所条項の想起精度低下 |
| **CLM-8B (Cold)** | 20.0% | 3/15 | 10.78 秒 | 8B Dense逐次計算。長文トピックベクトルに局所条件が埋没 |
| **CLM-8B (Cached)** | 40.0% | 6/15 | 0.78 秒 | 0.78秒に短縮するも、Bi-Encoder平均プーリングで15問中13問がFalse判定 |

### 3. Phase 3 語用論・文法論理トラップ（20問）

| モデル | 全体正解率 | 日本語正解率 | 英語正解率 | 境界線・留意点 |
| :--- | :---: | :---: | :---: | :--- |
| **Jev (Cloud API)** | **100.0% (20/20)** | **100.0% (11/11)** | **100.0% (9/9)** | 全カテゴリで誤答なし |
| **Lev (Local 8B)** | **100.0% (20/20)** | **100.0% (11/11)** | **100.0% (9/9)** | 全カテゴリで誤答なし |
| **Kev-4B (Local)** | 90.0% (18/20) | 90.9% (10/11) | 88.9% (8/9) | 多重入れ子例外、会話の公準（裏読み）でつまずき |
| **Qwen3.5-4B (SGLang)**| 90.0% (18/20) | 90.9% (10/11) | 88.9% (8/9) | 多重入れ子例外でつまずき |
| **Jeff-0.8B (Local)** | 75.0% (15/20) | 81.8% (9/11) | 66.7% (6/9) | 英語の複雑な推意、入れ子例外でつまずき |
| **CLM-8B (Local)** | 75.0% (15/20) | 63.6% (7/11) | 88.9% (8/9) | 日本語助詞ニュアンス、入れ子例外でつまずき |

---

## 🚀 再現手順

### 1. ADR-010 /v1/systemone サーバーの起動

#### Jeff-0.8B サーバー起動（ポート 8092 / RTX 3090 約 1.65 GB VRAM）:
```bash
python servers/jeff_server.py --port 8092 --host 127.0.0.1
```

#### Kev-4B サーバー起動（ポート 8090 / RTX 3090 約 7.94 GB VRAM）:
```bash
python servers/kev_server.py --port 8090 --host 127.0.0.1
```

#### Qwen3.5 SGLang アダプター起動（SGLang サーバー :8000 起動後、ポート 8002）:
```bash
python servers/sglang_systemone_adapter.py --port 8002 --host 127.0.0.1
```

### 2. 各ベンチマークの実行

#### levbench 公式ハーネスの実行（例: Jeff-0.8B）:
```bash
levbench run --url http://127.0.0.1:8092/v1/systemone --token test --tasks data/levbench_tasks
```

#### Phase 2 長文契約書ベンチマークの実行:
```bash
python scripts/run_phase2_long_context.py
```

#### Phase 3 語用論・文法論理トラップベンチマークの実行:
```bash
python scripts/run_phase3_pragmatic_logic.py
```

#### CLM-8B Action Caching 5シナリオ実機検証の実行:
```bash
python scripts/verify_action_caching_clm.py
```

---

## 📄 ライセンス

本リポジトリのコードおよびベンチマークデータは MIT License のもとで公開されています。
詳細は [LICENSE](LICENSE) を参照してください。
