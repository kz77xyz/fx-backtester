# FX Tester — 無料の裁量トレード検証ツール

ForexTester / TradingView 有料版の代わりに、**自分のデータ・全インジ・マルチタイムフレーム・描画**を無料・インストール最小で。ローカルで動くので回数制限もデータ課金もなし。

## 特徴
- **MTF 4分割**（15m / 1h / 4h / 日）＋ 1画面切替。上位足は再生で未確定バーが形成（先読みなし）
- **インジ無制限**（SMA/EMA/BB/RSI/ATR/MACD/ADX）。全足に反映・パラメータ/色変更可
- **描画ツール**（水平線・トレンドライン・フィボ）。価格/時間に固定、全足に表示、ドラッグで移動
- **発注**：成行・指値・逆指値（チャートクリックで価格指定）、SL/TPをクリック＋RR自動配置
- **バー送り再生**（速度可変）・日付ジャンプ
- **銘柄ダウンロード**：ボタンで Dukascopy から15分足を取得（無料・APIキー不要）
- **トレード記録**を `sessions/` にCSV保存 → Claude Code / codex 等のAIに読ませて弱点分析

## 使い方
```bash
pip install -r requirements.txt
python fxtester.py            # 既定 usdjpy_15m.csv（無ければUIの「取得」でDL）
python fxtester.py eurusd_15m.csv --bars 30000
```
ブラウザで http://localhost:8787 が開く。

CSV形式: `Datetime,Open,High,Low,Close`（UTC推奨）。

## 構成
- `fxtester.py` … ローカルサーバ（標準ライブラリ＋pandas）
- `fxtester.html` … UI（lightweight-charts）
- `signals.py` / `data.py` … インジ計算・CSVローダ

## ライセンス
MIT
