"""
簡易フォレックステスター（裁量トレード検証ツール）

ForexTester/TradingView 有料の代替。自分のCSVを無制限・オフラインで、
インジ無制限、速いバー送り検証。トレード記録は sessions/ に人間可読CSVで残し、
Claude Code / codex が読んで分析できる（このツールの肝）。

起動: python fxtester.py [csvパス] [--port 8787]
  例: python fxtester.py usdjpy_1h.csv

チャートは lightweight-charts(TradingView製OSS)、インジ計算は signals.add_indicators を流用。
"""
import sys
import json
import csv
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from data import load_csv
from signals import add_indicators

ROOT = Path(__file__).parent
SESSIONS = ROOT / "sessions"
HTML = ROOT / "fxtester.html"

# インジ カタログ: type → (signals用ind dictを作る関数, 返す列名, 配置ペイン)
# パラメータはフロントから渡る。計算は signals.add_indicators をそのまま再利用する。
def _ind_spec(itype: str, p: dict):
    t = itype.upper()
    if t == "SMA":
        n = int(p.get("period", 20));  return {"ma_periods": [n]}, [f"MA_{n}"], "price"
    if t == "EMA":
        n = int(p.get("period", 20));  return {"ema_periods": [n]}, [f"EMA_{n}"], "price"
    if t == "RSI":
        return {"rsi_period": int(p.get("period", 14))}, ["RSI"], "new"
    if t == "ATR":
        return {"atr_period": int(p.get("period", 14))}, ["ATR"], "new"
    if t == "BB":
        return ({"bb": {"period": int(p.get("period", 20)), "std": float(p.get("std", 2.0))}},
                ["BB_upper", "BB_mid", "BB_lower"], "price")
    if t == "MACD":
        return ({"macd": {"fast": int(p.get("fast", 12)), "slow": int(p.get("slow", 26)),
                          "signal": int(p.get("signal", 9))}},
                ["MACD", "MACD_signal", "MACD_hist"], "new")
    if t == "ADX":
        return {"adx_period": int(p.get("period", 14))}, ["ADX", "DI_plus", "DI_minus"], "new"
    raise ValueError(f"未知のインジ: {itype}")

TRADE_COLS = [
    "entry_time", "exit_time", "side", "lot", "entry", "exit",
    "pips", "pnl", "bars_held", "sl", "tp",
    "rsi_at_entry", "atr_at_entry", "exit_reason",
]

# モジュールグローバル（起動時に1度だけ構築）
BARS_JSON = b""
CSV_NAME = ""
PIP_SIZE = 0.01
MAX_BARS = 20000
DF = None   # 生OHLC(DatetimeIndex)。インジはここからオンデマンド計算


def _clean(series):
    return [None if (v != v) or v in (np.inf, -np.inf) else round(float(v), 6) for v in series]


def build_bars(csv_path: str, max_bars: int = 20000) -> bytes:
    """CSV→OHLCのみJSON化（インジは /api/indicator で都度計算）。
    巨大データ対策で直近 max_bars 本だけ読む（0以下で全件）。"""
    global DF
    DF = load_csv(csv_path)
    if max_bars and len(DF) > max_bars:
        DF = DF.iloc[-max_bars:]
    # index単位(ns/us/s)に依存せず POSIX秒(UTC)へ。lightweight-chartsはUTC秒を期待。
    times = DF.index.values.astype("datetime64[s]").astype("int64").tolist()
    payload = {
        "csv": Path(csv_path).name, "pip_size": PIP_SIZE, "time": times,
        "open": [round(float(v), 6) for v in DF["Open"]],
        "high": [round(float(v), 6) for v in DF["High"]],
        "low": [round(float(v), 6) for v in DF["Low"]],
        "close": [round(float(v), 6) for v in DF["Close"]],
    }
    return json.dumps(payload).encode("utf-8")


# tf(秒) → pandasリサンプル規則。MTFで各足ごとにインジ計算するため。
_TF_RULE = {900: "15min", 1800: "30min", 3600: "1h", 14400: "4h", 86400: "1D"}


def _tf_df(tf_sec: int):
    base = DF[["Open", "High", "Low", "Close"]]
    if not tf_sec or tf_sec == 900:
        return base.copy()
    rule = _TF_RULE.get(tf_sec)
    if not rule:
        return base.copy()
    r = base.resample(rule, label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return r


def compute_indicator(itype: str, params: dict, tf_sec: int = 0) -> dict:
    """指定インジをパラメータ・足(tf)付きで計算。signals.add_indicators を再利用。
    tf>0なら15mからリサンプルした足で計算し、各バーの開始時刻(UTC秒)も返す。"""
    ind, cols, pane = _ind_spec(itype, params)
    df = _tf_df(tf_sec)
    out = add_indicators(df, ind)
    times = out.index.values.astype("datetime64[s]").astype("int64").tolist()
    return {"type": itype, "pane": pane, "time": times,
            "series": {c: _clean(out[c]) for c in cols if c in out.columns}}


# 取得可能ペア → dukascopy instrument定数名（遅延importで起動を軽く）
SYMBOLS = {
    "USDJPY": "INSTRUMENT_FX_MAJORS_USD_JPY", "EURUSD": "INSTRUMENT_FX_MAJORS_EUR_USD",
    "GBPUSD": "INSTRUMENT_FX_MAJORS_GBP_USD", "AUDUSD": "INSTRUMENT_FX_MAJORS_AUD_USD",
    "NZDUSD": "INSTRUMENT_FX_MAJORS_NZD_USD", "USDCHF": "INSTRUMENT_FX_MAJORS_USD_CHF",
    "USDCAD": "INSTRUMENT_FX_MAJORS_USD_CAD", "EURJPY": "INSTRUMENT_FX_CROSSES_EUR_JPY",
    "GBPJPY": "INSTRUMENT_FX_CROSSES_GBP_JPY", "AUDJPY": "INSTRUMENT_FX_CROSSES_AUD_JPY",
}


def download_symbol(symbol: str, years: int = 2) -> str:
    """dukascopyから15分足を取得しCSV保存、パスを返す。無料・APIキー不要・UTC。"""
    from datetime import datetime, timedelta
    import dukascopy_python as d
    import dukascopy_python.instruments as inst
    sym = symbol.upper()
    if sym not in SYMBOLS:
        raise ValueError(f"未対応の銘柄: {symbol}")
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years)
    df = d.fetch(getattr(inst, SYMBOLS[sym]), d.INTERVAL_MIN_15, d.OFFER_SIDE_BID, start, end)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    df.index.name = "Datetime"
    out = str(ROOT / f"{sym.lower()}_15m.csv")
    df[["Open", "High", "Low", "Close"]].to_csv(out)
    return out


def append_trade(trade: dict):
    """確定トレードを sessions/ の CSV と JSON に追記。AIが読む先。"""
    SESSIONS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    base = SESSIONS / f"trades_{Path(CSV_NAME).stem}_{stamp}"

    row = {k: trade.get(k, "") for k in TRADE_COLS}
    csv_path = base.with_suffix(".csv")
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_COLS)
        if new:
            w.writeheader()
        w.writerow(row)

    # JSON は全フィールド（スナップショット込み）を1行JSONLで残す
    with open(base.with_suffix(".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(trade, ensure_ascii=False) + "\n")
    return str(csv_path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # 静かに

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path.startswith("/api/bars"):
            self._send(200, BARS_JSON)
        elif self.path.startswith("/api/symbols"):
            self._send(200, json.dumps({"symbols": list(SYMBOLS), "current": CSV_NAME}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/api/trade"):
            path = append_trade(body)
            self._send(200, json.dumps({"ok": True, "file": path}).encode())
        elif self.path.startswith("/api/indicator"):
            try:
                res = compute_indicator(body["type"], body.get("params", {}), int(body.get("tf", 0)))
                self._send(200, json.dumps(res).encode())
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}).encode())
        elif self.path.startswith("/api/download"):
            global BARS_JSON, CSV_NAME, PIP_SIZE
            try:
                sym = body["symbol"].upper()
                path = download_symbol(sym, int(body.get("years", 2)))
                CSV_NAME = Path(path).name
                PIP_SIZE = 0.01 if "jpy" in CSV_NAME.lower() else 0.0001
                BARS_JSON = build_bars(path, MAX_BARS)
                self._send(200, json.dumps({"ok": True, "csv": CSV_NAME}).encode())
            except Exception as e:
                self._send(400, json.dumps({"error": str(e)}).encode())
        else:
            self._send(404, b'{"error":"not found"}')


def main():
    global BARS_JSON, CSV_NAME, PIP_SIZE, MAX_BARS
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="usdjpy_15m.csv")  # MTFは15mを基準に上位足を合成
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--bars", type=int, default=20000, help="直近何本読むか(0で全件)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    csv_path = args.csv if Path(args.csv).is_absolute() else str(ROOT / args.csv)
    if not Path(csv_path).exists():
        sys.exit(f"CSVが見つからない: {csv_path}")

    CSV_NAME = Path(csv_path).name
    # JPYクロスは pip=0.01、その他は0.0001（簡易判定）
    PIP_SIZE = 0.01 if "jpy" in CSV_NAME.lower() else 0.0001
    MAX_BARS = args.bars
    print(f"読込中: {CSV_NAME} ...")
    BARS_JSON = build_bars(csv_path, MAX_BARS)
    print(f"完了。 http://localhost:{args.port}  (Ctrl+Cで終了)")

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
