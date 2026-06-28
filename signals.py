"""
シグナル評価モジュール（backtest と execution が共有する単一経路）

設計原則: エントリー/イグジット判定のコードはここ1箇所だけ。
バックテストも実取引も同じ関数を呼ぶことで「テストした通りに取引される」を保証する。
"""
import sys
import numpy as np
import pandas as pd


# ── インジケーター計算 ─────────────────────────────────────────────────────────

# 上位足リサンプリング用の対応表（JSON表記 → pandas rule）
_TF_RULE = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def add_indicators(df: pd.DataFrame, ind: dict) -> pd.DataFrame:
    """
    戦略JSONの indicators 定義に従って列を追加する。
    ind に "htf" があれば上位足の指標も計算して結合する（マルチタイムフレーム）。
    """
    df = _add_base_indicators(df, ind)

    # ── 上位足（HTF）: トレンドフィルター用 ────────────────────────────────────
    if "htf" in ind:
        df = _add_htf_indicators(df, ind["htf"])

    return df


def _add_base_indicators(df: pd.DataFrame, ind: dict) -> pd.DataFrame:
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]

    # 時間帯フィルター用（UTC時刻）。条件式で df['hour'] を使える。
    # 主要セッション(UTC): 東京0-9 / ロンドン7-16 / NY13-22 / ロンドン・NY重複13-16(最重要)
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour"] = df.index.hour

    # 寄り付きレンジ（セッション開始バーの高値/安値/方向）。
    # 「寄りでは入らず、寄りで動いた方向にブレイクしたら入る」戦略を表現するため。
    # ind["session_open"] = {"open_hour": 7} で有効化（UTC）。
    # look-ahead回避: 寄りバー(hour==open_hour)はopen_hour+1に閉じる。
    #   よって寄りバーの値は hour > open_hour のバーからのみ参照可能にする(それ以前はNaN)。
    so = ind.get("session_open")
    if so and isinstance(df.index, pd.DatetimeIndex):
        oh = so.get("open_hour", 7)
        day = pd.Series(df.index.normalize(), index=df.index)
        omask = df["hour"] == oh
        oh_high = h.where(omask)
        oh_low  = l.where(omask)
        oh_dir  = np.sign(c - df["Open"]).where(omask)  # 寄りバーの方向(+上/-下)
        df["SO_high"] = oh_high.groupby(day).transform("max")
        df["SO_low"]  = oh_low.groupby(day).transform("min")
        df["SO_dir"]  = oh_dir.groupby(day).transform("first")
        # 寄りバーが閉じる前(hour<=oh)＝未確定、寄りバーが無い日＝NaN。
        # dropna()で行ごと消えると価格が不連続になりフラクタル決済が壊れるため、
        # NaNではなく「ブレイク条件が絶対Falseになる番兵値」で埋める。
        invalid = (df["hour"] <= oh) | df["SO_high"].isna()
        df.loc[invalid, "SO_high"] = np.inf    # cross_above(Close, inf)=常にFalse
        df.loc[invalid, "SO_low"]  = -np.inf   # cross_below(Close, -inf)=常にFalse
        df.loc[invalid, "SO_dir"]  = 0
        df[["SO_high", "SO_low", "SO_dir"]] = df[["SO_high", "SO_low", "SO_dir"]].fillna(
            {"SO_high": np.inf, "SO_low": -np.inf, "SO_dir": 0})

    # セッションレンジ（指定時間帯[start_hour, end_hour)の高値/安値）。
    # ロンドン・ブレイクアウト用: アジア時間(00-07 UTC)の値幅を記録し、
    # その上抜け/下抜けに順張りでブレイクを取る戦略を表現する。
    # ind["session_range"] = {"start_hour": 0, "end_hour": 7} で有効化(UTC)。
    # look-ahead回避: レンジは end_hour で確定するため、hour >= end_hour のバーからのみ参照可。
    # 条件式で使える列: AR_high(レンジ高値), AR_low(レンジ安値), AR_width(値幅)。
    sr = ind.get("session_range")
    if sr and isinstance(df.index, pd.DatetimeIndex):
        sh = sr.get("start_hour", 0)
        eh = sr.get("end_hour", 7)
        day = pd.Series(df.index.normalize(), index=df.index)
        rmask = (df["hour"] >= sh) & (df["hour"] < eh)
        rng_high = h.where(rmask).groupby(day).transform("max")
        rng_low  = l.where(rmask).groupby(day).transform("min")
        df["AR_high"]  = rng_high
        df["AR_low"]   = rng_low
        df["AR_width"] = rng_high - rng_low
        # end_hour前(レンジ未確定)＝番兵で埋める。ブレイク条件が絶対Falseになる値。
        invalid = (df["hour"] < eh) | df["AR_high"].isna()
        df.loc[invalid, "AR_high"]  = np.inf    # cross_above(Close, inf)=常にFalse
        df.loc[invalid, "AR_low"]   = -np.inf   # cross_below(Close, -inf)=常にFalse
        df.loc[invalid, "AR_width"] = np.inf     # width<閾値 が常にFalse → 見送り
        df[["AR_high", "AR_low", "AR_width"]] = df[["AR_high", "AR_low", "AR_width"]].fillna(
            {"AR_high": np.inf, "AR_low": -np.inf, "AR_width": np.inf})

    # キリ番（心理的節目）。USDJPYなら 150.00 等の .00 や .50 はブレイクの効きが変わる。
    # ind["round_number"] = {"step": 0.5} で有効化(stepは節目の間隔。1.0=毎円, 0.5=半円)。
    # 条件式で使える列: RN_level(最寄りキリ番), RN_dist(最寄りキリ番までの絶対距離)。
    rn = ind.get("round_number")
    if rn:
        step = rn.get("step", 0.5)
        level = (c / step).round() * step
        df["RN_level"] = level
        df["RN_dist"]  = (c - level).abs()

    for p in ind.get("ma_periods", []):
        df[f"MA_{p}"] = c.rolling(p).mean()

    for p in ind.get("ema_periods", []):
        df[f"EMA_{p}"] = c.ewm(span=p, adjust=False).mean()

    if "rsi_period" in ind:
        p = ind["rsi_period"]
        d = c.diff()
        gain = d.clip(lower=0).ewm(alpha=1/p, min_periods=p).mean()
        loss = (-d.clip(upper=0)).ewm(alpha=1/p, min_periods=p).mean()
        df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    if "atr_period" in ind:
        p = ind["atr_period"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(span=p, adjust=False).mean()

    if "macd" in ind:
        f = ind["macd"].get("fast", 12)
        s = ind["macd"].get("slow", 26)
        sig = ind["macd"].get("signal", 9)
        m = c.ewm(span=f, adjust=False).mean() - c.ewm(span=s, adjust=False).mean()
        df["MACD"] = m
        df["MACD_signal"] = m.ewm(span=sig, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    if "bb" in ind:
        p = ind["bb"].get("period", 20)
        std = ind["bb"].get("std", 2.0)
        mid = c.rolling(p).mean()
        s_ = c.rolling(p).std()
        df["BB_upper"] = mid + std * s_
        df["BB_mid"]   = mid
        df["BB_lower"] = mid - std * s_

    if "adx_period" in ind:
        p = ind["adx_period"]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        um = h - h.shift()
        dm = l.shift() - l
        pdm = um.where((um > dm) & (um > 0), 0.0)
        mdm = dm.where((dm > um) & (dm > 0), 0.0)
        atr = tr.ewm(alpha=1/p, adjust=False).mean().replace(0, np.nan)
        pdi = 100 * pdm.ewm(alpha=1/p, adjust=False).mean() / atr
        mdi = 100 * mdm.ewm(alpha=1/p, adjust=False).mean() / atr
        di_sum = (pdi + mdi).replace(0, np.nan)
        dx = 100 * (pdi - mdi).abs() / di_sum
        df["ADX"]      = dx.ewm(alpha=1/p, adjust=False).mean().fillna(0)
        df["DI_plus"]  = pdi.fillna(0)
        df["DI_minus"] = mdi.fillna(0)

    return df


# ── 上位足（マルチタイムフレーム） ─────────────────────────────────────────────

def _add_htf_indicators(df: pd.DataFrame, htf: dict) -> pd.DataFrame:
    """
    上位足の指標を計算し、基準足のindexに結合する。
    列名には足のサフィックスが付く（例: 4h足のMA_50 → MA_50_4H）。
    条件式では df['MA_50_4H'] のように参照する。

    look-ahead bias回避: 上位足は「確定済みのバー」しか見えないようにする。
    - resample(label='right', closed='left') で各上位足バーのindex=その足の終値時刻
    - reindex(..., method='ffill') で各基準足バーには「直近で確定した上位足」だけが入る
      → まだ閉じていない上位足の値を未来予知してしまうのを防ぐ
    """
    tf = htf.get("timeframe", "4h")
    rule = _TF_RULE.get(tf, tf)
    suffix = "_" + tf.upper()

    agg = df.resample(rule, label="right", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    ).dropna()

    htf_df = _add_base_indicators(agg, htf)
    ind_cols = [col for col in htf_df.columns if col not in ("Open", "High", "Low", "Close")]

    merged = htf_df[ind_cols].add_suffix(suffix).reindex(df.index, method="ffill")
    return df.join(merged)


# ── 条件評価 ───────────────────────────────────────────────────────────────────

def cross_above(a: pd.Series, b) -> pd.Series:
    """a が b を下から上に抜けた「瞬間」だけTrue（イベント型エントリー）。
    前バーで a<=b かつ 現バーで a>b。b は Series でも定数でもよい。"""
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a.shift() <= b_prev) & (a > b)


def cross_below(a: pd.Series, b) -> pd.Series:
    """a が b を上から下に抜けた瞬間だけTrue。"""
    b_prev = b.shift() if isinstance(b, pd.Series) else b
    return (a.shift() >= b_prev) & (a < b)


def eval_signal(df: pd.DataFrame, exprs: list) -> pd.Series:
    """
    条件式リスト（文字列）をANDで評価してBoolシリーズを返す。
    式の中では df の列名（RSI, ATR, Close, MA_20 など）をそのまま使える。
    例: ["df['RSI'] < 30", "df['Close'] > df['MA_20']"]
    """
    if not exprs:
        return pd.Series(False, index=df.index)

    result = pd.Series(True, index=df.index)
    ns = {col: df[col] for col in df.columns}
    ns.update({"df": df, "np": np, "pd": pd,
               "cross_above": cross_above, "cross_below": cross_below})

    for expr in exprs:
        try:
            mask = eval(expr, {"__builtins__": __builtins__}, ns)
            result = result & mask.astype(bool)
        except Exception as e:
            print(f"  [warn] 条件エラー: {expr!r} → {e}", file=sys.stderr)
            result = result & False

    return result


def entry_signals(df: pd.DataFrame, strategy: dict):
    """戦略JSONから (long_signal, short_signal) のBoolシリーズを返す。"""
    ec = strategy.get("entry_conditions", {})
    long_sig  = eval_signal(df, ec.get("long",  []))
    short_sig = eval_signal(df, ec.get("short", []))
    return long_sig, short_sig
