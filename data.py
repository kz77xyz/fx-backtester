"""CSVローダ（fxtester用の最小版）。Datetime,Open,High,Low,Close を読む。"""
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    sep = "\t" if first.count("\t") > first.count(",") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns and "time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    elif "date" in df.columns:
        df["Datetime"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("日時列が見つかりません（datetime / date / time）")

    remap = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    df = df.rename(columns={k: v for k, v in remap.items() if k in df.columns})
    return df.set_index("Datetime")[["Open", "High", "Low", "Close"]].astype(float).sort_index()
