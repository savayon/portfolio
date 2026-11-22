# script.py — 평가 서버에서 자동 실행되는 추론 코드
#
# 규칙 준수 메모:
#  - 각 행의 예측은 그 행의 컬럼값 + 학습 때 확정된 상수(priors, b)만으로 계산된다.
#    test.csv의 다른 행이나 전체 분포를 읽는 연산은 없다(규칙 4/6번).
#  - 피처 생성 코드는 학습측 features.py와 동일하며, check_parity.py가 두 구현의
#    출력이 일치하는지 검사한다(제출 zip 구조를 건드리지 않으려고 인라인함).
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_VALUES = {
    "top_bottom": ["B", "T"],
    "game_type": ["F", "R"],
    "base_state": ["123", "12_", "1_3", "1__", "_23", "_2_", "__3", "___"],
    "pitcher_hand": [1, 2],
    "batter_hand": [1, 2],
    "pitcher_team_id": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25],
    "batter_team_id": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25],
    "count_state": [f"{b}-{s}" for b in range(4) for s in range(3)],
}

RATE_TO_N = {
    "asof_pitcher_success_rate": "asof_pitcher_n",
    "asof_pitcher_reverse_rate": "asof_pitcher_n",
    "asof_pitcher_middle_rate": "asof_pitcher_n",
    "asof_pitcher_ball_rate": "asof_pitcher_n",
    "asof_pitcher_strike_rate": "asof_pitcher_n",
    "asof_batter_success_rate": "asof_batter_n",
    "asof_batter_middle_rate": "asof_batter_n",
    "asof_pitcher_fastball_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_breaking_rate": "asof_pitcher_pitchmix_n",
    "asof_pitcher_offspeed_rate": "asof_pitcher_pitchmix_n",
}

SHRINK_K = 50.0


def add_features(df, priors):
    df = df.copy()

    b, s = df["balls_before"], df["strikes_before"]
    df["count_state"] = (b.astype(str) + "-" + s.astype(str))
    df["ball_minus_strike"] = b - s
    df["is_two_strikes"] = (s == 2).astype(np.int8)
    df["is_three_balls"] = (b == 3).astype(np.int8)
    df["pitcher_must_strike"] = ((b == 3) & (s < 2)).astype(np.int8)
    df["pitcher_can_waste"] = ((s == 2) & (b < 2)).astype(np.int8)

    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int8)

    for rate_col, n_col in RATE_TO_N.items():
        if rate_col not in df.columns:
            continue
        n = df[n_col].fillna(0).to_numpy(dtype=float)
        raw = df[rate_col].to_numpy(dtype=float)
        prior = priors[rate_col]
        obs = np.where(np.isnan(raw), prior, raw)
        df[rate_col + "_sm"] = (obs * n + prior * SHRINK_K) / (n + SHRINK_K)

    for w in (1, 3, 5):
        for kind in ("success", "middle"):
            col = f"asof_pitcher_prev{w}_game_{kind}_rate"
            base = f"asof_pitcher_{kind}_rate"
            if col in df.columns and base in df.columns:
                df[f"form_dev_{kind}_{w}"] = df[col] - df[base]

    df["log_pitcher_n"] = np.log1p(df["asof_pitcher_n"].fillna(0))
    df["log_batter_n"] = np.log1p(df["asof_batter_n"].fillna(0))

    df["abs_score_diff"] = df["score_diff_pitcher_team"].abs()
    df["late_and_close"] = ((df["inning"] >= 7) & (df["abs_score_diff"] <= 2)).astype(np.int8)

    for c, cats in CAT_VALUES.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=cats)
    return df


def apply_shift(p, b):
    """학습 때 확정한 로짓 오프셋을 행별로 적용 (단조 변환)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def main():
    test = pd.read_csv("./data/test.csv", encoding="utf-8-sig")
    sub = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")

    bundle = joblib.load("./model/model.pkl")
    clf, priors, b = bundle["clf"], bundle["priors"], bundle["b"]

    X = add_features(test.drop(columns=[ID_COL]), priors)
    if len(X):
        preds = apply_shift(clf.predict_proba(X)[:, 1], b)
    else:
        preds = []

    pred_map = dict(zip(test[ID_COL], preds))
    sub[TARGET_COL] = [pred_map.get(rid, cur) for rid, cur in zip(sub[ID_COL], sub[TARGET_COL])]

    os.makedirs("./output", exist_ok=True)
    sub.to_csv("./output/submission.csv", index=False, encoding="utf-8")
    print(f"Saved: ./output/submission.csv (rows={len(sub)})")


if __name__ == "__main__":
    main()
