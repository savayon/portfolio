# features.py — 피처 엔지니어링 (행 단위 독립, train 유래 상수만 사용)
#
# 대회 규칙 4/6번 준수: 각 행은 자기 자신의 컬럼값과 train.csv에서 유도한 상수
# (전역 평균 등)만으로 변환된다. test.csv의 다른 행이나 분포는 일절 쓰지 않는다.
import numpy as np
import pandas as pd

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

# asof rate 컬럼 -> 표본수 컬럼. 표본수가 적은 rate는 노이즈가 크므로
# 전역 평균 쪽으로 축소(empirical Bayes shrinkage)한다.
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

SHRINK_K = 50.0  # 사전분포 가중 표본수


def fit_priors(df):
    """train에서 shrinkage용 전역 평균을 구한다. 추론 때 이 값을 그대로 재사용."""
    return {c: float(df[c].mean()) for c in RATE_TO_N if c in df.columns}


def add_features(df, priors):
    df = df.copy()

    b, s = df["balls_before"], df["strikes_before"]
    df["count_state"] = (b.astype(str) + "-" + s.astype(str))
    df["ball_minus_strike"] = b - s
    df["is_two_strikes"] = (s == 2).astype(np.int8)
    df["is_three_balls"] = (b == 3).astype(np.int8)
    # 3볼 = 스트라이크를 넣어야 하는 압박 -> 가운데로 몰릴 위험(=제구 실패 정의)
    df["pitcher_must_strike"] = ((b == 3) & (s < 2)).astype(np.int8)
    # 2스트라이크 = 유인구 여유 -> 크게 벗어날 위험
    df["pitcher_can_waste"] = ((s == 2) & (b < 2)).astype(np.int8)

    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int8)

    # 표본수 기반 축소: n이 작으면 전역 평균, 크면 관측값
    for rate_col, n_col in RATE_TO_N.items():
        if rate_col not in df.columns:
            continue
        n = df[n_col].fillna(0).to_numpy(dtype=float)
        raw = df[rate_col].to_numpy(dtype=float)
        prior = priors[rate_col]
        obs = np.where(np.isnan(raw), prior, raw)
        df[rate_col + "_sm"] = (obs * n + prior * SHRINK_K) / (n + SHRINK_K)

    # 최근 폼 편차: 직전 N경기 - 통산
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


def season_base_rate_trend(df, target_season):
    """시즌별 정답률의 선형추세를 target_season으로 외삽한다.

    train.csv만 사용하므로 규칙상 허용되는 사전정보다(test 분포를 보지 않는다).
    """
    m = df.groupby("season")["control_success"].mean()
    coef = np.polyfit(m.index.to_numpy(dtype=float), m.to_numpy(), 1)
    return float(np.polyval(coef, target_season))


def shift_to_base_rate(p, target_rate):
    """예측 확률의 평균이 target_rate가 되도록 로짓을 평행이동(단조 변환이라 순위 불변)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sigmoid(logit + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return sigmoid(logit + (lo + hi) / 2)


def _self_check():
    # 로짓 평행이동: 평균이 목표치에 맞고, 순위는 보존되어야 한다
    p = np.array([0.1, 0.4, 0.6, 0.9])
    out = shift_to_base_rate(p, 0.3)
    assert abs(out.mean() - 0.3) < 1e-6, out.mean()
    assert list(np.argsort(out)) == list(np.argsort(p)), "단조 변환이어야 함"

    # shrinkage: n=0이면 전역 평균, n이 크면 관측값에 수렴
    df = pd.DataFrame({
        "balls_before": [0, 3], "strikes_before": [0, 2], "inning": [1, 9],
        "pitcher_hand": [1, 2], "batter_hand": [1, 1],
        "score_diff_pitcher_team": [0, -1],
        "asof_pitcher_n": [0, 100000], "asof_batter_n": [0, 0],
        "asof_pitcher_pitchmix_n": [0, 0],
        "asof_pitcher_success_rate": [np.nan, 0.9],
    })
    out = add_features(df, {c: 0.5 for c in RATE_TO_N})
    got = out["asof_pitcher_success_rate_sm"].to_numpy()
    assert abs(got[0] - 0.5) < 1e-9, f"n=0은 사전평균이어야: {got[0]}"
    assert abs(got[1] - 0.9) < 1e-3, f"n이 크면 관측값에 수렴해야: {got[1]}"
    assert out["pitcher_must_strike"].tolist() == [0, 0]
    assert out["is_two_strikes"].tolist() == [0, 1]
    print("features self-check OK")


if __name__ == "__main__":
    _self_check()
