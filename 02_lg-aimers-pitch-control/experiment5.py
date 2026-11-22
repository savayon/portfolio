# experiment5.py — 최종 제출 절차를 한 시즌 앞당겨 그대로 재현한 정직한 검증
#
# 2025를 2024로 치환해서, 제출 때 할 모든 단계(내부 시즌분할로 n_iter 선택 ->
# 재학습 -> 학습시점 상수 b로 보정)를 동일하게 수행한다. 어느 단계에서도
# 검증 시즌(2024)의 정답이나 분포를 보지 않으므로 규칙 준수 + 낙관편향 없음.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from features import add_features, fit_priors, season_base_rate_trend
from model import build_model
from validate import bss

ID_COL, TARGET_COL = "row_id", "control_success"
VAL_SEASON = 2024
LR = 0.015
STEP = 25


def solve_shift(p_ref, target):
    p_ref = np.clip(np.asarray(p_ref, float), 1e-6, 1 - 1e-6)
    logit = np.log(p_ref / (1 - p_ref))
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sig(logit + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def apply_shift(p, b):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def weights(seasons, decay=0.85):
    return decay ** (seasons.max() - np.asarray(seasons, float))


def main():
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    pool = df[df.season < VAL_SEASON]        # 제출 때의 train.csv 역할
    va = df[df.season == VAL_SEASON]         # 제출 때의 test.csv 역할
    priors = fit_priors(pool)
    prep = lambda d: add_features(d.drop(columns=[ID_COL, TARGET_COL]), priors)

    # --- 1) 내부 시즌분할로 n_iter 선택 (pool 안에서만) ---
    inner_last = pool.season.max()                 # 2023
    itr = pool[pool.season < inner_last]
    iva = pool[pool.season == inner_last]
    m0 = build_model(learning_rate=LR, max_iter=500, early_stopping=False).fit(
        prep(itr), itr[TARGET_COL].values, sample_weight=weights(itr.season)
    )
    # n_iter는 "미래 시즌으로의 전이 능력"만 보고 골라야 한다. 외삽 오차가 섞이면
    # 점수가 전부 0으로 눌려 선택이 무의미해지므로(실측), 내부 검증 시즌은 학습
    # 데이터이니 그 시즌의 실제 base rate로 보정해 모델 품질만 비교한다.
    yi = iva[TARGET_COL].values
    best = max(
        (
            (i, bss(yi, apply_shift(p[:, 1], solve_shift(p[:, 1], yi.mean()))))
            for i, p in enumerate(m0.staged_predict_proba(prep(iva)), 1)
            if i % STEP == 0
        ),
        key=lambda t: t[1],
    )
    n_iter = best[0]
    print(f"[1] inner split: train<={inner_last-1} val={inner_last} -> n_iter*={n_iter} (inner score {best[1]:.2f})")

    # --- 2) pool 전체로 재학습 (데이터가 늘어난 만큼 n_iter 비례 확대) ---
    scaled = int(round(n_iter * len(pool) / len(itr)))
    m = build_model(learning_rate=LR, max_iter=scaled, early_stopping=False).fit(
        prep(pool), pool[TARGET_COL].values, sample_weight=weights(pool.season)
    )
    print(f"[2] refit on pool with n_iter={scaled}")

    # --- 3) 학습시점 상수 b 확정 (pool 최신 시즌 예측 분포 기준) ---
    est = season_base_rate_trend(pool, VAL_SEASON)
    ref = pool.season == inner_last
    b = solve_shift(m.predict_proba(prep(pool[ref]))[:, 1], est)
    print(f"[3] est base rate={est:.4f} (actual {va[TARGET_COL].mean():.4f})  b={b:+.4f}")

    # --- 4) 추론: 행별로 상수 b만 적용 ---
    yva = va[TARGET_COL].values
    p = m.predict_proba(prep(va))[:, 1]
    pc = apply_shift(p, b)
    print(f"\nraw            {bss(yva, p):8.2f}  (mean={p.mean():.4f})")
    print(f"compliant calib{bss(yva, pc):8.2f}  (mean={pc.mean():.4f})   <== 제출 절차 그대로")


if __name__ == "__main__":
    main()
