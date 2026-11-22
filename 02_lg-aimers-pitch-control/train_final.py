# train_final.py — 제출용 최종 모델 학습
#
#   1) 전체 train으로 shrinkage prior 계산
#   2) 시즌 가중치(최근 우대)로 학습
#   3) 2025 base rate를 시즌 추세 외삽으로 추정하고, 그에 맞는 로짓 오프셋 b를 확정
#   4) 모델 + prior + b 를 한 파일로 저장 (추론은 행별 상수 적용만)
#
# n_iter는 "<=2023 학습 -> 2024 검증" 전이에서 고른 275(lr=0.015)를 전체 데이터
# 크기에 맞춰 확대한 값. 2024는 학습 데이터이므로 이 선택은 규칙상 정상적인
# 모델 선택이다. 더 예전 분할(<=2022 -> 2023)은 2023에 레짐 변화가 있어
# (base rate -0.029, AUC 0.526) 전이가 깨지므로 기준으로 쓰지 않는다.
import warnings

warnings.filterwarnings("ignore")

import os

import joblib
import numpy as np
import pandas as pd

from experiment5 import apply_shift, solve_shift, weights
from features import add_features, fit_priors, season_base_rate_trend
from model import build_model

ID_COL, TARGET_COL = "row_id", "control_success"
LR = 0.015
TARGET_SEASON = 2025
N_ITER_REF = 275      # <=2023 학습분(1,221,585행) 기준 최적 반복수
REF_SEASON_MAX = 2023


def main():
    df = pd.read_csv("data/train.csv", encoding="utf-8-sig")
    priors = fit_priors(df)
    X = add_features(df.drop(columns=[ID_COL, TARGET_COL]), priors)
    y = df[TARGET_COL].values

    ref_n = (df.season <= REF_SEASON_MAX).sum()
    n_iter = int(round(N_ITER_REF * len(df) / ref_n))
    print(f"n_iter: ref={N_ITER_REF} on {ref_n} rows -> final={n_iter} on {len(df)} rows")

    model = build_model(learning_rate=LR, max_iter=n_iter, early_stopping=False)
    model.fit(X, y, sample_weight=weights(df.season))

    est = season_base_rate_trend(df, TARGET_SEASON)
    ref = (df.season == df.season.max()).to_numpy()
    p_ref = model.predict_proba(X[ref])[:, 1]
    b = solve_shift(p_ref, est)
    print(f"est {TARGET_SEASON} base rate={est:.4f}  ref mean={p_ref.mean():.4f}  b={b:+.4f}")
    print(f"  -> shifted ref mean={apply_shift(p_ref, b).mean():.4f}")

    os.makedirs("submit/model", exist_ok=True)
    joblib.dump({"clf": model, "priors": priors, "b": float(b)}, "submit/model/model.pkl")
    print("Saved: submit/model/model.pkl")


if __name__ == "__main__":
    main()
