# validate.py — 시간 기준 홀드아웃 검증 유틸
#
# test.csv가 2025 단일 시즌인 반면 train.csv는 2019-2024뿐이라, 랜덤 분할 검증은
# 점수를 크게 부풀린다(인샘플 1997 vs 시간분할 415, 실제 LB 기준 549.51).
# 앞으로 모델을 바꿔가며 로컬 검증할 때는 항상 이 스킴을 써야 LB와 비교 가능하다.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

TRAIN_PATH = "data/train.csv"
ID_COL, TARGET_COL = "row_id", "control_success"


def bss(y_true, p):
    """Brier Skill Score, 대회 공식 산식 그대로."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = np.mean((p - y_true) ** 2)
    r = y_true.mean()
    brier_ref = r * (1 - r)
    return max(0.0, 100000 * (1 - brier / brier_ref))


def time_holdout_split(df, val_season=2024):
    """val_season 미만은 학습, val_season은 검증 — 미래 시즌 외삽을 흉내낸다."""
    train = df[df["season"] < val_season]
    val = df[df["season"] == val_season]
    return train, val


def evaluate(model, val_season=2024, train_path=TRAIN_PATH):
    """model: sklearn Pipeline/Estimator (fit/predict_proba 지원). val_season 이전 데이터로 학습 후 val_season으로 채점."""
    df = pd.read_csv(train_path, encoding="utf-8-sig")
    tr, va = time_holdout_split(df, val_season)

    from model import cast_categories

    Xtr, ytr = cast_categories(tr.drop(columns=[ID_COL, TARGET_COL])), tr[TARGET_COL].values
    Xva, yva = cast_categories(va.drop(columns=[ID_COL, TARGET_COL])), va[TARGET_COL].values

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xva)[:, 1]
    score = bss(yva, p)
    print(f"train={len(tr)} val(season={val_season})={len(va)} score={score:.2f}")
    return score


def _self_check():
    y = np.array([0, 0, 1, 1])
    assert bss(y, y) == 100000.0, "완벽 예측은 만점이어야 함"
    assert bss(y, [0.5, 0.5, 0.5, 0.5]) == 0.0, "기준 예측(평균)은 0점이어야 함"
    assert bss(y, 1 - y) < 0.001 or bss(y, 1 - y) == 0.0, "반대로 예측하면 0점(클립)이어야 함"
    print("bss() self-check OK")


if __name__ == "__main__":
    _self_check()

    import joblib
    from sklearn.base import clone

    model = clone(joblib.load("baseline_submit/model/rf.pkl"))
    evaluate(model, val_season=2024)
