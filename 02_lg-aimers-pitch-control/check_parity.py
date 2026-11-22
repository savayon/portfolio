# check_parity.py — 학습측 features.py와 제출측 script.py의 피처 생성이 같은지 검사
#
# script.py는 제출 zip 구조를 깨지 않으려고 피처 코드를 인라인 복사했다.
# 복사본이 학습측과 어긋나면 조용히 오예측하므로 실제 데이터로 대조한다.
import importlib.util
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from pandas.testing import assert_frame_equal

import features


def load_submit_script():
    spec = importlib.util.spec_from_file_location("submit_script", "submit/script.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sc = load_submit_script()
    assert sc.CAT_VALUES == features.CAT_VALUES, "CAT_VALUES 불일치"
    assert sc.RATE_TO_N == features.RATE_TO_N, "RATE_TO_N 불일치"
    assert sc.SHRINK_K == features.SHRINK_K, "SHRINK_K 불일치"

    df = pd.read_csv("data/train.csv", encoding="utf-8-sig", nrows=20000)
    priors = features.fit_priors(df)
    X = df.drop(columns=["row_id", "control_success"])
    assert_frame_equal(features.add_features(X, priors), sc.add_features(X, priors))

    # 실제 test.csv 형식(정답 컬럼 없음)에서도 동일해야 한다
    t = pd.read_csv("data/test.csv", encoding="utf-8-sig").drop(columns=["row_id"])
    assert_frame_equal(features.add_features(t, priors), sc.add_features(t, priors))

    print("parity OK: features.py == submit/script.py")


if __name__ == "__main__":
    main()
