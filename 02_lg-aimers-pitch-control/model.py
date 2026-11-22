# model.py — 제출용 모델 정의
#
# HistGradientBoostingClassifier는 결측치(NaN)와 범주형(category dtype)을
# 네이티브로 처리하므로 baseline의 OrdinalEncoder+SimpleImputer가 필요 없다.
#
# 주의: 제출 zip 구조상 model/ 폴더가 강제되므로, 커스텀 함수를 Pipeline에 넣어
# pickle하면 추론 시 "model" 모듈명이 model/ 폴더와 충돌해 unpickle이 깨진다.
# 그래서 카테고리 캐스팅은 pickle에 안 들어가게 fit/predict 호출부(train.py,
# submit/script.py)에서 인라인으로 처리하고, 여기서는 순정 estimator만 반환한다.
from sklearn.ensemble import HistGradientBoostingClassifier

# train.csv 전수 조사로 확인한 고정 카테고리 목록(data_description.md의 base_state
# 정의와 일치). HistGB가 fit 시점 카테고리 매핑을 자체 보관하므로 배치별
# .astype("category")로도 결과는 같음을 실측 확인했다(차이 0.0). 그래도 값->코드
# 계약을 코드에 명시해두는 편이 안전해서 고정 목록을 쓴다.
CAT_VALUES = {
    "top_bottom": ["B", "T"],
    "game_type": ["F", "R"],
    "base_state": ["123", "12_", "1_3", "1__", "_23", "_2_", "__3", "___"],
}
CAT_COLS = list(CAT_VALUES)


def build_model(**hgb_kwargs):
    params = dict(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        categorical_features="from_dtype",
        random_state=42,
    )
    params.update(hgb_kwargs)
    return HistGradientBoostingClassifier(**params)


def cast_categories(df):
    import pandas as pd

    df = df.copy()
    for c, cats in CAT_VALUES.items():
        df[c] = pd.Categorical(df[c], categories=cats)
    return df
