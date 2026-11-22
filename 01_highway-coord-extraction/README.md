# 시뮬레이션 기반 고속도로 위험 재정의 연구 — 학부연구생 (팀장)

2026.06 ~ 진행 중

> 본 연구는 **연구실 자산이자 현재 진행 중인 연구**여서, 연구의 세부 내용과 데이터·분석 코드는 공개가 어려운 점 양해 부탁드립니다.
> 이 폴더에는 연구 과정에서 제가 직접 작성한 작업물 중 공개 가능한 부분만 담았습니다.

## 좌표 추출

분석 대상 지점은 OSM 고속도로 본선에서 10km 간격으로 자동 추출했습니다.
43개 노선에서 상행 640개, 하행 492개를 추출했고, 이 중 상·하행 짝이 맞는 970개(485쌍)를 분석에 사용했습니다.

**상행** `extract_coords/extract_highway_coords.py`
1. Overpass API로 해당 노선 번호(`ref`)의 `highway=motorway` way를 수집합니다.
2. OSM 노드와 way로 방향 그래프(`networkx.DiGraph`)를 구성했습니다. 일방통행 방향이 그대로 보존되기 때문에, 연결 요소를 나누면 상행 차로와 하행 차로가 자연스럽게 분리됩니다.
3. 요청한 방향과 맞는 최장 경로를 선택합니다. 순환고속도로의 경우 사이클의 면적 부호(신발끈 공식)로 내선과 외선을 구분했습니다.
4. EPSG:5179로 투영한 뒤 10km마다 선형 보간합니다.

**하행** `extract_coords/get_overside_coords_v4.py`
- 상행 점에서 수직선을 그어 반대편 본선과 만나는 교점을 하행 좌표로 잡았습니다. 이 방식으로 같은 위치의 상·하행 좌표를 쌍으로 얻을 수 있습니다.
- 진행 방향이 같은 선분은 제외해 반대편 차로만 남기도록 했고, 교점이 없는 경우에는 임의 좌표를 만들지 않고 건너뛰도록 했습니다.
- 초기 버전(v1~v3)은 경로 탐색 방식이라 국도로 우회하거나 그래프가 끊기는 문제가 있었는데, v4에서 순수 기하 연산으로 바꿔 해결했습니다.

아래 명령으로 직접 실행해 보실 수 있습니다.

```bash
pip install requests networkx pyproj numpy pandas
python extract_coords/extract_highway_coords.py --ref 1 --direction up --output extract_coords/1/1_up
python extract_coords/get_overside_coords_v4.py --ref 1      # 또는 --all
```

결과 예시로 경부고속도로(1번) 추출 결과를 [`extract_coords/1/`](extract_coords/1/) 폴더에 담아 두었습니다.
`.geojson` 파일을 GitHub에서 여시면 추출된 좌표를 지도 위에서 바로 확인하실 수 있습니다. (좌표 데이터 © OpenStreetMap contributors, ODbL)

| 파일 | 내용 |
|---|---|
| `1_up_coords.csv` | 상행 10km 간격 좌표 |
| `1_up_map.geojson` | 상행 본선 궤적과 추출 지점 |
| `1_down_coords_projected.csv` | 하행 좌표 (짝이 되는 상행 좌표 포함) |
| `1_down_coords_projected.geojson` | 하행 추출 지점 |

## 경로 서버 구축 (Docker)

시뮬레이션 시나리오에 필요한 도로 거리·소요 시간을 얻기 위해, 경로 계산 서버를 Docker로 직접 빌드해 연구 파이프라인에 연결했습니다.

## 수행한 통계 분석

분석 코드는 진행 중인 연구 내용과 직결되어 있어, 사용한 기법만 정리해 두었습니다.

- OLS 다중회귀, 표준화 계수 비교, 다중공선성 진단(VIF)
- 짝지은 차분 회귀, Dominance Analysis
- 노선 단위 군집 표준오차, 군집 부트스트랩 (같은 노선 위의 지점끼리는 독립이 아니어서 보정)
- 그룹 비교: t-검정(독립·대응), Mann-Whitney U, Kruskal-Wallis, ANOVA
- 연관성: Spearman 순위상관, 카이제곱 검정과 Cramér's V
