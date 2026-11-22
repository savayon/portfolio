# -*- coding: utf-8 -*-
"""
get_overside_coords_v4.py — 순수 공간 연산(Spatial Spaghetti + Vector) 기반 하행선 수직 교점 추출
OSRM 우회 오류 및 NetworkX 단절 오류를 모두 해결한 최종 버전입니다.
"""

import os
import sys
import json
import time
import argparse
import math
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

EXTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
T_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
T_TO_WGS = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

def load_up_coords(ref: str) -> pd.DataFrame:
    csv_path = os.path.join(EXTRACT_DIR, ref, f"{ref}_up_coords.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"상행선 CSV 없음: {csv_path}")
    df = pd.read_csv(csv_path)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('lat', 'latitude', 'y'): col_map[c] = 'latitude'
        elif cl in ('lon', 'lng', 'longitude', 'x'): col_map[c] = 'longitude'
        elif cl in ('frag_id',): col_map[c] = 'frag_id'
    df = df.rename(columns=col_map)
    if 'frag_id' not in df.columns:
        df['frag_id'] = 1
    return df[['latitude', 'longitude', 'frag_id']].copy()

def exclude_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    keep_idx = []
    for _, grp in df.groupby('frag_id', sort=False):
        if len(grp) <= 2: continue
        keep_idx.extend(grp.index[1:-1].tolist())
    return df.loc[keep_idx].copy()

# ================= OVERPASS BASED EXTRACTION =================
def fetch_overpass_data(ref: str, max_retries=3):
    cache_dir = os.path.join(EXTRACT_DIR, "..", "data", "overpass_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ref}_overpass.json")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    q = f"""
    [out:json][timeout:60];
    area["ISO3166-1"="KR"][admin_level=2]->.searchArea;
    way["highway"="motorway"]["ref"~"(^|;){ref}(;|$)"](area.searchArea);
    out body; node(w); out skel;
    """
    headers = {'User-Agent': 'highway-coord-extractor/4.0'}
    servers = ['https://overpass-api.de/api/interpreter', 'https://overpass.kumi.systems/api/interpreter']
    for attempt in range(max_retries):
        for url in servers:
            try:
                r = requests.get(url, params={'data': q}, headers=headers, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False)
                    return data
                elif r.status_code == 429: time.sleep(3 * (attempt + 1))
            except Exception:
                continue
        time.sleep(3)
    return None

def extract_all_segments(data):
    """
    방향 상관없이 ref에 해당하는 모든 way의 모든 선분(Segment)을 2D 흩뿌림(Spaghetti) 상태로 추출합니다.
    NetworkX 그래프를 그리지 않습니다.
    """
    if not data or 'elements' not in data: return []
    nodes, ways = {}, []
    for el in data.get('elements', []):
        if el['type'] == 'node': nodes[el['id']] = (el['lon'], el['lat'])
        elif el['type'] == 'way': ways.append(el)

    segments = []
    for w in ways:
        nd = w.get('nodes', [])
        for i in range(len(nd) - 1):
            if nd[i] in nodes and nd[i+1] in nodes:
                u_lon, u_lat = nodes[nd[i]]
                v_lon, v_lat = nodes[nd[i+1]]
                ux, uy = T_TO_UTM.transform(u_lon, u_lat)
                vx, vy = T_TO_UTM.transform(v_lon, v_lat)
                segments.append((np.array([ux, uy]), np.array([vx, vy])))
    return segments

# ================= RAY-SEGMENT INTERSECTION =================
def ray_segment_intersection(P, normal, tangent, A, B, max_dist=5000.0):
    V = B - A
    v_norm = np.linalg.norm(V)
    if v_norm < 1e-9: return None, None
    
    # 역방향인지 확인 (내적이 0보다 크거나 같으면 동일 방향이므로 무시)
    if np.dot(tangent, V / v_norm) >= 0:
        return None, None

    det = normal[0] * (-V[1]) - normal[1] * (-V[0])
    if abs(det) < 1e-9: return None, None

    diff = A - P
    t = (diff[0] * (-V[1]) - diff[1] * (-V[0])) / det
    s = (normal[0] * diff[1] - normal[1] * diff[0]) / det

    # s는 선분 A-B 위(0~1), t는 반직선(ray) 위(3~max_dist)
    if 0.0 <= s <= 1.0 and 3.0 <= t <= max_dist:
        return t, P + t * normal
    return None, None

def compute_down_coords(target_df, full_df, segments, max_dist=5000.0):
    if not segments: return []
    all_utm = []
    for _, row in full_df.iterrows():
        x, y = T_TO_UTM.transform(row['longitude'], row['latitude'])
        all_utm.append(np.array([x, y]))
        
    results = []
    for idx, row in target_df.iterrows():
        try:
            pos = full_df.index.get_loc(idx)
        except KeyError:
            continue
            
        # Tangent vector (중앙 차분)
        if pos == 0:
            tangent = all_utm[1] - all_utm[0]
        elif pos == len(all_utm) - 1:
            tangent = all_utm[-1] - all_utm[-2]
        else:
            tangent = all_utm[pos+1] - all_utm[pos-1]
            
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-9: continue
        tangent = tangent / t_norm
        
        # 90도 회전 수직 벡터 (좌, 우 2개)
        n1 = np.array([-tangent[1], tangent[0]])
        n2 = np.array([tangent[1], -tangent[0]])
        
        P = all_utm[pos]
        best_t = float('inf')
        best_pt = None
        
        for A, B in segments:
            for normal in (n1, n2):
                t, pt = ray_segment_intersection(P, normal, tangent, A, B, max_dist)
                if t is not None and t < best_t:
                    best_t = t
                    best_pt = pt
                    
        if best_pt is not None:
            d_lon, d_lat = T_TO_WGS.transform(best_pt[0], best_pt[1])
            up_lat, up_lon = row['latitude'], row['longitude']
            results.append((round(d_lat, 6), round(d_lon, 6), round(up_lat, 6), round(up_lon, 6)))
            
    return results

def process_single_ref(ref: str, max_dist: float = 5000.0, no_geojson: bool = False):
    print(f"\n{'='*60}")
    print(f"  [v4] 고속도로 {ref}번 — 순수 공간 연산(Spaghetti+Vector) 추출")
    print(f"{'='*60}")
    
    try:
        full_df = load_up_coords(ref)
        target_df = exclude_endpoints(full_df)
        print(f"  -> 기종점 제외 완료: 총 {len(full_df)}개 중 {len(target_df)}개 좌표 추출 시도")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    print("  -> Overpass API에서 고속도로 뼈대 추출 중...")
    data = fetch_overpass_data(ref)
    segments = extract_all_segments(data)
    print(f"  -> {ref}번 노선 전체 선분(Segment) {len(segments)}개 추출 완료 (Spaghetti)")

    if not segments:
        print("  [ERROR] 선분 추출 실패")
        return False
        
    print(f"  -> 수직 교점 연산 진행 중 (최대 거리: {max_dist}m, 역방향 필터 적용)...")
    results = compute_down_coords(target_df, full_df, segments, max_dist)
    
    match_rate = len(results) / len(target_df) * 100 if len(target_df) > 0 else 0
    print(f"  -> 교점 연산 결과: {len(results)}/{len(target_df)} 매칭 성공 ({match_rate:.1f}%)")
    
    if results:
        out_path = os.path.join(EXTRACT_DIR, ref, f"{ref}_down_coords_projected.csv")
        df_out = pd.DataFrame(results, columns=['latitude', 'longitude', 'up_latitude', 'up_longitude'])
        df_out.insert(0, 'id', range(1, len(df_out) + 1))
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        
        if not no_geojson:
            features = []
            for _, r in df_out.iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [r['longitude'], r['latitude']]},
                    "properties": {"id": r['id'], "ref": ref, "type": "downstream"}
                })
            gj = {"type": "FeatureCollection", "features": features}
            gj_path = out_path.replace('.csv', '.geojson')
            with open(gj_path, 'w', encoding='utf-8') as f:
                json.dump(gj, f, ensure_ascii=False)
                
        return True
    return False

import multiprocessing

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--max-dist", type=float, default=5000.0)
    parser.add_argument("--no-geojson", action="store_true")
    args = parser.parse_args()

    if args.ref:
        process_single_ref(args.ref, args.max_dist, args.no_geojson)
    else:
        refs = sorted([d for d in os.listdir(EXTRACT_DIR) if os.path.isdir(os.path.join(EXTRACT_DIR, d)) and d.isdigit()], key=int)
        pool_args = [(str(r), args.max_dist, args.no_geojson) for r in refs]
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(process_single_ref, pool_args)

if __name__ == "__main__":
    main()
