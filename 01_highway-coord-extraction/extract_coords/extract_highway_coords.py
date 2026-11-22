import argparse
import sys
import os
import math
import json
import requests
import networkx as nx
from pyproj import Transformer

def calculate_distance(p1, p2):
    return math.sqrt(((p1[0]-p2[0])*88000)**2 + ((p1[1]-p2[1])*111000)**2)

def fetch_overpass_data(ref):
    print(f"[자동 수집] Overpass API에서 고속도로 {ref}번의 모든 차선 데이터를 가져옵니다...")
    q = f"""
    [out:json][timeout:180];
    area["ISO3166-1"="KR"][admin_level=2]->.searchArea;
    way["highway"="motorway"]["ref"~"(^|;){ref}(;|$)"](area.searchArea);
    out body;
    node(w);
    out skel;
    """
    headers = {'User-Agent': 'highway-coord-extractor/2.0'}
    
    servers = [
        'https://overpass-api.de/api/interpreter',
        'https://maps.mail.ru/osm/tools/overpass/api/interpreter',
        'https://overpass.kumi.systems/api/interpreter'
    ]
    
    for url in servers:
        try:
            r = requests.get(url, params={'data': q}, headers=headers, timeout=60)
            if r.status_code == 200:
                print(f"-> {url} 에서 데이터 수집 성공!")
                return r.json()
            else:
                print(f"Warning: {url} 통신 실패 ({r.status_code}). 다른 서버 시도 중...")
        except Exception as e:
            print(f"Warning: {url} 오류 ({e}). 다른 서버 시도 중...")
            
    print("Error: 모든 Overpass 서버와 통신할 수 없습니다.")
    sys.exit(1)

def calculate_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def is_clockwise(path_nodes, nodes):
    area = 0.0
    for i in range(len(path_nodes)-1):
        x1, y1 = nodes[path_nodes[i]]
        x2, y2 = nodes[path_nodes[i+1]]
        area += (x1 * y2 - x2 * y1)
    return area < 0

def build_graph_and_find_path(data, direction, ref):
    nodes = {}
    ways = []
    
    for el in data.get('elements', []):
        if el['type'] == 'node':
            nodes[el['id']] = (el['lon'], el['lat'])
        elif el['type'] == 'way':
            ways.append(el)
            
    if not nodes:
        print("Error: 해당하는 고속도로 데이터를 찾을 수 없습니다.")
        sys.exit(1)
        
    G = nx.DiGraph()
    for w in ways:
        nd = w['nodes']
        for i in range(len(nd)-1):
            u, v = nd[i], nd[i+1]
            if u in nodes and v in nodes:
                G.add_edge(u, v)
                
    print(f"-> 위상 수학 그래프 생성 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 간선")
    
    lats = [n[1] for n in nodes.values()]
    lons = [n[0] for n in nodes.values()]
    lat_spread = (max(lats) - min(lats)) * 111000
    lon_spread = (max(lons) - min(lons)) * 88000
    
    is_north_south = lat_spread >= lon_spread
    print(f"-> [분석] {'남북(North-South)' if is_north_south else '동서(East-West)'} 방향 고속도로로 감지되었습니다.")
    
    wcc = list(nx.weakly_connected_components(G))
    wcc.sort(key=len, reverse=True)
    
    if len(wcc) == 0:
        print("Error: 유효한 도로 네트워크가 없습니다.")
        sys.exit(1)
        
    print(f"-> 총 {len(wcc)}개의 차선 덩어리가 발견되었습니다.")
    
    valid_paths = []
    
    for i, comp in enumerate(wcc):
        comp_nodes = list(comp)
        if len(comp_nodes) < 10:
            continue
            
        comp_G = G.subgraph(comp_nodes)
        scc = list(nx.strongly_connected_components(comp_G))
        scc.sort(key=len, reverse=True)
        largest_scc = scc[0]
        
        is_circular = len(largest_scc) > 0.8 * len(comp_nodes)
        
        # 순환 고속도로 오인 방지 로직 (실제 순환 고속도로 번호만 허용)
        ring_highway_refs = ['100', '300', '400', '500', '600', '700']
        if is_circular and str(ref) not in ring_highway_refs:
            print(f"-> [보정] 거대 순환(SCC)이 감지되었으나, {ref}번은 순환 고속도로가 아니므로 일반 선형 로직으로 폴백합니다.")
            is_circular = False
            
        if is_circular:
            print(f"-> [분석] 360도 순환 고속도로(Ring Highway)로 감지되었습니다! (사이클 크기: {len(largest_scc)})")
            
            ring_G = G.subgraph(largest_scc)
            lats = [(n, nodes[n][1]) for n in largest_scc]
            start_node = max(lats, key=lambda x: x[1])[0]
            
            path = [start_node]
            current = start_node
            visited = set([start_node])
            
            while True:
                successors = list(ring_G.successors(current))
                if not successors:
                    break
                
                unvisited = [n for n in successors if n not in visited and n != start_node]
                if unvisited:
                    next_node = unvisited[0]
                else:
                    if start_node in successors:
                        next_node = start_node
                    else:
                        next_node = successors[0]
                
                if next_node == start_node:
                    path.append(next_node)
                    break
                    
                if next_node in visited:
                    break
                    
                visited.add(next_node)
                path.append(next_node)
                current = next_node
                
            clockwise = is_clockwise(path, nodes)
            print(f"-> 추출된 사이클 방향: {'시계방향(내선/상행)' if clockwise else '반시계방향(외선/하행)'}")
            
            if (clockwise and direction == 'up') or (not clockwise and direction == 'down'):
                valid_paths.append(path)
                print(f"-> 완벽한 {direction} 방향 360도 순환 조각을 찾았습니다! (노드 {len(path)}개)")
                
        else:
            # 새로운 논리: in-degree, out-degree와 물리적 거리를 종합적으로 판단
            start_nodes = [n for n in comp_nodes if comp_G.in_degree(n) == 0]
            end_nodes = [n for n in comp_nodes if comp_G.out_degree(n) == 0]
            
            # 만약 start_nodes나 end_nodes가 없거나, 너무 짧은 경로만 나오는 오류(왕복루프)를 방지하기 위해 
            # 양극단 최대 10개 노드들을 후보로 추가 (O(N^2) 성능 폭주 방지)
            if is_north_south:
                comp_nodes.sort(key=lambda nid: nodes[nid][1])
            else:
                comp_nodes.sort(key=lambda nid: nodes[nid][0])
                
            n_count = len(comp_nodes)
            margin = min(10, max(1, n_count // 100)) # 상하위 최대 10개로 제한
            bottom_nodes = comp_nodes[:margin]
            top_nodes = comp_nodes[-margin:]
            
            s_candidates = list(start_nodes)
            e_candidates = list(end_nodes)
            
            if (direction == 'up' and is_north_south) or (direction == 'down' and not is_north_south):
                s_candidates.extend(bottom_nodes)
                e_candidates.extend(top_nodes)
            else:
                s_candidates.extend(top_nodes)
                e_candidates.extend(bottom_nodes)
                
            best_path = None
            max_dist = -1
            
            for s in s_candidates:
                for e in e_candidates:
                    if nx.has_path(comp_G, s, e):
                        s_lon, s_lat = nodes[s]
                        e_lon, e_lat = nodes[e]
                        
                        if is_north_south:
                            goes_down = e_lat < s_lat # 북 -> 남
                        else:
                            goes_down = e_lon > s_lon # 서 -> 동
                            
                        is_down = goes_down
                        if (direction == 'down' and is_down) or (direction == 'up' and not is_down):
                            dist = calculate_distance((s_lon, s_lat), (e_lon, e_lat))
                            # 거리가 전체의 50% 이상 되는 직선 경로만 인정
                            if dist > max_dist and dist > (max(lat_spread, lon_spread) * 0.5):
                                max_dist = dist
                                best_path = nx.shortest_path(comp_G, s, e)
                                
            if best_path is None:
                # 마지막 안전망: 가장 거리가 긴 임의의 유효 경로 (최대 10개 제한)
                for s in bottom_nodes + top_nodes:
                    for e in bottom_nodes + top_nodes:
                        if s != e and nx.has_path(comp_G, s, e):
                            dist = calculate_distance(nodes[s], nodes[e])
                            if dist > max_dist:
                                max_dist = dist
                                best_path = nx.shortest_path(comp_G, s, e)

            if best_path:
                valid_paths.append(best_path)
                print(f"-> 완벽한 {direction} 방향 본선 조각을 찾았습니다! (노드 {len(best_path)}개)")
            
    if not valid_paths:
        print("Error: 요청하신 방향의 연속된 본선 경로를 찾지 못했습니다. 고속도로가 중간에 끊겨있을 수 있습니다.")
        sys.exit(1)
        
    return [[nodes[nid] for nid in p] for p in valid_paths]

def project_and_interpolate(coords):
    transformer = Transformer.from_crs("epsg:4326", "epsg:5179", always_xy=True)
    
    projected = []
    for lon, lat in coords:
        x, y = transformer.transform(lon, lat)
        projected.append((x, y, lon, lat))
        
    distances = [0.0]
    for i in range(1, len(projected)):
        p1 = projected[i-1]
        p2 = projected[i]
        dist = calculate_distance(p1, p2)
        distances.append(distances[-1] + dist)
        
    total_length_km = distances[-1] / 1000.0
    print(f"-> 총 연장 길이: {total_length_km:.2f} km")
    
    interval = 10000.0
    results = []
    
    target_dist = 0.0
    idx = 0
    
    while target_dist <= distances[-1]:
        while idx < len(distances) - 1 and distances[idx+1] < target_dist:
            idx += 1
            
        p1 = projected[idx]
        if idx < len(projected) - 1:
            p2 = projected[idx+1]
            d1 = distances[idx]
            d2 = distances[idx+1]
            
            if d2 == d1:
                ratio = 0
            else:
                ratio = (target_dist - d1) / (d2 - d1)
                
            interp_lon = p1[2] + (p2[2] - p1[2]) * ratio
            interp_lat = p1[3] + (p2[3] - p1[3]) * ratio
        else:
            interp_lon = p1[2]
            interp_lat = p1[3]
            
        results.append({
            'dist_km': target_dist / 1000.0,
            'lon': interp_lon,
            'lat': interp_lat
        })
        target_dist += interval
        
    if results[-1]['dist_km'] < total_length_km:
        results.append({
            'dist_km': total_length_km,
            'lon': projected[-1][2],
            'lat': projected[-1][3]
        })
        
    return results, coords

def save_geojson(raw_coords, interpolated_coords, folder_path, base_name):
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    geojson["features"].append({
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": raw_coords
        },
        "properties": {
            "name": "Highway Route",
            "stroke": "#0000FF",
            "stroke-width": 4
        }
    })
    
    for i, pt in enumerate(interpolated_coords):
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [pt['lon'], pt['lat']]
            },
            "properties": {
                "id": i,
                "distance_km": round(pt['dist_km'], 2),
                "marker-color": "#FF0000",
                "marker-size": "medium"
            }
        })
        
    file_path = os.path.join(folder_path, f"{base_name}_map.geojson")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"-> GeoJSON 시각화 파일 생성: {file_path}")

def save_csv(results, folder_path, base_name):
    file_path = os.path.join(folder_path, f"{base_name}_coords.csv")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Seq,Distance(km),Lon,Lat\n")
        for i, res in enumerate(results):
            f.write(f"{i+1},{res['dist_km']:.2f},{res['lon']:.6f},{res['lat']:.6f}\n")
    print(f"-> CSV 좌표 파일 생성: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="고속도로 좌표 완전 자동 추출기 (NetworkX 위상 수학 기반)")
    parser.add_argument('--ref', required=True, help="고속도로 노선 번호 (예: 15)")
    parser.add_argument('--direction', required=True, choices=['up', 'down'], help="방향 (up: 상행, down: 하행)")
    parser.add_argument('--output', required=True, help="결과물이 저장될 폴더 이름 (예: 15_down)")
    
    args = parser.parse_args()
    
    folder_name = args.output.replace('.csv', '')
    os.makedirs(folder_name, exist_ok=True)
    
    print("\n=======================================================")
    print(f" 고속도로 {args.ref}번 ({'하행' if args.direction=='down' else '상행'}) 10km 간격 좌표 추출 (AI 자동화)")
    print("=======================================================")
    
    data = fetch_overpass_data(args.ref)
    valid_paths_coords = build_graph_and_find_path(data, args.direction, args.ref)
    
    print("\n[저장 중...]")
    base_name = os.path.basename(args.output)
    
    all_results = []
    global_seq = 1
    
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for idx, raw_coords in enumerate(valid_paths_coords):
        results, track_coords = project_and_interpolate(raw_coords)
        
        # CSV 누적용 데이터 정리
        for r in results:
            r['frag_id'] = idx + 1
            r['seq'] = global_seq
            all_results.append(r)
            global_seq += 1
            
        # GeoJSON 본선 (조각별 라인)
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": track_coords
            },
            "properties": {
                "name": f"Highway {args.ref} {args.direction} Track (Frag {idx+1})",
                "stroke": "#0000FF",
                "stroke-width": 4
            }
        })
        
        # GeoJSON 10km 포인트
        for i, pt in enumerate(results):
            geojson["features"].append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pt['lon'], pt['lat']]
                },
                "properties": {
                    "frag_id": idx + 1,
                    "distance_km": round(pt['dist_km'], 2),
                    "marker-color": "#FF0000",
                    "marker-size": "medium"
                }
            })
            
    # 통합 CSV 저장
    csv_file_path = os.path.join(folder_name, f"{base_name}_coords.csv")
    with open(csv_file_path, 'w', encoding='utf-8') as f:
        f.write("id,distance_km,longitude,latitude,frag_id\n")
        for res in all_results:
            f.write(f"{res['seq']},{res['dist_km']:.2f},{res['lon']:.6f},{res['lat']:.6f},{res['frag_id']}\n")
    print(f"-> 통합 CSV 좌표 파일 생성: {csv_file_path}")

    # 통합 GeoJSON 저장
    geojson_file_path = os.path.join(folder_name, f"{base_name}_map.geojson")
    with open(geojson_file_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    print(f"-> 통합 GeoJSON 시각화 파일 생성: {geojson_file_path}")
        
    print("\n[모든 작업이 완료되었습니다!]")
    print(f"결과 폴더: {folder_name}/")
    print("  - _coords.csv: 10km 간격 좌표 데이터 (모든 조각 통합)")
    print("  - _map.geojson: 지도 시각화 검증 파일 (모든 조각 통합)")

if __name__ == "__main__":
    main()
