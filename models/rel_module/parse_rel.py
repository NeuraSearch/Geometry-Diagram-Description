# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))

import re
import torch
from functools import partial
from collections import defaultdict

from image_structure import Point, Line, Circle

is_function = lambda x, min, max: min <= x < max

def parse_rel(geo_rels, sym_geo_rels, ocr_results, threshold=0.5):
    """Main function to parse the relations.

    Args:
        geo_rels (List[Dict]): each dict contain relation between (points and lines), (points and circles),
            The Dict keys must be: "pl_rels" (P, L), "pc_rels" (P, C).
        sym_geo_rels (List[Dict]): each dict contain relations between sym and geo, (#sym, #relevant_geo)
            The Dict keys must be: "text_symbol_geo_rel", "head_symbol_geo_rel", 
                "[None|double_|triple_|quad_|penta_]angle_symbols_geo_rel",
                "[None|double_|triple_|quad_]bar_symbols_geo_rel",
                "[None|double_|parallel_]parallel_symbols_geo_rel",
                "perpendicular_symbols_geo_rel".
        ocr_results (List[List]): each list contains ocr results for the text_symbols.
        threshold (float): the threshold for sym and geo relation prediction.
    
    Returns:
        parse_results (List(Dict)): each dict contains the parsed relations:
            keys: {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}
    """
    
    text_symbols_parse_results = []     # [Dict("angle": [] or [Point], "line": [] or [Point]), ...]
    other_symbols_parse_results = []    # [Dict("parallel": [lines, ...]), ...]
    # parse each data
    for per_geo_rel, per_sym_geo_rel, per_ocr_res in zip(geo_rels, sym_geo_rels, ocr_results):
        
        """ 1. parse points, lines, circles objects, and geo relations. """
        points, lines, circles = parse_geo_rel_per_data(per_geo_rel)
        
        if len(points) == 0:
            text_symbols_parse_results.append(None)
            other_symbols_parse_results.append(None)
            continue
        
        """ 2. parse text_symbols and geos. """
        # {"angle": [point] or [] , "length": [line] or []}
        # print("per_ocr_res: ", per_ocr_res)

        text_symbols_geos_rel, points, lines, circles = parse_text_symbol_rel_per_data(
            per_sym_geo_rel["text_symbol_geo_rel"], 
            per_ocr_res, 
            points, lines, circles, 
            per_sym_geo_rel["head_symbol_geo_rel"],
            per_sym_geo_rel["text_symbols"],
            per_sym_geo_rel["head_symbols"],
        )
        text_symbols_parse_results.append(text_symbols_geos_rel)
        
        """ 3. parse congruent. """
        # {"[]angle_symbols", "bar_symbols", "parallel_symbols", "perpendicular"}
        other_symbols_geos_rel = {}
        for sym, sym_rel in per_sym_geo_rel.items():
            if sym_rel != None:
                
                if "angle" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_congruent_angle_geo(sym_rel, points, lines)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "bar" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_congruent_bar_geo(sym_rel, lines, points)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "parallel" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_parallel_geo(sym_rel, lines)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "perpendicular" in sym:
                    res = extract_perperdicular_geo(sym_rel, points, lines)
                    if res != None:
                        other_symbols_geos_rel["perpendicular"] = res
                        
        other_symbols_parse_results.append(other_symbols_geos_rel)
    
    return text_symbols_parse_results, other_symbols_parse_results
      
def parse_geo_rel_per_data(geo_rel):
    """
        Args:
            geo_rel (Dict[]): geo rel of one data. contains "pl_rels" Tensor(P, L), "pc_rels" Tensor(P, C).
        Returns:
            points: (List[Point])
            lines: (List[Line])
            circles: (List[Circle])
    """
    
    pl_rels = geo_rel["pl_rels"]
    pc_rels = geo_rel["pc_rels"]
    if pl_rels == None and pc_rels == None:
        return [], [], []

    points = []
    lines = []
    circles = []
    
    """ Extract Points and Lines """
    if pl_rels != None:
        num_points = pl_rels.size(0)
        num_lines = pl_rels.size(1)
        
        points = [Point(ids=i) for i in range(num_points)]
        lines = [Line(ids=i) for i in range(num_lines)]
        
        for p in range(num_points):
            for l in range(num_lines):
                # print(pl_rels[p, :])
                if pl_rels[p, l].item() == 1:
                    # print("l1: ", l)
                    points[p].rel_endpoint_lines.append(lines[l])
                    lines[l].rel_endpoint_points.append(points[p])
                elif pl_rels[p, l].item() == 2:
                    # print("l2: ", l)
                    points[p].rel_online_lines.append(lines[l])
                    lines[l].rel_online_points.append(points[p])
    
    """ Extract Points and Circles """
    if pc_rels != None:
        num_points = pc_rels.size(0)
        num_circles = pc_rels.size(1)
        
        if len(points) == 0:    # we will reuse points if list already contains points objects
            points = [Point(ids=i) for i in range(num_points)]
        circles = [Circle(ids=i) for i in range(num_circles)]
        
        for p in range(num_points):
            for c in range(num_circles):
                if pc_rels[p, c].item() == 1:
                    circles[c].rel_on_circle_points.append(points[p])
                elif pc_rels[p, c].item() == 2:
                    circles[c].rel_center_points.append(points[p])
    
    return points, lines, circles

def parse_text_symbol_rel_per_data(text_sym_geo_rel, ocr, points, lines, circles, sym_head, text_symbols_class, head_symbols_class):
    """
    Args:
        text_sym_geo_rel (Tensor[N_ts, P+L+C+H]): relations between text symbols and all geos (may include head_symbol).
        ocr (List[str]): ocr results for each text symbol.
        points (List[Point]): _description_
        lines (List[Line]): _description_
        circles (List[Circle]): _description_
        sym_head (Tensor[N_sh, P+L+C]): relations between head symbols and Points, LPL, PLP, PCP.
    """

    # {"angle", "length"}
    parse_res = defaultdict(list)
    
    if text_sym_geo_rel == None:
        return parse_res, points, lines, circles
    
    """ construct function for determine ids to correct geo. """
    func, geo_start_ids = build_ids_assignment(points, lines, circles, sym_head)
    
    """ parse text symbol and geo, head rel. """
    class_0_cache = {}
    class_1_P_cache = {}
    class_2_L_cache = {}
    total_text_sym = text_sym_geo_rel.size(0)
    for i in range(total_text_sym):
        text_sym_class = torch.argmax(text_symbols_class[i]).item()
        if text_sym_class == 0:
            max_value_cand, max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1), descending=True)
            for val, idx in zip(max_value_cand.tolist(), max_idx_cand.tolist()):
                if (idx not in class_0_cache) and (val > 0.5):
                    if func["points"](idx):
                        which_point_ids = idx - geo_start_ids["points"]
                        res_ = re.findall(r"[A-Z]{1}", ocr[i])
                        if len(res_) > 0:
                            points[which_point_ids].ref_name = res_[0]
                            class_0_cache[idx] = None
                            break
                    elif func["lines"](idx):
                        which_line_ids = idx - geo_start_ids["lines"]
                        res_ = re.findall(r"[a-z]{1}", ocr[i])
                        if len(res_) > 0:
                            lines[which_line_ids].ref_name = res_[0]
                            class_0_cache[idx] = None
                            break
                    elif func["circles"](idx):
                        which_circle_ids = idx - geo_start_ids["circles"]
                        circles[which_circle_ids].ref_name = ocr[i]
                        class_0_cache[idx] = None
                        break
                    elif func["head"](idx):
                        which_head_ids = idx - geo_start_ids["head"]
                        this_head_rel = sym_head[which_head_ids].squeeze(-1)
                        this_head_class = torch.argmax(head_symbols_class[which_head_ids]).item()
                        
                        if this_head_class == 0:
                            if len(points) > 0:
                                P_max_value_cand, P_max_idx_cand = torch.sort(this_head_rel[0: len(points)], descending=True)
                                for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                                    if (idx not in class_0_cache) and (val > 0.5):
                                        res_ = re.findall(r"[A-Z]{1}", ocr[i])
                                        if len(res_) > 0:
                                            points[idx].ref_name = res_
                                            class_0_cache[idx] = None
                                            break
                        elif this_head_class == 1:  # LPL
                            if len(points) > 0 and len(lines) > 1:
                                P_max_value_cand, P_max_idx_cand = torch.sort(this_head_rel[0: len(points)], descending=True)
                                L_max_value_cand, L_max_idx_cand = torch.sort(this_head_rel[len(points):len(points)+len(lines)], descending=True)
                                selected_point = []
                                selected_lines = []
                                for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                                    if (idx not in class_1_P_cache) and (val > 0.5):
                                        selected_point.append(idx)
                                        break
                                if len(selected_point) == 1:
                                    for val, idx in zip(L_max_value_cand.tolist(), P_maxL_max_idx_cand_idx_cand.tolist()):
                                        if val > 0.5:
                                            selected_lines.append(idx)
                                            if len(selected_lines) == 2:
                                                break
                                
                                if len(selected_lines) == 2 and len(ocr[i]) > 0:
                                    res = resolve_LPL(ocr[i])
                                    if res[1] == "name":
                                        points[selected_point[0]].angle_name = res[0]
                                    elif res[1] == "degree":
                                        parse_res["angle"].append([lines[selected_lines[0]], points[selected_point[0]], lines[selected_lines[1]], res[0]])                                
                                    class_1_P_cache.append(selected_point[0])
                                    
                        elif this_head_class == 2: # PLP
                            if len(points) > 1 and len(lines) > 0:
                                P_max_value_cand, P_max_idx_cand = torch.sort(this_head_rel[0: len(points)], descending=True)
                                L_max_value_cand, L_max_idx_cand = torch.sort(this_head_rel[len(points):len(points)+len(lines)], descending=True)
                                selected_line = []
                                selected_points = []
                                
                                for val, idx in zip(L_max_value_cand.tolist(), L_max_idx_cand.tolist()):
                                    if (idx not in class_2_L_cache) and (val > 0.5):
                                        selected_line.append(idx)
                                        break
                                if len(selected_line) == 1:
                                    for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                                        if val > 0.5:
                                            selected_points.append(idx)
                                            if len(selected_points) == 2:
                                                break
                                
                                if len(selected_points) == 2 and len(ocr[i]) > 0:
                                    res = resolve_PLP(ocr[i])
                                    if res != None:
                                        parse_res["length"].append([points[selected_points[0]], lines[selected_line[0]], points[selected_points[1]], res])
                                        class_2_L_cache.append(selected_line[0])

                        elif this_head_class == 3:  # PCP
                            if len(points) > 1 and len(circles) > 0:
                                P_max_value_cand, P_max_idx_cand = torch.sort(this_head_rel[0: len(points)], descending=True)
                                C_max_value_cand, C_max_idx_cand = torch.sort(this_head_rel[len(points)+len(lines): len(points)+len(lines)+len(circles)], descending=True)
                                selected_circle= []
                                selected_points = []
                        
                                for val, idx in zip(C_max_value_cand.tolist(), C_max_idx_cand.tolist()):
                                    if val > 0.5:
                                        selected_circle.append(idx)
                                if len(selected_circle) == 1:
                                    for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                                        if val > 0.5:
                                            selected_points.append(idx)
                                            if len(selected_points) == 2:
                                                break
                                
                                if len(selected_points) == 2 and len(ocr[i]) > 0:
                                    res = resolve_PCP(circles, which_circle_ids, ocr_str)
                                    if res != None:
                                        parse_res["angle"].append([points[selected_points[0]], circles[selected_circle[0]], selected_points[1], res])
                        
                        else:
                            raise ValueError(f"Unknown head_symbol class: {this_head_class}")

        elif text_sym_class == 1:   # LPL
            if len(points) > 0 and len(lines) > 1:
                P_max_value_cand, P_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[0: len(points)], descending=True)
                L_max_value_cand, L_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[len(points):len(points)+len(lines)], descending=True)
                selected_point = []
                selected_lines = []
                for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                    if (idx not in class_1_P_cache) and (val > 0.5):
                        selected_point.append(idx)
                        break
                if len(selected_point) == 1:
                    for val, idx in zip(L_max_value_cand.tolist(), P_maxL_max_idx_cand_idx_cand.tolist()):
                        if val > 0.5:
                            selected_lines.append(idx)
                            if len(selected_lines) == 2:
                                break
                
                if len(selected_lines) == 2 and len(ocr[i]) > 0:
                    res = resolve_LPL(ocr[i])
                    if res[1] == "name":
                        points[selected_point[0]].angle_name = res[0]
                    elif res[1] == "degree":
                        parse_res["angle"].append([lines[selected_lines[0]], points[selected_point[0]], lines[selected_lines[1]], res[0]])
                    class_1_P_cache.append(selected_point[0])
                
        elif text_sym_class == 2:   # PLP
            if len(points) > 1 and len(lines) > 0:
                P_max_value_cand, P_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[0: len(points)], descending=True)
                L_max_value_cand, L_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[len(points):len(points)+len(lines)], descending=True)
                selected_line = []
                selected_points = []
                
                for val, idx in zip(L_max_value_cand.tolist(), L_max_idx_cand.tolist()):
                    if (idx not in class_2_L_cache) and (val > 0.5):
                        selected_line.append(idx)
                        break
                if len(selected_line) == 1:
                    for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                        if val > 0.5:
                            selected_points.append(idx)
                            if len(selected_points) == 2:
                                break
                
                if len(selected_points) == 2 and len(ocr[i]) > 0:
                    res = resolve_PLP(ocr[i])
                    if res != None:
                        parse_res["length"].append([points[selected_points[0]], lines[selected_line[0]], points[selected_points[1]], res])
                        class_2_L_cache.append(selected_line[0])

        elif text_sym_class == 3: # PCP
            if len(points) > 1 and len(circles) > 0:
                P_max_value_cand, P_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[0: len(points)], descending=True)
                C_max_value_cand, C_max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1)[len(points)+len(lines): len(points)+len(lines)+len(circles)], descending=True)
                selected_circle= []
                selected_points = []
        
                for val, idx in zip(C_max_value_cand.tolist(), C_max_idx_cand.tolist()):
                    if val > 0.5:
                        selected_circle.append(idx)
                if len(selected_circle) == 1:
                    for val, idx in zip(P_max_value_cand.tolist(), P_max_idx_cand.tolist()):
                        if val > 0.5:
                            selected_points.append(idx)
                            if len(selected_points) == 2:
                                break
                
                if len(selected_points) == 2 and len(ocr[i]) > 0:
                    res = resolve_PCP(circles, which_circle_ids, ocr_str)
                    if res != None:
                        parse_res["angle"].append([points[selected_points[0]], circles[selected_circle[0]], selected_points[1], res])
                            
        else:
            raise ValueError(f"Unknown text_sym_class: {text_sym_class}")
            
    return parse_res, points, lines, circles

def build_ids_assignment(points, lines, circles, sym_head):

    func = {}
    geo_start_ids = {}
    prev_end = 0
    
    for geo in ["points", "lines", "circles", "head"]:
        
        if geo in ["points"]:
            total_num = len(points)
        elif geo in ["lines"]:
            total_num = len(lines)
        elif geo in ["circles"]:
            total_num = len(circles)
        elif geo in ["head"]:
            total_num = sym_head.size(0) if sym_head != None else 0
        
        if total_num > 0:
            geo_start_ids[geo] = prev_end
            func[geo] = partial(is_function, min=prev_end, max=prev_end+total_num)
            prev_end += total_num
        else:
            func[geo] = partial(is_function, min=-10, max=-5)
    
    return func, geo_start_ids

def resolve_LPL(ocr_str):

    if len(re.findall(r"[A-Z]", ocr_str)) == 1:     # name
        return ocr_str, "name"
    elif len(re.findall(r"\d", ocr_str)) > 0:       # degree
        try:
            angle_num = int(re.sub(" ", "", ocr_str))
            return angle_num, "degree"
        except ValueError as e:
            pass
        
    return None, None

def resolve_PLP(ocr_str):

    if len(ocr_str) > 0:
        return ocr_str
    
    return None

def resolve_PCP(circles, which_circle_ids, ocr_str):
    
    if len(re.findall(r"\d", ocr_str)) > 0:
        return int(re.sub(" ", "", ocr_str))
        
    return None

def extract_congruent_angle_geo(symbol_geo_rel, points, lines):
    
    total_angles = symbol_geo_rel.size(0)
    
    results = []
    if len(points) > 0 and len(lines) > 1:  # LPL
        point_idx_cache = {}
        selected_point = []
        selected_lines = []
        for i in range(total_angles):
            max_P_value_cand, max_P_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1)[0:len(points)], descending=True)
            max_L_value_cand, max_L_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1)[len(points) : len(points) + len(lines)], descending=True)
            for val, idx in zip(max_P_value_cand.tolist(), max_P_idx_cand.tolist()):
                if (idx not in point_idx_cache) and (val > 0.5):
                    selected_point.append(idx)
                    point_idx_cache[idx] = None
                    break
            if len(selected_point) == 1:
                for val, idx in zip(max_L_value_cand.tolist(), max_L_idx_cand.tolist()):
                    if val > 0.5:
                        selected_lines.append(idx)
                        if len(selected_lines) == 2:
                            break
            
            if len(selected_lines) == 2:
                results.append([lines[selected_lines[0]], points[selected_point[0]], lines[selected_lines[1]]])
        
    return results if len(results) > 1 else None

def extract_congruent_bar_geo(symbol_geo_rel, lines, points):
    
    total_angles = symbol_geo_rel.size(0)
    
    results = []
    if len(points) > 1 and len(lines) > 0:  # PLP
        line_idx_cache = {}
        selected_line = []
        selected_points = []
        for i in range(total_angles):
            max_P_value_cand, max_P_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1)[0:len(points)], descending=True)
            max_L_value_cand, max_L_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1)[len(points) : len(points) + len(lines)], descending=True)
            for val, idx in zip(max_L_value_cand.tolist(), max_L_idx_cand.tolist()):
                if (idx not in line_idx_cache) and (val > 0.5):
                    selected_line.append(idx)
                    line_idx_cache[idx] = None
                    break
            if len(selected_line) == 1:
                for val, idx in zip(max_P_value_cand.tolist(), max_P_idx_cand.tolist()):
                    if val > 0.5:
                        selected_points.append(idx)
                        if len(selected_points) == 2:
                            break
            
            if len(selected_points) == 2:
                results.append([points[selected_points[0]], lines[selected_line[0]], points[selected_points[1]]])
        
    return results if len(results) > 1 else None

def extract_parallel_geo(symbol_geo_rel, lines):
    
    total_angles = symbol_geo_rel.size(0)
    
    results = []
    if len(lines) > 1:
        line_idx_cache = {}
        selected_line = []
        for i in range(total_angles):
            max_l_value_cand, max_l_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1), descending=True)
            for val, idx in zip(max_l_value_cand.tolist(), max_l_idx_cand.tolist()):
                if (idx not in line_idx_cache) and (val > 0.5):
                    selected_line.append(idx)
                    line_idx_cache[idx] = None
                    break

            if len(selected_line) == 1:
                results.append([lines[selected_line[0]]])
        
    return results if len(results) > 1 else None

def extract_perperdicular_geo(perpendicular_geo_rel, points, lines):
    
    total_sym = perpendicular_geo_rel.size(0)
    
    results = []
    if len(points) > 0 and len(lines) > 1:  # LPL
        point_idx_cache = {}
        selected_point = []
        selected_lines = []
        for i in range(total_sym):
            max_P_value_cand, max_P_idx_cand = torch.sort(perpendicular_geo_rel[i].squeeze(-1)[0 : len(points)], descending=True)
            max_L_value_cand, max_L_idx_cand = torch.sort(perpendicular_geo_rel[i].squeeze(-1)[len(points) : len(points) + len(lines)], descending=True)
            for val, idx in zip(max_P_value_cand.tolist(), max_P_idx_cand.tolist()):
                if (idx not in point_idx_cache) and (val > 0.5):
                    selected_point.append(idx)
                    point_idx_cache[idx] = None
                    break
            if len(selected_point) == 1:
                for val, idx in zip(max_L_value_cand.tolist(), max_L_idx_cand.tolist()):
                    if val > 0.5:
                        selected_lines.append(idx)
                        if len(selected_lines) == 2:
                            break
            
            if len(selected_lines) == 2:
                results.append([lines[selected_lines[0]], points[selected_point[0]], lines[selected_lines[1]]])
                        
    return results if len(results) > 0 else None