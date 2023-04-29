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
            parse_results.append(None)
            continue
        
        """ 2. parse text_symbols and geos. """
        # {"angle": [point] or [] , "length": [line] or []}
        # print("per_ocr_res: ", per_ocr_res)

        text_symbols_geos_rel, points, lines, circles = parse_text_symbol_rel_per_data(
            per_sym_geo_rel["text_symbol_geo_rel"], 
            per_ocr_res, 
            points, lines, circles, 
            per_sym_geo_rel["head_symbol_geo_rel"]
        )
        text_symbols_parse_results.append(text_symbols_geos_rel)
        
        # for p in points:
        #     print(p.ref_name)

        # for l in lines:
        #     print(l.ref_name)

        # for c in circles:
        #     print(c.ref_name)
        
        # input()
        
        """ 3. parse congruent. """
        # {"[]angle_symbols", "bar_symbols", "parallel_symbols", "perpendicular"}
        other_symbols_geos_rel = {}
        for sym, sym_rel in per_sym_geo_rel.items():
            
            if sym_rel != None:
                
                if "angle" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_congruent_geo(sym_rel, points)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "bar" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_congruent_geo(sym_rel, lines)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "parallel" in sym:
                    assert sym not in other_symbols_geos_rel
                    res = extract_congruent_geo(sym_rel, lines)
                    if res != None:
                        other_symbols_geos_rel[sym] = res
                elif "perpendicular" in sym:
                    res = extract_perperdicular_geo(sym_rel, points)
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

def parse_text_symbol_rel_per_data(text_sym_geo_rel, ocr, points, lines, circles, sym_head):
    """
    Args:
        text_sym_geo_rel (Tensor[N_ts, P+L+C+P+L+C+H]): relations between text symbols and all geos (may include head_symbol).
        ocr (List[str]): ocr results for each text symbol.
        points (List[Point]): _description_
        lines (List[Line]): _description_
        circles (List[Circle]): _description_
        sym_head (Tensor[N_sh, P+L+C]): relations between head symbols and LPL, PLP, PCP.
    """

    # {"angle", "length"}
    parse_res = defaultdict(list)
    
    if text_sym_geo_rel == None:
        return parse_res, points, lines, circles
    
    """ construct function for determine ids to correct geo. """
    func, geo_start_ids = build_ids_assignment(points, lines, circles, sym_head)
    
    """ parse text symbol and geo, head rel. """
    idx_cache = {}
    total_text_sym = text_sym_geo_rel.size(0)
    for i in range(total_text_sym):

        max_value_cand, max_idx_cand = torch.sort(text_sym_geo_rel[i].squeeze(-1), descending=True)
        for val, idx in zip(max_value_cand.tolist(), max_idx_cand.tolist()):
            if (idx not in idx_cache) and (val > 0.5):
                max_ids = idx
                        
                # if relevant to P, L, C, we just assign ocr of this sym to P, L, C
                if func["points"](max_ids):
                    which_point_ids = max_ids - geo_start_ids["points"]
                    res_ = re.findall(r"[A-Z]{1}", ocr[i])
                    if len(res_) > 0:
                        points[which_point_ids].ref_name = res_[0]
                        idx_cache[idx] = None
                        break
                elif func["lines"](max_ids):
                    which_line_ids = max_ids - geo_start_ids["lines"]
                    res_ = re.findall(r"[a-z]{1}", ocr[i])
                    if len(res_) > 0:
                        lines[which_line_ids].ref_name = res_[0]
                        idx_cache[idx] = None
                        break
                elif func["circles"](max_ids):
                    which_circle_ids = max_ids - geo_start_ids["circles"]
                    circles[which_circle_ids].ref_name = ocr[i]
                    break

                elif func["LPL"](max_ids):
                    which_point_ids = max_ids - geo_start_ids["LPL"]
                    ocr_str = ocr[i]
                    res = resolve_LPL(points, which_point_ids, ocr_str)
                    if res != None:
                        parse_res["angle"].append(res)    # (Point, angle_degree_in_int)
                        idx_cache[idx] = None
                        break
                
                elif func["PLP"](max_ids):
                    which_line_ids = max_ids - geo_start_ids["PLP"]
                    ocr_str = ocr[i]
                    res = resolve_PLP(lines, which_line_ids, ocr_str)
                    if res != None:
                        parse_res["length"].append(res)
                        idx_cache[idx] = None
                        break
                
                elif func["PCP"](max_ids):
                    which_circle_ids = max_ids - geo_start_ids["PCP"]
                    ocr_str = ocr[i]
                    res = resolve_PCP(circles, which_circle_ids, ocr_str)
                    if res != None:
                        parse_res["angle"].append(res)
                        idx_cache[idx] = None
                        break
                        
                elif func["head"](max_ids):
                    which_head_ids = max_ids - geo_start_ids["head"]
                    this_head_rel = sym_head[which_head_ids]
                    this_head_point_max_ids = torch.argmax(this_head_rel).item()
                    
                    
                    if this_head_point_max_ids - len(points) < 0:
                        
                        res_ = re.findall(r"[A-Z]{1}", ocr[i])
                        if len(res_) > 0:
                            points[this_head_point_max_ids].ref_name = res_[0]
                            idx_cache[idx] = None
                            break
                        # head_sym points to Points, LPL, PLP, PCP,
                        # since func["points"], func["lines"], func["circles"] is consistent to
                        #       func["LPL"] - len(points),    funcp["PLP"] - len(points),  func["PCP"] - len(points)
                        # we use func["points"], func["lines"], func["circles"] to determine which geo this head points to
                    elif func["points"](this_head_point_max_ids - len(points)):
                        which_ids_head_point = this_head_point_max_ids - len(points) - geo_start_ids["points"]
                        res = resolve_LPL(points, which_ids_head_point, ocr[i])
                        if res != None:
                            parse_res["angle"].append(res)
                            idx_cache[idx] = None
                            break
                    elif func["lines"](this_head_point_max_ids - len(points)):
                        which_ids_head_point = this_head_point_max_ids - len(points) - geo_start_ids["lines"]
                        res = resolve_PLP(lines, which_ids_head_point, ocr[i])
                        if res != None:
                            parse_res["length"].append(res)
                            idx_cache[idx] = None
                            break
                    elif func["circles"](this_head_point_max_ids - len(points)):
                        which_ids_head_point = this_head_point_max_ids - len(points) - geo_start_ids["circles"]
                        res = resolve_PCP(circles, which_ids_head_point, ocr[i])
                        if res != None:
                            parse_res["angles"].append(res)
                            idx_cache[idx] = None
                            break
            else:
                continue
                
    return parse_res, points, lines, circles

def build_ids_assignment(points, lines, circles, sym_head):

    func = {}
    geo_start_ids = {}
    prev_end = 0
    
    for geo in ["points", "lines", "circles", "LPL", "PLP", "PCP", "head"]:
        
        if geo in ["points", "LPL"]:
            total_num = len(points)
        elif geo in ["lines", "PLP"]:
            total_num = len(lines)
        elif geo in ["circles", "PCP"]:
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

def resolve_LPL(points, which_point_ids, ocr_str):

    if len(re.findall(r"[A-Z]", ocr_str)) == 1:     # name
        points[which_point_ids].angle_name = ocr_str
    elif len(re.findall(r"\d", ocr_str)) > 0:       # degree
        try:
            angle_num = int(re.sub(" ", "", ocr_str))
            return ([points[which_point_ids], angle_num])
        except ValueError as e:
            pass
        
    return None

def resolve_PLP(lines, which_line_ids, ocr_str):

    if len(ocr_str) > 0:
        return (lines[which_line_ids], ocr_str)
    
    return None

def resolve_PCP(circles, which_circle_ids, ocr_str):
    
    if len(re.findall(r"\d", ocr_str)) > 0:
        try:
            angle_num = int(re.sub(" ", "", ocr_str))
            return (circles[which_circle_ids], angle_num)
        except ValueError as e:
            pass
    
    return None

def extract_congruent_geo(symbol_geo_rel, geo):
    
    total_angles = symbol_geo_rel.size(0)
    
    idx_cache = {}
    results = []
    for i in range(total_angles):
        max_value_cand, max_idx_cand = torch.sort(symbol_geo_rel[i].squeeze(-1), descending=True)
        for val, idx in zip(max_value_cand.tolist(), max_idx_cand.tolist()):
            if (idx not in idx_cache) and (val > 0.5):
                results.append(geo[idx])
                idx_cache[idx] = None
                break
    
    return results if len(results) > 1 else None

def extract_perperdicular_geo(perpendicular_geo_rel, geo):
    
    total_sym = perpendicular_geo_rel.size(0)
    
    idx_cache = {}
    points = []
    for i in range(total_sym):
        max_value_cand, max_idx_cand = torch.sort(perpendicular_geo_rel[i].squeeze(-1), descending=True)
        for val, idx in zip(max_value_cand.tolist(), max_idx_cand.tolist()):
            if (idx not in idx_cache) and (val > 0.5):
                points.append(geo[idx])
                idx_cache[idx] = None
                break

    return points if len(points) > 0 else None