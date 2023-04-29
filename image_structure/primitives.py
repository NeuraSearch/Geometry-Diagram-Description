# coding:utf-8

import re
from collections import defaultdict

class Point:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"p{ids}"
        
        self.ref_name = None
        self.angle_name = None
        
        self.rel_endpoint_lines = []
        self.rel_online_lines = []
    
    def __str__(self):
        if self.ref_name != None:
            return self.ref_name
        else:
            return self.ids_name

    @property
    def angle(self):
        if self.angle_name != None:
            return self.angle_name
        else:
            return str(self)
    
    def can_perpendicular(self):
        return len(self.rel_endpoint_lines) > 1

    def __eq__(self, other):
        return self.ids == other.ids

class Line:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"l{ids}"

        self.ref_name = None
        
        self.rel_endpoint_points = []
        self.rel_online_points = []
    
    def __str__(self):
        if self.ref_name != None:
            return self.ref_name
        else:
            if len(self.rel_endpoint_points) > 1:
                return f"{self.rel_endpoint_points[0]}{self.rel_endpoint_points[-1]}"
            else:
                return self.ids_name

    def __eq__(self, other):
        return self.ids == other.ids

class Circle:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"c{ids}"

        self.ref_name = None
                
        self.rel_on_circle_points = []
        self.rel_center_points = []
            
def convert_parse_to_natural_language(text_symbols_parse_results, other_symbols_parse_results):
    """
    Args:
        text_symbols_parse_results: List[ Dict{} ], 
            each Dict: {"angle": [point, point, ...] or [], "line": [line, line, ...] or []}
        other_symbols_parse_results: List[ Dict{}], each Dict: {"parallel": [line, line, ...]}
    
    Returns:
        results List[ Dict{}, ...]: each dict contains keys: "angle", "length", values: [str, str, ...] or [],
            and keys: "congruent_angle", "congruent_line", "parallel", "perpendicular", values: [str, str, ...] or []
    """
    
    results = []    # List[ Dict{}, Dict{}, ...]         
    
    # generate for text_symbols
    for per_data_result in text_symbols_parse_results:
        angles_res, lines_res = generate_for_text_symbols(per_data_result)
        
        # print("per_data_result: ", per_data_result)
        
        per_data_text_symbol_nl = {
            "angle": angles_res,    # [str, str, ...] or []
            "length": lines_res,
        }
        # print("angles_res: ", angles_res)
        # print("lines_res: ", lines_res)
        
        results.append(per_data_text_symbol_nl)
        
    
    for idx, per_data_result in enumerate(other_symbols_parse_results):
        
        per_data_other_symbol_nl = defaultdict(list)    # value: [] or [str, str, str]
        for sym_key, sym_val in per_data_result.items():
            
            if "angle" in sym_key:  # congruent angle
                res = generate_for_congruent_angle(sym_val)
                if res != None:
                    per_data_other_symbol_nl["congruent_angle"].append(res)
            
            elif "bar" in sym_key:  # congruent bar
                res = generate_for_congruent_bar(sym_val)
                if res != None:
                    per_data_other_symbol_nl["congruent_line"].append(res)
                    
            elif "parallel" in sym_key: # parallel line
                res = generate_for_parallel(sym_val)
                if res != None:
                    per_data_other_symbol_nl["parallel"].append(res)
                
            elif "perpendicular" in sym_key:    # perpendicular
                res = generate_for_perpendicular(sym_val)
                if res != None:
                    # here, res is either List[str, str, ...] or []
                    per_data_other_symbol_nl["perpendicular"] = res
        
        # print("per_data_result: ", per_data_result)
        # print("per_data_other_symbol_nl: ", per_data_other_symbol_nl)
        # print()
        # print()
        # print()
        
        results[idx].update(per_data_other_symbol_nl)    
    
    return results
          
def generate_for_text_symbols(per_data_result):
    
    # angles
    angles_res = []
    angles_info = per_data_result["angle"]
    for info in angles_info:
        point = info[0]
        degree = info[1]

        if type(point) == Point:
        
            if point.angle_name:
                angles_res.append(f"Angle {point.angle_name} has degree of {str(degree)}.")
            else:
                # find two lines for angle
                angle_name = extract_two_edges_for_point(point)
                if angle_name != None:
                    angles_res.append(f"Angle {angle_name[0]}{angle_name[1]}{angle_name[2]} has degree of {str(degree)}.")
        
        elif type(point) == Circle:
            
            angle_name = extract_angle_from_circle(point)
            if angle_name != None:
                angles_res.append(f"Angle {angle_name[0]}{angle_name[1]}{angle_name[2]} has degree of {str(degree)}.")
        
    # length
    lines_res = []
    lines_info = per_data_result["length"]
    for info in lines_info:
        line = info[0]
        length = info[1]
        
        if line.ref_name:
            lines_res.append(f"The length of Line {line.ref_name} is {str(length)}.")
        else:
            # find two points for the line
            line_name = extract_two_points_line(line)
            if line_name != None:
                lines_res.append(f"The length of Line {line_name[0]}{line_name[1]} is {str(length)}.")
    
    return angles_res, lines_res

def generate_for_congruent_angle(points):
    
    angles_name = []
    
    for point in points:
        if point.angle_name:
            angles_name.append(point.angle_name)
        else:
            angle = extract_two_edges_for_point(point)
            if angle != None:
                angles_name.append(f"{angle[0]}{angle[1]}{angle[2]}")
    
    if len(angles_name) > 1:
        res = f"Angle {angles_name[0]} has the same degree with "
        for angle in angles_name[1:]:
            res += f"Angle {angle} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_congruent_bar(lines):
    
    lines_name = []
    
    for line in lines:
        if line.ref_name:
            lines_name.append(line.ref_name)
        else:
            line_name = extract_two_points_line(line)
            if line_name != None:
                lines_name.append(f"{line_name[0]}{line_name[1]}")
    
    if len(lines_name) > 1:
        res = f"Line {lines_name[0]} has the same length with "
        for line in lines_name[1:]:
            res += f"Line {line} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_parallel(lines):

    lines_name = []
    
    for line in lines:
        if line.ref_name:
            lines_name.append(line.ref_name)
        else:
            line_name = extract_two_points_line(line)
            if line_name != None:
                lines_name.append(f"{line_name[0]}{line_name[1]}")
    
    if len(lines_name) > 1:
        res = f"Line {lines_name[0]} is parallel with "
        for line in lines_name[1:]:
            res += f"Line {line} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_perpendicular(points):
    
    res = []
    
    # each point in points constitutes the perpendicular with lines of its own
    for point in points:
        lines = extract_two_edges_for_point(point, force=True)
        if lines != None:
            if lines[1] != None:
                res.append(f"Line {lines[0]} is perpendicular with Line {lines[2]} at Point {lines[1]}.")
            else:
                res.append(f"Line {lines[0]} is perpendicular with Line {lines[2]}.")

    if len(res) > 0:
        return res
    else:
        return None
    
def extract_two_edges_for_point(point, force=False):
    
    if point.ref_name or force:
        mid_name = point.ref_name
        
        
        candidates = []
        # 1. select from lines, where point is their endpoint
        if len(point.rel_endpoint_lines) > 0:
            for line in point.rel_endpoint_lines:
                for l_p in line.rel_endpoint_points:
                    if l_p.ref_name and l_p != point:
                        if len(candidates) < 2:
                            candidates.append(l_p)
                            break   # each line provide one point
                
                if len(candidates) < 1:
                    for l_p in line.rel_online_points:
                        if l_p.ref_name and l_p != point:
                            if len(candidates) < 2:
                                candidates.append(l_p)
                                break   # each line provide one point    
        if len(candidates) >= 2:
            return [candidates[0].ref_name, mid_name, candidates[1].ref_name]
        else:
            # 2. select from lines, where point is their online point
            if len(point.rel_online_lines) > 0:
                for line in point.rel_online_lines:
                    for l_p in line.rel_endpoint_points:
                        if l_p.ref_name and l_p != point:
                            if len(candidates) < 2:
                                candidates.append(l_p)
                                break   # each line provide one point
                        
                    if len(candidates) < 1:
                        for l_p in line.rel_online_points:
                            if l_p.ref_name and l_p != point:
                                if len(candidates) < 2:
                                    candidates.append(l_p)
                                    break   # each line provide one point    
        if len(candidates) >= 2:
            return [candidates[0].ref_name, mid_name, candidates[1].ref_name]
        else:
            return None
    else:
        return None

def extract_two_points_line(line):
    
    candidates = []
    for p in line.rel_endpoint_points:
        if p.ref_name and p not in candidates:
            candidates.append(p)
        if len(candidates) == 2:
            break
    
    if len(candidates) >= 2:
        return [candidates[0].ref_name, candidates[1].ref_name]
    else:
        for p in line.rel_online_points:
            if p.ref_name and p not in candidates:
                candidates.append(p)
            if len(candidates) == 2:
                break
    
    if len(candidates) == 2:
        return [candidates[0].ref_name, candidates[1].ref_name]
    else:
        return None

def extract_angle_from_circle(circle):
    
    if len(circle.rel_center_points) == 0:
        return None
    
    center_name = None
    for center in circle.rel_center_points:
        if center.ref_name:
            center_name = center.ref_name
            break
    
    if center_name == None:
        return None
    
    other_points = []
    for point in circle.rel_center_points:
        if point.ref_name:
            other_points.append(point.ref_name)
            if len(other_points) == 2:
                break
    
    if len(other_points) == 2:
        return [other_points[0].ref_name, center_name, other_points[1].ref_name]
    else:
        return None