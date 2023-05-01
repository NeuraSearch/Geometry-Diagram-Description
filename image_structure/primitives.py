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
            each Dict: {"angle": [line1, point1, line2] or [], "length": [point1, line1, point2] or []}
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
    for angle_triple in angles_info:
        # angle_triple: [line1, point1, line2, degree]
        point = angle_triple[1]
        line_1 = angle_triple[0]
        line_2 = angle_triple[2]
        degree = angle_triple[3]
        
        if type(point) == Point:
            if point.angle_name:
                angles_res.append(f"Angle {point.angle_name} has degree of {str(degree)}.")
            else:
                point_name = point.ref_name
                if line_1.ref_name and line_2.ref_name:
                    if point_name:
                        angles_res.append(f"Line {line_1.ref_name} and Line {line_2.ref_name} cross at Point {point_name} has degree of {str(degree)}.")
                    else:
                        angles_res.append(f"The Angle between Line {line_1.ref_name} and Line {line_2.ref_name} has degree of {str(degree)}.")
                        
        elif type(point) == Circle:
            circle_name = point.ref_name
            if len(point.rel_center_points) > 0:
                center_point = point.rel_center_points[0]
                
                if center_point.ref_name:
                    if line_1.ref_name and line_2.ref_name:
                        if circle_name:
                            angles_res.append(f"Line {line_1.ref_name} and Line {line_2.ref_name} cross at the Center Point {center_point.ref_name} of Circle {circle_name} has degree of {str(degree)}.")
                        else:
                            angles_res.append(f"Line {line_1.ref_name} and Line {line_2.ref_name} cross at Point {center_point.ref_name} has degree of {str(degree)}.")
            
    # length
    lines_res = []
    lines_info = per_data_result["length"]
    for line_triple in lines_info:
        # line_triple: [point1, line1, point2, length]
        line = line_triple[1]
        point_1 = line_triple[0]
        point_2 = line_triple[2]
        length = line_triple[3]
        
        if line.ref_name:
            lines_res.append(f"The length of Line {line.ref_name} is {str(length)}.")
        else:
            if point_1.ref_name and point_2.ref_name:
                lines_res.append(f"The length of Line {point_1.ref_name}{point_2.ref_name} is {str(length)}.")
    
    return angles_res, lines_res

def generate_for_congruent_angle(sym_val):
    
    angles_name = []
    
    for triple in sym_val:
        # triple: [line1, point1, line2]
        point = triple[1]
        line_1 = triple[0]
        line_2 = triple[2]

        if point.angle_name:
            angles_name.append(point.angle_name)
        else:
            point_name = point.ref_name
            if line_1.ref_name and line_2.ref_name:
                if point_name:
                    angles_name.append(f"between Line {line_1.ref_name} and Line {line_2.ref_name} cross at Point {point_name}")
                else:
                    angles_name.append(f"between Line {line_1.ref_name} and Line {line_2.ref_name}")
    
    if len(angles_name) > 1:
        res = f"Angle {angles_name[0]} has the same degree with "
        for angle in angles_name[1:]:
            res += f"Angle {angle} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_congruent_bar(sym_val):
    
    lines_name = []
    
    for triple in sym_val:
        # triple: [point1, line1, point2]
        line = triple[1]
        point_1 = triple[0]
        point_2 = triple[2]
        
        if line.ref_name:
            lines_name.append(f"{line.ref_name}")
        else:
            if point_1.ref_name and point_2.ref_name:
                lines_name.append(f"{point_1.ref_name}{point_2.ref_name}")

    if len(lines_name) > 1:
        res = f"Line {lines_name[0]} has the same length with "
        for line in lines_name[1:]:
            res += f"Line {line} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_parallel(sym_val):

    lines_name = []
    
    for line_list in sym_val:
        # line_list: [line]
        line = line_list[0]
        
        if line.ref_name:
            lines_name.append(f"{line.ref_name}")
    
    if len(lines_name) > 1:
        res = f"Line {lines_name[0]} is parallel with "
        for line in lines_name[1:]:
            res += f"Line {line} and "
        
        res = re.sub(r".and.$", "", res)
        
        return res
    else:
        return None

def generate_for_perpendicular(sym_val):
    
    res = []
    
    for triple in sym_val:
        # triple: [line1, point1, line2]
        point = triple[1]
        line_1 = triple[0]
        line_2 = triple[2]

        point_name = point.ref_name
        if line_1.ref_name and line_2.ref_name:
            if point_name:
                res.append(f"Line {line_1.ref_name} is perpendicular with Line {line_2.ref_name} at Point {point_name}.")
            else:
                res.append(f"Line {line_1.ref_name} is perpendicular with Line {line_2.ref_name}.")

    if len(res) > 0:
        return res
    else:
        return None