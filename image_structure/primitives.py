# coding:utf-8

import re
from collections import defaultdict

class Point:
    def __init__(self, ids, ref_fake_name):
        self.ids = ids
        self.ids_name = f"p{ids}"
        
        self.ref_name = None
        self.ref_fake_name = ref_fake_name
        self.angle_name = None
        
        self.rel_endpoint_lines = []
        self.rel_online_lines = []
    
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

    def __str__(self):
        if self.ref_name:
            return self.ref_name
        else:
            return self.ref_fake_name

class Line:
    def __init__(self, ids, ref_fake_name):
        self.ids = ids
        self.ids_name = f"l{ids}"

        self.ref_name = None
        self.ref_fake_name = ref_fake_name
        
        self.rel_endpoint_points = []
        self.rel_online_points = []
    
    def __eq__(self, other):
        return self.ids == other.ids

    def __str__(self):
        if self.ref_name:
            return self.ref_name
        else:
            return self.ref_fake_name

class Circle:
    def __init__(self, ids, ref_fake_name=None):
        self.ids = ids
        self.ids_name = f"c{ids}"

        self.ref_name = None
        self.ref_fake_name = ref_fake_name
                
        self.rel_on_circle_points = []
        self.rel_center_points = []

    def __str__(self):
        if self.ref_name:
            return self.ref_name
        else:
            return self.ref_fake_name
            
def convert_parse_to_natural_language(text_symbols_parse_results, other_symbols_parse_results, points, lines, circles):
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
    
    # describe the image in our own language
    for per_data_points, per_data_lines, per_data_circles in zip(points, lines, circles):
        primitives_info = {}
        
        if len(per_data_points) > 0:
           primitives_info["points"] = generate_for_points(per_data_points)
        
        if len(per_data_lines) > 0:
            primitives_info["lines"] = generate_for_lines(per_data_lines)
        
        if len(per_data_circles) > 0:
            primitives_info["circles"] = generate_for_circles(per_data_circles)
        
        results.append(primitives_info)
    
    # generate for text_symbols
    for idx, per_data_result in enumerate(text_symbols_parse_results):
        if per_data_result == None:
            results.append({})
            continue
        angles_res, lines_res = generate_for_text_symbols(per_data_result)
        
        # print("per_data_result: ", per_data_result)
        
        per_data_text_symbol_nl = {
            "angle": angles_res,    # [str, str, ...] or []
            "length": lines_res,
        }
        # print("angles_res: ", angles_res)
        # print("lines_res: ", lines_res)
        
        results[idx].update(per_data_text_symbol_nl)
    
    # print("results: ", results)
    # print("*"*100)
    # print()
    
    for idx, per_data_result in enumerate(other_symbols_parse_results):
        if per_data_result == None:
            continue
        per_data_other_symbol_nl = defaultdict(list)    # value: [] or [str, str, str]
        for sym_key, sym_val in per_data_result.items():
            print("sym_key: ", sym_key)
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
    
    # print("final results: ", results)
    # print("-"*100)
    # print()
    # input()
    
    return results

def generate_for_points(per_data_points):
    points_res = []
    
    for point in per_data_points:
        points_res.append(f"Point {point}")
    
    res = "The diagram contains "
    for point in points_res:
        res = res + point + ", "
    
    return res

def generate_for_lines(per_data_lines):
    lines_res = []
    endpoints_res = []
    onlines_res = []    
    
    for line in per_data_lines:
        endpoints_per_line = []
        onlines_per_line = []
        
        lines_res.append(f"Line {line}")
        if len(line.rel_endpoint_points) > 0:
            for point in line.rel_endpoint_points:
                endpoints_per_line.append(f"Point {point}")
            endpoints_res.append(endpoints_per_line)
        if len(line.rel_online_points) > 0:
            for point in line.rel_online_points:
                onlines_per_line.append(f"Point {point}")
            onlines_res.append(onlines_per_line)
    
    res = "The digram contains "
    for line, endpoints, onlines in zip(lines_res, endpoints_res, onlines_res):
        res += f"Line {line}, "
        
        if len(endpoints) > 0:
            temp = "which has endpoints: "
            for end in endpoints:
                temp += f"Point {end}, "
            res += temp
        elif len(onlines) > 0:
            temp = "In addition, there are "
            for on in onlines:
                temp += f"Point {on}, "
            res += temp
            res += "on the line."
    
    return res

def generate_for_circles(per_data_circles):
    # self.rel_on_circle_points = []
    # self.rel_center_points = []

    circles_res = []
    oncircles_res = []
    center_res = []    
    
    for circle in per_data_circles:
        oncircles_on_circle = []
        center_on_circle = []
        
        circles_res.append(f"Circle {circle}")
        if len(circle.rel_on_circle_points) > 0:
            for point in circle.rel_on_circle_points:
                oncircles_on_circle.append(f"Point {point}")
            oncircles_res.append(oncircles_on_circle)
        if len(circle.rel_center_points) == 1:
            center_on_circle.append(f"Point {circle.rel_center_points[0]}")
            center_res.append(center_on_circle)
    
    res = "The digram contains "
    for circle, oncircles, center in zip(circles_res, oncircles_res, center_res):
        res += f"Circle {circle}, "

        if len(center) == 1:
            temp = f"whose center point is {center[0]}, "
            res += temp
        
        if len(oncircles) > 0:
            temp = "which has "
            for on in oncircles:
                temp += f"Point {on}, "
            res += temp
            res += "on its arc."
    
    return res

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
            # print("Name Point: ", Point)
            if point.angle_name:
                # print("Name Point angle_name: ", point.angle_name)
                angles_res.append(f"Angle {point.angle_name} has degree of {str(degree)}.")
            else:
                angles_res.append(f"Line {line_1} and Line {line_2} cross at Point {point} has degree of {str(degree)}.")
                        
        elif type(point) == Circle:
            if len(point.rel_center_points) > 0:
                center_point = point.rel_center_points[0]
                angles_res.append(f"Line {line_1} and Line {line_2} cross at the Center Point {center_point} of Circle {[point]} has degree of {str(degree)}.")
            else:
                angles_res.append(f"Line {line_1} and Line {line_2} cross at the Center Point of Circle {[point]} has degree of {str(degree)}.")
            
    # length
    lines_res = []
    lines_info = per_data_result["length"]
    for line_triple in lines_info:
        # line_triple: [point1, line1, point2, length]
        line = line_triple[1]
        point_1 = line_triple[0]
        point_2 = line_triple[2]
        length = line_triple[3]
        
        lines_res.append(f"The length of Line {line} between Point {point_1} and Point {point_2} is {str(length)}.")
    
    return angles_res, lines_res

def generate_for_congruent_angle(sym_val):
    
    angles_name = []
    # print("congruent_angle...")
    for triple in sym_val:
        # triple: [line1, point1, line2]
        point = triple[1]
        line_1 = triple[0]
        line_2 = triple[2]
        if point.angle_name:
            # print("congruent_angle point.angle_name: ", point.angle_name)
            angles_name.append(point.angle_name)
        else:
            angles_name.append(f"between Line {line_1} and Line {line_2} cross at Point {point}")
    
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
    # print("congruent_bar...")
    for triple in sym_val:
        # triple: [point1, line1, point2]
        line = triple[1]
        point_1 = triple[0]
        point_2 = triple[2]
        
        lines_name.append(f"{line} between Point {point_1} and Point {point_2}")
    
    # print("lines_name: ", lines_name)
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
    # print("parallel...")
    for line_list in sym_val:
        # line_list: [line]
        line = line_list[0]
        # print("parallel line.ref_name: ", line.ref_name)
        lines_name.append(f"{line}")
    
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
    # print("perpendicular...")
    for triple in sym_val:
        # triple: [line1, point1, line2]
        point = triple[1]
        line_1 = triple[0]
        line_2 = triple[2]

        res.append(f"Line {line_1} is perpendicular with Line {line_2} at Point {point}.")

    if len(res) > 0:
        return res
    else:
        return None