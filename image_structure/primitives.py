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

class Circle:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"c{ids}"

        self.ref_name = None
        
        self.rel_on_circle_points = []
        self.rel_center_points = []

def convert_parse_to_natural_language(parse_result):
    
    if parse_result == None:
        return None
    
    natural_language_results = defaultdict(list)
    
    for rel_name, rel_results in parse_result.items():
        
        if rel_name == "angle":
            for rel in rel_results:
                point = rel[0]
                degree = rel[1]
                res = generate_angle_degree(point, degree)
                if res != None:
                    natural_language_results[rel_name].append(res)
        
        elif rel_name == "length":
            for rel in rel_results:
                line = rel[0]
                length = rel[1]
                res = generate_line_length(line, length)
                if res != None:
                    natural_language_results[rel_name].append(res)
        
        elif "angle" in rel_name:
            res = generate_congruent_angles(rel_results)
            if res != None:
                natural_language_results["congruent_angle"].append(res)
        
        elif rel_name == "congruent_bar":
            res = generate_congruent_bars(rel_results)
            if res != None:
                natural_language_results["congruent_line"].append(res)  
        
        elif rel_name == "parallel":
            res = generate_parallel(rel_resulst)(rel_results)
            if res != None:
                natural_language_results["parallel"].append(res)           
        
        elif rel_name == "perpendicular":
            for per_perpendicular in rel_results:
                res = generate_perpendicular(per_perpendicular)

                if res != None:
                    natural_language_results["perpendicular"].append(res)           

def generate_angle_degree(point, degree):
    if point.angle_name:
        return f"Angle {point.angle_name} has degree of {str(degree)}."
    else:
        if point.ref_name:
            middle_name = point.ref_name
            
            temp = []
            if len(point.rel_endpoint_lines) > 1:
                for line in point.rel_endpoint_lines:
                    for l_p in line.rel_endpoint_points:
                        if l_p.ref_name and l_p != point:
                            temp.append(l_p)
                        if len(temp) == 2:
                            return f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name} has degree of {str(degree)}."
                    for l_p in line.rel_online_points:
                        if l_p.ref_name and l_p != point:
                            temp.append(l_p)
                        if len(temp) == 2:
                            return f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name} has degree of {str(degree)}."

                for line in point.rel_online_lines:
                    for l_p in line.rel_endpoint_points:
                        if l_p.ref_name and l_p != point:
                            temp.append(l_p)
                        if len(temp) == 2:
                            return f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name} has degree of {str(degree)}."
                    for l_p in line.rel_online_points and l_p != point:
                        if l_p.ref_name:
                            temp.append(l_p)
                        if len(temp) == 2:
                            return f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name} has degree of {str(degree)}."
    
    return None

def generate_line_length(line, length):
    if line.ref_name:
        return f"Line {line.ref_name} has length of {str(length)}."
    else:
        temp = []
        
        for p in line.rel_endpoint_points:
            if p.ref_name and p not in temp:
                temp.append(p)
            if len(temp) == 2:
                return f"Line {temp[0].ref_name}{temp[1].ref_name} has length of {str(length)}."
        
        for p in line.rel_online_points:
            if p.ref_name and p not in temp:
                temp.append(p)
            if len(temp) == 2:
                return f"Line {temp[0].ref_name}{temp[1].ref_name} has length of {str(length)}."
    
    return None

def generate_congruent_angles(rel_resulst):
    
    angle_cache = []
    for point in rel_resulst:
        if point.angle_name:
            angle_cache.append(point.angle_name)
        else:
            if point.ref_name:
                middle_name = point.ref_name
                
                temp = []
                if len(point.rel_endpoint_lines) > 1:
                    for line in point.rel_endpoint_lines:
                        for l_p in line.rel_endpoint_points:
                            if l_p.ref_name and l_p != point:
                                temp.append(l_p)
                            if len(temp) == 2:
                                angle_cache.append(f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name}")
                                break
                        if len(temp) == 2:
                            break
                        for l_p in line.rel_online_points:
                            if l_p.ref_name and l_p != point:
                                temp.append(l_p)
                            if len(temp) == 2:
                                angle_cache.append(f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name}")
                                break
                        if len(temp) == 2:
                            break
                
                if len(point.rel_online_lines) > 1 and len(temp) < 2: 
                    for line in point.rel_online_lines:
                        for l_p in line.rel_endpoint_points:
                            if l_p.ref_name and l_p != point:
                                temp.append(l_p)
                            if len(temp) == 2:
                                angle_cache.append(f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name}")
                                break
                        if len(temp) == 2:
                            break
                        for l_p in line.rel_online_points and l_p != point:
                            if l_p.ref_name:
                                temp.append(l_p)
                            if len(temp) == 2:
                                angle_cache.append(f"Angle {temp[0].ref_name}{middle_name}{temp[1].ref_name}")
                                break
                        if len(temp) == 2:
                            break
    
    if len(angle_cache) > 1:
        temp = f"Angle {angle_cache[0]} has the same degree with "
        for angle in angle_cache[1:]:
            temp += f"{angle} and "
        
        temp = re.sub(r".and.$", "", temp)
        
        return temp

    else:
        return None

def generate_congruent_bars(rel_resulst):
    
    line_cache = []
    
    for line in rel_resulst:
        if line.ref_name:
            angle_cache.append(line.ref_name)
        else:
            temp = []
        
            for p in line.rel_endpoint_points:
                if p.ref_name and p not in temp:
                    temp.append(p)
                if len(temp) == 2:
                    line_cache.append(f"Line {temp[0].ref_name}{temp[1].ref_name}")
                    break
            
            for p in line.rel_online_points:
                if p.ref_name and p not in temp:
                    temp.append(p)
                if len(temp) == 2:
                    line_cache.append(f"Line {temp[0].ref_name}{temp[1].ref_name}")
                    break
    
    if len(line_cache) > 1:
        temp = f"Line {line_cache[0]} has the same length with "
        for line in line_cache[1:]:
            temp += f"{line} and "
        
        temp = re.sub(r".and.$", "", temp)
        
        return temp

    else:
        return None

def generate_parallel(rel_resulst):
    
    line_cache = []
    
    for line in rel_resulst:
        if line.ref_name:
            angle_cache.append(line.ref_name)
        else:
            temp = []
        
            for p in line.rel_endpoint_points:
                if p.ref_name and p not in temp:
                    temp.append(p)
                if len(temp) == 2:
                    line_cache.append(f"Line {temp[0].ref_name}{temp[1].ref_name}")
                    break
            
            for p in line.rel_online_points:
                if p.ref_name and p not in temp:
                    temp.append(p)
                if len(temp) == 2:
                    line_cache.append(f"Line {temp[0].ref_name}{temp[1].ref_name}")
                    break
    
    if len(line_cache) > 1:
        temp = f"Line {line_cache[0]} is parallel with "
        for line in line_cache[1:]:
            temp += f"{line} and "
        
        temp = re.sub(r".and.$", "", temp)
        
        return temp

    else:
        return None

def generate_perpendicular(per_perpendicular):
    
    point = per_perpendicular
    
    point_name = point.ref_name
    lines_name = []

    if len(point.rel_endpoint_lines) > 0:
        for line in point.rel_endpoint_lines:
            if line.ref_name:
                if len(lines_name) < 2:
                    lines_name.append(line.ref_name)
                else:
                    break
    
    if len(point.rel_online_lines) > 0 and len(lines_name) < 2:
        for line in point.rel_online_lines:
            if line.ref_name:
                if len(lines_name) < 2:
                    lines_name.append(line.ref_name)
                else:
                    break
    
    if len(lines_name) == 2:
        if point != None:
            return f"Line {lines_name[0]} is perpendicular with Line {lines_name[1]} at Point {point}."
        else:
            return f"Line {lines_name[0]} is perpendicular with Line {lines_name[1]}."
    
    return None
            