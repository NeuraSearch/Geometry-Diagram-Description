# coding:utf-8

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
                angle = rel[0].angle
                degree = rel[1]
                natural_language_results[rel_name].append(f"the degree of Angle {angle} is {degree}.")
        
        elif rel_name == "length":
            for rel in rel_results:
                line = rel[0]
                length = rel[1]
                natural_language_results[rel_name].append(f"the length of Line {str(line) is {length}}")
        
        elif rel_name == "congruent_angle":
            for rel in rel_results:
                if len(rel) != 0:
                    angles = ["Angle " + ang.angle for ang in rel]
                    natural_language_results[rel_name].append("The degrees of " + " and ".join(angles) + " are the same.")
        
        elif rel_name == "congruent_bar":
            for rel in rel_results:
                if len(rel) != 0:
                    lines = ["Line " + str(line) for line in rel]
                    natural_language_results[rel_name].append("The lengths of " + " and ".join(lines) + " are the same.")
        
        elif rel_name == "parallel":
            for rel in rel_results:
                if len(rel) != 0:
                    lines = ["Line " + str(line) for line in rel]
                    natural_language_results[rel_name].append(" and ".join(lines) + " are parallel.")
        
        elif rel_name == "perpendicular":
            for point_list in rel_results:
                for point in point_list:
                    if len(point.rel_endpoint_lines) > 1:
                        line_0 = point.rel_endpoint_lines[0]
                        line_1 = point.rel_endpoint_lines[1]
                
                        natural_language_results[rel_name].append(f"Line {str(line_0)} is perpendicular to Line {str(line_1)} at Point {str(point)}.")
                    else:
                        continue