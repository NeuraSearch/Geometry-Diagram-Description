# coding:utf-8

import bisect
import operator
import itertools

class Point:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"p{ids}"
        
        self.ref_name = None
        
        self.endpoint_lines = []
        self.onlines = []
    
    def __str__(self):
        if self.ref_name != None:
            return f"point {self.ref_name}"
        else:
            return self.ids_name

class Line:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"l{ids}"

        self.ref_name = None
        
        self.endpoints = []
        self.onlines = []
    
    def __str__(self):
        if self.ref_name != None:
            return f"line {self.ref_name}"
        else:
            if len(self.endpoints) > 1:
                return f"line {self.endpoints[0]}{self.endpoints[-1]}"
            else:
                return self.ids_name

class Circle:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"c{ids}"

        self.ref_name = None
        
        self.on_circle = []
        self.centers = []
    

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
    """
    
    for per_geo_rel, per_sym_geo_rel, per_ocr_res in zip(geo_rels, sym_geo_rels, ocr_results):
        
        points, lines, circles = parse_geo_rel_per_data(per_geo_rel)
        

def parse_sym_geo_rel_per_data(sym_geo_rel, points, lines, circles, ocr_res):
    pass

def text_symbel_rel(text_symbol_geo_rel, points, lines, circles, ocr_res, head_symbol_geo_rel=None):
    num_ts = text_symbol_geo_rel.size(0)

    all_geo_num = [len(points), len(lines), len(points), len(lines), len(circles)]
    if head_symbol_geo_rel != None:
        all_geo_num.append(head_symbol_geo_rel.size(0))
    all_geo_num_accum = list(itertools.accumulate(all_geo_num, operator.add))
    
    for i in range(num_ts):
        
        for j in range(sum(all_geo_num)):
            
            rel = text_symbol_geo_rel[i, j].item()
            
            if rel == 1:
                rel_belong_to = bisect.bisect_right(all_geo_num_accum, j)

                if rel_belong_to == 0:  # point
                    pass
    
    
        
def parse_geo_rel_per_data(geo_rel):
    """
        geo_rel (Dict[]): contains "pl_rels" (P, L), "pc_rels" (P, C).
    """

    pl_rels = geo_rel["pl_rels"]
    pc_rels = geo_rel["pc_rels"]

    points = []
    lines = []
    circles = []
    if pl_rels != None:
        num_points = pl_rels.size(0)
        num_lines = pl_rels.size(1)
        
        points = [Point(ids=i) for i in range(num_points)]
        lines = [Line(ids=i) for i in range(num_lines)]
        
        for i in range(num_points):
            for j in range(num_lines):
                if pl_rels[i, j].item() == 1:
                    points[i].endpoint_lines.append(j)
                    lines[j].endpoints.append(i)
                elif pl_rels[i, j].item() == 2:
                    points[i].onlines.append(j)
                    lines[j].onlines.append(i)
    
    if pc_rels != None:
        num_points = pl_rels.size(0)
        num_circles = pc_rels.size(1)
        
        points = [Point(ids=i) for i in range(num_points)]
        circles = [Circle(ids=i) for i in range(num_circles)]
        
        for i in range(num_points):
            for j in range(num_lines):
                if pl_rels[i, j].item() == 1:
                    circles[j].on_circle.append(i)
                elif pl_rels[i, j].item() == 2:
                    circles[j].centers.append(i)
    
    return points, lines, circles
    