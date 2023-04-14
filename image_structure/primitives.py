# coding:utf-8

class Point:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"p{ids}"
        
        self.ref_name = None
        self.angle_name = None
        
        self.rel_endpoint_lines = []
        self.rel_online_lines = []
    
    # def __str__(self):
    #     if self.ref_name != None:
    #         return f"point {self.ref_name}"
    #     else:
    #         return self.ids_name

class Line:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"l{ids}"

        self.ref_name = None
        
        self.rel_endpoint_points = []
        self.rel_online_points = []
    
    # def __str__(self):
    #     if self.ref_name != None:
    #         return f"line {self.ref_name}"
    #     else:
    #         if len(self.endpoints) > 1:
    #             return f"line {self.endpoints[0]}{self.endpoints[-1]}"
    #         else:
    #             return self.ids_name

class Circle:
    def __init__(self, ids):
        self.ids = ids
        self.ids_name = f"c{ids}"

        self.ref_name = None
        
        self.rel_on_circle_points = []
        self.rel_center_points = []
    