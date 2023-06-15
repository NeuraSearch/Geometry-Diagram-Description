import numpy as np
import math
import json

THRESH_DIST = 4
MAXLEN = 1e12

def get_point_dist(p0, p1):
    dist = math.sqrt((p0[0]-p1[0])**2
                    +(p0[1]-p1[1])**2)
    return dist

def get_angle_bet2vec(vec1, vec2):
    vector_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    length_prod = get_point_dist([0,0], vec1)*get_point_dist([0,0], vec2)
    cos_value = vector_prod * 1.0 / (length_prod * 1.0 + 1e-8)
    cos_value = min(1,max(-1,cos_value))
    return (math.acos(cos_value) / math.pi) * 180

def sub_vec(vec1,vec2):
    return (np.array(vec1)-np.array(vec2)).tolist()

def isLineIntersectRectangle(point1, point2, rectangle, pixes_change=0):
    linePointX1, linePointY1 = point1[0], point1[1]
    linePointX2, linePointY2 = point2[0], point2[1]
    rectangleLeftTopX, rectangleLeftTopY, rectangleRightBottomX, rectangleRightBottomY = \
        rectangle[0]-pixes_change, rectangle[1]-pixes_change, \
        rectangle[0]+rectangle[2]-1+pixes_change, rectangle[1]+rectangle[3]-1+pixes_change
        
    lineHeight = linePointY1 - linePointY2
    lineWidth = linePointX2 - linePointX1

    c = linePointX1 * linePointY2 - linePointX2 * linePointY1

    if ((lineHeight * rectangleLeftTopX + lineWidth * rectangleLeftTopY + c >= 0 and  
        lineHeight * rectangleRightBottomX + lineWidth * rectangleRightBottomY + c <= 0) or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleLeftTopY + c <= 0 and 
        lineHeight * rectangleRightBottomX + lineWidth * rectangleRightBottomY + c >= 0) or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleRightBottomY + c >= 0 and
        lineHeight * rectangleRightBottomX + lineWidth * rectangleLeftTopY + c <= 0 )or 
        (lineHeight * rectangleLeftTopX + lineWidth * rectangleRightBottomY + c <= 0 and 
        lineHeight * rectangleRightBottomX + lineWidth * rectangleLeftTopY + c >= 0)):

        if (rectangleLeftTopX > rectangleRightBottomX):
            rectangleLeftTopX, rectangleRightBottomX = rectangleRightBottomX, rectangleLeftTopX

        if (rectangleLeftTopY < rectangleRightBottomY):
            rectangleLeftTopY, rectangleRightBottomY = rectangleRightBottomY, rectangleLeftTopY 
 
        if ((linePointX1 < rectangleLeftTopX and linePointX2 < rectangleLeftTopX)
            or (linePointX1 > rectangleRightBottomX and linePointX2 > rectangleRightBottomX)
            or (linePointY1 > rectangleLeftTopY and linePointY2 > rectangleLeftTopY)
            or (linePointY1 < rectangleRightBottomY and linePointY2 < rectangleRightBottomY)):
            return False
        else:
            return True
    else:
        return False
        
def IsTrangleOrArea(x1,y1,x2,y2,x3,y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
 
def IsInsideTrangle(trangle_point1, trangle_point2, trangle_point3, judge_point):
    '''
        trangle_point1: endpoint
        trangle_point2: endpoint
        trangle_point3: crosspoint
        judge_point: judge_point
    '''
    thresh_value= 2.0
    increase_ratio = 3.0
    x1, y1 = trangle_point1[0], trangle_point1[1]
    x2, y2 = trangle_point2[0], trangle_point2[1]
    x3, y3 = trangle_point3[0], trangle_point3[1] # intersection point
    x, y = judge_point[0], judge_point[1] # judgment point
    # Expand the length of the sides of the triangle
    x1, y1 = increase_ratio*x1+(1-increase_ratio)*x3, increase_ratio*y1+(1-increase_ratio)*y3
    x2, y2 = increase_ratio*x2+(1-increase_ratio)*x3, increase_ratio*y2+(1-increase_ratio)*y3
    ABC = IsTrangleOrArea(x1,y1,x2,y2,x3,y3)
    PBC = IsTrangleOrArea(x,y,x2,y2,x3,y3)
    PAC = IsTrangleOrArea(x1,y1,x,y,x3,y3)
    PAB = IsTrangleOrArea(x1,y1,x2,y2,x,y)
    return abs(ABC-PBC-PAC-PAB)<thresh_value

def get_angle_point(relation, elem_dict, line_dict, img_name=None, text_class_name='angle', text_content=None):
    
    """
        According to the sign within the triangle to determine angle direction, and returns the nearest neighbor point
        Consider special circumstances:
        1. Angle is very large and it still can not include the symbol center after enlargement the triangle, so choose the only reasonable Angle
        2. If the edge of the corner passes through the symbol rectangle (referring to the head sometimes happens), then choose the similar Angle
        3. If two corresponding edges are the same (Angle=180), then select the end point of the edge
    """
    sym = relation[0]
    sym_bbox = elem_dict[sym]['bbox']
    sym_center_loc = [sym_bbox[0]+sym_bbox[2]/2, sym_bbox[1]+sym_bbox[3]/2]

    crosspoint = relation[1][0]
    crosspoint_loc = elem_dict[crosspoint]['loc'][0]

    l1_point_list, l2_point_list = line_dict[relation[1][1]], line_dict[relation[1][2]]
    point_num_l1, point_num_l2 = len(l1_point_list), len(l2_point_list)

    for index in range(point_num_l1):
        if l1_point_list[index][0]==crosspoint:
            crosspoint_l1_index = index 
            break
    for index in range(point_num_l2):
        if l2_point_list[index][0]==crosspoint:
            crosspoint_l2_index = index 
            break
    
    angle_num = 0
    near_p1, near_p2 = crosspoint_l1_index, crosspoint_l2_index

    for point_l1_index in [0, point_num_l1-1]:
        if point_l1_index!=crosspoint_l1_index:
            for point_l2_index in [0, point_num_l2-1]:
                if point_l2_index!=crosspoint_l2_index:
                    angle_num += 1 
                    t1, t2 = point_l1_index, point_l2_index
                    if IsInsideTrangle(l1_point_list[point_l1_index][1], l2_point_list[point_l2_index][1], \
                                                crosspoint_loc, sym_center_loc):
                        # Return nearest neighbor
                        if point_l1_index==0: 
                            near_p1-=1 
                        else:
                            near_p1+=1
                        if point_l2_index==0: 
                            near_p2-=1 
                        else:
                            near_p2+=1

    if near_p1!=crosspoint_l1_index and near_p2!=crosspoint_l2_index:

        if elem_dict[sym]['sym_class']=='head' and text_class_name=='degree' and text_content.isdigit(): 
            # Determine if the head position fits the Angle
            degree_gt= int(text_content)
            angle_degree = get_angle_bet2vec(sub_vec(l1_point_list[near_p1][1], crosspoint_loc), \
                                sub_vec(l2_point_list[near_p2][1], crosspoint_loc))
            if abs(degree_gt-angle_degree)>20:
                if isLineIntersectRectangle(crosspoint_loc, l1_point_list[near_p1][1], sym_bbox):
                    near_p2 = 2*crosspoint_l2_index-near_p2
                else:
                    near_p1 = 2*crosspoint_l1_index-near_p1
        return l1_point_list[near_p1], l2_point_list[near_p2]
        
    elif angle_num==1:  # There's only one possible Angle
        if t1==0: 
            near_p1-=1 
        else:
            near_p1+=1
        if t2==0: 
            near_p2-=1 
        else:
            near_p2+=1
        return l1_point_list[near_p1], l2_point_list[near_p2]

    elif relation[1][1]==relation[1][2]: 
        return l1_point_list[0], l1_point_list[-1]

    else:
        return None, None

def get_elem_dict(primitives):
    elem_dict=dict()
    for prim_item in primitives:
        for item in prim_item:
            elem_dict[item['id']] = item
    return elem_dict

def get_annotation_json(path_file):
    with open(path_file, 'r') as file:
        contents = json.load(file)
    return contents

def mid_vec(vec1,vec2):
    return ((np.array(vec1)+np.array(vec2))/2).tolist()

def get_intersection(list_a, list_b):
    result_list = list(set(list_a)&set(list_b))
    return result_list


class point(): 
    def __init__(self, loc_list):
        self.x=loc_list[0]
        self.y=loc_list[1] 
        self.loc_list = loc_list

def cross(p1,p2,p3): 
    x1=p2.x-p1.x
    y1=p2.y-p1.y
    x2=p3.x-p1.x
    y2=p3.y-p1.y
    return x1*y2-x2*y1     

def IsIntersec(p1,p2,p3,p4): 
    """
        Determine whether two line segments intersect
    """
    if(max(p1.x,p2.x)>=min(p3.x,p4.x)  
            and max(p3.x,p4.x)>=min(p1.x,p2.x)
            and max(p1.y,p2.y)>=min(p3.y,p4.y)
            and max(p3.y,p4.y)>=min(p1.y,p2.y)): 
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
        and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            return True
        else:
            return False
    else:
        return False

def judge_arc_major(p1_name, p2_name, c_name, text_name, elem_dict, is_degree=True):
    """
        Judge whether the arc is major or minor according to:
        1. Whether intersect or not between the line connecting text to the center of a circle and the line connecting the end of arc
        2. The degree is greater than 180 or not.
    """
    p1 = point(elem_dict[p1_name]['loc'][0])
    p2 = point(elem_dict[p2_name]['loc'][0])
    p3 = point(elem_dict[c_name]['loc'][0])

    box_loc = elem_dict[text_name]['bbox']
    p4 = point([box_loc[0]+box_loc[2]/2, box_loc[1]+box_loc[3]/2])
    dis2center = get_point_dist(p3.loc_list, p4.loc_list)
    radius = elem_dict[c_name]['loc'][1]
    
    value = elem_dict[text_name]['text_content']
    if (is_degree and value.isdigit() and int(value)<180):
        return False
    elif (is_degree and value.isdigit() and int(value)>=180):
        return True
    elif not IsIntersec(p1, p2, p3, p4) and dis2center>radius:
        return True
    else:
        return False


