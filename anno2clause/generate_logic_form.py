from utils import *
from itertools import combinations
import json
import math 

def push_list_by_value(list_cur, name, value, loc, is_arc_major=False):
    is_exist_value = False
    name1 = name 
    name2 = name[-1]+name[1:-1]+name[0]
    for k in range(len(list_cur)):
        if value==list_cur[k]['value'] and value!='':
            is_exist_value = True
            is_exist_name = False
            for exist_name in list_cur[k]['name']:
                if exist_name == name1 or exist_name == name2:
                    is_exist_name = True
                    break
            if not is_exist_name: 
                list_cur[k]['name'].append(name)
            break
    if not is_exist_value:
        list_cur.append({'value':value, 'name':[name], 'loc': loc, 'is_arc_major':is_arc_major})

def push_list_by_name(list_cur, name_list):
    is_exist_name = False
    record_k = None
    for k in range(len(list_cur)):
        if list_cur[k]['value']!='':
            for item in list_cur[k]['name']:
                item1 = item
                item2 = item[-1]+item[1:-1]+item[0]
                if item1 in name_list or item2 in name_list:
                    is_exist_name = True
                    record_k = k
    if not is_exist_name:
        list_cur.append({'value':'', 'name':name_list})
    else:
        for item in name_list:
            item1 = item
            item2 = item[-1]+item[1:-1]+item[0]
            if not item1 in list_cur[record_k]['name'] and \
                not item2 in list_cur[record_k]['name']:
                list_cur[record_k]['name'].append(item)

def get_normal_logic(list_cur, pre_seq, angle_com2sim_dict=None):
    """
        list_cur: [{'value':value, 'name':[name], 'loc': [x, y], 'is_arc_major': bool]}]
        pre_seq: ['l \\widehat ', 'm \\angle ', 'm \\widehat ']
        angle_com2sim_dict: complex angle (\angle ABC) -> simple angle (\angle 1) 
    """
    list_logic_form = [] 
    for item in list_cur:
        seq_each = ''
        pre_seq_new = pre_seq[:]
        # Distinguish major arcs or minor arcs
        if 'is_arc_major' in item and item['is_arc_major']:
            if pre_seq == 'l \\widehat ': pre_seq_new = 'l major \\widehat '
            if pre_seq == 'm \\widehat ': pre_seq_new = 'm major \\widehat '
        for ind, name in enumerate(item['name']):
            if ind>0:
                seq_each = seq_each + ' = '
            seq_each += pre_seq_new + name
            if not angle_com2sim_dict is None:
                if name in angle_com2sim_dict:
                    seq_each += " = " + pre_seq_new + angle_com2sim_dict[name]
                    angle_com2sim_dict[name] = ''
                name_flat = name[-1]+name[1:-1]+name[0]
                if name_flat in angle_com2sim_dict:
                    seq_each += " = " + pre_seq_new + angle_com2sim_dict[name_flat]
                    angle_com2sim_dict[name_flat] = ''

        if item['value']!='': 
            seq_each += ' = ' + item['value']

        if seq_each!='': list_logic_form.append(seq_each)

    if not angle_com2sim_dict is None:
        for key, value in angle_com2sim_dict.items():
            if value!='':
                seq_each = pre_seq+key+' = '+pre_seq+value
                list_logic_form.append(seq_each)

    return list_logic_form

def get_point_name(points, circles, textpoint2point, point2circle, 
                    elem_dict, id2name_dict):
    '''
        points: point_list
        circles: circle_list
        textpoint2point: [s0, [p0]]
        point2circle: relation [p0, c0, oncircle/center]
        id2name_dict: {id -> point_name}
        position_dict:{point_name -> loc_list}
    '''
    point_instances = list() # point name list
    point_positions = dict() # point name -> loc
    circle_instances = list() # circle name list

    for item in textpoint2point:
        sym_id, geo_id = item[0], item[1][0]  # s0, p0
        id2name_dict[geo_id] = elem_dict[sym_id]['text_content'] 
        point_instances.append(id2name_dict[geo_id]) 
        point_positions[id2name_dict[geo_id]] =  elem_dict[geo_id]['loc'][0]

    candidate_point_name = list(set([chr(i) for i in range(65,91)]) - set(point_instances)) 
    candidate_point_name.sort()
    index = 0

    for item in points:
        geo_id = item['id'] 
        if not geo_id in id2name_dict: 
            id2name_dict[geo_id] = candidate_point_name[index]
            point_instances.append(id2name_dict[geo_id])
            point_positions[id2name_dict[geo_id]] =  elem_dict[geo_id]['loc'][0]
            index +=1
    
    circle_id2center_id = dict() # circle id -> center id

    circle_name_num = dict() # Count the circle with the same center
    circle_index = dict() # the index of concentric circles

    for item in point2circle:
        if item[2]=='center':
            circle_name_num[item[0]] = circle_name_num.get(item[0],0) + 1
            circle_index[item[0]] = 0

    for item in point2circle:
        if item[2]=='center':
            circle_id2center_id[item[1]] = item[0]
            circle_name = id2name_dict[item[0]]
            # Consider concentric circles, where the same letter represents multiple circles as name+index
            if circle_name_num[item[0]]>1:
                circle_name = circle_name+str(circle_index[item[0]])
                circle_index[item[0]]+=1 
            id2name_dict[item[1]] = circle_name
            circle_instances.append(circle_name)

    for item in circles:
        geo_id = item['id']
        if not geo_id in circle_id2center_id:
            id2name_dict[geo_id] = candidate_point_name[index]
            point_instances.append(candidate_point_name[index])
            circle_instances.append(candidate_point_name[index])
            point_positions[candidate_point_name[index]] = elem_dict[geo_id]['loc'][0]
            index +=1
            
    return point_instances, point_positions, circle_instances

def get_line_instances(lines, point2line, elem_dict, id2name_dict, line_dict, line_name):
    '''
        point2line: relation [p0, l0, endpoint/online]
        line_dict: line id -> [[point id, point loc]]
    '''
    line_instances = []
    for item in point2line:
        if not item[1] in line_dict:
            line_dict[item[1]]=[]
        line_dict[item[1]].append([item[0], elem_dict[item[0]]['loc'][0]])

    # The points on the line are sorted on the x or y axes (the dimension with high variance)
    for item in lines:
        point_list = line_dict[item['id']]
        coord_x = [point[1][0] for point in point_list]
        coord_y = [point[1][1] for point in point_list]
        if max(coord_x)-min(coord_x)>max(coord_y)-min(coord_y):
            line_dict[item['id']].sort(key=lambda x:x[1][0])
        else:
            line_dict[item['id']].sort(key=lambda x:x[1][1])
        
        for point_a, point_b in list(combinations(line_dict[item['id']], 2)):
            line_instances.append(id2name_dict[point_a[0]]+id2name_dict[point_b[0]])
    
    return line_instances

def get_line_name(linename_each, line_name, line_dict, id2name_dict, elem_dict, image_id):

    for id in line_dict:
        line_name[id] = id2name_dict[line_dict[id][0][0]]+id2name_dict[line_dict[id][-1][0]]
    for item in linename_each:
        text_content = elem_dict[item[0]]['text_content']
        line_name[item[1][1]] = "line "+text_content
    
def get_PointLiesOnLine(id2name_dict, line_dict, line_name, image_id):

    list_PointLiesOnLine = list()
    for id, point_list in line_dict.items():
        name_record = list()
        for item in point_list: 
            name_record.append(id2name_dict[item[0]])
        try:
            if len(line_name[id].split()[-1])==1: 
                logic_form_each = line_name[id] + ' lieson ' + ' '.join(name_record)
            else:
                logic_form_each = 'line ' + ' '.join(name_record)
        except:
            print(image_id, "error in PointLiesOnLine!")
            continue

        list_PointLiesOnLine.append(logic_form_each)
    return list_PointLiesOnLine

def get_PointLiesOnCircle(elem_dict, id2name_dict, point2circle, circles, circle_dict, image_id):

    """
        circle_dict: circle id -> [[point id, dis]]
    """
    list_PointLiesOnCircle = list()
    for item in circles:
        circle_dict[item['id']] = []

    for item in point2circle:
        if item[2]!='center':
            center_loc = elem_dict[item[1]]['loc'][0]
            point_loc = elem_dict[item[0]]['loc'][0]
            circle_dict[item[1]].append([item[0], sub_vec(point_loc, center_loc)])
    
    for id in circle_dict:
        circle_dict[id].sort(key=lambda x:math.atan2(x[1][0], x[1][1]))
        name_record = list()
        for item in circle_dict[id]: 
            name_record.append(id2name_dict[item[0]])
        try:
            logic_form_each = "\\odot "+id2name_dict[id]+' lieson '+' '.join(name_record)
        except:
            print(image_id, "error in PointLiesOnLine!")
            continue

        list_PointLiesOnCircle.append(logic_form_each)

    return list_PointLiesOnCircle

def get_Parallel(parallel, elem_dict, line_name, image_id):
    
    list_Parallel2Line = list()
    parallel_list = []
    double_parallel_list = []
    triple_parallel_list = []

    def get_logic_form(item_list):
        for index, item in enumerate(item_list):
            if index==0:
                logic_form_each = line_name[item]
            else:
                logic_form_each += ' \\parallel ' + line_name[item]
        list_Parallel2Line.append(logic_form_each)

        
    for item in parallel:

        if not (len(item)==2 and item[1][0][0]=='l'):
            print(image_id+" does not conform to the relationship of parallel symbol!")
            continue

        if elem_dict[item[0]]['sym_class']=='parallel':
            parallel_list.append(item[1][0])
        if elem_dict[item[0]]['sym_class']=='double parallel':
            double_parallel_list.append(item[1][0])
        if elem_dict[item[0]]['sym_class']=='triple parallel':
            triple_parallel_list.append(item[1][0])

    if len(parallel_list)>1: get_logic_form(parallel_list) 
    if len(double_parallel_list)>1: get_logic_form(double_parallel_list)
    if len(triple_parallel_list)>1: get_logic_form(triple_parallel_list)

    return list_Parallel2Line

def get_Perpendicular(id2name_dict, perpendicular, line_name, image_id):

    list_Perpendicular = list()
    for item in perpendicular:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l'):
            print(image_id+" does not conform to the relationship of perpendicular symbol!")
            continue
        cross_point = id2name_dict[item[1][0]]
        line1 = line_name[item[1][1]]
        line2 = line_name[item[1][2]]
        try:
            logic_form_each = line1 + ' \\perp ' + line2 + ' on ' + cross_point 
        except:
            print(image_id, 'error in Perpendicular!')
            continue     
        list_Perpendicular.append(logic_form_each)

    return list_Perpendicular

def get_Textlen(id2name_dict, textlen, elem_dict, sym2head_dict, circles, 
                                    line_len_list, arc_len_list, image_id, head2sym_dict):
    '''
        The relation forms are as follows:
        1. [p, p, l]
        2. [p, p, c]
        3. [p, p]
        4. [] # If it is similar with diameter of circle, otherwise remove it 
    '''
    for item in textlen:
        sym = item[0]
        value = elem_dict[sym]['text_content']
        if sym in head2sym_dict:
            bboxt = elem_dict[head2sym_dict[sym]]['bbox']
        else:
            bboxt = elem_dict[sym]['bbox']
        try:
            if len(item[1])>=2:
                if not (item[1][0][0]=='p' and item[1][1][0]=='p'):
                    print(image_id+" does not conform to the relationship of text length!")
                    continue 
                name = id2name_dict[item[1][0]]+id2name_dict[item[1][1]]   
                if len(item[1])==3 and item[1][2][0]=='c': # arc
                    is_arc_major = judge_arc_major(item[1][0], item[1][1], item[1][2], sym, elem_dict, is_degree=False)
                    push_list_by_value(arc_len_list, name, value, bboxt, is_arc_major)
                else: # line
                    push_list_by_value(line_len_list, name, value, bboxt)
            elif len(item[1])==0:
                # only consider the situation of circle diameter
                if len(circles)>0 and (sym in sym2head_dict) and len(sym2head_dict[sym])==2:
                    for item in circles:
                        diameter = item['loc'][1]*2
                        head1_bbox = elem_dict[sym2head_dict[sym][0]]["bbox"]
                        head2_bbox = elem_dict[sym2head_dict[sym][1]]["bbox"]
                        head1_center = [head1_bbox[0]+head1_bbox[2]/2, head1_bbox[1]+head1_bbox[3]/2]
                        head2_center = [head2_bbox[0]+head2_bbox[2]/2, head2_bbox[1]+head2_bbox[3]/2]
                        if abs(get_point_dist(head1_center, head2_center)/diameter-1)<0.15:
                            name = '\\Phi '+id2name_dict[item['id']] # represent circle by a single letter
                            push_list_by_value(line_len_list, name, value, bboxt)
                            break
        except:
            print(image_id, "error in Length!")
            continue

def get_Bar(id2name_dict, bar, elem_dict, line_len_list, arc_len_list, image_id):

    def get_logic_form(item_list):
        name_list = []
        geo_type = 'line'
        for item in item_list:
            try:
                point1, point2 = id2name_dict[item[0]], id2name_dict[item[1]]
                name_list.append(point1+point2)
                if item[2][0]=='l':  
                    geo_type = 'line'
                else:
                    geo_type = 'arc'
            except:
                print(image_id, 'error in bar!')
                continue
        if geo_type=='line':
            push_list_by_name(line_len_list, name_list)
        else:
            push_list_by_name(arc_len_list, name_list)
                    
    bar_list, double_bar_list, triple_bar_list, quad_bar_list  = [], [], [], []

    for item in bar:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='p' and (item[1][2][0]=='l' or item[1][2][0]=='c')):
            print(image_id+" does not conform to the relationship of bar symbol!")
            continue         
        if elem_dict[item[0]]['sym_class']=='bar':
            bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='double bar':
            double_bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='triple bar':
            triple_bar_list.append(item[1])
        if elem_dict[item[0]]['sym_class']=='quad bar':
            quad_bar_list.append(item[1])

    if len(bar_list)>1: get_logic_form(bar_list)
    if len(double_bar_list)>1: get_logic_form(double_bar_list)
    if len(triple_bar_list)>1: get_logic_form(triple_bar_list)
    if len(quad_bar_list)>1: get_logic_form(quad_bar_list)

def get_Symangle(id2name_dict, symangle, elem_dict, line_dict, angle_degree_list, image_id):

    def get_logic_form(item_list):
        item_new_list = []
        for i in range(len(item_list)):
            crosspoint = item_list[i][1][0]
            crosspoint_loc = elem_dict[crosspoint]['loc'][0]
            # Get the near neighbor endpoints on both sides of the angle
            point1, point2 = get_angle_point(item_list[i], elem_dict, line_dict) 
            if point1==None or point2==None:
                print(image_id+" The equiangular symbol cannot find the corresponding angle!")
                continue
            angle_degree = get_angle_bet2vec(sub_vec(point1[1], crosspoint_loc), sub_vec(point2[1], crosspoint_loc))
            # Add neighbor points and angle degree
            item_new_list.append([crosspoint, [item_list[i][1][1], point1[0]], [item_list[i][1][2], point2[0]], angle_degree]) 
        # Merge the neighbor angle
        item_merge_list = []
        is_merge = [False]*len(item_new_list)
        for i in range(len(item_new_list)):
            if not is_merge[i]:
                for j in range(i+1, len(item_new_list)):
                    if not is_merge[j] \
                        and (item_new_list[i][0]==item_new_list[j][0]) \
                            and abs(item_new_list[i][3]-item_new_list[j][3])>10:
                        is_merge[j] = True
                        item_new_list[i][3] = item_new_list[i][3] + item_new_list[j][3]
                        if  item_new_list[i][1][0]==item_new_list[j][1][0]:
                            item_new_list[i][1] = item_new_list[j][2]
                        elif  item_new_list[i][1][0]==item_new_list[j][2][0]:
                            item_new_list[i][1] = item_new_list[j][1]
                        elif  item_new_list[i][2][0]==item_new_list[j][1][0]:
                            item_new_list[i][2] = item_new_list[j][2]
                        elif  item_new_list[i][2][0]==item_new_list[j][2][0]:
                            item_new_list[i][2] = item_new_list[j][1]
                item_merge_list.append(item_new_list[i])

        name_list = []
        for item in item_merge_list:
            try:
                point1, crosspoint, point2 = id2name_dict[item[1][1]], id2name_dict[item[0]], \
                                             id2name_dict[item[2][1]]
                name = point1+crosspoint+point2
                name_list.append(name) 
            except:
                print(image_id,"error in sym angle!")
                continue
        push_list_by_name(angle_degree_list, name_list)

    angle_list, double_angle_list, triple_angle_list =[], [], []
    quad_angle_list, penta_angle_list = [], []

    for item in symangle:
        if not (len(item[1])==3 and item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l'):
            print(image_id+" does not conform to the relationship of equiangular symbol!")
            continue              
        if elem_dict[item[0]]['sym_class']=='angle':
            angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='double angle':
            double_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='triple angle':
            triple_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='quad angle':
            quad_angle_list.append(item)
        if elem_dict[item[0]]['sym_class']=='penta angle':
            penta_angle_list.append(item)

    if len(angle_list)>0: get_logic_form(angle_list)
    if len(double_angle_list)>0: get_logic_form(double_angle_list)
    if len(triple_angle_list)>0: get_logic_form(triple_angle_list)
    if len(quad_angle_list)>0: get_logic_form(quad_angle_list)
    if len(penta_angle_list)>0: get_logic_form(penta_angle_list)

def get_Textangle(id2name_dict, textangle, text_class_name, elem_dict, line_dict, \
                        angle_com2sim_dict, angle_degree_list, arc_degree_list, image_id, head2sym_dict):

    for item in textangle:
        sym = item[0]
        if sym in head2sym_dict:
            bboxt = elem_dict[head2sym_dict[sym]]['bbox']
        else:
            bboxt = elem_dict[sym]['bbox']

        if not ((item[1][0][0]=='p' and item[1][1][0]=='l' and item[1][2][0]=='l') or \
            (item[1][0][0]=='p' and item[1][1][0]=='p' and item[1][2][0]=='c')):
            print(image_id+" does not conform to the relationship of angle text!")
            continue
        
        try:
            if item[1][2][0]=='l':
                if text_class_name=='angle':
                    point1, point2 = get_angle_point(item, elem_dict, line_dict, image_id, 'angle', elem_dict[sym]['text_content']) 
                    if point1==None or point2==None:
                        print(image_id+" The angle indicator cannot find the corresponding angle!")
                        continue
                    point1, crosspoint, point2 = id2name_dict[point1[0]], \
                                                    id2name_dict[item[1][0]], \
                                                        id2name_dict[point2[0]]
                    angle_com_name = point1+crosspoint+point2
                    angle_sim_name = elem_dict[sym]['text_content']
                    angle_com2sim_dict[angle_com_name] = angle_sim_name
                elif text_class_name=='degree':
                    point1, point2 = get_angle_point(item, elem_dict, line_dict, image_id, 'degree', elem_dict[sym]['text_content']) 
                    if point1==None or point2==None:
                        print(image_id+" The angle degree cannot find the corresponding angle!")
                        continue
                    name = id2name_dict[point1[0]]+id2name_dict[item[1][0]]+id2name_dict[point2[0]]
                    value = elem_dict[sym]['text_content']
                    push_list_by_value(angle_degree_list, name, value, bboxt)            

            elif item[1][2][0]=='c':
                point1, point2, center = id2name_dict[item[1][0]], \
                                            id2name_dict[item[1][1]], \
                                                id2name_dict[item[1][2]]
                name = point1 + point2
                value = elem_dict[sym]['text_content']
                is_arc_major = judge_arc_major(item[1][0], item[1][1], item[1][2], sym, elem_dict)
                push_list_by_value(arc_degree_list, name, value, bboxt, is_arc_major)  
                                          
        except:
            print(image_id, 'error in text angle/degree!')
            continue

def get_logic_form(relation_each, elem_dict, image_id):

    logic_item = dict()
    id2name_dict = dict()
    line_dict = dict()  # line id -> [point id, point loc]
    line_name = dict()  # line id -> name of line 
    circle_dict = dict()  # circle id -> [point id, point loc]
    sym2head_dict = dict() # sym(head) id -> head id 
    head2sym_dict = dict() # head id -> sym(head) id
    head_len_dict = dict() 

    line_len_list = list() # [{'value':#, 'name':[#]}]
    arc_len_list = list() 
    angle_degree_list = list() 
    arc_degree_list = list() 
    angle_com2sim_dict = dict() # complex angle (\angle ABC) -> simple angle (\angle 1)  

    points = relation_each['geos']['points']
    lines = relation_each['geos']['lines']
    circles = relation_each['geos']['circles']

    textpoint2point_each, point2circle_each, point2line_each = [], [], []
    parallel_each, perpendicular_each, Bar_each = [], [], []
    textlen_each, symangle_each, linename_each = [], [], []
    textangle_each, degree_each = [], []

    for item in relation_each['relations']["geo2geo"]:
        if item[1][0]=='c':
            point2circle_each.append(item)
        elif item[1][0]=='l':
            point2line_each.append(item)

    for item in relation_each['relations']["sym2sym"]:
        sym2head_dict[item[0]]=item[1] # list 
        if len(item[1])==2:
            head2sym_dict[item[1][0]]=item[0] # scale
            head2sym_dict[item[1][1]]=item[0] 
        else:
            head2sym_dict[item[1][0]]=item[0] 

    for item in relation_each['relations']["sym2geo"]:
        if 'parallel' in elem_dict[item[0]]['sym_class']: parallel_each.append(item)
        if elem_dict[item[0]]['text_class']=='point': textpoint2point_each.append(item)
        if elem_dict[item[0]]['sym_class']=='perpendicular': perpendicular_each.append(item)
        if 'bar' in elem_dict[item[0]]['sym_class']: Bar_each.append(item)
        if elem_dict[item[0]]['text_class']=='len': textlen_each.append(item)
        if 'angle' in elem_dict[item[0]]['sym_class']: symangle_each.append(item)
        if elem_dict[item[0]]['text_class']=='angle': textangle_each.append(item)
        if elem_dict[item[0]]['text_class']=='degree': degree_each.append(item)
        if elem_dict[item[0]]['text_class']=='line': linename_each.append(item)

        if elem_dict[item[0]]['sym_class']=='head_len': 
            head_len_id = elem_dict[item[0]]['id']
            sym_id = head2sym_dict[head_len_id]
            if elem_dict[sym_id]['sym_class']=='head':
                sym_id = head2sym_dict[sym_id]
            if not sym_id in head_len_dict:
                head_len_dict[sym_id] = []
            head_len_dict[sym_id] += item[1]
    
        if elem_dict[item[0]]['sym_class']=='head':
            head_id = item[0]
            sym_id = head2sym_dict[head_id]
            elem_dict[head_id]['text_content'] = elem_dict[sym_id]['text_content']
            if elem_dict[sym_id]['text_class']=='point': textpoint2point_each.append(item)
            if elem_dict[sym_id]['text_class']=='len': textlen_each.append(item)
            if elem_dict[sym_id]['text_class']=='angle': textangle_each.append(item)
            if elem_dict[sym_id]['text_class']=='degree': degree_each.append(item)
    
    for key, value in head_len_dict.items():
        textlen_each.append([key, value])

    # get point, line and circle
    get_point_name(points, circles, textpoint2point_each, point2circle_each, elem_dict, id2name_dict)
    get_line_instances(lines, point2line_each, elem_dict, id2name_dict, line_dict, line_name)
    get_line_name(linename_each, line_name, line_dict, id2name_dict, elem_dict, image_id)
    # PointLiesOnLine
    list_PointLiesOnLine =  get_PointLiesOnLine(id2name_dict, line_dict, line_name, image_id)
    # PointLiesOnCircle
    list_PointLiesOnCircle = get_PointLiesOnCircle(elem_dict, id2name_dict, point2circle_each, circles, circle_dict, image_id)
    # Parallel
    list_Parallel = get_Parallel(parallel_each, elem_dict, line_name, image_id)
    # Perpendicular
    list_Perpendicular = get_Perpendicular(id2name_dict, perpendicular_each, line_name, image_id)

    # Text_len
    get_Textlen(id2name_dict, textlen_each, elem_dict, sym2head_dict, circles, line_len_list, arc_len_list, image_id, head2sym_dict)
    # Bar
    get_Bar(id2name_dict, Bar_each, elem_dict, line_len_list, arc_len_list, image_id)
    # Degree
    get_Textangle(id2name_dict, degree_each, 'degree', elem_dict, line_dict, \
                                            angle_com2sim_dict, angle_degree_list, arc_degree_list, image_id, head2sym_dict)
    # Sym_angle
    get_Symangle(id2name_dict, symangle_each, elem_dict, line_dict, angle_degree_list, image_id)
    # Text_angle
    get_Textangle(id2name_dict, textangle_each, 'angle', elem_dict, line_dict, \
                                            angle_com2sim_dict, angle_degree_list, arc_degree_list, image_id, head2sym_dict)
   
    list_Linelen = get_normal_logic(line_len_list, pre_seq='')
    list_Arclen = get_normal_logic(arc_len_list, pre_seq='l \\widehat ')
    list_Angledeg = get_normal_logic(angle_degree_list, pre_seq='m \\angle ', angle_com2sim_dict=angle_com2sim_dict)
    list_Arcdeg = get_normal_logic(arc_degree_list, pre_seq='m \\widehat ')

    stru_logic_forms = list_PointLiesOnLine+list_PointLiesOnCircle
    dup_stru_logic_forms = list(set(stru_logic_forms))
    dup_stru_logic_forms.sort(key=stru_logic_forms.index)
    logic_item["parsing_stru_seqs"] = dup_stru_logic_forms

    sem_logic_forms = list_Parallel+list_Perpendicular+list_Linelen+list_Arclen+list_Angledeg+list_Arcdeg
    dup_sem_logic_forms = list(set(sem_logic_forms))
    dup_sem_logic_forms.sort(key=sem_logic_forms.index)
    logic_item["parsing_sem_seqs"] = dup_sem_logic_forms
    
    return logic_item, id2name_dict, line_dict, circle_dict

if __name__ == '__main__':

    path_input = r"./diagram_annotation.json"
    anno_all = get_annotation_json(path_input)
    logic_json = dict()

    for key in anno_all:
        relation_each = anno_all[key]
        points = relation_each['geos']['points']
        lines = relation_each['geos']['lines']
        circles = relation_each['geos']['circles']
        symbols = relation_each['symbols']
        elem_dict = get_elem_dict([points, lines, circles, symbols])
        logic_item, id2name_dict, line_dict, circle_dict = get_logic_form(relation_each, elem_dict, key)
        logic_json[key] = logic_item

    with open('clause_forms.json', 'w') as f:
        json.dump(logic_json, f, indent=2)
    
