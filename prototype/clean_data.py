# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import yaml
import codecs
import json

with codecs.open("config.yaml", "r", "utf-8") as file:
    config = yaml.safe_load(file)

args = {}
for _, config in config.items():
    for key, val in config.items():
        args[key] = val

from argparse import Namespace
args = Namespace(**args)
print(args)

with open(MAIN_PATH / args.test_annot_path, 'rb') as file:
    contents = json.load(file)

start = True
for key, value in contents.items():
    # if key == "1601":
    #     start = True
    
    if start:
        points = value["geos"]["points"]
        lines = value["geos"]["lines"]
        
        original_points = {}
        original_lines = {}
        
        for p in points:
            id = p["id"]
            original_points[id] = f"p{len(original_points)}"
        for l in lines:
            id = l["id"]
            original_lines[id] = f"l{len(original_lines)}"
        
        point_list = []
        for point in value["geos"]["points"]:
            point_list.append(
                {"id": original_points[point["id"]],
                 "loc": point["loc"]}
            )
        contents[key]["geos"]["points"] = point_list

        line_list = []
        for line in value["geos"]["lines"]:
            line_list.append(
                {"id": original_lines[line["id"]],
                 "loc": line["loc"]}
            )
        contents[key]["geos"]["lines"] = line_list
        
        geo2geo_list = []
        for rel in value["relations"]["geo2geo"]:
            temp = []
            for ent in rel:
                if ent in original_points:
                    temp.append(original_points[ent])
                elif ent in original_lines:
                    temp.append(original_lines[ent])
                else:
                    temp.append(ent)
            geo2geo_list.append(temp)
        contents[key]["relations"]["geo2geo"] = geo2geo_list

        sym2geo_list = []
        for rel in value["relations"]["sym2geo"]:
            temp = []
            for ent in rel[1]:
                if ent in original_points:
                    temp.append(original_points[ent])
                elif ent in original_lines:
                    temp.append(original_lines[ent])
                else:
                    temp.append(ent)
            sym2geo_list.append([rel[0], temp])
        contents[key]["relations"]["sym2geo"] = sym2geo_list

with open(MAIN_PATH / (args.test_annot_path + "_revised"), 'w') as file:
    contents = json.dump(contents, file, indent=4)