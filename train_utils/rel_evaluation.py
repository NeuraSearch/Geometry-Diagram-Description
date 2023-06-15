# coding:utf-8

def evaluate_rel(all_golden_rel, all_predict_rel, metric_logger):
    
    for image_id in all_golden_rel.keys():
        golden_rel = all_golden_rel[image_id]
        predict_rel = all_predict_rel[image_id]
        golden_geo2geo = golden_rel["golden_geo2geo"]
        golden_sym2geo = golden_rel["golden_sym2geo"]
        
        # geo2geo
        golden_endpoint_num = golden_geo2geo["endpoint_num"]
        golden_online_num = golden_geo2geo["online_num"]
        golden_center_num = golden_geo2geo["center_num"]
        golden_oncircle_num = golden_geo2geo["oncircle_num"]
        
        predict_endpoint_num = predict_rel["endpoint_num"]
        predict_online_num = predict_rel["online_num"]
        predict_center_num = predict_rel["center_num"]
        predict_oncircle_num = predict_rel["oncircle_num"]
        
        correct_endpoint_num = min(golden_endpoint_num, predict_endpoint_num)
        incorrect_endpoint_num = abs(golden_endpoint_num - predict_endpoint_num)
        for _ in range(correct_endpoint_num):
            metric_logger.update(endpoint=1.0)
            metric_logger.update(all=1.0)
        for _ in range(incorrect_endpoint_num):
            metric_logger.update(endpoint=0.0)
            metric_logger.update(all=0.0)

        correct_online_num = min(golden_online_num, predict_online_num)
        incorrect_online_num = abs(golden_online_num - predict_online_num)
        for _ in range(correct_online_num):
            metric_logger.update(online=1.0)
            metric_logger.update(all=1.0)
        for _ in range(incorrect_online_num):
            metric_logger.update(online=0.0)
            metric_logger.update(all=0.0)

        correct_center_num = min(golden_center_num, predict_center_num)
        incorrect_center_num = abs(golden_center_num - predict_center_num)
        for _ in range(correct_center_num):
            metric_logger.update(center=1.0)
            metric_logger.update(all=1.0)
        for _ in range(incorrect_center_num):
            metric_logger.update(center=0.0)
            metric_logger.update(all=0.0)

        correct_oncircle_num = min(golden_oncircle_num, predict_oncircle_num)
        incorrect_oncircle_num = abs(golden_oncircle_num - predict_oncircle_num)
        for _ in range(correct_oncircle_num):
            metric_logger.update(oncircle=1.0)
            metric_logger.update(all=1.0)
        for _ in range(incorrect_oncircle_num):
            metric_logger.update(oncircle=0.0)
            metric_logger.update(all=0.0)
        
        # sym2geo
        golden_text_symbol_num = golden_sym2geo["text_symbol_num"]
        predict_text_symbol_num = predict_rel["text_symbol_num"] if "text_symbol_num" in predict_rel else 0
        correct_text_num = min(golden_text_symbol_num, predict_text_symbol_num)
        incorrect_text_num = abs(golden_text_symbol_num - predict_text_symbol_num)
        for _ in range(correct_text_num):
            metric_logger.update(text_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_text_num):
            metric_logger.update(text_symbol=0.0)
            metric_logger.update(all_sym=0.0)
        
        golden_angle_symbol_num = golden_sym2geo["angle_symbol_num"]
        predict_angle_symbol_num = predict_rel["angle_symbols_geo_rel"] if "angle_symbols_geo_rel" in predict_rel else 0
        correct_angle_num = min(golden_angle_symbol_num, predict_angle_symbol_num)
        incorrect_angle_num = abs(golden_angle_symbol_num - predict_angle_symbol_num)
        for _ in range(correct_angle_num):
            metric_logger.update(angle_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_angle_num):
            metric_logger.update(angle_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_angle_symbol_num = golden_sym2geo["double_angle_symbol_num"]
        predict_angle_symbol_num = predict_rel["double_angle_symbols_geo_rel"] if "double_angle_symbols_geo_rel" in predict_rel else 0
        correct_angle_num = min(golden_angle_symbol_num, predict_angle_symbol_num)
        incorrect_angle_num = abs(golden_angle_symbol_num - predict_angle_symbol_num)
        for _ in range(correct_angle_num):
            metric_logger.update(angle_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_angle_num):
            metric_logger.update(angle_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_angle_symbol_num = golden_sym2geo["triple_angle_symbol_num"]
        predict_angle_symbol_num = predict_rel["triple_angle_symbols_geo_rel"] if "triple_angle_symbols_geo_rel" in predict_rel else 0
        correct_angle_num = min(golden_angle_symbol_num, predict_angle_symbol_num)
        incorrect_angle_num = abs(golden_angle_symbol_num - predict_angle_symbol_num)
        for _ in range(correct_angle_num):
            metric_logger.update(angle_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_angle_num):
            metric_logger.update(angle_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_angle_symbol_num = golden_sym2geo["quad_angle_symbol_num"]
        predict_angle_symbol_num = predict_rel["quad_angle_symbols_geo_rel"] if "quad_angle_symbols_geo_rel" in predict_rel else 0
        correct_angle_num = min(golden_angle_symbol_num, predict_angle_symbol_num)
        incorrect_angle_num = abs(golden_angle_symbol_num - predict_angle_symbol_num)
        for _ in range(correct_angle_num):
            metric_logger.update(angle_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_angle_num):
            metric_logger.update(angle_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_angle_symbol_num = golden_sym2geo["penta_angle_symbol_num"]
        predict_angle_symbol_num = predict_rel["penta_angle_symbols_geo_rel"] if "penta_angle_symbols_geo_rel" in predict_rel else 0
        correct_angle_num = min(golden_angle_symbol_num, predict_angle_symbol_num)
        incorrect_angle_num = abs(golden_angle_symbol_num - predict_angle_symbol_num)
        for _ in range(correct_angle_num):
            metric_logger.update(angle_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_angle_num):
            metric_logger.update(angle_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_bar_symbol_num = golden_sym2geo["bar_symbol_num"]
        predict_bar_symbol_num = predict_rel["bar_symbols_geo_rel"] if "bar_symbols_geo_rel" in predict_rel else 0
        correct_bar_num = min(golden_bar_symbol_num, predict_bar_symbol_num)
        incorrect_bar_num = abs(golden_bar_symbol_num - predict_bar_symbol_num)
        for _ in range(correct_bar_num):
            metric_logger.update(bar_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_bar_num):
            metric_logger.update(bar_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_bar_symbol_num = golden_sym2geo["double_bar_symbol_num"]
        predict_bar_symbol_num = predict_rel["double_bar_symbols_geo_rel"] if "double_bar_symbols_geo_rel" in predict_rel else 0
        correct_bar_num = min(golden_bar_symbol_num, predict_bar_symbol_num)
        incorrect_bar_num = abs(golden_bar_symbol_num - predict_bar_symbol_num)
        for _ in range(correct_bar_num):
            metric_logger.update(bar_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_bar_num):
            metric_logger.update(bar_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_bar_symbol_num = golden_sym2geo["triple_bar_symbol_num"]
        predict_bar_symbol_num = predict_rel["triple_bar_symbols_geo_rel"] if "triple_bar_symbols_geo_rel" in predict_rel else 0
        correct_bar_num = min(golden_bar_symbol_num, predict_bar_symbol_num)
        incorrect_bar_num = abs(golden_bar_symbol_num - predict_bar_symbol_num)
        for _ in range(correct_bar_num):
            metric_logger.update(bar_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_bar_num):
            metric_logger.update(bar_symbol=0.0)
            metric_logger.update(all_sym=0.0)
        
        golden_bar_symbol_num = golden_sym2geo["quad_bar_symbol_num"]
        predict_bar_symbol_num = predict_rel["quad_bar_symbols_geo_rel"] if "quad_bar_symbols_geo_rel" in predict_rel else 0
        correct_bar_num = min(golden_bar_symbol_num, predict_bar_symbol_num)
        incorrect_bar_num = abs(golden_bar_symbol_num - predict_bar_symbol_num)
        for _ in range(correct_bar_num):
            metric_logger.update(bar_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_bar_num):
            metric_logger.update(bar_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_parallel_symbol_num = golden_sym2geo["parallel_symbol_num"]
        predict_parallel_symbol_num = predict_rel["parallel_symbols_geo_rel"] if "parallel_symbols_geo_rel" in predict_rel else 0
        correct_parallel_num = min(golden_parallel_symbol_num, predict_parallel_symbol_num)
        incorrect_parallel_num = abs(golden_parallel_symbol_num - predict_parallel_symbol_num)
        for _ in range(correct_parallel_num):
            metric_logger.update(parallel_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_parallel_num):
            metric_logger.update(parallel_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_parallel_symbol_num = golden_sym2geo["double_parallel_symbol_num"]
        predict_parallel_symbol_num = predict_rel["double_parallel_symbols_geo_rel"] if "double_parallel_symbols_geo_rel" in predict_rel else 0
        correct_parallel_num = min(golden_parallel_symbol_num, predict_parallel_symbol_num)
        incorrect_parallel_num = abs(golden_parallel_symbol_num - predict_parallel_symbol_num)
        for _ in range(correct_parallel_num):
            metric_logger.update(parallel_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_parallel_num):
            metric_logger.update(parallel_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_parallel_symbol_num = golden_sym2geo["triple_parallel_symbol_num"]
        predict_parallel_symbol_num = predict_rel["triple_parallel_symbols_geo_rel"] if "triple_parallel_symbols_geo_rel" in predict_rel else 0
        correct_parallel_num = min(golden_parallel_symbol_num, predict_parallel_symbol_num)
        incorrect_parallel_num = abs(golden_parallel_symbol_num - predict_parallel_symbol_num)
        for _ in range(correct_parallel_num):
            metric_logger.update(parallel_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_parallel_num):
            metric_logger.update(parallel_symbol=0.0)
            metric_logger.update(all_sym=0.0)

        golden_perpendicular_symbol_num = golden_sym2geo["perpendicular_symbol_num"]
        predict_perpendicular_symbol_num = predict_rel["perpendicular"] if "perpendicular" in predict_rel else 0
        correct_perpendicular_num = min(golden_perpendicular_symbol_num, predict_perpendicular_symbol_num)
        incorrect_perpendicular_num = abs(golden_perpendicular_symbol_num - predict_perpendicular_symbol_num)
        for _ in range(correct_perpendicular_num):
            metric_logger.update(perpendicular_symbol=1.0)
            metric_logger.update(all_sym=1.0)
        for _ in range(incorrect_perpendicular_num):
            metric_logger.update(perpendicular_symbol=0.0)
            metric_logger.update(all_sym=0.0)