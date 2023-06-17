# coding:utf-8

def evaluate_rel(all_golden_geo2geo, all_golden_sym2geo_text, all_golden_sym2geo,
                 all_pred_geo2geo, all_pred_sym2geo_text, all_pred_sym2geo,
                 metric_logger):
    
    for image_id in all_golden_geo2geo.keys():
        golden_geo2geo = all_golden_geo2geo[image_id]
        golden_sym2geo_text = all_golden_sym2geo_text[image_id]
        golden_sym2geo = all_golden_sym2geo[image_id]
        pred_geo2geo = all_pred_geo2geo[image_id]
        pred_sym2geo_text = all_pred_sym2geo_text[image_id]
        pred_sym2geo = all_pred_sym2geo[image_id]
        
        # endpoint
        update_metric(golden_geo2geo, pred_geo2geo, "endpoint", metric_logger)
        # online
        update_metric(golden_geo2geo, pred_geo2geo, "online", metric_logger)
        # oncircle
        update_metric(golden_geo2geo, pred_geo2geo, "oncircle", metric_logger)
        # center
        update_metric(golden_geo2geo, pred_geo2geo, "center", metric_logger)
        
        # text-point
        update_metric(golden_sym2geo_text, pred_sym2geo_text, "point", metric_logger, force_name="text_symbol")
        # text-angle
        update_metric(golden_sym2geo_text, pred_sym2geo_text, "angle", metric_logger, force_name="text_symbol")
        # text-len
        update_metric(golden_sym2geo_text, pred_sym2geo_text, "len", metric_logger, force_name="text_symbol")
        # text-degree
        update_metric(golden_sym2geo_text, pred_sym2geo_text, "degree", metric_logger, force_name="text_symbol")

        # angle
        for gold_key, gold_val in golden_sym2geo.items():
            flag = False
            if "angle" in gold_key:
                for pred_key, pred_val in pred_sym2geo.items():
                    if "angle" in pred_key:
                        if gold_val == pred_val:
                            metric_logger.update(congruent_angle=1.0)
                            pred_sym2geo.pop(pred_key)
                            flag = True
                            break
                if not flag: metric_logger.update(congruent_angle=0.0)
            elif "bar" in gold_key:
                for pred_key, pred_val in pred_sym2geo.items():
                    if "bar" in pred_key:
                        if gold_val == pred_val:
                            metric_logger.update(congruent_bar=1.0)
                            pred_sym2geo.pop(pred_key)
                            flag = True
                            break 
                if not flag: metric_logger.update(congruent_bar=0.0)
            elif "parallel" in gold_key:
                for pred_key, pred_val in pred_sym2geo.items():
                    if "parallel" in pred_key:
                        if gold_val == pred_val:
                            metric_logger.update(parallel=1.0)
                            pred_sym2geo.pop(pred_key)
                            flag = True
                            break 
                if not flag: metric_logger.update(parallel=0.0)
            elif gold_key == "perpendicular":
                for pred_key, pred_val in pred_sym2geo.items():
                    if pred_key == "perpendicular":
                        if gold_val == pred_val:
                            metric_logger.update(perpendicular=1.0)
                            flag = True
                            break
                if not flag: metric_logger.update(perpendicular=0.0)
    
    return metric_logger

def update_metric(gold, pred, name, metric_logger, force_name=None):
    # geo2geo
    correct = min(gold[name], pred[name])
    incorrect = abs(gold[name] - pred[name])
    for _ in range(correct):
        if force_name:
            metric_logger.update(text_symbol=1.0)
        else:
            if name == "endpoint":
                metric_logger.update(endpoint=1.0)
            elif name == "online":
                metric_logger.update(online=1.0)
            elif name == "oncircle":
                metric_logger.update(oncircle=1.0)
            elif name == "center":
                metric_logger.update(center=1.0)
    for _ in range(incorrect):
        if force_name:
            metric_logger.update(text_symbol=0.0)
        else:
            if name == "endpoint":
                metric_logger.update(endpoint=0.0)
            elif name == "online":
                metric_logger.update(online=0.0)
            elif name == "oncircle":
                metric_logger.update(oncircle=0.0)
            elif name == "center":
                metric_logger.update(center=0.0)
    