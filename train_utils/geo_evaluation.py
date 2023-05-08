# coding:utf-8

import os
import math
import json
import codecs

from .eval_equ import Equations

_equ = Equations()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count

class GeoEvaluation:
    
    def __init__(self):
        self.calculation_acc = AverageMeter()
        self.calculation_no_result = AverageMeter()
        self.proving_acc = AverageMeter()
        self.proving_no_result = AverageMeter()

        self.cal_angle = AverageMeter()
        self.cal_length = AverageMeter()
        self.cal_other = AverageMeter()

        self.prove_parallel = AverageMeter()
        self.prove_triangle = AverageMeter()
        self.prove_quadrilateral = AverageMeter()
        self.prove_congruent = AverageMeter()
        self.prove_similarity = AverageMeter()
        
        self.cal_wrong_predictions = {}
        self.pro_wrong_predictions = {}

    def geo_evaluation(self, num_beam, batch_data, images_id, target, pred,
                        source_nums, choice_nums, label, problem_form, problem_type,
                        metric_logger):

        batch_size = len(source_nums)

        for b in range(batch_size):
            img_id = images_id[b]
            if problem_form[b] == 'calculation':
                choice = self.evaluate_calculation(pred[b*num_beam:(b+1)*num_beam], choice_nums[b], source_nums[b], target[b], num_beam, label[b])
                if choice is None:
                    assert img_id not in self.cal_wrong_predictions
                    self.cal_wrong_predictions[img_id] = {
                        "problems_types": batch_data["problems_types"][b],
                        "problem": batch_data["problem"][b],
                        "golden_program": batch_data["program"][b],
                        "predict_program": pred[b*num_beam:(b+1)*num_beam],
                        "numbers": batch_data["numbers"][b],
                        "choice_numbers": batch_data["choice_numbers"][b],
                        "label": batch_data["label"][b],
                    }
                    metric_logger.update(cal_acc=0.0)
                    metric_logger.update(cal_no_res=1.0)
                    self.calculation_acc.update(0.0)
                    self.calculation_no_result.update(1.0)
                elif choice == label[b]:
                    metric_logger.update(cal_acc=1.0)
                    metric_logger.update(cal_no_res=0.0)
                    self.calculation_acc.update(1.0)
                    self.calculation_no_result.update(0.0)
                else:
                    assert img_id not in self.cal_wrong_predictions
                    self.cal_wrong_predictions[img_id] = {
                        "problems_types": batch_data["problems_types"][b],
                        "problem": batch_data["problem"][b],
                        "golden_program": batch_data["program"][b],
                        "predict_program": pred[b*num_beam:(b+1)*num_beam],
                        "numbers": batch_data["numbers"][b],
                        "choice_numbers": batch_data["choice_numbers"][b],
                        "label": batch_data["label"][b],
                    }

                flag = 1.0 if choice == label[b] else 0
                if problem_type[b] == 'angle':
                    metric_logger.update(angle=flag)
                    self.cal_angle.update(flag)
                elif problem_type[b] == 'length':
                    metric_logger.update(length=flag)
                    self.cal_length.update(flag)
                else:
                    metric_logger.update(other=flag)
                    self.cal_other.update(flag)

            else:
                assert problem_form[b] == 'proving'
                success = self.evaluate_proving(pred[b*num_beam:(b+1)*num_beam], target[b], num_beam)
                if success is None:
                    assert img_id not in self.pro_wrong_predictions
                    self.pro_wrong_predictions[img_id] = {
                        "problems_types": batch_data[b]["problems_types"],
                        "problem": batch_data[b]["problem"],
                        "golden_program": batch_data[b]["program"],
                        "predict_program": pred[b*num_beam:(b+1)*num_beam],
                    }
                    metric_logger.update(pro_acc=0.0)
                    metric_logger.update(pro_no_res=1.0)
                    self.proving_acc.update(0)
                    self.proving_no_result.update(1.0)
                else:
                    metric_logger.update(pro_acc=1.0)
                    metric_logger.update(pro_no_res=0.0)   
                    self.proving_acc.update(1.0)
                    self.proving_no_result.update(0)

                flag = 0 if success is None else 1.0
                if problem_type[b] == 'parallel':
                    metric_logger.update(parallel=flag)
                    self.prove_parallel.update(flag)
                elif problem_type[b] == 'triangle':
                    metric_logger.update(triangle=flag)
                    self.prove_triangle.update(flag)
                elif problem_type[b] == 'quadrilateral':
                    metric_logger.update(quadrilateral=flag)
                    self.prove_quadrilateral.update(flag)
                elif problem_type[b] == 'congruent':
                    metric_logger.update(congruent=flag)
                    self.prove_congruent.update(flag)
                elif problem_type[b] == 'similarity':
                    self.prove_congruent.update(similarity=flag)
                    self.prove_similarity.update(flag)
                else:
                    assert problem_type[b] == 'proportions'
                    # The proportion problems are also related to triangle
                    metric_logger.update(triangle=flag)
                    self.prove_triangle.update(flag)

    def save(self, save_dir, epoch=None):
        if len(self.cal_wrong_predictions) != 0:
            file_name = "wrong_cal_predictions.json" if epoch == None else f"wrong_{epoch}_cal_predictions.json"
            with codecs.open(os.path.join(save_dir, "wrong_cal_predictions.json"), "w", "utf-8") as file:
                json.dump(self.cal_wrong_predictions, file, indent=2)
                
        if len(self.pro_wrong_predictions) != 0:
            file_name = "wrong_pro_predictions.json" if epoch == None else f"wrong_{epoch}_pro_predictions.json"
            with codecs.open(os.path.join(save_dir, "wrong_pro_predictions.json"), "w", "utf-8") as file:
                json.dump(self.pro_wrong_predictions, file, indent=2)
                
    def evaluate_calculation(self, top_k_predictions, choice_nums, source_nums, target, num_beam, label):
        choice = None
        for i in range(num_beam):
            if choice is not None:
                break
            hypo = top_k_predictions[i].split()
            try:
                res = _equ.excuate_equation(hypo, source_nums)
                # print("res: ", res)
                # print("choice_nums: ", choice_nums)
                # print()
            except:
                res = None

            if res is not None and len(res) > 0:

                for j in range(4):
                    if choice_nums[j] is not None and math.fabs(res[-1] - choice_nums[j]) < 0.001:
                        if j == label:
                            choice = j
                            break
                        else:
                            continue

        return choice

    def evaluate_proving(self, top_k_predictions, target, num_beam):
        success = None
        target = target.split()
        for i in range(num_beam):
            if success is not None:
                break
            hypo = top_k_predictions[i].split()

            if hypo == target:
                success = True

        return success