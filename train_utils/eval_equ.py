from inspect import getmembers, isfunction
import itertools
import math


import math


def g_equal(n1):  # 0
    return n1


def g_double(n1):  # 1
    return n1*2


def g_half(n1):  # 2
    return n1/2


def g_add(n1, n2):  # 3
    return n1 + n2


def g_minus(n1, n2):  # 4
    return math.fabs(n1 - n2)


def g_sin(n1):  # 5
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.sin(n1/180*math.pi)
    return False


def g_cos(n1):  # 6
    if n1 % 15 == 0 and 0 <= n1 <= 180:
        return math.cos(n1/180*math.pi)
    return False


def g_tan(n1):  # 7
    if n1 % 15 == 0 and 5 <= n1 <= 85:
        return math.tan(n1/180*math.pi)
    return False


def g_asin(n1):  # 8
    if -1 < n1 < 1:
        n1 = math.asin(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def g_acos(n1):  # 9
    if -1 < n1 < 1:
        n1 = math.acos(n1)
        n1 = math.degrees(n1)
        return n1
    return False


def gougu_add(n1, n2):  # 13
    return math.sqrt(n1*n1+n2*n2)


def gougu_minus(n1, n2):  # 14
    if n1 != n2:
        return math.sqrt(math.fabs(n1*n1-n2*n2))
    return False


def g_bili(n1, n2, n3):  # 16
    if n1 > 0 and n2 > 0 and n3 > 0:
        return n1/n2*n3
    else:
        return False


def g_mul(n1, n2):  # 17
    return n1*n2


def g_divide(n1, n2):  # 18
    if n1 > 0 and n2 > 0:
        return n1/n2
    return False


def cal_circle_area(n1):  # 19
    return n1*n1*math.pi


def cal_circle_perimeter(n1):  # 20
    return 2*math.pi*n1


def cal_cone(n1, n2):  # 21
    return n1*n2*math.pi

constant = [30, 60, 90, 180, 360, math.pi, 0.618]
op_dict = {0: g_equal, 1: g_double, 2: g_half, 3: g_add, 4: g_minus,
          5: g_sin, 6: g_cos, 7: g_tan, 8: g_asin, 9: g_acos,
          10: gougu_add, 11: gougu_minus, 12: g_bili,
          13: g_mul, 14: g_divide, 15: cal_circle_area, 16: cal_circle_perimeter, 17: cal_cone}
op_list = [op_dict[key] for key in sorted(op_dict.keys())]


class Equations:
    def __init__(self):

        self.op_list = op_list
        self.op_num = {}
        self.call_op = {}
        self.exp_info = None
        self.results = []
        self.max_step = 3
        self.max_len = 7
        for op in self.op_list:
            self.call_op[op.__name__] = op
            # self.call_op[op] = eval(op)
            self.op_num[op.__name__] = self.call_op[op.__name__].__code__.co_argcount

    def str2exp(self, inputs):
        inputs = inputs.split(',')
        exp = inputs.copy()
        for i, s in enumerate(inputs):
            if 'n' in s or 'v' in s or 'c' in s:
                exp[i] = s.replace('n', 'N_').replace('v', 'V_').replace('c', 'C_')
            else:
                exp[i] = op_dict[int(s[2:])]
            exp[i] = exp[i].strip()

        self.exp = exp
        return exp

    def excuate_equation(self, exp, source_nums=None):
        # exp:  ['g_minus', 'C_3', 'N_0', 'g_minus', 'V_0']
        # source_nums:  [56.0]
        # print("exp: ", exp)
        # print("source_nums: ", source_nums)
        # print("self.op_list: ", self.op_list)
        if source_nums is None:
            source_nums = self.exp_info['nums']
        vars = []
        idx = 0
        while idx < len(exp):
            op = exp[idx]
            # if op not in self.op_list:
            if op not in self.call_op:
                # print("op not in: ", op)
                return None
            op_nums = self.op_num[op]
            if idx + op_nums >= len(exp):
                # print("idx: ", idx)
                # print("op_nums: ", op_nums)
                # print("op: ", op)
                return None
            excuate_nums = []
            for tmp in exp[idx + 1: idx + 1 + op_nums]:
                if tmp[0] == 'N' and int(tmp[-1]) < len(source_nums):
                    excuate_nums.append(source_nums[int(tmp[-1])])
                elif tmp[0] == 'V' and int(tmp[-1]) < len(vars):
                    excuate_nums.append(vars[int(tmp[-1])])
                elif tmp[0] == 'C' and int(tmp[-1]) < len(constant):
                    excuate_nums.append(constant[int(tmp[-1])])
                else:
                    # print("Here: ", tmp[0])
                    return None
            idx += op_nums + 1
            v = self.call_op[op](*excuate_nums)
            if v is None:
                # print("excuate_nums: ", excuate_nums)
                # print("op: ", op)
                return None
            vars.append(v)
        return vars
