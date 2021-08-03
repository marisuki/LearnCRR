# import numpy as np
# import pyspark
import random
#import constraint_graph


equiv_bias = 1e-6
bias = 0.1


def reverse(op):
    if op == ">":
        return "<="
    elif op == ">=":
        return "<"
    elif op == "<":
        return ">="
    elif op == "<=":
        return ">"
    elif op == "=":
        return "!="
    elif op == "!=":
        return "="
    else:
        print("error op.")


def swap_op(op):
    if op == ">":
        return "<"
    elif op == ">=":
        return "<="
    elif op == "<":
        return ">"
    elif op == "<=":
        return ">="
    elif op == "=":
        return "="
    elif op == "!=":
        return "!="
    else:
        print("error op.")


def predicate_reverse(pred):
    return (pred[0], pred[1], reverse(pred[2]), pred[3])


def minus(val1, val2, op, val_type=1e-5):
    # val1 op val2
    x = val1 - val2
    if op == "=":
        return abs(x) < equiv_bias
    elif op == ">=":
        return x > -equiv_bias
    elif op == "<=":
        return x < equiv_bias
    elif op == ">":
        if val_type > 0.5:
            return x > 1.0 - equiv_bias
        else:
            return x > equiv_bias
    elif op == "<":
        if val_type > 0.5:
            return x < equiv_bias - 1.0
        else:
            return x < -equiv_bias
    elif op == "!=":
        if val_type > 0.5:
            return abs(x) > 1.0 - equiv_bias
        else:
            return abs(x) > equiv_bias
    else:
        return False


def dom_filter(dom, val, op, data_type):
    reconstruct = []
    for x in dom:
        if minus(x, val, op, data_type):
            reconstruct.append(x)
    return reconstruct


def read_from_dc_file(file_root):
    import os
    f = open(os.path.join(file_root, "dc.txt"), "r")
    dcs = []
    for line in f:
        line = line.split(",")
        for x in range(int(len(line)/4)):
            dcs.append((line[4*x+0], line[4*x+1], line[4*x+2], "", line[4*x+3]))
    return dcs


def tup_dist(t1: [float], t2: [float]):
    return sum([abs(t1[i] - t2[i]) for i in range(len(t1))])


def minimal_distance_projection(table, attrs: [int], origin: [float], mask_tuple: {int}):
    sampled = random.sample(range(len(table) - 1), min(1000, len(table) - 1))
    dis, candidate = 0x7f7f7f7f, -1
    for tid in sampled:
        if tid in mask_tuple: continue
        dist = sum([abs(origin[i] - table[tid][1][i]) for i in attrs])
        if dist < dis:
            dis = dist
            candidate = tid
    return table[candidate][1]


def dc_classify(dc):
    single, multiple = [], []
    for pred in dc:
        if pred[3] == 2 or pred[3] == 3:
            single.append(pred)
        elif pred[3] == 1:
            multiple.append(pred)
        else:
            # add other predicates here.
            print("DC predicate type without definition in [database.dc_classify]. Skip.")
    return single, multiple


def f_detect(pred_set, pass_by, true_set, x, data_precision):
    # print(x)
    status = True
    for pred in pred_set:
        if not status: break
        if pred[3] == 3:
            tup_i = x
            if (tup_i[0], pred[0]) in pass_by: continue
            if (tup_i[0], pred[0]) not in true_set:
                if not minus(tup_i[1][pred[0]], pred[1], pred[2], data_precision[pred[0]]):
                    status = False
            else:
                status = False
        elif pred[3] == 2:
            tup_i = x
            if (tup_i[0], pred[0]) in pass_by or (tup_i[0], pred[1]) in pass_by: continue
            if (tup_i[0], pred[0]) not in true_set and (tup_i[0], pred[1]) not in true_set:
                if not minus(tup_i[1][pred[0]], tup_i[1][pred[1]], pred[2],
                             min(data_precision[pred[0]], data_precision[pred[1]])):
                    status = False
            else:
                status = False
        elif pred[3] == 1:
            tup_i = x[0]
            tup_j = x[1]
            if (tup_i[0], pred[0]) in pass_by or (tup_j[0], pred[1]) in pass_by: continue
            if (tup_i[0], pred[0]) not in true_set and (tup_j[0], pred[1]) not in true_set:
                if not minus(tup_i[1][pred[0]], tup_j[1][pred[1]], pred[2], min(data_precision[pred[0]], data_precision[pred[1]])):
                    status = False
            else:
                status = False
        else:
            # add other predicates here.
            print("Predicate type without definition in viol detect [database.f_detect]. Skip.")
    return status


def exe_vio_accel_spark(table, shuffled_indexes_single, shuffled_indexes_multiple,  dc, forgive, pass_cells, data_precision, collect=True):
    # pyspark, return values
    # forgive: fresh variables: when considering: false; shine: true, pass (when asking repair context)
    single, multiple = dc_classify(dc)
    if pass_cells is None:
        pass_cells = set()
    data_s = shuffled_indexes_single
    if len(single) != 0:
        data_s = data_s.filter(lambda x: not f_detect(single, pass_cells, forgive, table[x], data_precision))
        data_se = set(data_s.collect())
    else:
        data_se = set()
    if len(multiple) != 0: 
        if data_se:
            data_m = shuffled_indexes_multiple.filter(lambda x: not (x[0] in data_se or x[1] in data_se)) # construct tuple pair
        else:
            data_m = shuffled_indexes_multiple
        data_m = data_m.filter(lambda x: x[0] != x[1])
        data_m = data_m.filter(lambda x: f_detect(multiple, pass_cells, forgive, [table[x[0]], table[x[1]]], data_precision))
        if collect:
            return data_m.collect()
        else:
            return data_m
        # data = data.map(lambda x: (x[0][0], x[1][0]))
    else:
        data_s = shuffled_indexes_single.filter(lambda x: x not in data_se)
        if collect:
            return data_s.collect()
        else:
            return data_s
        # data = data.map(lambda x: x[0])
    # data.map(lambda x: (x, dc)).foreach(print)
    #if collect and data_m: return data.collect()
    #return data


def set_prf_calc(truth: set, x: set):
    tp = truth.intersection(x)
    return len(tp)*1.0/len(x), len(tp)*1.0/len(truth)


def vairance_tp(table):
    import statistical
    import pyspark as spark
    sparkConf = spark.SparkConf()
    sparkConf.setMaster("local[*]")
    sparkContext = spark.SparkContext.getOrCreate(sparkConf)
    tp = sparkContext.parallelize(table).map(lambda x: x[1])
    aver = statistical.average(tp)
    vari = statistical.variance(tp, aver)
    return vari


class database:
    def __init__(self):
        self.table = []  # int: index, array(double)
        # self.table_broadcast = None
        self.value_map = {}  # (int: column --> (string: enumerate value or value: float -> int: value))
        # self.reverse_value = {} # (int: column, int: value) -> string: enumerate value
        self.dcs = []  # (int, int/double, string(op), type=1,2,3)
        # self.attr2plc_type = {} # (attr -> (plc, type))
        self.modify = {}  # (int:index, int:column) -> (float: ori, double : new value)
        self.modifyHistory = {}  # (int:index, int:column) -> (float: ori, double : new value)
        self.fv = []  # (int:index, int:column)
        # self.suspectSet = [] # [Array[Array[int: index]:tuple list]: all related tuple list given dc]
        # self.minimumDist = {} # int: index -> double: modify unit
        self.schema = {}  # attrStr -> int, type: Enumerate/Value
        self.data_precision = {}  # int: attrId -> float
        self.emptyCell = []  # tupleId, attrId
        self.error_change = []  # ((int, int):cell, origin, newVal)
        self.dom = {}  # attrId -> Enumerate: (min, max), Value: {val_i, ...}
        self.restricted_attr = {}
        # self.disjoint_cluster: constraint_graph.constraintGraph = None
    
    """
    def copy_with_partial_db(self, block_left: int, block_right: int):
        # print(partial_db_index)
        from copy import copy
        ans = database()
        tables = []
        if block_left == -1:
            block_left, block_right = 0, len(self.table)
        for tid in range(block_left, block_right):
            tables.append((tid-block_left, copy(self.table[tid][1])))
        ans.table = tables
        else:
            ans.table = self.table.copy()
        ans.value_map = self.value_map
        ans.dcs = self.dcs
        ans.schema = self.schema
        ans.data_precision = self.data_precision
        ans.emptyCell = self.emptyCell
        ans.dom = self.dom
        ans.disjoint_cluster = self.disjoint_cluster
        ans.error_change = self.error_change
        # ans.modifyHistory = self.modifyHistory
        # ans.modify = self.modify
        # ans.fv = self.fv
        return ans
    """
    def add_table(self, path: str, schema: [(str, str)], regex: str, first_line_omit: bool, max_index=-1):
        #self.disjoint_cluster = constraint_graph.constraintGraph(len(schema))
        db_input = open(path, "r")
        index, omit = 0, False
        for i in range(len(schema)):
            self.schema[schema[i][0]] = (i, schema[i][1])
            self.schema[i] = (schema[i][0], schema[i][1])
            if schema[i][1] == "Enumerate":
                self.dom[i] = (0x7f7f7f7f, -1)
            else:
                self.dom[i] = set()
            self.value_map[i] = {}
        for line in db_input:
            if max_index != -1 and index > max_index:
                break
            if first_line_omit and not omit:
                omit = not omit
                continue
            repack = []
            spl = line.split(regex)
            for column in range(0, len(spl)):
                if spl[column] == "":
                    self.emptyCell.append((index, column))
                    repack.append(-1)
                    continue
                if schema[column][1] == "Value":
                    repack.append(float(spl[column]))
                    ori = self.dom[column]
                    # self.dom[column] = (min(float(spl[column]), ori[0]), max(float(spl[column]), ori[1]))
                    self.dom[column] |= {float(spl[column])}
                elif schema[column][1] == "Enumerate":
                    if spl[column] not in self.value_map[column]:
                        self.value_map[column][spl[column]] = len(self.value_map[column])
                    repack.append(self.value_map[column][spl[column]])
                else:
                    print("error type.")
            self.table.append((index, repack))
            index += 1
        for i in range(len(schema)):
            if schema[i][1] == "Enumerate":
                self.dom[i] = (0, len(self.value_map[i]) - 1)

    def add_dc(self, dcs: [[(str, str, str, str, str)]]):  # t1, attr, op, t2(t1), attr/ t1, attr, op, <emp>, val
        tot_attr = []
        for dc in dcs:
            dci = []
            single = True
            attr = []
            for pred in dc:
                if pred[3] == "": # constant: type 3
                    dci.append((self.schema[pred[1]][0], float(pred[4]), pred[2], 3))
                    attr.append(self.schema[pred[1]][0])
                elif pred[0] == pred[3]: # single tuple: type 2
                    dci.append((self.schema[pred[1]][0], self.schema[pred[4]][0], pred[2], 2))
                    attr += [self.schema[pred[1]][0], self.schema[pred[4]][0]]
                else: 
                    dci.append((self.schema[pred[1]][0], self.schema[pred[4]][0], pred[2], 1))
                    attr += [self.schema[pred[1]][0], self.schema[pred[4]][0]]
                    single = False
            self.disjoint_cluster.add_connection(list(set(attr)))
            self.dcs.append((single, dci))
            tot_attr += attr
        self.restricted_attr = set(tot_attr)

    def value_type(self, type_map: [(str, str)]):
        for (attr, typ) in type_map:
            print(attr, typ)
            if self.schema[attr][1] == "Value":
                if typ == "Int":
                    self.data_precision[self.schema[attr][0]] = 1.0
                elif typ == "Float":
                    self.data_precision[self.schema[attr][0]] = 1e-1
                else:
                    print("[Database.value_type] Value type error.")
        for k in self.schema.keys():
            if self.schema[k][0] not in self.data_precision:
                self.data_precision[self.schema[k][0]] = 1.0

    def add_error(self, error_rate=0.0, related=True, pointed_attrs=None):
        if error_rate < equiv_bias: return
        if related: rel_attr = self.restricted_attr
        else: rel_attr: set = {i for i in range(int(len(self.schema)/2))}
        if pointed_attrs: rel_attr = set(pointed_attrs)
        if related and not rel_attr:
            print("[Database.add_error] Related attrs is empty, pls add constraints first if consider an addition of related errors. ADDING ERRORS Skipped.")
            return
        row_bound = 1.0 - pow(1.0 - error_rate, len(rel_attr))
        cell_bound = error_rate / row_bound
        changed = []
        for tupId in range(len(self.table)):
            tup = self.table[tupId]
            reconstruct = tup[1]
            if random.random() < row_bound:
                for i in rel_attr:
                    if random.random() < cell_bound:
                        if (tupId, i) in self.emptyCell: continue
                        if self.schema[i][1] == "Enumerate":
                            if type(self.dom[i]) == type(()) and self.dom[i][1] == -1: continue
                            new_value = random.randint(self.dom[i][0], self.dom[i][1])
                            if abs(new_value - tup[1][i]) > equiv_bias:
                                changed.append(((tup[0], i), tup[1][i], new_value))
                                reconstruct[i] = new_value
                        else:
                            new_value = random.random() * (max(self.dom[i]) - min(self.dom[i])) + min(self.dom[i])
                            if abs(new_value - tup[1][i]) > equiv_bias:
                                changed.append(((tup[0], i), tup[1][i], new_value))
                                reconstruct[i] = new_value
            self.table[tupId] = (tup[0], reconstruct)
        self.error_change = changed
    
    def load_error(modifications): #modifications: cell->(ori, tar)
        self.modifyHistory = modifications
        for cell in modifications:
            self.table[cell[0]][1][cell[1]] = modifications[cell][1]
    
    def rule_based_error(self, error_rate=0.0, related=True, pointed_attrs=None, fds=[]):
        #bart with /run.sh and jdk, depend on ant and postgresql.
        return

    def evaluate(self, average=None, variance=None): # continuous requires non-None aver and variance
        if len(self.modify) != 0:
            self.persist()
        if len(self.error_change) == 0:
            return -1, -1, -1, 0
        discrete_prec, discrete_prec_all, discrete_rec = 0.0, 0.0, 0.0
        dis_tr, dis_nr, dis_tn = 0.0, 0.0, 0.0
        # self.error_change, self.modifyHistory
        changed_cells = {rec[0] for rec in self.error_change}
        for record in self.error_change:  # cell, ori, new
            if self.schema[record[0][1]][1] == "Enumerate" and record[0] in self.modifyHistory:
                if int(record[1]) == int(self.modifyHistory[record[0]][1]):
                    discrete_prec += 1
                # discrete_prec_all += 1
            if self.schema[record[0][1]][1] == "Enumerate":
                discrete_rec += 1
        discrete_prec_all = 0
        for k in self.modifyHistory:
            if abs(self.modifyHistory[k][0] - self.modifyHistory[k][1]) > 1e-3:
                discrete_prec_all += 1
        all_related: dict = {}  # count all value typed modifications, cell -> ([ori, noi, rep], type:-1->fresh)
        for record in self.error_change:
            all_related[record[0]] = [[record[1], record[2], record[2]], 1]
        for cell in self.modifyHistory:
            if self.schema[cell[1]][1] != "Value":
                continue
            if cell in all_related:
                all_related[cell][0][2] = self.modifyHistory[cell][1]
                all_related[cell][1] = 2
            else:
                all_related[cell] = [
                [self.modifyHistory[cell][0], self.modifyHistory[cell][0], self.modifyHistory[cell][1]], 0]
        value_counter = 0
        for cell in self.fv:
            if self.schema[cell[1]][1] == "Value":
                value_counter += 1
                if cell in all_related:
                    all_related[cell][0][2] = -1
                    all_related[cell][1] = -1
                else:
                    all_related[cell] = [[self.table[cell[0]][1][cell[1]], self.table[cell[0]][1][cell[1]], -1], -1]
            else:
                discrete_prec_all += 1
                if cell in changed_cells:
                    discrete_prec += 0.5
        # discrete_prec += (len(self.fv) - value_counter) * 0.5
        # discrete_prec_all += (len(self.fv) - value_counter) * 1.0
        # arr[xxxxx]
        if not variance: variance = vairance_tp(self.table)
        # vairance = abs(variance)
        for cell in all_related:
            if all_related[cell][1] == -1:
                #dis = abs(self.fv_distance(cell[1], all_related[cell][0][0]))/variance[cell[1]]
                dis_tr += abs(self.fv_distance(cell[1], all_related[cell][0][0]))/variance[cell[1]]
                dis_nr += abs(self.fv_distance(cell[1], all_related[cell][0][1]))/variance[cell[1]]
                dis_tn += abs(all_related[cell][0][0]-all_related[cell][0][1])/variance[cell[1]]
                #dis_tn += dis
            else:
                dis_tr += abs(all_related[cell][0][0] - all_related[cell][0][-1])/variance[cell[1]]
                dis_nr += abs(all_related[cell][0][1] - all_related[cell][0][-1])/variance[cell[1]]
                dis_tn += abs(all_related[cell][0][0] - all_related[cell][0][1])/variance[cell[1]]
        if discrete_prec_all < 1e-5: discrete_prec_all += 1e-5
        if discrete_rec < 1e-5: discrete_rec += 1e-5
        if dis_nr + dis_tn < 1e-5: dis_nr += 1e-5
        precision = discrete_prec * 1.0 / (discrete_prec_all) * 1.0
        recall = discrete_prec * 1.0 / (discrete_rec) * 1.0
        if precision + recall > 1e-5: f_score = 2 * precision * recall / (precision + recall)
        else: f_score = -1
        mnad = dis_tr
        #accuracy = len(set([i[0] for i in self.error_change]).intersection(set(self.modifyHistory.keys())))*1.0
        #accuracy += len(set([i[0] for i in self.error_change]).intersection(set(self.fv)))*0.5
        #accuracy /= (len(self.fv) + len(self.modifyHistory))
        accuracy = 1.0 - dis_tr/(dis_nr+dis_tn)
        return precision, recall, f_score, dis_tr, mnad, accuracy, len(self.modifyHistory), len(self.fv)

    def persist(self):
        for k in self.modify.keys():
            if k in self.modifyHistory:
                self.modifyHistory[k] = (self.modifyHistory[k][0], self.modify[k][1])
            else:
                self.modifyHistory[k] = self.modify[k]
            self.table[k[0]][1][k[1]] = self.modify[k][1]
        self.modify.clear()

    def merge_repair(self, repair: {(int, int): (float, float)}, fresh_variable: [(int, int)]):
        if fresh_variable is None:
            fresh_variable = []
        self.fv += fresh_variable
        self.fv = set(self.fv)
        for k in repair:
            if k in self.fv:
                self.fv.remove(k)
            if repair[k][0] != repair[k][1]:
                self.modify[k] = repair[k]
        self.fv = list(self.fv)

    def weight(self, cell: (int, int)):
        if self.schema[cell[1]][1] == "Enumerate":
            return 1.0, -1
        else:
            return max(abs(min(self.dom[cell[1]]) - self.table[cell[0]][1][cell[1]])
                       , abs(max(self.dom[cell[1]]) - self.table[cell[0]][1][cell[1]])), -1

    def fv_distance(self, column, compare_val):
        return max(abs(compare_val - min(self.dom[column])), abs(compare_val - max(self.dom[column])))

    @staticmethod
    def f_detect(pred_set, pass_by, true_set, x, data_precision):
        # print(x)
        status = True
        for pred in pred_set:
            if not status: break
            if pred[3] == 3:
                tup_i = x
                if (tup_i[0], pred[0]) in pass_by: continue
                if (tup_i[0], pred[0]) not in true_set:
                    if not minus(tup_i[1][pred[0]], pred[1], pred[2], data_precision[pred[0]]):
                        status = False
                else:
                    status = False
            elif pred[3] == 2:
                tup_i = x
                if (tup_i[0], pred[0]) in pass_by or (tup_i[0], pred[1]) in pass_by: continue
                if (tup_i[0], pred[0]) not in true_set and (tup_i[0], pred[1]) not in true_set:
                    if not minus(tup_i[1][pred[0]], tup_i[1][pred[1]], pred[2],
                                min(data_precision[pred[0]], data_precision[pred[1]])):
                        status = False
                else:
                    status = False
            elif pred[3] == 1:
                tup_i = x[0]
                tup_j = x[1]
                if (tup_i[0], pred[0]) in pass_by or (tup_j[0], pred[1]) in pass_by: continue
                if (tup_i[0], pred[0]) not in true_set and (tup_j[0], pred[1]) not in true_set:
                    if not minus(tup_i[1][pred[0]], tup_j[1][pred[1]], pred[2], min(data_precision[pred[0]], data_precision[pred[1]])):
                        status = False
                else:
                    status = False
            else:
                print("error.")
        return status


    def exe_vio(self, dc, forgive, shine):
        # return indexes
        if len(dc) == 0: return []
        single, multiple = dc_classify(dc)
        viol = []
        if len(single) != 0:
            for tup in self.table:
                if f_detect(single, forgive, shine, tup, self.data_precision):
                    viol.append(tup)
        else:
            viol = self.table
        if len(multiple) != 0:
            tmp = []
            for tup1 in viol:
                for tup2 in viol:
                    if f_detect(multiple, forgive, shine, (tup1, tup2), self.data_precision):
                        tmp.append((tup1[0], tup2[0]))
            viol = tmp
        else:
            tmp = []
            for tup in viol:
                tmp.append(tup[0])
            viol = tmp
        return viol

    def persist_file(self, file_name: str, proj=None):
        f = open(file_name + ".csv", "w")
        if len(self.modify) != 0:
            self.persist()
        empty_cells = set(self.emptyCell+self.fv)
        reverse_map = {}
        for col in self.value_map.keys():
            for str_val in self.value_map[col]:
                reverse_map[(col, self.value_map[col][str_val])] = str_val
        for tup in self.table:
            ans = ""
            for col in range(len(tup[1])):
                if proj and col not in proj: continue
                if self.schema[col][1] == "Value":
                    if (tup[0], col) in empty_cells:
                        ans += ","
                        continue
                    if self.data_precision[col] > 0.5:
                        ans += str(int(tup[1][col])) + ","
                    else:
                        ans += str(tup[1][col]) + ","
                else:
                    if (tup[0], col) in empty_cells:
                        ans += ","
                        continue
                    ans += reverse_map[(col, tup[1][col])] + ","
            ans = ans[:-1]
            if ans[-1] != "\n": ans = ans + "\n"
            f.write(ans)
        f.close()
    
    def export_proj(self, file_name, proj_attrs):
        self.persist_file(file_name, proj_attrs)


