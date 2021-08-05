import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from core import database
import numpy as np
import time
from copy import copy
import random
from sklearn.linear_model import BayesianRidge
import sklearn
from core import sklearn_cnd
from sklearn.metrics import mean_squared_error
mse = mean_squared_error
    

# dic: {attr -> [a1, a2, flg=0, 1, 2, 3]} &1, 2\
# initiate: [dom-, dom+]
def intersection2d(di, d2d):
    for dj, A in d2d:
        if intersection(di, dj):
            return True
    return False

def intersection(di, dj):
    for attr in di:
        if not attr_intersection(di, dj, attr): return False
    return True

def attr_intersection(di, dj, attr):
    if di[attr][0] < dj[attr][0] and dj[attr][0] < di[attr][1]: return True
    if di[attr][0] > dj[attr][0] and di[attr][0] < dj[attr][1]: return True
    return False

def attr_intersect(di, dj, attr):
    if di[attr][2] & 1:
        if dj[attr][2] & 2:
            if di[attr][2] & 2 and dj[attr][2] & 1:
                if di[attr][1] >= dj[attr][0]: return True
                else: return False
            elif di[attr][0] <= dj[attr][1]: return True
        if di[attr][2] & 2:
            if dj[attr][2] & 1 == 0 and dj[attr][2] & 2 == 0: return True
            if dj[attr][2] & 1 and dj[attr][0] <= di[attr][1]: return True
            if dj[attr][2] & 2 and dj[attr][1] >= di[attr][0]: return True
        else:
            if dj[attr][2] & 2 == 0: return True
            if dj[attr][2] & 2 and di[attr][0] <= dj[attr][1]: return True
    else:
        if di[attr][2] & 2 == 0: return True
        if dj[attr][2] & 1 and di[attr][1] >= dj[attr][0]: return True
        if dj[attr][2] & 1 == 0: return True
    return False

def AND2d(di, dj):
    ans = dict()
    for attr in di:
        ans[attr] = [0, 0, 3]
        ans[attr][0] = max(dj[attr][0], di[attr][0])
        ans[attr][1] = min(dj[attr][1], di[attr][1])
        ans[attr][2] = 3
    return ans

def AND1d(d, attr, rang):
    ans = {x: [d[x][0], d[x][1], d[x][2]] for x in d}
    ans[attr][0] = max(ans[attr][0], rang[0])
    ans[attr][1] = min(ans[attr][1], rang[1])
    return ans

def examine_valuation(x, attr, op, const, data_precision):
    #print(x, attr, op, const, data_precision)
    return database.minus(x[attr], const, op, data_precision[attr])

def cond_filter(tb, cond, instance, data_precision):
    ans = []
    for i in instance:
        flag = True
        for attr in cond:
            if not examine_valuation(tb[i], attr, ">", cond[attr][0], data_precision):
                flag = False
            if not examine_valuation(tb[i], attr, "<=", cond[attr][1], data_precision):
                flag = False
            if not flag: break
        if flag: ans.append(i)
    return ans

def pred_filter(tb, attr, rang, instance):
    ans = []
    for i in instance:
        if not examine_valuation(tb[i], attr, ">", rang[0], db.data_precision):
            continue
        if not examine_valuation(tb[i], attr, "<=", rang[1], db.data_precision):
            continue
        ans.append(i)
    return ans

def mid(tb, instance, attr):
    domainB = list(set([tb[i][attr] for i in instance]))
    domainB = sorted(domainB)
    if len(domainB) <= 1: return 1, -1
    if len(domainB) == 2:
        return 2, domainB[0]
    return len(domainB), domainB[int(len(domainB)/2)]


def regress(reg, X_train, y_train, X_test, y_test, test_func, rhoA):
    ans = None
    if len(y_train) == 0: return ans, -1, -1, -1
    reg.fit(X_train, y_train)
    flg, rmse, pred = test_func(reg, X_test, y_test, rhoA)
    # print(flg,rmse)
    return copy(reg), rmse, flg, pred


def test(reg, X_test, y_test, rho):
    pred = reg.predict(X_test)
    rmse = mse(y_test, pred)
    if np.max(np.abs(y_test-pred)) > rho:
        flg = False
    else: flg = True
    return flg, rmse, pred


def dependence_sel(tb, IC, attrs, functionals, target, rho, fraction=0.9):
    # find best func, indep for target under IC
    # ans: reg, rmse, flg, pred, y_train, src
    ans = (None, 0x3f3f3f3f, False, None, None, -1)
    for func_name, params in functionals:
        params['func_name'] = func_name
        X_train, y_train, x_train = sklearn_cnd.generate(attrs, tb, IC, target, params=params)
        for i in range(len(X_train)):
            reg = sklearn_cnd.init(func_name, y_train, params)
            train_ic = np.random.choice(np.array(len(IC)), size=(int(len(IC)*fraction),), replace=False)
            test_ic = np.array(list(filter(lambda x: x not in train_ic, range(len(IC)))))
            pred, rmse, flg, pred = regress(reg, X_train[i][train_ic], y_train[train_ic], X_train[i][test_ic], y_train[test_ic], test, rho[target])
            if rmse < ans[1]: 
                ans = (reg, rmse, flg, pred, y_train[test_ic], i)
    return ans


def fusion_test(tb, IC, attrs, reg, src, target, rho):
    X_train, y_train, x_train = sklearn_cnd.generate(attrs, tb, IC, target, params=params)
    flg, rmse, pred = test(reg, X_train[src], y_train, rho[target])
    return flg, rmse


def discrete_separation(queue, schema):
    for attr in schema:
        tmp = []
        if schema[attr] == "Enumerate":
            for C, sep, avail_ans in queue:
                for val in range(C[attr][0]+1, C[attr][1]):
                    Ci = {a: C[a] for a in C}
                    Ci[attr] = [val-1, val, 3]
                    tmp.append((Ci, sep, None))
            queue = tmp
    return queue


def separation(tb, schema, k, functionals, dom, rho, max_P, max_p, targets, data_precision, 
                 sep_attr=None, force=False, init_condition=None):
    """
    data space separation.
    tb: data table, [[val0, ...valn],...]
    schema: attrID -> type e.g., 1 -> 'Enumerate', 0 -> 'Value'
    k: lower bound of sz of instance of each functionals, |k| = |functionals|, k=[12, 23,...]
    functionals: [("func_name" e.g., 'linear', 'bayesian',..., params:{e.g., 'n_order': 7}), ...]
    rho: validation bias for regression
    max_P: maximum number of separation in total: for continuous
    max_p: maximum number of separation for each attr: for continuous e.g., date, la, lo, year
    targets: target attribute in regression model
    sep_attr: None for all attrs in schema, can be pointed: e.g., [1, 2] (attrID, means: date, lo e.g.)
    force: default False, force to use the whole dataset
    init_condition: condition for init, prior knowledge to prune conditions.
    return CRRs: [(cnd, reg, targetA, src)], rmse, tot_time.
    """
    rules, rules_cnd = [], []
    st_tot = time.time()
    tot_y, tot_pred = np.array([]), np.array([])
    for A in targets:
        print("Regress on " + str(A))
        attrs_excA = list(filter(lambda x: x!=A, list(schema.keys())))
        queue = []
        inst = range(len(tb))
        if not init_condition: C_init = {k: [dom[k][0]-1, dom[k][1], 3] for k in dom}
        else: C_init = init_condition
        # test whole regress
        reg, rmse, flg, pred, y, src_id = dependence_sel(tb, inst, schema.keys(), functionals, A, rho)
        print("Overall: " + str(rmse) + ", Target: " + str(A) + ", Flag: " + str(flg) + ", SZ= " + str(len(y)))
        if flg or force or len(tb) < 2*min(k) or max_P <= 0 or max_p <= 0:
            rules.append((C_init, reg, A, src_id))
            rules_cnd.append((C_init, A))
            tot_y = np.append(tot_y, y)
            tot_pred = np.append(tot_pred, pred)
            continue
        else:
            queue.append((C_init, {i: 0 for i in schema}, (reg, rmse, flg, pred, y, src_id)))
        queue = discrete_separation(queue, schema)
        while queue:
            queue = list(filter(lambda cc: not intersection2d(cc[0], rules_cnd), queue))
            cond, sep_map, avail_ans = queue[0]
            queue = queue[1:]
            IC = cond_filter(tb, cond, inst, data_precision=data_precision)
            if len(IC) == 0: continue
            opt_sep = (-1, 0x3f3f3f3f, [])
            for B in sep_attr:
                dsz, b = mid(tb, IC, B)
                candidate = [AND1d(cond, B, [cond[B][0], b, 3]), AND1d(cond, B, [b, cond[B][1], 3])]
                candidate = [(condx, cond_filter(tb, condx, inst, data_precision)) for condx in candidate]
                candidate = list(filter(lambda x: len(x[1])!=0, candidate))
                if min([len(candx[1]) for candx in candidate]) < min(k): continue
                for condx, ICi in candidate:
                    # funs with size limit.
                    restricted_funcs = list(filter(lambda x: x[0] <= len(ICi), list(zip(k, functionals))))
                    restricted_funcs = [x[1] for x in restricted_funcs]
                    ansx = dependence_sel(tb, ICi, schema.keys(), restricted_funcs, A, rho)
                    if opt_sep[1] > ansx[1]: 
                        if opt_sep[0] != B:
                            opt_sep = (B, ansx[1], [(condx, ansx)])
                    if opt_sep[0] == B:
                        opt_sep = (B, min(opt_sep[1], ansx[1]), opt_sep[2] + [(condx, ansx)])
            if opt_sep[0] == -1:
                # fail to make separation
                if avail_ans:
                    rules.append((cond, avail_ans[0], A, avail_ans[5]))
                    rules_cnd.append((cond, A))
                    tot_y = np.append(tot_y, avail_ans[4])
                    tot_pred = np.append(tot_pred, avail_ans[3])
                else:
                    IC_avail = cond_filter(tb, cond, inst, data_precision=data_precision)
                    tmp_ans = dependence_sel(tb, IC_avail, schema.keys(), functionals, A, rho)
                    rules.append((cond, tmp_ans[0], A, tmp_ans[5]))
                    rules_cnd.append((cond, A))
                    tot_y = np.append(tot_y, tmp_ans[4])
                    tot_pred = np.append(tot_pred, tmp_ans[3])
            else:
                for condition, ansx in opt_sep[2]:
                    reg, rmse, flg, pred, y, src_id = ansx
                    if flg or max(sep_map.values()) >= max_p or sum(sep_map.values()) >= max_P:
                        rules.append((condition, reg, A, src_id))
                        rules_cnd.append((condition, A))
                        tot_y = np.append(tot_y, y)
                        tot_pred = np.append(tot_pred, pred)
                    else:
                        sep_tmp = {sepx: sep_map[sepx] for sepx in sep_map}
                        sep_tmp[B] = sep_map[B] + 1
                        queue.append((condition, sep_tmp, ansx))
    tot_time = time.time() - st_tot
    rmse = mse(tot_pred, tot_y)
    return rules, rmse, tot_time


def extend(tb, rulesx, attrs, rho, data_precision):
    rules = rulesx
    queue = [(rules[i][-2], i) for i in range(len(rules))]
    while queue:
        A, i = queue[0]
        queue = queue[1:]
        sz_iter = len(rules)
        for j in range(i + 1, sz_iter):
            if rules[j][-1] != A: continue
            flgB, leg = False, False
            condi, condj = rules[i][0], rules[j][0]
            cond_ans = {B: condi[B] for B in condi}
            for B in condi:
                if condi[B][0] == condj[B][0] and condi[B][1] == condj[B][1]: continue
                if condi[B][0] == condj[B][1]:
                    if flgB: 
                        leg = True
                        break
                    cond_ans[B][0] = condj[B][0]
                    flgB = True
                if condi[B][1] == condj[B][0]:
                    if flgB:
                        leg=True
                        break
                    cond_ans[B][1] = condj[B][1]
                    flgB = True
            if not leg: # fusable
                ICj = cond_filter(tb, condj, range(len(tb)), data_precision)
                flg, rmse = fusion_test(tb, ICj, attrs, rules[i][1], rules[i][3], rules[i][2], rho)
                if flg: 
                    queue.append((A, len(rules)))
                    rules.append((cond_ans, rules[i][1], A, rules[i][3]))
    return rules

