import numpy as np
from disjoint import disjointSet
import database

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

def y_translation(X, y, f, max_rho):
    """
    X, y: np.array()
    f: predict, fit
    """
    pred = f.predict(X)-y
    delta = np.sum(pred)*1.0/np.shape(pred)[0]
    diff = pred - delta
    if np.max(diff) > max_rho: return None
    else: return delta


def xy_translation(X, y, f, max_rho, Delta):
    """
    Delat: f(X-Delta) +delta = f'(x)
    """
    #print(X)
    #print(X+Delta)
    XX = X+Delta
    pred = np.array([f.predict([XX[i]]) for i in range(len(XX))]) - y
    #print(pred)
    delta = np.sum(pred)*1.0/np.shape(pred)[0]
    diff = pred - delta
    #print(diff)
    if np.max(diff) > max_rho: return None
    else: return delta

def translation_equiv(tb, rules, db, max_rho, xy=False):
    # rules: [(condition, reg, A, src_id)]
    tb = [x[1] for x in db.table]
    #print(list(zip(range(len(rules)), range(len(rules)))))
    process = []
    for i in range(len(rules)):
        for j in range(len(rules)):
            if i > j: continue
            process.append((i, j))
    ds = disjointSet(len(rules))
    translate_p = {i: {j: None for j in range(len(rules))} for i in range(len(rules))}
    #print(process)
    res = {}
    for i, j in process:
        #print(ds.par)
        if ds.find(i) == ds.find(j):
            continue
        #print(i, j)
        ci,ri,A,s = rules[i]
        cj,rj,A,s = rules[j]
        #print(ci, cj, s, A)
        Delta = np.array([ci[x][0]-cj[x][0] for x in s])
        #print(Delta)
        inst = cond_filter(tb, cj, range(len(tb)), db.data_precision)
        X = np.array([[tb[t][attr] for attr in s] for t in inst])
        y = np.array([[tb[t][A]] for t in inst])
        #print(X, y, Delta)
        if xy: trans = xy_translation(X, y, ri, max_rho, Delta)
        else: trans = y_translation(X, y, ri, max_rho)
        if trans: 
            ds.combine(i, j)
            translate_p[i][j] = (Delta, trans)
            res[(i, j)] = inst + cond_filter(tb, ci, range(len(tb)), db.data_precision)
    #single = ans[1].export()
    ans = [[cond_filter(tb, c, range(len(tb)), db.data_precision), r, a, src] for c,r,a,src in rules]
    for i in range(len(rules)):
        if ds.find(i) != i:
            ans[ds.find(i)][0] += ans[i][0]
    opt = [ans[i] for i in ds.export()]
    return translate_p, ds, opt

def predict(qry, opt, X, y):
    #print(qry, y[qry], X[qry])
    for rule in opt:
        inst, reg, a, src = rule
        if qry in inst:
            return (np.sum(y[qry] - reg.predict([X[qry]])))**2
    return None

def rule_predict(qry, rules, X, y, db, tb):
    for rule in rules:
        c, reg, a, src = rule
        inst = cond_filter(tb, c, range(len(tb)), db.data_precision)
        if qry in inst:
            return (np.sum(y[qry] - reg.predict([X[qry]])))**2
    return None