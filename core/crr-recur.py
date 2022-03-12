import database
import copy
import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn_cnd
from sklearn.linear_model import BayesianRidge
import random
import time

mp = {}
test_id = []

class Range:
    def build(self, db:database.database, source_attr: list):
        # source_attr: attributes in filtering
        self.l = {attr: min(db.dom[attr]) for attr in source_attr}
        self.r = {attr: max(db.dom[attr]) for attr in source_attr}
    
    def __init__(self, l: dict, r: dict):
        self.l = copy.copy(l)
        self.r = copy.copy(r)
        self.inv = set()
    
    def _binary(self, attr):
        lx, rx = copy.copy(self.l), copy.copy(self.r)
        lx[attr] = (self.l[attr] + self.r[attr])/2.0
        ly, ry = copy.copy(self.l), copy.copy(self.r)
        ry[attr] = (self.l[attr] + self.r[attr])/2.0
        return [Range(lx, rx), Range(ly, ry)]
    
    def _triary(self, attr):
        lx, rx = copy.deepcopy(self.l), copy.deepcopy(self.r)
        ly, ry = copy.deepcopy(self.l), copy.deepcopy(self.r)
        lz, rz = copy.deepcopy(self.l), copy.deepcopy(self.r)
        rx[attr] = (2*self.l[attr] + self.r[attr])/3.0
        ly[attr], ry[attr] = (2*self.l[attr] + self.r[attr])/3.0, (self.l[attr] + 2*self.r[attr])/3.0
        lz[attr] = (self.l[attr] + 2*self.r[attr])/3.0
        return [Range(lx, rx), Range(ly, ry), Range(lz, rz)]
    
def partition(rangex: Range, part_meth, attr):
    if part_meth == 'binary': return rangex._binary(attr)
    elif part_meth == 'triary': return rangex._triary(attr)
    else: return [None]

def dataset(db:database.database, src: list, target: int, rate=0.1, const_factor=True, illegal=-100):
    # src are attrs in X, independent variables.
    X, y, plc = [], [], 0
    assert type(target) is int or len(target) == 1
    tb = np.array([t[1] for t in db.table])
    ids = np.array(list(range(len(db.table))))
    np.random.shuffle(ids)
    # np.random.shuffle(tb)
    prev = tb[0][target]
    for i, t in zip(range(len(tb)), tb):
        tx, ty = [], []
        if t[target] < illegal: t[target] = prev
        else: prev = t[target]
        for i, v in zip(range(len(t)), t):
            if i in src: 
                tx.append(v)
                if i not in mp: 
                    mp[i] = plc
                    plc += 1
            elif i in target: ty.append(v)
        if const_factor: X.append(tx + [1.])
        else: X.append(tx)
        y.append(ty)
    X, y = np.array(X), np.array(y)
    train_id, test_id = np.sort(ids[:int(np.shape(X)[0]*(1-rate))]), np.sort(ids[int(np.shape(X)[0]*(1-rate)):])
    #print(np.shape(X))
    return X[train_id], y[train_id], X[test_id], y[test_id], train_id, test_id
    

def filter_sv(X: np.array, Y: np.array, rangex: Range):
    ans = []
    for x in X:
        flag = True
        for attr in rangex.l:
            if x[mp[attr]] >= rangex.l[attr] and x[mp[attr]] < rangex.r[attr]:
                continue
            else: 
                flag = False
                break
        ans.append(flag)
    return X[np.array(ans)], Y[np.array(ans)]

def test_model(models, trainX, trainY, rangex: Range, eps: float, mod):
    error, md, deltax = np.inf, None, -1
    for model in models:
        if mod == 'linear':
            bias = (trainY - model.predict(trainX))
        elif mod == 'bayesian':
            bias = np.abs(trainY - model.predict(np.vander(trainX.T[0], N=4)))
        if (np.max(bias) - np.min(bias))/2.0 <= eps:
            err = np.sum(bias)/(np.shape(trainY)[0]*1.0)
            if err < error:
                error, md, deltax = err, model, (np.max(bias) + np.min(bias))/2.0
    return error, md, deltax

# model sharing NG
def alg1_discover_ns(models, db, target, src, source_range: Range, partition_attr, mod = 'linear', part_meth='binary', eps=0.1, edge_sz=30):
    trainX, trainY, testX, testY, train_id, test_id = dataset(db, src, target)
    print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY))
    que = [source_range]
    model = []
    crr = []
    while que:
        r0 = que[0]
        que = que[1:]
        subX, subY = filter_sv(trainX, trainY, r0)
        if np.shape(subX)[0] == 0: continue
        # error, ext_md, deltax = test_model(model, subX, subY, r0, eps, mod)
        error, ext_md, deltax = np.inf, None, -1.0
        if error <= eps:
            r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
            crr.append([ext_md, deltax, r0])
        elif np.shape(subX)[0] <= edge_sz:
            if mod == 'linear':
                reg = LinearRegression()
                reg.fit(subX, subY)
            elif mod == 'bayesian':
                init = [1 / np.var(trY), 1.]
                reg = BayesianRidge(tol=random.random()*1e-5, fit_intercept=False, compute_score=True)
                reg.set_params(alpha_init=init[0], lambda_init=init[1])
                reg.fit(np.vander(subX.T[0], N=4), subY)
            r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
            crr.append([reg, 0., r0])
            model.append(reg)
        else:
            trX, trY = subX[:int(np.shape(subX)[0]*(0.9)-0.5),:], subY[:int(np.shape(subX)[0]*(0.9)-0.5)]
            vlX, vlY = subX[int(np.shape(subX)[0]*(0.9)-0.5):,:], subY[int(np.shape(subX)[0]*(0.9)-0.5):]
            if mod == 'linear':
                reg = LinearRegression()
                reg.fit(trX, trY)
                loss = np.sum(np.abs(vlY - reg.predict(vlX)))/(np.shape(vlX)[0]*1.0)
            elif mod == 'bayesian':
                init = [1 / np.var(trY), 1.]
                reg = BayesianRidge(tol=random.random()*1e-5, fit_intercept=False, compute_score=True)
                reg.set_params(alpha_init=init[0], lambda_init=init[1])
                #print(trX.T[0])
                reg.fit(np.vander(trX.T[0], N=4), trY)
                loss = np.sum(np.abs(vlY - reg.predict(np.vander(vlX.T[0], N=4))))/(np.shape(vlX)[0]*1.0)
            if np.shape(trX)[0] <= edge_sz or loss <= eps:
                r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
                crr.append([reg, 0., r0])
                model.append(reg)
            else:
                rs = partition(r0, part_meth=part_meth,attr=partition_attr)
                if rs is not None:
                    que += rs
    return crr, testX, testY, model, test_id


def alg1_discover(models, db, target, src, source_range: Range, partition_attr, mod = 'linear', part_meth='binary', eps=0.1, edge_sz=30):
    trainX, trainY, testX, testY, train_id, test_id = dataset(db, src, target)
    print(np.shape(trainX), np.shape(trainY), np.shape(testX), np.shape(testY))
    que = [source_range]
    model = []
    crr = []
    while que:
        r0 = que[0]
        que = que[1:]
        subX, subY = filter_sv(trainX, trainY, r0)
        if np.shape(subX)[0] == 0: continue
        error, ext_md, deltax = test_model(model, subX, subY, r0, eps, mod)
        if error <= eps:
            r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
            crr.append([ext_md, deltax, r0])
        elif np.shape(subX)[0] <= edge_sz:
            if mod == 'linear':
                reg = LinearRegression()
                reg.fit(subX, subY)
            elif mod == 'bayesian':
                init = [1 / np.var(trY), 1.]
                reg = BayesianRidge(tol=random.random()*1e-5, fit_intercept=False, compute_score=True)
                reg.set_params(alpha_init=init[0], lambda_init=init[1])
                reg.fit(np.vander(subX.T[0], N=4), subY)
            r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
            crr.append([reg, 0., r0])
            model.append(reg)
        else:
            trX, trY = subX[:int(np.shape(subX)[0]*(0.9)-0.5),:], subY[:int(np.shape(subX)[0]*(0.9)-0.5)]
            vlX, vlY = subX[int(np.shape(subX)[0]*(0.9)-0.5):,:], subY[int(np.shape(subX)[0]*(0.9)-0.5):]
            if mod == 'linear':
                reg = LinearRegression()
                reg.fit(trX, trY)
                loss = np.sum(np.abs(vlY - reg.predict(vlX)))/(np.shape(vlX)[0]*1.0)
            elif mod == 'bayesian':
                init = [1 / np.var(trY), 1.]
                reg = BayesianRidge(tol=random.random()*1e-5, fit_intercept=False, compute_score=True)
                reg.set_params(alpha_init=init[0], lambda_init=init[1])
                #print(trX.T[0])
                reg.fit(np.vander(trX.T[0], N=4), trY)
                loss = np.sum(np.abs(vlY - reg.predict(np.vander(vlX.T[0], N=4))))/(np.shape(vlX)[0]*1.0)
            if np.shape(trX)[0] <= edge_sz or loss <= eps:
                r0 = Range({0: np.min(subX.T[0])}, {0: np.max(subX.T[0])})
                crr.append([reg, 0., r0])
                model.append(reg)
            else:
                rs = partition(r0, part_meth=part_meth,attr=partition_attr)
                if rs is not None:
                    que += rs
    return crr, testX, testY, model, test_id


def alg2_compation(db, crr, model, source):
    compat = []
    reverse = {int(t[1][source]*86400): i for i, t in zip(range(len(db.table)), db.table)}
    # model = set([m for m, d, r in crr])
    #print(crr)
    ans = {} #{reg: [[], [], []] for reg in model}
    for i, rule in zip(range(len(crr)), crr):
        dom = list(range(reverse[int(86400*rule[-1].l[source])], reverse[int(86400*rule[-1].r[source])]))
        for x in dom: ans[x] = i
        #ans[rule[0]][2] += dom
        #ans[rule[0]][1].append(dom)
        #ans[rule[0]][0].append(rule[1])
    # ans = [[reg, ans[reg][0], ans[reg][1], ans[reg][2]] for reg in ans]
    #print(ans)
    return ans

def pack_crr(db, crr, model, source):
    compat = []
    reverse = {int(t[1][source]*86400): i for i, t in zip(range(len(db.table)), db.table)}
    # model = set([m for m, d, r in crr])
    #$print(crr)
    ans = []
    for rule in crr:
        dom = list(range(reverse[int(86400*rule[-1].l[source])], reverse[int(86400*rule[-1].r[source])]))
        ans.append([rule[0], [rule[1]], [dom], dom])
    #print(ans)
    return ans

def test(db, testX, testY, crr, mod='linear'):
    err, cnt, rel = 0., 0, 0
    st = time.time()
    #print(crr)
    for ci in crr:
        reg, delta, rangex = ci[0], ci[1], ci[2]
        ans = []
        for x in testX:
            flag = True
            for attr in rangex.l:
                if x[mp[attr]] >= rangex.l[attr] and x[mp[attr]] < rangex.r[attr]:
                    continue
                else: 
                    flag = False
                    break
            ans.append(flag)
        tX, tY = testX[np.array(ans)], testY[np.array(ans)]
        if np.shape(tX)[0] == 0: continue
        rel += 1
        if mod=='linear':
            #if np.max(np.abs(tY - reg.predict(tX))) > 10: continue
            err += np.sum((tY - reg.predict(tX) - delta)**2)
            cnt += np.shape(tX)[0]
            # /(np.shape(tX)[0]*1.0))
        else:
            err += np.sum(np.abs(tY-reg.predict(np.vander(tX.T[0], N=4))))/(np.shape(tX)[0]*1.0)
        # print(cnt, err)
    return [len(db.table)/1000., time.time()-st, np.sqrt(err/cnt), rel]


def test_com(db, testX, testY, crr, mod='linear', test_id=None, maps=None):
    err, cnt, rel = 0., 0, 0
    st = time.time()
    ans, lab = [], []
    #print(crr)
    fg, bias = [], []
    related = []
    for ite, i in zip(test_id, range(len(testX))):
        x, y = testX[i], testY[i]
        #flag, delt = False, None
        if ite not in maps: continue
        ci = crr[maps[ite]]
        related.append(maps[ite])
        reg, delta, rangex = ci[0], ci[1], ci[2]
        #print(reg.predict(np.array([x]))[0])
        ans.append(reg.predict(np.array([x]))[0] + delta)
        lab.append(y)
        """
        for ci in crr:
            reg, delta, rangex, dom = ci[0], ci[1], ci[2], ci[3]
            if ite not in dom: continue
            for d, ri in zip(delta, rangex):
                if ite in ri:
                    delt, flag = d, True
                    ans.append(reg.predict(np.array([x]))[0] + delt)
                    lab.append(y)
                    break
            if flag: break
        """
        #fg.append(flag)
        #if flag: bias.append([delt])
    #fg, bias = np.array(fg).T, np.array(bias)
    #print(fg)
    #print(np.shape(reg.predict(testX[fg])), np.shape(testY[fg]), np.shape(bias))
    #print(np.shape(lab), np.shape(ans))
    rmse = np.sqrt(mean_squared_error(ans, lab))
    dt = time.time()-st
    #print(rmse)
    return [len(db.table)/1000., dt, rmse, len(set([crr[ri][0] for ri in related]))], 
    """
    for ci in crr:
        reg, delta, rangex = ci[0], ci[1], ci[2]
        ans = []
        for ite, x in zip(test_id, testX):
            flag = False
            if ite in ci[3]: flag = True
            for attr in rangex.l:
                if x[mp[attr]] >= rangex.l[attr] and x[mp[attr]] < rangex.r[attr]:
                    continue
                else: 
                    flag = False
                    break
            ans.append(flag)
        tX, tY = testX[np.array(ans)], testY[np.array(ans)]
        if np.shape(tX)[0] == 0: continue
        rel += 1
        if mod=='linear':
            #if np.max(np.abs(tY - reg.predict(tX))) > 10: continue
            err += np.sum((tY - reg.predict(tX))**2)
            cnt += np.shape(tX)[0]
            # /(np.shape(tX)[0]*1.0))
        else:
            err += np.sum(np.abs(tY-reg.predict(np.vander(tX.T[0], N=4))))/(np.shape(tX)[0]*1.0)
        # print(cnt, err)
    return [len(db.table)/1000., time.time()-st, np.sqrt(err/cnt), rel]
    """

def deltaTRecur(db: database.database, target: int, temporal: int):
    ans = {}
    for t in db.table:
        if int(t[1][target]) not in ans:
            ans[int(t[1][target])] = [(t[1][temporal], t[1][target])]
        else:
            ans[int(t[1][target])].append((t[1][temporal], t[1][target]))
    train, test = {k: [] for k in ans}, []
    crr = {}
    for k in ans:
        trainX, trainY = [], []
        ans[k] = np.sort(ans[k])
        pre1, pre2 = -1, -1
        tr = []
        for i, ti in zip(range(len(ans[k])), ans[k]):
            if i == 0: pre1 = pre2 = 0
            else:
                if i == 1: pre2 = 0
                else: pre2 = ans[k][i-2][0]
                pre1 = ans[k][i-1][0]
            if random.random() <= 0.1:
                test.append(ti)
            else:
                tr.append(ti)
                trainX.append([pre2, pre1, 1.])
                trainY.append([ti[0]])
        if len(tr) <= 2: continue
        else: train[k] = tr
        reg = LinearRegression()
        reg.fit(np.array(trainX[2:]), np.array(trainY[2:]))
        crr[k] = [reg, 0., Range({target: k}, {target: k+1})]
    # test
    st = time.time()
    err, cnt = 0., 0
    for ti in test:
        flag = True
        candidate, bias = None, np.inf
        for k in train:
            if len(train[k]) == 0: continue
            ite = (np.abs(np.array([tj[0] for tj in train[k]]) - ti[0])).argmin()
            if np.abs(train[k][ite][0] - ti[0]) < 1e-4: continue
            #    err += (k-ti[1])**2
            #    cnt += 1
            #    flag = False
            #    break
            if train[k][ite][0] > ti[0]: ite -= 1
            if ite >= 1:
                diff = np.abs(ti[0] - crr[k][0].predict(np.array([[train[k][ite-1][0], train[k][ite][0], 1.]])))
                if diff < bias: candidate, bias = k, diff
        if flag and candidate:
            err += (k-ti[1])**2
            cnt += 1
    err = np.sqrt(err*1.0/cnt)
    #print(time.time() - st, err)
    return [len(db.table)/1000., time.time() - st, err, len(crr)]


import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
# auto-regression
def AR(db: database.database, source: int, target: int, rate=0.1):
    train, test, plc = [], [], 0
    for i, t in zip(range(len(db.table)), db.table):
        if plc >= 2 and random.random() <= rate:
            test.append((t[1][source], t[1][target], plc))
        else:
            train.append([t[1][source], t[1][target]])
            plc += 1
    lab, ans = [], []
    st = time.time()
    for t, v, pos in test:
        history = [x[1] for x in train[pos-2: pos]]
        ext = [x[0] for x in train[max(0, pos-2): pos]]
        model = ARIMA(history, order=(0, 2, 2))
        model = model.fit()
        out = model.forecast()
        lab.append(out[0])
        ans.append(v)
        #print(out[0], v)
    rmse = np.sqrt(mean_squared_error(y_true=ans, y_pred=lab))
    #print(rmse)
    return [len(db.table)/1000., time.time()-st, rmse, len(test)]

from pmdarima.preprocessing import FourierFeaturizer
from pmdarima import auto_arima
import pandas as pd
def DHR1(db: database.database, source: int, target: int, rate=0.1, period=24):
    #data = pd.read_csv(file, sep=";", index_col=0)
    train, test, plc = [], [], 0
    for i, t in zip(range(len(db.table)), db.table):
        if plc >= 2 and random.random() <= rate:
            test.append((t[1][source], t[1][target], plc))
        else:
            train.append([t[1][source], t[1][target]])
            plc += 1
    lab, ans = [], []
    st = time.time()
    for t, v, pos in test:
        history = [[x[1]] for x in train[pos-2: pos]]
        ext = [[x[0]] for x in train[max(0, pos-2): pos]]
        dfy = pd.DataFrame(train)
        dfy.set_index(keys=0)
        # print(dfy.iloc[:10])
        four_terms = FourierFeaturizer(period, 1)
        y_prime, exog = four_terms.fit_transform(y=dfy[1])
        #exog['date'] = y_prime.index
        #exog = exog.set_index(exog['date'])
        #exog.index.freq = 'D'
        #exog = exog.drop(columns=['date'])
        reg = auto_arima(y=y_prime, D=1, exogenous=exog, seasonal=True, m=7)
        res = reg.predict(n_periods=1)
        print(res)

import torch
device = torch.device("cpu")
dtype = torch.float
def DHR(db: database.database, source: int, target: int, rate=0.1, period=24):
    train, test, plc = [], [], 0
    for i, t in zip(range(len(db.table)), db.table):
        if plc >= 2 and random.random() <= rate:
            test.append((t[1][source], t[1][target], plc))
        else:
            train.append([t[1][source], t[1][target]])
            plc += 1
    lab, ans = [], []
    st = time.time()
    for t, v, pos in test:
        trainY = torch.from_numpy(np.array([x[1] for x in train]).T).to(device)
        trainX = torch.from_numpy(np.array([x[0] for x in train]).T).to(device)
        lr = 1e-6
        w_c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
        w_s = torch.randn((), device=device, dtype=dtype, requires_grad=True)
        w_const = torch.randn((), device=device, dtype=dtype, requires_grad=True)
        for i in range(1000):
            loss = torch.sum(torch.pow(trainY - torch.cos(trainX*w_c) - torch.sin(trainX*w_s) - w_const, 2))
            loss.backward()
            with torch.no_grad():
                w_c -= lr*w_c.grad
                w_s -= lr*w_s.grad
                w_const -= lr*w_const.grad
                w_c.grad=None
                w_s.grad=None
                w_const.grad=None
        lab.append(v)
        ans.append((torch.cos(w_c*t) + torch.sin(t*w_s) + w_const).cpu().detach().numpy())
    rmse = float(np.sqrt(mean_squared_error(y_true=lab, y_pred=ans)))
    print(len(db.table), rmse)
    return [len(db.table), time.time()-st, rmse, len(ans)]


from sklearn.linear_model import LogisticRegression
def MCReg(db: database.database, source: int, target: int):
    trainX, trainY, testX, testY, train_id, test_id = dataset(db, [source], [target])
    trainY = np.rint(trainY*10).ravel()
    # print(trainY)
    reg = LogisticRegression(random_state=0, solver='newton-cg', max_iter=500).fit(trainX, trainY)
    st = time.time()
    res = reg.predict(testX)
    # print(testY.T)
    rmse = np.sqrt(mean_squared_error(testY, res.T/10.0))
    return [len(db.table)/1000., time.time()-st, rmse, 1.0]


def air(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t3.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    print(r.l, r.r)
    mod = 'linear'
    crr, testX, testY, model, test_id = alg1_discover(None, db, [11], [0], r, partition_attr=0, part_meth='binary', mod=mod, eps=1.0, edge_sz=5)
    print(len(crr))
    print(len(alg2_compation(crr)))
    st = time.time()
    print(test(testX, testY, crr, mod, test_id=test_id))
    print(time.time()-st)

def air_crr(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t4.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    #mod = 'bayesian'
    mod = 'linear'
    crr, testX, testY, model, test_id = alg1_discover(None, db, [1], [0, -1, -2, -3, -4], r, partition_attr=0, part_meth='binary', mod=mod, eps=0.7, edge_sz=5)
    pac = alg2_compation(db, crr, model, source=0)
    #crr = pack_crr(db, crr, model, source=0)
    #print(len(crr))
    #print(len(alg2_compation(crr)))
    st = time.time()
    #print(test(testX, testY, crr, mod))
    #print(time.time()-st)
    #return np.array(test(db, testX, testY, crr, mod))
    return np.array(test_com(db, testX, testY, crr, mod, test_id=test_id, maps=pac))[0]


def air_recur(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t4.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    #mod = 'bayesian'
    mod = 'linear'
    # crr, testX, testY = alg1_discover(None, db, [11], [0, -1, -2, -3, -4], r, partition_attr=0, part_meth='binary', mod=mod, eps=0.5, edge_sz=2)
    return np.array(deltaTRecur(db, target=11, temporal=0))

def air_AR(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t4.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(AR(db, 0, 11))

def air_MC(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t4.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(MCReg(db, source=0, target=11))

def air_DHR(sz=1000):
    schema= [('Time', 'Value'), ('CO(GT)', 'Value'), ('PT08.S1(CO)', 'Value'), ('NMHC(GT)', 'Value'), ('C6H6(GT)', 'Value'), ('PT08.S2(NMHC)', 'Value'), ('NOx(GT)', 'Value'), ('PT08.S3(NOx)', 'Value'), ('NO2(GT)', 'Value'), ('PT08.S4(NO2)', 'Value'), ('PT08.S5(O3)', 'Value'), ('T', 'Value'), ('RH', 'Value'), ('AH', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/AirQualityUCI_t4.csv",
             schema,
             ";", False, data_size)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(DHR(db, 0, 11))


#birdmap
def bird_crr(sz=1000):
    schema = [('decimalLatitude', 'Value'), ('decimalLongitude', 'Value'), ('Class', 'Enumerate'), ('year', 'Value'), ('month', 'Value'), ('day', 'Value'), ('date', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/abalone.data", schema, ",", first_line_omit=False)
    db.value_type(
        [('Time', 'Float'), ('CO(GT)', 'Float'), ('PT08.S1(CO)', 'Float'), ('NMHC(GT)', 'Float'), ('C6H6(GT)', 'Float'), ('PT08.S2(NMHC)', 'Float'), ('NOx(GT)', 'Float'), ('PT08.S3(NOx)', 'Float'), ('NO2(GT)', 'Float'), ('PT08.S4(NO2)', 'Float'), ('PT08.S5(O3)', 'Float'), ('T', 'Float'), ('RH', 'Float'), ('AH', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
        )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    #mod = 'bayesian'
    mod = 'linear'
    crr, testX, testY, model, test_id = alg1_discover(None, db, [11], [0, -1, -2, -3, -4], r, partition_attr=0, part_meth='binary', mod=mod, eps=1.0, edge_sz=3)
    crr = alg2_compation(db, crr, model, source=0)
    #print(len(crr))
    #print(len(alg2_compation(crr)))
    st = time.time()
    #print(test(testX, testY, crr, mod))
    #print(time.time()-st)
    return np.array(test(db, testX, testY, crr, mod, test_id=test_id))

def power_crr(sz=1000):
    schema = [('Time', 'Value'), ('Global_active_power', 'Value'), ('Global_reactive_power', 'Value'), ('Voltage', 'Value'), ('Global_intensity', 'Value'), ('Sub_metering_1', 'Value'), ('Sub_metering_2', 'Value'), ('Sub_metering_3', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]    
    db = database.database()
    data_size = sz
    db.add_table("datasets/power_t3.csv", schema, ";", first_line_omit=False, max_index=sz)
    db.value_type(
        [('Time', 'Float'), ('Global_active_power', 'Float'), ('Global_reactive_power', 'Float'), ('Voltage', 'Float'), ('Global_intensity', 'Float'), ('Sub_metering_1', 'Float'), ('Sub_metering_2', 'Float'), ('Sub_metering_3', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
            )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    #mod = 'bayesian'
    mod = 'linear'
    crr, testX, testY, model, test_id = alg1_discover(None, db, [1], [0, -1, -2, -3, -4], r, partition_attr=0, part_meth='binary', mod=mod, eps=0.7, edge_sz=5)
    pac = alg2_compation(db, crr, model, source=0)
    #crr = pack_crr(db, crr, model, source=0)
    #print(len(crr))
    #print(len(alg2_compation(crr)))
    st = time.time()
    #print(test(testX, testY, crr, mod))
    #print(time.time()-st)
    #return np.array(test(db, testX, testY, crr, mod))
    return np.array(test_com(db, testX, testY, crr, mod, test_id=test_id, maps=pac))[0]

def power_ar(sz=1000):
    schema = [('Time', 'Value'), ('Global_active_power', 'Value'), ('Global_reactive_power', 'Value'), ('Voltage', 'Value'), ('Global_intensity', 'Value'), ('Sub_metering_1', 'Value'), ('Sub_metering_2', 'Value'), ('Sub_metering_3', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]    
    db = database.database()
    data_size = sz
    db.add_table("datasets/power_t3.csv", schema, ";", first_line_omit=False, max_index=sz)
    db.value_type(
        [('Time', 'Float'), ('Global_active_power', 'Float'), ('Global_reactive_power', 'Float'), ('Voltage', 'Float'), ('Global_intensity', 'Float'), ('Sub_metering_1', 'Float'), ('Sub_metering_2', 'Float'), ('Sub_metering_3', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
            )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(AR(db, 0, 1))

def power_MC(sz=1000):
    schema = [('Time', 'Value'), ('Global_active_power', 'Value'), ('Global_reactive_power', 'Value'), ('Voltage', 'Value'), ('Global_intensity', 'Value'), ('Sub_metering_1', 'Value'), ('Sub_metering_2', 'Value'), ('Sub_metering_3', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]    
    db = database.database()
    data_size = sz
    db.add_table("datasets/power_t3.csv", schema, ";", first_line_omit=False, max_index=sz)
    db.value_type(
        [('Time', 'Float'), ('Global_active_power', 'Float'), ('Global_reactive_power', 'Float'), ('Voltage', 'Float'), ('Global_intensity', 'Float'), ('Sub_metering_1', 'Float'), ('Sub_metering_2', 'Float'), ('Sub_metering_3', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
            )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(MCReg(db, source=0, target=1))

def power_DHR(sz=1000):
    schema = [('Time', 'Value'), ('Global_active_power', 'Value'), ('Global_reactive_power', 'Value'), ('Voltage', 'Value'), ('Global_intensity', 'Value'), ('Sub_metering_1', 'Value'), ('Sub_metering_2', 'Value'), ('Sub_metering_3', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]    
    db = database.database()
    data_size = sz
    db.add_table("datasets/power_t3.csv", schema, ";", first_line_omit=False, max_index=sz)
    db.value_type(
        [('Time', 'Float'), ('Global_active_power', 'Float'), ('Global_reactive_power', 'Float'), ('Voltage', 'Float'), ('Global_intensity', 'Float'), ('Sub_metering_1', 'Float'), ('Sub_metering_2', 'Float'), ('Sub_metering_3', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
            )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    return np.array(DHR(db, 0, 1))

def power_recur(sz=1000):
    schema = [('Time', 'Value'), ('Global_active_power', 'Value'), ('Global_reactive_power', 'Value'), ('Voltage', 'Value'), ('Global_intensity', 'Value'), ('Sub_metering_1', 'Value'), ('Sub_metering_2', 'Value'), ('Sub_metering_3', 'Value'), ('pre1', 'Value'), ('pre2', 'Value'), ('pret1', 'Value'), ('pret2', 'Value')]    
    db = database.database()
    data_size = sz
    db.add_table("datasets/power_t3.csv", schema, ";", first_line_omit=False, max_index=sz)
    db.value_type(
        [('Time', 'Float'), ('Global_active_power', 'Float'), ('Global_reactive_power', 'Float'), ('Voltage', 'Float'), ('Global_intensity', 'Float'), ('Sub_metering_1', 'Float'), ('Sub_metering_2', 'Float'), ('Sub_metering_3', 'Float'), ('pre1', 'Float'), ('pre2', 'Float'), ('pret1', 'Float'), ('pret2', 'Float')]
            )
    r = Range(None, None)
    r.build(db, [0])
    #print(r.l, r.r)
    #mod = 'bayesian'
    mod = 'linear'
    # crr, testX, testY = alg1_discover(None, db, [11], [0, -1, -2, -3, -4], r, partition_attr=0, part_meth='binary', mod=mod, eps=0.5, edge_sz=2)
    return np.array(deltaTRecur(db, target=1, temporal=0))

def tax_crr(sz=1000):
    schema = [('FName', 'Enumerate'), ('LName', 'Enumerate'), ('Gender', 'Enumerate'), ('AreaCode', 'Enumerate'),
            ('Phone', 'Enumerate'), ('City', 'Enumerate'), ('State', 'Enumerate'), ('Zip', 'Enumerate'), ('MaritalStatus', 'Enumerate'),
            ('HasChild', 'Enumerate'), ('Salary', 'Value'), ('Rate', 'Value'), ('SingleExemp', 'Value'),
            ('MarriedExemp', 'Value'), ('ChildExemp', 'Value'), ('Tax', 'Value')]
    value_type = [('Salary', 'Int'), ('Rate', 'Float'), ('SingleExemp', 'Int'),
                ('MarriedExemp', 'Int'), ('ChildExemp', 'Int'), ('Tax', 'Float')]

def run(func, sz):
    st = time.time()
    ans = func(sz=sz)
    cnt = 1
    for i in range(4):
        ans += func(sz=sz)
        cnt += 1
    ans = ans*1.0 / 5.0
    return (time.time()-st)/5.0, ans
    

if __name__ == "__main__":
    f = open("out/power-recur.txt", "w")
    for i in range(1, 10):
        tot, res = run(air_crr, sz=i*2000)
        print(tot, res)
        for i in range(len(res)):
            f.write(str(float(res[i])) + " ")
        f.write(str(tot) + "\n")
    #print(power_DHR())
    # air_DHR()
    #print(run(func=air_DHR))
    # print(air_MC())
    # print(air_DHR())