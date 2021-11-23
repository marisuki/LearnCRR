import lineartree
import condregression
import pprint
import numpy as np

from sklearn.linear_model import *
from lineartree import LinearTreeClassifier, LinearTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import copy
import translation
import time
import database

db = database.database()

schema = [("Time", 'Value'), ('decimalLatitude', 'Value'), ('decimalLongitude', 'Value')]

value_type = [('decimalLatitude', 'Float'), ('decimalLongitude', 'Float'), ('Time', 'Int')]

db.add_table("datasets/lalo_b0r.csv", schema, ",", first_line_omit=False)

db.value_type(value_type)

tb = [x[1] for x in db.table]

attrs = list(filter(lambda x: type(x) != type(""), [attrx[0] for attrx in db.schema.values()]))

dom = {k: [min(db.dom[k]), max(db.dom[k])] for k in db.dom}

print("Table sz= " + str(len(tb)))

schema = {A: db.schema[A][1] for A in attrs}

X = np.array([[x[0], x[1]] for x in tb])
y = np.array([x[2] for x in tb])
kf = KFold(n_splits=10)
score = 0

#regressor = DecisionTreeRegressor(random_state=0)
#print(cross_val_score(regressor, X, y, cv=10))
# rules: [(condition, reg, A, src_id)]

def cond_dfs(s, A, src):
    C_init = {k: [dom[k][0]-1, dom[k][1], 3] for k in dom}
    cond = {0: C_init}
    res = {}
    que = [0]
    while que:
        top = que[0]
        que = que[1:]
        if 'children' not in s[top]:
            res[top] = (cond[top], s[top]['models'], A, src)
            continue
        C0 = copy.deepcopy({k: cond[top][k] for k in cond[top]})
        C1 = copy.deepcopy({k: cond[top][k] for k in cond[top]})
        C0[s[top]['col']][1] = s[top]['th']
        C1[s[top]['col']][0] = s[top]['th']
        cond[s[top]['children'][0]] = C0
        cond[s[top]['children'][1]] = C1
        que += [s[top]['children'][0], s[top]['children'][1]]
    return res.values()


compress, sums = 0, 0
dt1, dt2 = 0,0
regans = None
valid = -1
tot_t1 = 0
tot_t2 = 0
loss_r = 0
loss_t = 0
for train_index, test_index in kf.split(X):
    st = time.time()
    #regressor = DecisionTreeRegressor(random_state=0)
    regr = LinearTreeRegressor(LinearRegression(), criterion='mae', max_depth=10)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #regressor.fit(X_train, y_train)
    #print(X)
    regr.fit(X_train, y_train)
    dt1 += time.time() - st
    #print(regr.summary())
    #print(cond_dfs(regr.summary(), 1, [0]))
    #translation_equiv(tb, rules, db, max_rho, xy=False):
    rules = cond_dfs(regr.summary(), 2, [0, 1])
    #print(len(rules))
    st = time.time()
    ans = translation.translation_equiv(tb, list(rules), db, 10, xy=False)
    #print(ans[0], ans[1].export())
    compress += len(ans[1].export())
    sums += len(rules)
    print(time.time()-st)

    ls1, ls2 = 0, 0
    false = []
    dt1, dt2 = 0, 0
    for qry in test_index:
        st = time.time()
        qes = translation.predict(qry, ans[-1], X, y)
        dt1 += time.time()-st
        if qes: ls1 += qes
        else: false.append(qry)
        st = time.time()
        qes = translation.rule_predict(qry, rules, X, y, db, tb)
        dt2 += time.time()-st
        if qes: ls2 += qes
        else: false.append(qry)
    ls1 = ls1*1.0/len(test_index)
    ls2 = ls2*1.0/len(test_index)
    loss_r += ls2
    loss_t += ls1
    tot_t1 += dt1
    tot_t2 += dt2
    #break
    #func = [('linear', {})]
    #ans = condregression.separation(tb, schema, k=[10, 10, 10], functionals=func, dom=dom, data_precision=db.data_precision, 
    #                            rho=[0.5, 0.5], sep_attr=[0], max_P=12, max_p=12, targets=[1], src=[0])
    #bias = regressor.predict(X_test)-y_test
    #print(regressor.get_n_leaves(), np.shape(X_train))
    #print(regr.summary(), np.shape(X_train))

    #print(ans)
    #print(mean_squared_error(y_test, regressor.predict(X_test)))
    #print(mean_squared_error(y_test, regr.predict(X_test)))
    #print(cross_val_score(regressor, X, y, cv=10))
    #print(regr.score(X_test, y_test))"""
#print(1.0-compress*1.0/sums)
print(compress, sums)
#print(dt1/10)
print(tot_t1/10.0, tot_t2/10)
print(loss_t/10.0, loss_r/10.0)
