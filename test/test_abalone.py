import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import condregression
from core import database

db = database.database()
schema = [("sex", "Enumerate"), ("length", "Value"), ("diameter", "Value"), ("height", "Value"), ("whole_weight", "Value"),
          ("shucked_weight", "Value"), ("viscera_weight", "Value"), ("shell_weight", "Value"), ("rings", "Value")]
db.add_table("datasets/abalone.data", schema, ",", first_line_omit=False)
db.value_type([ ("length", "Float"), ("diameter", "Float"), ("height", "Float"), ("whole_weight", "Float"),
          ("shucked_weight", "Float"), ("viscera_weight", "Float"), ("shell_weight", "Float"), ("rings", "Float")])
tb = [x[1] for x in db.table]
attrs = list(filter(lambda x: type(x) != type(""), [attrx[0] for attrx in db.schema.values()]))
dom = {k: [min(db.dom[k]), max(db.dom[k])] for k in db.dom}

print("Table sz= " + str(len(tb)))

schema = {A: db.schema[A][1] for A in attrs}



func = [('bayesian', {'n_order': 5}), ('bayesian', {'n_order': 7}), ('linear', {})]

rho = [0.3 for A in attrs]

print(condregression.separation(tb, schema, k=[10, 10, 10], functionals=func, dom=dom, data_precision=db.data_precision, 
                                rho=rho, sep_attr=range(len(schema)), max_P=3, max_p=2, targets=[1,2,3,4,5,6,7]))

# loss 0.0007707487325764856