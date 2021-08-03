import condregression
import database

db = database.database()
schema = [('decimalLatitude', 'Value'), ('decimalLongitude', 'Value'), ('Class', 'Enumerate'), ('year', 'Value'), 
          ('month', 'Value'), ('day', 'Value'), ('date', 'Value')]
#schema = [('decimalLatitude', 'Value'), ('decimalLongitude', 'Value')]
#db.add_table("dataset/birdBrief.csv", schema, ",", first_line_omit=True)
db.add_table("datasets/birdmap-samples.csv", schema, ",", first_line_omit=False)
typex = [('decimalLatitude', 'Float'), ('decimalLongitude', 'Float'),
         ('day', 'Int'), ('month', 'Int'), ('year', 'Int'), ('date', 'Int')]
#typex = [('decimalLatitude', 'Float'), ('decimalLongitude', 'Float')]
db.value_type(typex)
tb = [x[1] for x in db.table]
attrs = list(filter(lambda x: type(x) != type(""), [attrx[0] for attrx in db.schema.values()]))
dom = {k: [min(db.dom[k]), max(db.dom[k])] for k in db.dom}

print(len(tb))

schema = {A: db.schema[A][1] for A in attrs}

func = [('bayesian', {'n_order': 5}), ('bayesian', {'n_order': 7}), ('linear', {})]

print(condregression.separation(tb, schema, k=[10, 10, 10], functionals=func, dom=dom, data_precision=db.data_precision, 
                                rho=[0.5, 0.5], sep_attr=[6], max_P=10, max_p=5, targets=[1]))