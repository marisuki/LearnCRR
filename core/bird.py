import database

def bird_crr_sc(f, bias=0, sz=1000):
    schema = [('decimalLatitude', 'Value'), ('decimalLongitude', 'Value'), ('Class', 'Value'), ('date', 'Value')]
    db = database.database()
    data_size = sz
    db.add_table("datasets/birdmap-time.csv", schema, ",", first_line_omit=False, max_index=sz)
    typex = [('decimalLatitude', 'Float'), ('decimalLongitude', 'Float'), ('date', 'Int'), ('Class', 'Int')]
    db.value_type(typex)
    ans = []
    ans = None
    timer = 0
    for i in db.dom[2]:
        dbi = db.filter(condition=[(2, i, '=', 3)])
        tb =  [t[1] for t in dbi.table]
        for i in range(2, len(tb)):
            tb[i] += [tb[i-1][0], tb[i-2][0], tb[i-1][1], tb[i-2][1]]
            for x in range(0, len(tb[i])-1):
                if x == 2: f.write(str(tb[i][x]+bias) + ", ")
                else: f.write(str(tb[i][x]) + ", ")
            f.write(str(tb[i][len(tb[i])-1]))
            f.write("\n")

if __name__ == "__main__":
    f = open("datasets/bird_lpp.txt", "w")
    for i in range(11):
        bird_crr_sc(f, bias = i, sz=-1)