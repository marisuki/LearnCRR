class disjointSet:
    def __init__(self, length):
        self.par = [i for i in range(length)]
        self.rank = [0 for i in range(length)]

    def find(self, x):
        # print(self.par, x)
        if self.par[x] == x: return x
        while self.par[x] != x:
            x = self.par[x]
        return x

    def combine(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            if self.rank[x] > self.rank[y]:
                self.par[y] = x
            else:
                if self.rank[x] == self.rank[y]:
                    self.rank[x] += 1
                self.par[x] = y

    def export_cluster(self, x):
        ans = []
        for i in range(len(self.par)):
            self.par[i] = self.find(i)
        for i in range(len(self.par)):
            if self.par[i] == self.par[x]:
                ans.append(i)
        return ans
    
    def export(self):
        rep = set()
        for x in range(len(self.par)):
            if self.find(x) not in rep:
                rep |= {self.find(x)}
        return rep