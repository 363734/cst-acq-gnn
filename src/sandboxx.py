


d = {"time":0.33, 'tt':9, 'all':{"acc":1}, "sudoku":{"acc":1},"koi":{"acc":1},"rrr":{"acc":1}}

k = [l for l in d if type(d[l]) is dict if l != "all"]
print(k)
k= sorted(k)
print(k)