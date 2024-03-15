
import torch


a = torch.tensor([1 for i in range(4)])
b = torch.tensor([2 for i in range(4)])
c = torch.tensor([3 for i in range(4)])
d = torch.tensor([1, 1, 0, 0])
print(a)
print(b)
print(torch.cat((a,b,c)))
abc = torch.stack((a,b,c),dim=1)
print(abc)
print(torch.cat((abc,abc),dim=1))
ab = torch.stack((a,b), dim=0)
print(ab)

print(a[d.nonzero(as_tuple=True)])
# true positives : 15553/19217
# true negative : 4163/7918

e = torch.add(a,1)
print(e)

dic={"a":1, 'b':2}
print(len(dic))
print(int(True))
print(int(False))

print(list(dic))


l = [("name",(1,2)), ("name",(3,2)),("name",(10,1)),("name",(5,2)),("namee",(1,2)),("name",(1,3))]
print(l)
k=sorted(l)
print(k)
print(l)