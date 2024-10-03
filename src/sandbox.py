
import torch


a = torch.tensor([1 for i in range(4)])
b = torch.tensor([2 for i in range(4)])
c = torch.tensor([3 for i in range(4)])
d = torch.tensor([1, 1, 0, 0])


n=5
z = [ [1,2,3], [1,2], [1,2,3,4]]
to = [d[:n] if len(d) >= n else d + [0] * (n - len(d)) for d in z]
print(to)
