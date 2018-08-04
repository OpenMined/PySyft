1. Non-leaf variables do not register gradiets.

```
x = torch.autograd.Variable(torch.FloatTensor([1,2,3,4,5]))
y = x + x
z = y + y
z.backward()'''
# y will not have a gradient pointer but x and z will.
