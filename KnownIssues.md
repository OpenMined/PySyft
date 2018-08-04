1. Non-leaf variables do not register/track gradients.

All variables which have been created as the result of an operation have unreliable gradients. They are said to be non-leaf, as opposed as leaf variables which are directly assign (ex: x = Variable(torch.FloatTensor([1,2,3,4,5])))

```
x = torch.autograd.Variable(torch.FloatTensor([1,2,3,4,5]))
x.send(bob)
y = x + x
z = y.sum()
z.backward()
```
Only `x` will have a correct `x.grad` pointing to a remote gradient. `y`, `z` should have a pointer pointing to None, however these pointers can sometimes be broken.
