1. Non-leaf variables do not register gradiets.

```y = x + x
z = y + y
z.backward()'''
# y will not have a gradient pointer but x and z will.
