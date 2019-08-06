# Computational graph

Python project implementing computational graph and differentiation on graph

```python
    x = Var("x")
    y = Var("y")
    a = Const(2)
    
    z = x ** 2 + y * a
    print(z)
    # ((x ^ 2) + (y * 2))
    z = z / 2
    print(repr(z))
    # Div(Add(Pow(x, 2), Mul(y, 2)), 2)
    print(z.compute({x: 2, y: 3}))

    # 5.0
    
    dzdx = z.differentiate(x)
    print(dzdx)
    # (((1 / 2) * ((1 * (((2 * (x ^ (2 - 1))) * 1) + (((x ^ 2) * Log(2)) * 0))) + (1 * ((2 * 0) + (y * 0))))) + ((-((x ^ 2) + (y * 2))) / (2 ^ 2)) * 0))
    print(dzdx.compute())
    # 2.0
    print(z.differentiate(y).compute({x:2, y:1}))
    # 1.0
```

## TODO List
- [ ] Name the project
- [ ] Tests
- [ ] Differentiation
- [ ] Graph simplification
