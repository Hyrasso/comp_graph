# Computational graph

Python project implementing computational graph

```python
    x = Variable("x")
    y = Variable("y")
    a = Constant(2)

    z = x + y * a
    print(z)
    # Add(Variable(x), Mul(Variable(y), Constant(2)))
    z = z / 2
    print(z)
    # Div(Add(Variable(x), Mul(Variable(y), Constant(2))), Constant(2))
    print(z.evaluate(variables={x: 2, y: 3}))
    # 4.0

```

## TODO List
- [ ] Name the project
- [ ] Tests
- [ ] Differentiation
- [ ] Graph simplification
