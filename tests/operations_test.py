import unittest
from src import *
from src.operations import Add, Mul

class Test_Operations(unittest.TestCase):
    def test_add(self):
        a = Var("A")
        b = Var("B")

        z = a + b
        self.assertEqual(z, Add(a, b))
        ctx = {a: 0, b: 2}
        z.set_context(ctx)
        self.assertEqual(z.grad(), (Const(1), Const(1)))
        self.assertEqual(z.differentiate(a).compute(), 1)
        self.assertEqual(z.differentiate(b).compute(), 1)
        self.assertEqual(z.compute(), 0 + 2)

    def test_mul(self):
        a = Var("A")
        b = Var("B")

        z = a * b
        self.assertEqual(z, Mul(a, b))
        ctx = {a: 0, b: 2}
        z.set_context(ctx)
        self.assertEqual(z.grad(), (b, a))
        self.assertEqual(z.differentiate(a).compute(), b.compute())
        self.assertEqual(z.differentiate(b).compute(), a.compute())
        self.assertEqual(z.compute(), 0 * 2)

