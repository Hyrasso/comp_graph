import unittest
from src import *
from src.tensor import *

class Test_tensor(unittest.TestCase):

    def test_tensor_add(self):
        t1 = Tensor(((Const(1), Const(1)), (Const(1), Const(1))))
        t2 = Tensor(((Const(2), Const(2)), (Const(2), Const(2))))
        r = t1 + t2
        r = r.compute()
        self.assertEqual(r[0, 0], 3)
        self.assertEqual(r[1, 1], 3)
    
    def test_tensor_pow(self):
        t1 = Tensor(((Const(1), Const(1)), (Const(1), Const(4))))
        r = t1 ** 2
        r = r.compute()
        self.assertEqual(r[0, 0], 1)
        self.assertEqual(r[1, 1], 16)

    def test_iter_sum(self):
        t = Tensor((Const(1), Const(2), Const(3)))
        r = iter_sum(t)
        r = r.compute()
        self.assertEqual(r, 6)
    
    def test_matmul(self):
        m = Tensor(((Const(1), Const(0)),
                    (Const(0), Const(1))))
        x = Var("x")
        y = Var("y")

        X = Tensor.from_vector((x, y))
        r = matmul(m, X)
        self.assertEqual(r.shape, (1, 2))
        self.assertEqual(r[0, 0], x)
        self.assertEqual(r[0, 1], y)
