import unittest
from src.node import *
from src.variable import Const, Var

class TestNode(Node):
    def compute(self):
        return 0
    def grad(self):
        return ()

class Test_Node_base_class(unittest.TestCase):
    def test_abstract(self):
        with self.assertRaises(TypeError):
            f = Node()

        with self.assertRaises(TypeError):
            class TestNode(Node):
                ...
            TestNode()

        with self.assertRaises(TypeError):
            class TestNode(Node):
                def grad(self):
                    return ()
            TestNode()

        with self.assertRaises(TypeError):
            class TestNode(Node):
                def compute(self):
                    return 0
            TestNode()

        class TestNode(Node):
            def compute(self):
                return 0
            def grad(self):
                return ()
        TestNode()

class Test_Node_Const(unittest.TestCase):
    def test_init(self):
        obj1 = object()
        obj2 = object()
        testf = TestNode(obj1, obj2)
        self.assertEqual(len(testf.args), 2)
        self.assertEqual(testf.args[0], obj1)
        self.assertEqual(testf.args[1], obj2)

    def test_repr(self):
        obj1 = object()
        obj2 = object()

        testf = TestNode(obj1, obj2)
        self.assertEqual(repr(testf), f"TestNode({obj1!r}, {obj2!r})")

    def test_eq(self):
        c1 = Const(object())
        c2 = Const(object())
        self.assertEqual(c1, c1)
        self.assertEqual(c2, c2)
        self.assertNotEqual(c1, c2)
        c1 = Const(0)
        c2 = Const(0)
        c3 = Const(1)
        self.assertEqual(c1, c2)
        self.assertNotEqual(c3, c1)
        self.assertNotEqual(c3, c2)

    def test_hash(self):
        c1 = Const(object())
        c2 = Const(object())
        self.assertNotEqual(c1.__hash__(), c2.__hash__())

        c1 = Const(0)
        c2 = Const(0)
        self.assertNotEqual(c1.__hash__(), c2.__hash__())

    def test_set_context(self):
        o = object()
        c1 = Const(o)
        # TODO: should pass
        # ctx = {c1: 1}
        # with c1.set_context(ctx):
        #     self.assertEqual(c1.value, o)
        o1, o2 = object(), object()
        v1, v2 = Var(), Var()
        z = v1 + v2 + c1
        with z.set_context({v1:0}):
            self.assertEqual(v1.value, 0)
            self.assertEqual(v2.value, v2)
            self.assertEqual(c1.value, o)
            with z.set_context({v2:2}):
                self.assertEqual(v1.value, 0)
                self.assertEqual(v2.value, 2)
                self.assertEqual(c1.value, o)
            self.assertEqual(v1.value, 0)
            self.assertEqual(v2.value, v2)
            self.assertEqual(c1.value, o)



    def test_call(self):
        obj = object()
        c = Const(obj)
        self.assertEqual(c(), obj)
        
        a, b = 1, 2
        f = Const(a) + Const(b)
        self.assertEqual(f(), a + b, "depends on _add__ being implemented for const")

class Test_Const(unittest.TestCase):
    def test_init(self):
        obj = object()
        c = Const(obj)
        self.assertEqual(c.args, (obj,))
        with self.assertRaises(TypeError):
            Const()

    def test_compute(self):
        obj = object()
        c = Const(obj)
        self.assertEqual(c.compute(), obj)

    def test_grad(self):
        c = Const(object())
        grad = c.grad()
        self.assertEqual(grad, (Const(0),))
        self.assertEqual(grad[0].compute(), 0)
    
    def test_differentiate(self):
        c = Const(object())
        dcd0 = c.differentiate(Const(0))
        self.assertEqual(dcd0, Const(0))
        self.assertEqual(dcd0.compute(), 0)

