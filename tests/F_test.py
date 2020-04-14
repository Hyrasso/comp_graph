import unittest
from src.F import *

class TestF(F):
    def compute(self):
        return 0
    def grad(self):
        return ()

class Test_F_base_class(unittest.TestCase):
    def test_abstract(self):
        with self.assertRaises(TypeError):
            f = F()

        with self.assertRaises(TypeError):
            class TestF(F):
                ...
            TestF()

        with self.assertRaises(TypeError):
            class TestF(F):
                def grad(self):
                    return ()
            TestF()

        with self.assertRaises(TypeError):
            class TestF(F):
                def compute(self):
                    return 0
            TestF()

        class TestF(F):
            def compute(self):
                return 0
            def grad(self):
                return ()
        TestF()

class Test_F_Const(unittest.TestCase):
    def test_init(self):
        obj1 = object()
        obj2 = object()
        testf = TestF(obj1, obj2)
        self.assertEqual(len(testf.args), 2)
        self.assertEqual(testf.args[0], obj1)
        self.assertEqual(testf.args[1], obj2)

    def test_repr(self):
        obj1 = object()
        obj2 = object()

        testf = TestF(obj1, obj2)
        self.assertEqual(repr(testf), f"TestF({obj1!r}, {obj2!r})")

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
        ctx = {o:1}
        with c1.set_context(ctx):
            self.assertEqual(F.context, ctx)
            self.assertEqual(c1.compute(), o)
        with c1.set_context():
            self.assertEqual(F.context, {})
        o1, o2 = object(), object()
        with c1.set_context({o1:0}):
            self.assertEqual(F.context, {o1:0})
            with c1.set_context({o2:2}):
                self.assertEqual(F.context, {o1:0, o2:2})
                self.assertEqual(c1.compute(), o)


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

