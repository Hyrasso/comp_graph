import unittest
import graph

class Graph(unittest.TestCase):

    def test_add_evaluation(self):
        a = graph.Variable("A")
        b = graph.Variable("B")

        z = a + b
        res = z.evaluate(variables={a: 1, b: 2})
        self.assertEquals(res, 3)

        z = b + a
        res = z.evaluate(variables={a: 1, b: 2})
        self.assertEquals(res, 3)
        
        z = a + 2
        res = z.evaluate(variables={a: 1, b: 2})
        self.assertEquals(res, 3)

    def test_mul_evaluation(self):
        a = graph.Variable("A")
        b = graph.Variable("B")

        z = a * b
        res = z.evaluate(variables={a: 2, b: 3})
        self.assertEquals(res, 6)

        z = b * a
        res = z.evaluate(variables={a: 2, b: 3})
        self.assertEquals(res, 6)
        
        z = a * 3
        res = z.evaluate(variables={a: 2, b: 3})
        self.assertEquals(res, 6)
