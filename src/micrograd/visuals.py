from graphviz import Digraph
from src.micrograd.engine import Value


class DrawNN:
    def __init__(self) -> None:
        pass

    @staticmethod
    def trace(root: Value):
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    def draw(self, root: Value):
        dot = Digraph(
            format="svg",
            graph_attr={"rankdir": "LR"},
        )
        nodes, edges = DrawNN.trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(
                name=uid,
                label=f"{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}",
                shape="record",
            )
            if n._op:
                dot.node(name=uid + n._op, label=n._op)
                dot.edge(tail_name=uid + n._op, head_name=uid)

        for n1, n2 in edges:
            dot.edge(tail_name=str(id(n1)), head_name=str(id(n2)) + n2._op)

        return dot
