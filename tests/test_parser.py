import unittest
import sys
import os
import networkx as nx

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from causal_reasoning.graph.graph import Graph
from causal_reasoning.graph.node import Node
from causal_reasoning.utils.parser import (
    parse_input_graph,
    _parse_default_graph,
    parse_edges,
    _edge_string_to_edge_tuples,
    list_tuples_into_list_nodes,
    tuple_into_node,
    parse_tuples_str_int_list,
    parse_tuple_str_int,
    parse_edges,
    parse_to_string_list,
)


class TestParseDefaultGraph(unittest.TestCase):
    def test_simple_graph(self):
        edges = [("A", "B"), ("B", "C")]
        latents = []
        custom_card = {}
        (n_nodes, children, node_cardinalities, parents, node_labels, dag) = (
            _parse_default_graph(edges, latents, custom_card)
        )

        # number of nodes
        self.assertEqual(n_nodes, 3)
        # node labels set
        self.assertEqual(node_labels, {"A", "B", "C"})
        # children
        self.assertEqual(children, {"A": ["B"], "B": ["C"], "C": []})
        # parents
        self.assertEqual(parents, {"A": [], "B": ["A"], "C": ["B"]})
        # default cardinalities = 2 for all
        self.assertEqual(node_cardinalities, {"A": 2, "B": 2, "C": 2})
        # DAG edges
        self.assertEqual(set(dag.nodes()), {"A", "B", "C"})
        self.assertEqual(set(dag.edges()), {("A", "B"), ("B", "C")})

    def test_custom_cardinalities_override(self):
        edges = [("X", "Y"), ("Y", "Z")]
        latents = []
        custom_card = {"X": 5, "Z": 7}
        (_, _, node_cardinalities, _, _, _) = _parse_default_graph(
            edges, latents, custom_card
        )
        # X and Z from custom, Y default=2
        self.assertEqual(node_cardinalities, {"X": 5, "Y": 2, "Z": 7})

    def test_latent_child_not_allowed_raises(self):
        edges = [("A", "L1")]
        latents = ["L1"]
        custom_card = {}
        with self.assertRaises(Exception) as cm:
            _parse_default_graph(edges, latents, custom_card)
        self.assertIn(
            "Invalid latent node: L1. Latent has income arrows.", str(cm.exception)
        )

    def test_latent_not_present_raises(self):
        edges = [("A", "B")]
        latents = ["L2"]
        custom_card = {}
        with self.assertRaises(Exception) as cm:
            _parse_default_graph(edges, latents, custom_card)
        self.assertIn(
            "Invalid latent node: L2. Not present in the graph.", str(cm.exception)
        )

    def test_latent_cardinality_default(self):
        # latents appear only on left side
        edges = [("L", "A")]
        latents = ["L"]
        custom_card = {}
        (_, _, node_cardinalities, _, _, _) = _parse_default_graph(
            edges, latents, custom_card
        )
        # L default to 0, A default to 2
        self.assertEqual(node_cardinalities, {"L": 0, "A": 2})

    def test_empty_edges(self):
        edges = []
        latents = []
        custom_card = {}
        (n_nodes, children, node_cardinalities, parents, node_labels, dag) = (
            _parse_default_graph(edges, latents, custom_card)
        )
        self.assertEqual(n_nodes, 0)
        self.assertEqual(children, {})
        self.assertEqual(parents, {})
        self.assertEqual(node_labels, set())
        self.assertEqual(node_cardinalities, {})
        self.assertEqual(len(dag.nodes()), 0)
        self.assertEqual(len(dag.edges()), 0)

    def test_parse_input_graph_wrapper(self):
        edges = [("A", "B")]
        latents = []
        custom_card = {"A": 3}

        direct = _parse_default_graph(edges, latents, custom_card)
        wrapper = parse_input_graph(edges, latents, custom_card)

        self.assertEqual(wrapper[:5], direct[:5])

        dag_direct, dag_wrapper = direct[5], wrapper[5]
        self.assertEqual(set(dag_direct.nodes()), set(dag_wrapper.nodes()))
        self.assertEqual(set(dag_direct.edges()), set(dag_wrapper.edges()))


class TestEdgeStringToTuples(unittest.TestCase):

    def test_simple_single_edge(self):
        """
        A single 'A->B' (no commas) should yield [('A', 'B')].
        """
        s = "A->B"
        expected = [("A", "B")]
        self.assertEqual(_edge_string_to_edge_tuples(s), expected)

    def test_single_edge_with_spaces(self):
        """
        A single ' A -> B ' with extra spaces should be trimmed to [('A','B')].
        """
        s = "   A   ->   B   "
        expected = [("A", "B")]
        self.assertEqual(_edge_string_to_edge_tuples(s), expected)

    def test_two_edges_comma_separated(self):
        """
        Two edges separated by a comma with and without spaces:
        'X->Y, Z->W' should yield [('X','Y'),('Z','W')].
        """
        s1 = "X->Y, Z->W"
        s2 = "X->Y,Z->W"
        expected = [("X", "Y"), ("Z", "W")]
        self.assertEqual(_edge_string_to_edge_tuples(s1), expected)
        self.assertEqual(_edge_string_to_edge_tuples(s2), expected)

    def test_multiple_edges_mixed_spacing(self):
        """
        Three edges with various spaces: 'A->B ,C -> D, E->F' should parse correctly.
        """
        s = "A->B ,C -> D,   E->F   "
        expected = [("A", "B"), ("C", "D"), ("E", "F")]
        self.assertEqual(_edge_string_to_edge_tuples(s), expected)

    def test_invalid_format_raises_value_error(self):
        """
        If a part does not contain '->', split will fail and raise ValueError.
        For example 'A-B' or 'A->B, C-D' should raise a ValueError for the invalid fragment.
        """
        with self.assertRaises(ValueError):
            _edge_string_to_edge_tuples("A-B")
        # Even if the first edge is valid, the second is invalid:
        with self.assertRaises(ValueError):
            _edge_string_to_edge_tuples("A->B, C-D")

    def test_empty_string_behaviour(self):
        """
        If an empty string is passed, it splits into [''], and split('->') will raise.
        """
        with self.assertRaises(ValueError):
            _edge_string_to_edge_tuples("")


class TestParseEdges(unittest.TestCase):

    def test_list_of_tuples_valid(self):
        """
        A well‐formed list of 2‐tuples, with str/int entries, should return
        a list of (str, str) pairs.
        """

        input_list = [(1, 2), ("A", 3), (4, "B")]
        expected = [("1", "2"), ("A", "3"), ("4", "B")]
        output = parse_edges(input_list)
        self.assertEqual(output, expected)

    def test_list_with_non_tuple_element_raises(self):
        """
        If the list contains something other than a 2‐tuple (e.g. a list or a single value),
        parse_edges should raise an Exception.
        """
        bad_list = [(1, 2), ["not", "a", "tuple"], ("A", "B")]
        with self.assertRaises(Exception) as cm:
            parse_edges(bad_list)
        self.assertIn("not recognized", str(cm.exception))

    def test_list_with_invalid_inner_type_raises(self):
        """
        If the inner 2‐tuple has an element that is neither str nor int, parse_edges should raise.
        E.g. a float or None.
        """
        bad_list = [
            (1, 2.5)
        ]  # 2.5 is a float; no conversion path to str in code, so it should raise.
        with self.assertRaises(Exception) as cm:
            parse_edges(bad_list)
        self.assertIn("not recognized", str(cm.exception))

    def test_tuple_input_valid(self):
        """
        If `state` is a single 2‐tuple of (u, v), parse_edges should return [(u, v)]
        with u, v coerced to strings if they are ints.
        """
        # Pure strings
        self.assertEqual(parse_edges(("A", "B")), [("A", "B")])

        # Pure ints → strings
        self.assertEqual(parse_edges((1, 2)), [("1", "2")])

        # Mixed int and str
        self.assertEqual(parse_edges((3, "C")), [("3", "C")])

    def test_tuple_with_invalid_inner_type_raises(self):
        """
        If tuple elements aren’t str or int, parse_edges should raise.
        """
        with self.assertRaises(Exception):
            parse_edges((None, "A"))
        with self.assertRaises(Exception):
            parse_edges(("A", object()))

    def test_graph_input_all_str_or_int_nodes(self):
        """
        If `state` is an nx.DiGraph whose nodes are strings or ints,
        parse_edges should produce a list of (left_str, right_str) for each edge.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")  # both strings
        G.add_edge(1, 2)  # ints
        G.add_edge("X", 3)  # mixed

        result = parse_edges(G)
        # We expect edges converted to strings:
        #   [("A", "B"), ("1", "2"), ("X", "3")]  in any order
        self.assertEqual(set(result), {("A", "B"), ("1", "2"), ("X", "3")})

    def test_graph_input_invalid_node_labels_raises(self):
        """
        If the graph has an edge whose source or target is neither str nor int,
        parse_edges should raise an Exception.
        """
        G1 = nx.DiGraph()

        class Foo:
            pass

        # Edge with a node of type Foo:
        G1.add_edge(Foo(), "B")
        with self.assertRaises(Exception):
            parse_edges(G1)

        G2 = nx.DiGraph()
        G2.add_edge("A", Foo())
        with self.assertRaises(Exception):
            parse_edges(G2)

    def test_invalid_type_raises(self):
        """
        Any other type (e.g. an int, float, set, dict, None, etc.) not covered above
        should trigger the final “not recognized” exception.
        """
        for bad in [123, 3.14, {"A", "B"}, {"A": "B"}, None]:
            with self.assertRaises(Exception):
                parse_edges(bad)


class TestTupleIntoNode(unittest.TestCase):
    def setUp(self):
        self.nodeA = Node(
            children=[],  # no children yet
            parents=[],  # no parents yet
            latentParent=None,  # no latent parent
            isLatent=False,  # not a latent node
            label="A",  # string label "A"
            cardinality=2,  # arbitrary cardinality
        )
        self.nodeB = Node(
            children=[],
            parents=[],
            latentParent=None,
            isLatent=False,
            label="B",
            cardinality=2,
        )

        # 2) Build the directed graph structure (networkx.DiGraph).
        #    We add nodes "A" and "B" (using string labels) and one edge.
        self.DAG = nx.DiGraph()
        self.DAG.add_node("A")
        self.DAG.add_node("B")
        self.DAG.add_edge("A", "B")

        # 3) Now fill in all Graph constructor arguments:

        # numberOfNodes: total count of actual Node objects
        numberOfNodes = 2

        # currNodes: a list of Node objects currently “in” the graph
        currNodes = [self.nodeA, self.nodeB]

        # dagComponents: Suppose we treat the entire graph as one strongly connected component.
        #                For a simple A->B chain, you can put both nodes together.
        dagComponents = [[self.nodeA, self.nodeB]]

        # exogenous: list of nodes with no incoming edges. Here, “A” has no parents.
        exogenous = [self.nodeA]

        # endogenous: list of nodes with at least one incoming edge. Here, “B” has A→B.
        endogenous = [self.nodeB]

        # topologicalOrder: one valid topological ordering is [A, B]
        topologicalOrder = [self.nodeA, self.nodeB]

        # cComponentToUnob: if you have any mapping from C‐component index → an unobserved Node,
        #                   you could fill it here. For our minimal example, we’ll leave it empty.
        cComponentToUnob = {}

        # graphNodes: a dictionary mapping each node’s string label → the Node instance
        graphNodes = {"A": self.nodeA, "B": self.nodeB}

        # node_set: a Python set containing all Node objects
        node_set = {self.nodeA, self.nodeB}

        # topologicalOrderIndexes: map each Node → its index in the above topologicalOrder list
        topologicalOrderIndexes = {self.nodeA: 0, self.nodeB: 1}

        # Finally, construct the Graph object itself:
        self.graph = Graph(
            numberOfNodes=numberOfNodes,
            currNodes=currNodes,
            dagComponents=dagComponents,
            exogenous=exogenous,
            endogenous=endogenous,
            topologicalOrder=topologicalOrder,
            DAG=self.DAG,
            cComponentToUnob=cComponentToUnob,
            graphNodes=graphNodes,
            node_set=node_set,
            topologicalOrderIndexes=topologicalOrderIndexes,
        )

    def test_tuple_none_returns_none(self):
        # If tuple_label_value is None, should return None
        self.assertIsNone(tuple_into_node(None, self.graph))

    def test_tuple_valid_label_sets_value_and_returns_node(self):
        # Before: node "A" exists with value None
        self.assertIsNone(self.graph.graphNodes["A"].value)

        node = tuple_into_node(("A", 42), self.graph)
        # It should return the same DummyNode instance
        self.assertIs(node, self.graph.graphNodes["A"])
        # And the node’s value should now be updated
        self.assertEqual(node.value, 42)

        # Also test a second label
        node_b = tuple_into_node(("B", -7), self.graph)
        self.assertIs(node_b, self.graph.graphNodes["B"])
        self.assertEqual(node_b.value, -7)

    def test_tuple_invalid_label_raises_exception(self):
        # If label not in graph, should raise an Exception
        with self.assertRaises(Exception) as cm:
            tuple_into_node(("X", 100), self.graph)
        self.assertIn("Node 'X' not present", str(cm.exception))

        # Even if graph has some nodes, "Z" is not defined
        with self.assertRaises(Exception):
            tuple_into_node(("Z", 0), self.graph)


class TestListTuplesIntoListNodes(unittest.TestCase):
    def setUp(self):
        # 1) Create three Node objects: A, B, and C
        self.nodeA = Node(
            children=[],
            parents=[],
            latentParent=None,
            isLatent=False,
            label="A",
            cardinality=2,
        )
        self.nodeB = Node(
            children=[],
            parents=[],
            latentParent=None,
            isLatent=False,
            label="B",
            cardinality=2,
        )
        self.nodeC = Node(
            children=[],
            parents=[],
            latentParent=None,
            isLatent=False,
            label="C",
            cardinality=2,
        )

        # 2) Build the directed graph structure with edges A→B and B→C
        self.DAG = nx.DiGraph()
        for lbl in ("A", "B", "C"):
            self.DAG.add_node(lbl)
        self.DAG.add_edge("A", "B")
        self.DAG.add_edge("B", "C")  # new edge

        # 3) Fill in Graph constructor arguments

        numberOfNodes = 3
        currNodes = [self.nodeA, self.nodeB, self.nodeC]

        # Now all three form a single chain component A→B→C
        dagComponents = [[self.nodeA, self.nodeB, self.nodeC]]

        # Exogenous: nodes with no incoming edges (only A)
        exogenous = [self.nodeA]

        # Endogenous: nodes with at least one incoming edge (B and C)
        endogenous = [self.nodeB, self.nodeC]

        # One valid topological ordering is [A, B, C]
        topologicalOrder = [self.nodeA, self.nodeB, self.nodeC]

        cComponentToUnob = {}

        graphNodes = {
            "A": self.nodeA,
            "B": self.nodeB,
            "C": self.nodeC,
        }

        node_set = {self.nodeA, self.nodeB, self.nodeC}

        topologicalOrderIndexes = {
            self.nodeA: 0,
            self.nodeB: 1,
            self.nodeC: 2,
        }

        # Construct the Graph
        self.graph = Graph(
            numberOfNodes=numberOfNodes,
            currNodes=currNodes,
            dagComponents=dagComponents,
            exogenous=exogenous,
            endogenous=endogenous,
            topologicalOrder=topologicalOrder,
            DAG=self.DAG,
            cComponentToUnob=cComponentToUnob,
            graphNodes=graphNodes,
            node_set=node_set,
            topologicalOrderIndexes=topologicalOrderIndexes,
        )

    def test_list_none_returns_none(self):
        self.assertIsNone(list_tuples_into_list_nodes(None, self.graph))

    def test_list_empty_returns_none(self):
        self.assertIsNone(list_tuples_into_list_nodes([], self.graph))

    def test_list_single_valid_tuple(self):
        result = list_tuples_into_list_nodes([("A", 10)], self.graph)
        self.assertEqual(len(result), 1)
        node = result[0]
        self.assertIs(node, self.graph.graphNodes["A"])
        self.assertEqual(node.value, 10)

    def test_list_multiple_valid_tuples(self):
        input_list = [("A", 1), ("C", 3)]
        result = list_tuples_into_list_nodes(input_list, self.graph)

        self.assertEqual(len(result), 2)
        node_a, node_c = result
        self.assertIs(node_a, self.graph.graphNodes["A"])
        self.assertEqual(node_a.value, 1)
        self.assertIs(node_c, self.graph.graphNodes["C"])
        self.assertEqual(node_c.value, 3)

        self.assertIsNone(self.graph.graphNodes["B"].value)

    def test_list_contains_invalid_label_raises(self):
        bad_list = [("A", 5), ("X", 7)]
        with self.assertRaises(Exception) as cm:
            list_tuples_into_list_nodes(bad_list, self.graph)
        self.assertIn("Node 'X' not present", str(cm.exception))

    def test_list_contains_none_tuple_raises(self):
        with self.assertRaises(TypeError):
            list_tuples_into_list_nodes([None], self.graph)

    def test_list_contains_invalid_type_raises(self):
        # If the list contains something that’s not a 2-tuple, unpacking also fails.
        with self.assertRaises(TypeError):
            list_tuples_into_list_nodes([("A", 1), "not a tuple"], self.graph)

        # If it’s a 2-tuple of wrong types, but label not in graph, it raises from tuple_into_node
        with self.assertRaises(Exception):
            list_tuples_into_list_nodes([("X", 1)], self.graph)


class TestParseTuplesStrIntList(unittest.TestCase):
    def test_list_of_valid_tuples(self):
        """
        A list of 2-tuples with str/int elements should be parsed element-wise.
        E.g. [("X","10"), (20, "30")] → [("X", 10), ("20", 30)].
        """
        input_list = [("X", "10"), (20, "30"), ("Z", 40)]
        expected = [("X", 10), ("20", 30), ("Z", 40)]
        result = parse_tuples_str_int_list(input_list)
        self.assertEqual(result, expected)

    def test_list_with_non_tuple_element_raises(self):
        """
        If the list contains a non-tuple item, the function should fall through
        and raise its “not recognized” Exception.
        """
        bad_list = [("A", "1"), ["B", "2"], ("C", "3")]
        with self.assertRaises(Exception) as cm:
            parse_tuples_str_int_list(bad_list)
        self.assertIn("Input format for", str(cm.exception))

    def test_single_tuple(self):
        """
        A single 2-tuple should be wrapped in a list and parsed.
        E.g. ("A","5") → [("A",5)] or (6, "7") → [("6",7)].
        """
        self.assertEqual(parse_tuples_str_int_list(("A", "5")), [("A", 5)])
        self.assertEqual(parse_tuples_str_int_list((6, "7")), [("6", 7)])

    def test_tuple_with_non_numeric_string_raises_value_error(self):
        """
        If the tuple’s second element is a non-numeric string, int(...) will raise ValueError.
        That propagates out of parse_tuples_str_int_list.
        """
        with self.assertRaises(ValueError):
            parse_tuples_str_int_list(("A", "not-a-number"))

    def test_tuple_with_wrong_length_raises_value_error(self):
        """
        If the tuple is not length 2, unpacking in parse_tuple_str_int triggers ValueError.
        """
        with self.assertRaises(ValueError):
            parse_tuples_str_int_list(("A", "1", "extra"))

    def test_invalid_type_raises(self):
        """
        Non-list, non-tuple inputs should raise the generic “not recognized” Exception.
        """
        for bad in [123, 3.14, "string", None, {"A": 1}]:
            with self.assertRaises(Exception) as cm:
                parse_tuples_str_int_list(bad)
            self.assertIn("Input format for", str(cm.exception))


class TestParseTupleStrInt(unittest.TestCase):
    def test_str_str_numeric(self):
        """('A','123') → ('A', 123)"""
        self.assertEqual(parse_tuple_str_int(("A", "123")), ("A", 123))

    def test_int_int(self):
        """(5, 6) → ('5', 6)"""
        self.assertEqual(parse_tuple_str_int((5, 6)), ("5", 6))

    def test_int_str(self):
        """(7, '8') → ('7', 8)"""
        self.assertEqual(parse_tuple_str_int((7, "8")), ("7", 8))

    def test_invalid_first_element_type(self):
        """First element not str/int (e.g. None) → custom Exception"""
        with self.assertRaises(Exception) as cm:
            parse_tuple_str_int((None, 1))
        self.assertIn("Tuple input format for", str(cm.exception))

    def test_invalid_second_element_type(self):
        """Second element neither str nor int (e.g. float) → custom Exception"""
        with self.assertRaises(Exception) as cm:
            parse_tuple_str_int(("A", 2.5))
        self.assertIn("Tuple input format for", str(cm.exception))

    def test_non_numeric_string_second_raises_value_error(self):
        """Second element is non‐numeric string → int(...) raises ValueError"""
        with self.assertRaises(ValueError):
            parse_tuple_str_int(("A", "hello"))

    def test_wrong_length_tuple_raises_value_error(self):
        """Tuple of length ≠ 2 → unpacking into two variables raises ValueError"""
        with self.assertRaises(ValueError):
            parse_tuple_str_int(("A", 1, "extra"))

    def test_non_tuple_input_raises_exception(self):
        """Any non‐tuple input → custom Exception with type in message"""
        for bad in ["A", 123, None, ["A", 1]]:
            with self.assertRaises(Exception) as cm:
                parse_tuple_str_int(bad)
            self.assertIn("Input format for", str(cm.exception))


class TestParseToStringList(unittest.TestCase):
    def test_string_input(self):
        """A pure string should be wrapped in a single-element list."""
        self.assertEqual(parse_to_string_list("hello"), ["hello"])
        self.assertEqual(parse_to_string_list(""), [""])

    def test_int_input(self):
        """An integer should be converted to string and wrapped in a list."""
        self.assertEqual(parse_to_string_list(123), ["123"])
        self.assertEqual(parse_to_string_list(0), ["0"])

    def test_list_of_str(self):
        """A list of strings should be returned unchanged."""
        data = ["a", "b", "c"]
        result = parse_to_string_list(data)
        self.assertIs(result, data)  # same list object
        self.assertEqual(result, ["a", "b", "c"])

    def test_empty_list(self):
        """An empty list yields itself (an empty list)."""
        data = []
        result = parse_to_string_list(data)
        self.assertIs(result, data)
        self.assertEqual(result, [])

    def test_list_of_ints(self):
        """
        A list of ints passes the loop without raising but is returned unchanged
        (the code attempts conversion but does not rebuild the list).
        """
        data = [1, 2, 3]
        result = parse_to_string_list(data)
        self.assertIs(result, data)
        # Currently ints remain ints under existing implementation
        self.assertEqual(result, [1, 2, 3])

    def test_mixed_list_int_and_str(self):
        """
        A mixed list of int and str passes and is returned unchanged.
        """
        data = [10, "20", 30]
        result = parse_to_string_list(data)
        self.assertIs(result, data)
        self.assertEqual(result, [10, "20", 30])

    def test_list_with_invalid_type_raises(self):
        """Any list containing a non-str, non-int element should raise."""
        with self.assertRaises(Exception) as cm:
            parse_to_string_list([1, None, "x"])
        self.assertIn("not recognized", str(cm.exception))

        with self.assertRaises(Exception) as cm:
            parse_to_string_list([[], "a"])
        self.assertIn("not recognized", str(cm.exception))

    def test_other_types_raise(self):
        """Inputs that are not str, int, or list should raise."""
        for bad in [3.14, {"a": 1}, {"a", "b"}, None]:
            with self.assertRaises(Exception) as cm:
                parse_to_string_list(bad)
            self.assertIn("not recognized", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
