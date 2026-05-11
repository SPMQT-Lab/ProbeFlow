"""Tests for the scan provenance graph (provenance/graph.py)."""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.provenance.graph import (
    ImageNode,
    MeasurementNode,
    ScanGraph,
    graph_from_dict,
    graph_to_dict,
    materialize_image,
)


@pytest.fixture
def simple_graph():
    g = ScanGraph(scan_id="scan-1")
    root = ImageNode(
        source_scan_id="scan-1",
        operation="root",
        units="m",
        array=np.zeros((4, 4), dtype=np.float64),
    )
    g.add(root, root=True)
    return g, root


class TestImageNodeBasics:
    def test_root_detection(self):
        n = ImageNode(
            source_scan_id="s", operation="root", array=np.zeros((2, 2)),
        )
        assert n.is_root
        assert not n.is_virtual

    def test_derived_is_virtual_until_array_set(self):
        n = ImageNode(
            source_scan_id="s",
            operation="smooth",
            parent_ids=("parent-id",),
            params={"sigma": 1.0},
        )
        assert not n.is_root
        assert n.is_virtual

    def test_release_clears_array_on_derived_node(self):
        n = ImageNode(
            source_scan_id="s",
            operation="smooth",
            parent_ids=("p",),
            array=np.ones((3, 3)),
        )
        assert not n.is_virtual
        n.release()
        assert n.is_virtual

    def test_release_root_raises(self):
        n = ImageNode(
            source_scan_id="s", operation="root", array=np.zeros((2, 2)),
        )
        with pytest.raises(ValueError):
            n.release()

    def test_id_is_unique_per_instance(self):
        a = ImageNode(source_scan_id="s", operation="root", array=np.zeros((2, 2)))
        b = ImageNode(source_scan_id="s", operation="root", array=np.zeros((2, 2)))
        assert a.id != b.id


class TestScanGraphTopology:
    def test_add_root_records_in_root_ids(self, simple_graph):
        graph, root = simple_graph
        assert graph.root_ids == (root.id,)
        assert root.id in graph

    def test_add_derived_validates_parent(self):
        g = ScanGraph(scan_id="s")
        with pytest.raises(ValueError):
            g.add(
                ImageNode(
                    source_scan_id="s",
                    operation="smooth",
                    parent_ids=("nonexistent",),
                ),
            )

    def test_add_root_must_be_image_node(self):
        g = ScanGraph(scan_id="s")
        with pytest.raises(ValueError):
            g.add(
                MeasurementNode(source_scan_id="s", operation="root"),
                root=True,
            )

    def test_add_root_must_carry_array(self):
        g = ScanGraph(scan_id="s")
        with pytest.raises(ValueError):
            g.add(ImageNode(source_scan_id="s", operation="root"), root=True)

    def test_add_collision_raises(self, simple_graph):
        graph, root = simple_graph
        clone = ImageNode(
            id=root.id,
            source_scan_id="s",
            operation="root",
            array=np.zeros((2, 2)),
        )
        with pytest.raises(ValueError):
            graph.add(clone, root=True)

    def test_parents_and_children(self, simple_graph):
        graph, root = simple_graph
        derived = ImageNode(
            source_scan_id="scan-1",
            operation="smooth",
            parent_ids=(root.id,),
        )
        graph.add(derived)
        assert graph.parents(derived.id) == (root,)
        assert graph.children(root.id) == (derived,)

    def test_image_and_measurement_filters(self, simple_graph):
        graph, root = simple_graph
        meas = MeasurementNode(
            source_scan_id="scan-1",
            operation="line_profile",
            parent_ids=(root.id,),
            result={"length_m": 1e-9},
        )
        graph.add(meas)
        assert len(graph.image_nodes()) == 1
        assert len(graph.measurement_nodes()) == 1
        assert graph.measurement_nodes()[0] is meas

    def test_iteration_and_length(self, simple_graph):
        graph, root = simple_graph
        derived = ImageNode(
            source_scan_id="scan-1",
            operation="smooth",
            parent_ids=(root.id,),
        )
        graph.add(derived)
        assert len(graph) == 2
        ids = {n.id for n in graph}
        assert ids == {root.id, derived.id}


class TestMaterialize:
    def test_returns_root_array_directly(self, simple_graph):
        graph, root = simple_graph
        out = materialize_image(graph, root.id, op_registry=lambda name: None)
        assert out is root.array

    def test_walks_single_parent_recipe(self, simple_graph):
        graph, root = simple_graph
        derived = ImageNode(
            source_scan_id="scan-1",
            operation="add_constant",
            parent_ids=(root.id,),
            params={"value": 3.0},
        )
        graph.add(derived)

        def registry(name):
            assert name == "add_constant"
            return lambda arr, value: arr + value

        out = materialize_image(graph, derived.id, op_registry=registry)
        assert np.all(out == 3.0)
        assert derived.array is out

    def test_materialize_caches_result(self, simple_graph):
        graph, root = simple_graph
        calls = {"n": 0}

        def registry(name):
            def op(arr, **_kw):
                calls["n"] += 1
                return arr + 1.0
            return op

        derived = ImageNode(
            source_scan_id="scan-1",
            operation="bump",
            parent_ids=(root.id,),
        )
        graph.add(derived)
        materialize_image(graph, derived.id, op_registry=registry)
        materialize_image(graph, derived.id, op_registry=registry)
        assert calls["n"] == 1

    def test_release_then_materialize_recomputes(self, simple_graph):
        graph, root = simple_graph
        calls = {"n": 0}

        def registry(name):
            def op(arr, **_kw):
                calls["n"] += 1
                return arr + 1.0
            return op

        derived = ImageNode(
            source_scan_id="scan-1",
            operation="bump",
            parent_ids=(root.id,),
        )
        graph.add(derived)
        materialize_image(graph, derived.id, op_registry=registry)
        derived.release()
        materialize_image(graph, derived.id, op_registry=registry)
        assert calls["n"] == 2

    def test_walks_multi_parent_recipe(self, simple_graph):
        graph, root = simple_graph
        root2 = ImageNode(
            source_scan_id="scan-1",
            operation="root",
            array=np.full((4, 4), 5.0),
        )
        graph.add(root2, root=True)
        diff = ImageNode(
            source_scan_id="scan-1",
            operation="subtract",
            parent_ids=(root2.id, root.id),
        )
        graph.add(diff)

        def registry(name):
            return lambda a, b: a - b

        out = materialize_image(graph, diff.id, op_registry=registry)
        assert np.all(out == 5.0)

    def test_recursion_through_chain(self, simple_graph):
        graph, root = simple_graph
        a = ImageNode(
            source_scan_id="scan-1",
            operation="add",
            parent_ids=(root.id,),
            params={"v": 1.0},
        )
        graph.add(a)
        b = ImageNode(
            source_scan_id="scan-1",
            operation="add",
            parent_ids=(a.id,),
            params={"v": 2.0},
        )
        graph.add(b)

        def registry(name):
            return lambda arr, v: arr + v

        out = materialize_image(graph, b.id, op_registry=registry)
        assert np.all(out == 3.0)

    def test_materialize_measurement_raises(self, simple_graph):
        graph, root = simple_graph
        meas = MeasurementNode(
            source_scan_id="scan-1",
            operation="count",
            parent_ids=(root.id,),
            result=42,
        )
        graph.add(meas)
        with pytest.raises(TypeError):
            materialize_image(graph, meas.id, op_registry=lambda name: None)


class TestSerialization:
    def test_roundtrip_preserves_topology(self):
        g = ScanGraph(scan_id="scan-x")
        root = ImageNode(
            source_scan_id="scan-x",
            operation="root",
            units="m",
            array=np.zeros((3, 3)),
        )
        g.add(root, root=True)
        derived = ImageNode(
            source_scan_id="scan-x",
            operation="align_rows",
            parent_ids=(root.id,),
            params={"method": "median"},
            plugin_version="0.1.0",
        )
        g.add(derived)
        meas = MeasurementNode(
            source_scan_id="scan-x",
            operation="line_profile",
            parent_ids=(derived.id,),
            params={"p0": [0, 0], "p1": [1, 1]},
            result={"length_m": 1.4e-9, "samples": [0.0, 0.5, 1.0]},
            units="m",
        )
        g.add(meas)

        payload = graph_to_dict(g)
        g2 = graph_from_dict(payload)

        assert g2.scan_id == "scan-x"
        assert g2.root_ids == (root.id,)
        assert g2.get(root.id).operation == "root"
        assert g2.get(derived.id).params == {"method": "median"}
        assert g2.get(derived.id).plugin_version == "0.1.0"
        assert g2.get(meas.id).result == {
            "length_m": 1.4e-9,
            "samples": [0.0, 0.5, 1.0],
        }
        # Arrays are deliberately dropped during serialization.
        assert g2.get(root.id).array is None
        assert g2.get(derived.id).array is None

    def test_numpy_array_result_is_listified(self):
        g = ScanGraph(scan_id="s")
        root = ImageNode(
            source_scan_id="s",
            operation="root",
            array=np.zeros((2, 2)),
        )
        g.add(root, root=True)
        meas = MeasurementNode(
            source_scan_id="s",
            operation="profile",
            parent_ids=(root.id,),
            result=np.array([0.0, 0.5, 1.0]),
        )
        g.add(meas)
        payload = graph_to_dict(g)
        g2 = graph_from_dict(payload)
        assert g2.get(meas.id).result == [0.0, 0.5, 1.0]
