import numpy as np
import pandas as pd

from graphviz import Graph as DOT_Graph
from itertools import combinations, product
from functools import reduce, partial

class Graph():
    def __init__(self, vertices : int, edges : list[tuple]) -> None:
        self.num_vertices = vertices
        self.num_edges = len(edges)

        self.vertice_values = np.zeros((vertices,), dtype=int)

        self.edge_values = np.zeros((self.num_edges,), dtype=int)
        self.edge_mapping = {f"{v1}-{v2}":edge_idx for edge_idx, (v1, v2) in enumerate(edges)}
        self.edges = edges

        self.adj_matrix = np.zeros((vertices, vertices), dtype=int)
        self.edge_adj_matrix = np.zeros((self.num_edges, self.num_edges), dtype=int)

        for v1, v2 in edges:
            self.adj_matrix[v1][v2] = 1
            self.adj_matrix[v2][v1] = 1

        for e1, e2 in product(edges, repeat=2):
            e1_idx = self.get_edge_index(e1)
            e2_idx = self.get_edge_index(e2)
            if e1_idx == e2_idx:
                continue
            if e1[0] == e2[0] or e1[1] == e2[1] or e1[0] == e2[1] or e1[1] == e2[0]:
                self.edge_adj_matrix[e1_idx][e2_idx] = 1
                self.edge_adj_matrix[e2_idx][e1_idx] = 1
                

    def has_edge(self, v1:int, v2:int) -> bool:
        return True if self.adj_matrix[v1][v2] == 1 else False

    def has_vertex(self, v:int) -> bool:
        return v in self.vertices

    def vertex_order(self, v:int) -> int:
        return np.sum(self.adj_matrix[v])

    def get_vertex_color(self, v:int) -> int:
        return self.vertice_values[v]

    def get_vertex_colors(self) -> list:
        return self.vertice_values

    def get_edge_color(self, v1:int, v2:int):
        if f"{v1}-{v2}" in self.edge_mapping.keys():
            return self.edge_values[self.edge_mapping[f"{v1}-{v2}"]]
        elif f"{v2}-{v1}" in self.edge_mapping.keys():
            return self.edge_values[self.edge_mapping[f"{v2}-{v1}"]]

    def get_edge_index(self, e):
        if f"{e[0]}-{e[1]}" in self.edge_mapping.keys():
            return self.edge_mapping[f"{e[0]}-{e[1]}"]
        elif f"{e[1]}-{e[0]}" in self.edge_mapping.keys():
            return self.edge_mapping[f"{e[1]}-{e[0]}"]

    def get_edge_colors(self):
        return self.edge_values

    def set_vertex_color(self, v:int, color:int) -> None:
        self.vertice_values[v] = color

    def set_vertex_colors(self, colors:list) -> None:
        if len(colors) != self.num_vertices:
            raise ValueError("Incorrect number of colors")
        self.vertice_values = colors

    def set_edge_color(self, v1:int, v2:int, color:int):
        if f"{v1}-{v2}" in self.edge_mapping.keys():
            self.edge_values[self.edge_mapping[f"{v1}-{v2}"]] = color
        elif f"{v2}-{v1}" in self.edge_mapping.keys():
            self.edge_values[self.edge_mapping[f"{v2}-{v1}"]] = color

    def set_edge_colors(self, colors:list[tuple]):
        if len(colors) != self.num_edges:
            raise ValueError("Incorrect number of colors")
        self.edge_values = colors

    def get_adjacent_vertices(self) -> np.ndarray:
        return np.argwhere(self.adj_matrix)

    def get_vertices_adjacent_to_vertex(self, v:int):
        return np.argwhere(self.adj_matrix[v])

    def __get_adj_matrix(self):
        return self.adj_matrix

    def get_conflicted_vertex_pairs(self):
        conflict_pairs = []
        for v1, v2 in self.get_adjacent_vertices():
            if self.get_vertex_color(v1) == self.get_vertex_color(v2):
                if not edge_in((v1, v2), conflict_pairs):
                    conflict_pairs.append((v1, v2))

        return conflict_pairs

    def get_vertices_in_conflict(self):
        conflicts = []
        for v1, v2 in self.get_adjacent_vertices():
            if self.get_vertex_color(v1) == self.get_vertex_color(v2):
                if v1 not in conflicts:
                    conflicts.append(v1)
                if v2 not in conflicts:
                    conflicts.append(v2)
        return conflicts

    def num_vertex_conflicts(self):
        return len(self.get_vertices_in_conflict())

    def __get_edges_connected_to_vertex(self, v:int):
        return [(v, v2[0]) for v2 in np.argwhere(self.adj_matrix[v])]

    def get_adjacent_edge_indexes(self):
        return np.argwhere(self.edge_adj_matrix)

    def get_edges_in_conflict(self):
        conflicts = []
        for e1, e2 in self.get_adjacent_edge_indexes():
            if self.edge_values[e1] == self.edge_values[e2]:
                if e1 not in conflicts:
                    conflicts.append(e1)
                if e2 not in conflicts:
                    conflicts.append(e2)
        return conflicts

    def num_edge_conflicts(self):
        return len(self.get_edges_in_conflict())


    def get_dot(self):
        dot = DOT_Graph()
        for v, val in enumerate(self.vertice_values):
            dot.node(str(v), f"{str(v)}: {str(val)}")

        edges = map(lambda x : (str(x[0]), str(x[1])), self.edges)
        for e in edges:
            dot.edge(e[0], e[1], label=str(f"{self.get_edge_index(e)}: {self.get_edge_color(e[0], e[1])}"))
        # dot.edges(edges)

        # dot.render('Machine.gv.pdf', view=True)
        return dot

def edge_eq(e1, e2):
    return (e1[0] == e2[0] and e1[1] == e2[1]) or (e1[0] == e2[1] and e1[1] == e2[0])

def edge_in(e, edges):
    return reduce(lambda x, y : x or y, map(partial(edge_eq, e), edges), False)


if __name__ == '__main__':
    vs = 5
    edges = [(0,1),(2,3),(3,1),(0,4),(3,4)]
    g = Graph(vs, edges)
    g.set_vertex_colors([0,1,1,1,2])
    g.set_edge_colors([0, 1, 1, 3, 5])
    print(g.get_edges_in_conflict())
    g.get_dot().render('Machine.gv.pdf', view=True)