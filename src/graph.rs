use std::{
    collections::{HashMap, HashSet, VecDeque},
    hash::Hash,
};

/// Space complexity: O(V+E) on average, O(V^2) worst case
#[derive(Clone, Debug)]
pub struct Graph<T> {
    edges: HashMap<T, HashSet<T>>, // Adjacency list
}

impl<T> Graph<T> {
    pub fn new() -> Graph<T> {
        Graph {
            edges: HashMap::new(),
        }
    }
}

impl<'a, T> Graph<T>
where
    T: Hash + Eq + Copy,
{
    /// Time complexity: O(1)
    pub fn add_vertex(&mut self, v: T) {
        if !self.has_vertex(&v) {
            self.edges.insert(v, HashSet::new());
        }
    }

    /// Time complexity: O(1)
    pub fn add_edge(&mut self, u: T, v: T) {
        self.add_vertex(u);
        self.add_vertex(v);
        self.add_directed_edge(&u, v);
        self.add_directed_edge(&v, u);
    }

    /// Time complexity: O(1)
    fn add_directed_edge(&mut self, from: &T, to: T) {
        if let Some(s) = self.edges.get_mut(from) {
            s.insert(to);
        }
    }

    /// Time complexity: O(E)
    pub fn remove_vertex(&mut self, v: &T) {
        if let Some(us) = self.edges.remove(v) {
            for u in us {
                self.remove_directed_edge(&u, v);
            }
        }
    }

    /// Time complexity: O(1)
    pub fn remove_edge(&mut self, u: &T, v: &T) {
        self.remove_directed_edge(u, v);
        self.remove_directed_edge(v, u);
    }

    /// Time complexity: O(1)
    fn remove_directed_edge(&mut self, from: &T, to: &T) {
        if let Some(s) = self.edges.get_mut(from) {
            s.remove(to);
        }
    }

    /// Time complexity: O(1)
    pub fn has_vertex(&self, v: &T) -> bool {
        self.edges.contains_key(v)
    }

    /// Time complexity: O(1)
    pub fn has_edge(&self, u: &T, v: &T) -> bool {
        if let Some(s) = self.edges.get(u) {
            s.contains(v)
        } else {
            false
        }
    }

    /// Time complexity: O(V)
    pub fn vertices(&self) -> VerticesIterator<T> {
        VerticesIterator::new(self)
    }

    /// Time complexity: O(V+E)
    pub fn edges(&self) -> EdgesIterator<T> {
        EdgesIterator::new(self)
    }

    /// Time complexity: O(E)
    pub fn neighbours(&self, v: &T) -> NeighboursIterator<T> {
        NeighboursIterator::new(self, v)
    }

    /// Time complexity: O(V+E)
    pub fn bfs(&'a self, start: &'a T) -> BFSIterator<T> {
        BFSIterator::new(self, start)
    }

    /// Time complexity: O(V+E)
    pub fn dfs(&'a self, start: &'a T) -> DFSIterator<T> {
        DFSIterator::new(self, start)
    }

    /// Time complexity: O(V+E)
    pub fn has_path(&self, u: &T, v: &T) -> bool {
        for w in self.bfs(u) {
            if w == v {
                return true;
            }
        }
        false
    }
}

pub struct NeighboursIterator<'a, T>(Option<std::collections::hash_set::Iter<'a, T>>);

impl<'a, T> NeighboursIterator<'a, T>
where
    T: Eq + Hash,
{
    pub fn new(graph: &'a Graph<T>, vertex: &T) -> Self {
        NeighboursIterator {
            0: match graph.edges.get(vertex) {
                Some(neighbours) => Some(neighbours.iter()),
                None => None,
            },
        }
    }
}

impl<'a, T> Iterator for NeighboursIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            Some(iter) => iter.next(),
            None => None,
        }
    }
}

pub struct VerticesIterator<'a, T>(std::collections::hash_map::Iter<'a, T, HashSet<T>>);

impl<'a, T> VerticesIterator<'a, T>
where
    T: Eq + Hash,
{
    pub fn new(graph: &'a Graph<T>) -> Self {
        VerticesIterator {
            0: graph.edges.iter(),
        }
    }
}

impl<'a, T> Iterator for VerticesIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(v, _)| v)
    }
}

pub struct EdgesIterator<'a, T> {
    graph: &'a Graph<T>,
    vertices: VerticesIterator<'a, T>,
    vertex: Option<&'a T>,
    neighbours: NeighboursIterator<'a, T>,
}

impl<'a, T> EdgesIterator<'a, T>
where
    T: Eq + Hash,
{
    pub fn new(graph: &'a Graph<T>) -> Self {
        EdgesIterator {
            graph,
            vertices: VerticesIterator::new(graph),
            vertex: None,
            neighbours: NeighboursIterator(None),
        }
    }
}

impl<'a, T> Iterator for EdgesIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    type Item = (&'a T, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.neighbours.next() {
                Some(u) => return Some((self.vertex.unwrap(), u)),
                None => match self.vertices.next() {
                    Some(v) => {
                        self.vertex = Some(v);
                        self.neighbours = NeighboursIterator::new(self.graph, v);
                    }
                    None => return None,
                },
            }
        }
    }
}

pub struct BFSIterator<'a, T> {
    graph: &'a Graph<T>,
    queue: VecDeque<&'a T>,
    visited: HashSet<&'a T>,
}

impl<'a, T> BFSIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    pub fn new(graph: &'a Graph<T>, start: &'a T) -> Self {
        BFSIterator {
            graph,
            queue: VecDeque::from([start]),
            visited: HashSet::from([start]),
        }
    }
}

impl<'a, T> Iterator for BFSIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.queue.pop_front() {
            Some(v) => {
                if !self.graph.has_vertex(v) {
                    None
                } else {
                    for u in self.graph.neighbours(&v) {
                        if !self.visited.contains(&u) {
                            self.visited.insert(u);
                            self.queue.push_back(u);
                        }
                    }
                    Some(v)
                }
            }
            None => None,
        }
    }
}

pub struct DFSIterator<'a, T> {
    graph: &'a Graph<T>,
    stack: Vec<&'a T>,
    visited: HashSet<&'a T>,
}

impl<'a, T> DFSIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    pub fn new(graph: &'a Graph<T>, start: &'a T) -> Self {
        DFSIterator {
            graph,
            stack: vec![start],
            visited: HashSet::from([start]),
        }
    }
}

impl<'a, T> Iterator for DFSIterator<'a, T>
where
    T: Hash + Eq + Copy,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            Some(v) => {
                if !self.graph.has_vertex(v) {
                    None
                } else {
                    for u in self.graph.neighbours(&v) {
                        if !self.visited.contains(&u) {
                            self.visited.insert(u);
                            self.stack.push(u);
                        }
                    }
                    Some(v)
                }
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_vertex_exists() {
        let g = Graph {
            edges: HashMap::from([(1, HashSet::new())]),
        };
        assert!(g.has_vertex(&1));
    }

    #[test]
    fn test_has_vertex_not_exists() {
        let g = Graph::new();
        assert!(!g.has_vertex(&1));
    }

    #[test]
    fn test_has_edge_exists() {
        let g = Graph {
            edges: HashMap::from([(1, HashSet::from([2])), (2, HashSet::from([1]))]),
        };
        assert!(g.has_edge(&1, &2));
    }

    #[test]
    fn test_has_edge_one_vertex_exists() {
        let g = Graph {
            edges: HashMap::from([(1, HashSet::new())]),
        };
        assert!(!g.has_edge(&1, &2));
    }

    #[test]
    fn test_has_edge_no_edge_neither_vertex_exists() {
        let g = Graph::new();
        assert!(!g.has_edge(&1, &2));
    }

    #[test]
    fn test_add_vertex() {
        let mut g = Graph::new();
        g.add_vertex(1);
        assert!(g.has_vertex(&1));
    }

    #[test]
    fn test_add_edge_both_vertices_exist() {
        let mut g = Graph::new();
        g.add_vertex(1);
        g.add_vertex(2);
        g.add_edge(1, 2);
        assert!(g.has_edge(&1, &2));
    }

    #[test]
    fn test_add_edge_one_vertex_exist() {
        let mut g = Graph::new();
        g.add_vertex(1);
        g.add_edge(1, 2);
        assert!(g.has_edge(&1, &2));
    }

    #[test]
    fn test_add_edge_vertices_do_not_exist() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        assert!(g.has_edge(&1, &2));
    }

    #[test]
    fn test_remove_vertex_singleton() {
        let mut g = Graph::new();
        g.add_vertex(1);
        g.remove_vertex(&1);
        assert!(!g.has_vertex(&1));
    }

    #[test]
    fn test_remove_vertex_connected() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.remove_vertex(&1);
        assert!(!g.has_vertex(&1));
        assert!(g.has_vertex(&2));
        assert!(g.has_vertex(&3));
    }

    #[test]
    fn test_remove_edge_exists() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.remove_edge(&1, &2);
        assert!(!g.has_edge(&1, &2));
        assert!(g.has_vertex(&1));
        assert!(g.has_vertex(&2));
    }

    #[test]
    fn test_remove_edge_not_exists() {
        let mut g = Graph::new();
        g.remove_edge(&1, &2);
        assert!(!g.has_edge(&1, &2));
    }

    #[test]
    fn test_vertices() {
        let mut g = Graph::new();
        assert_eq!(g.vertices().count(), 0);
        g.add_vertex(1);
        assert_eq!(g.vertices().count(), 1);
        g.add_edge(1, 2);
        assert_eq!(g.vertices().count(), 2);
        g.add_edge(3, 4);
        assert_eq!(g.vertices().count(), 4);
    }

    #[test]
    fn test_edges() {
        let mut g = Graph::new();
        assert_eq!(g.edges().count(), 0);
        g.add_vertex(1);
        assert_eq!(g.edges().count(), 0);
        g.add_vertex(2);
        assert_eq!(g.edges().count(), 0);
        g.add_edge(1, 2);
        assert_eq!(g.edges().count(), 2);
        g.add_edge(1, 3);
        assert_eq!(g.edges().count(), 4);
        g.add_edge(4, 5);
        assert_eq!(g.edges().count(), 6);
        g.add_edge(5, 6);
        assert_eq!(g.edges().count(), 8);
        g.add_edge(4, 6);
        assert_eq!(g.edges().count(), 10);
        g.remove_edge(&1, &2);
        assert_eq!(g.edges().count(), 8);
        g.remove_edge(&1, &3);
        assert_eq!(g.edges().count(), 6);
        g.remove_edge(&4, &5);
        assert_eq!(g.edges().count(), 4);
        g.remove_edge(&4, &6);
        assert_eq!(g.edges().count(), 2);
        g.remove_edge(&5, &6);
        assert_eq!(g.edges().count(), 0);
    }

    #[test]
    fn test_neighbours_vertex_connected() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        let mut n = g.neighbours(&1).collect::<Vec<_>>();
        n.sort();
        assert_eq!(n, vec![&2, &3]);
    }

    #[test]
    fn test_neighbours_vertex_singleton() {
        let mut g = Graph::new();
        g.add_vertex(1);
        let n = g.neighbours(&1).collect::<Vec<_>>();
        assert!(n.is_empty());
    }

    #[test]
    fn test_neighbours_vertex_not_exists() {
        let g = Graph::new();
        let n = g.neighbours(&1).collect::<Vec<_>>();
        assert!(n.is_empty());
    }

    #[test]
    fn test_bfs_singleton() {
        let mut g = Graph::new();
        g.add_vertex(1);
        let vs = g.bfs(&1).collect::<Vec<_>>();
        assert_eq!(vs, vec![&1]);
    }

    #[test]
    fn test_bfs_start_vertex_does_not_exist() {
        let g = Graph::new();
        let vs = g.bfs(&1).collect::<Vec<_>>();
        assert!(vs.is_empty());
    }

    #[test]
    fn test_bfs_pair() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2]);
    }

    #[test]
    fn test_bfs_path() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_bfs_star() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(1, 5);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5]);
    }

    #[test]
    fn test_bfs_star_and_paths() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(2, 5);
        g.add_edge(3, 6);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5, &6]);
    }

    #[test]
    fn test_bfs_cycle() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 1);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3]);
    }

    #[test]
    fn test_bfs_star_cycle_and_path() {
        let mut g = Graph::new();
        // Star
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        // Cycle
        g.add_edge(2, 3);
        g.add_edge(2, 4);
        g.add_edge(3, 4);
        // Path
        g.add_edge(2, 5);
        g.add_edge(5, 6);
        g.add_edge(6, 7);
        let mut vs = g.bfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5, &6, &7]);
    }

    #[test]
    fn test_dfs_singleton() {
        let mut g = Graph::new();
        g.add_vertex(1);
        let vs = g.dfs(&1).collect::<Vec<_>>();
        assert_eq!(vs, vec![&1]);
    }

    #[test]
    fn test_dfs_start_vertex_does_not_exist() {
        let g = Graph::new();
        let vs = g.dfs(&1).collect::<Vec<_>>();
        assert!(vs.is_empty());
    }

    #[test]
    fn test_dfs_pair() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2]);
    }

    #[test]
    fn test_dfs_path() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4]);
    }

    #[test]
    fn test_dfs_star() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(1, 5);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5]);
    }

    #[test]
    fn test_dfs_star_and_paths() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        g.add_edge(2, 5);
        g.add_edge(3, 6);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5, &6]);
    }

    #[test]
    fn test_dfs_cycle() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 1);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3]);
    }

    #[test]
    fn test_dfs_star_cycle_and_path() {
        let mut g = Graph::new();
        // Star
        g.add_edge(1, 2);
        g.add_edge(1, 3);
        g.add_edge(1, 4);
        // Cycle
        g.add_edge(2, 3);
        g.add_edge(2, 4);
        g.add_edge(3, 4);
        // Path
        g.add_edge(2, 5);
        g.add_edge(5, 6);
        g.add_edge(6, 7);
        let mut vs = g.dfs(&1).collect::<Vec<_>>();
        vs.sort();
        assert_eq!(vs, vec![&1, &2, &3, &4, &5, &6, &7]);
    }

    #[test]
    fn test_has_path_pair() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        assert!(g.has_path(&1, &2));
    }

    #[test]
    fn test_has_path_path() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        assert!(g.has_path(&1, &4));
    }

    #[test]
    fn test_has_path_target_vertex_does_not_exist() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        assert!(!g.has_path(&1, &3));
    }
}
