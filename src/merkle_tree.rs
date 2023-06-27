use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use crate::hash_map::HashMap;

type Digest = [u8; 8];

const DEFAULT_LEAF: Digest = [0; 8];

/// Space complexity: O(n)
pub struct MerkleTree {
    height: usize,
    values: HashMap<usize, Digest>,
    default_nodes: Vec<Digest>,
}

impl MerkleTree {
    pub fn new(height: usize) -> Self {
        Self::with_default_leaf(DEFAULT_LEAF, height)
    }

    pub fn with_default_leaf(default_leaf: Digest, height: usize) -> Self {
        let default_nodes = default_nodes(default_leaf, height);
        let default_root = default_nodes[height];

        let mut values = HashMap::new();
        values.insert(1, default_root);

        Self {
            height,
            values,
            default_nodes,
        }
    }

    /// Time complexity: O(1)
    pub fn root(&self) -> Digest {
        *self.values.get(&1).unwrap()
    }

    /// Time complexity: O(log n)
    pub fn insert(&mut self, index: usize, value: impl AsRef<[u8]>) -> Result<Digest, Error> {
        self.bounds_check(index)?;

        let leaf_index = self.leaf_index(index);
        let leaf = hash(value);
        self.values.insert(leaf_index, leaf);

        let mut node_index = self.path_index_at_height(index, 0);
        let mut node = leaf;

        for height in 0..self.height {
            let sibling_index = sibling_index(node_index);
            let sibling = *self
                .values
                .get(&sibling_index)
                .unwrap_or(&self.default_nodes[height]);

            let parent_index = self.path_index_at_height(index, height + 1);
            let parent = hash_pair(node, sibling);
            self.values.insert(parent_index, parent);

            node_index = parent_index;
            node = parent;
        }

        let root = node;

        Ok(root)
    }

    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn path(&self, index: usize) -> Result<Vec<usize>, Error> {
        self.bounds_check(index)?;

        let sibling_indices = (0..self.height)
            .map(|height| {
                let node_index = self.path_index_at_height(index, height);
                sibling_index(node_index)
            })
            .collect();

        Ok(sibling_indices)
    }

    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn prove(&self, index: usize) -> Result<Vec<Digest>, Error> {
        let path = self.path(index)?;

        let proof = path
            .iter()
            .zip(&self.default_nodes)
            .map(|(index, default_node)| *self.values.get(index).unwrap_or(default_node))
            .collect();

        Ok(proof)
    }

    /// Time complexity: O(log n)
    pub fn validate(&self, value: impl AsRef<[u8]>, proof: Vec<Digest>) -> Result<bool, Error> {
        if proof.len() != self.height {
            return Err(Error::IncorrectProofLength {
                len: proof.len(),
                height: self.height,
            });
        }

        let mut node = hash(value);
        for sibling in proof {
            node = hash_pair(node, sibling);
        }

        let proof_root = node;

        Ok(proof_root == self.root())
    }

    /// Time complexity: O(n)
    /// Space complexity: O(n)
    ///
    /// Returns the indicies whose value is `value`.
    pub fn indicies_of(&self, value: impl AsRef<[u8]>) -> Option<Vec<usize>> {
        let node = hash(value);

        let first_index = self.leaf_index(0);

        let mut indexes = self
            .values
            .iter()
            .filter(|(k, v)| **k > first_index && **v == node)
            .map(|(k, _)| *k - first_index)
            .collect::<Vec<usize>>();

        indexes.sort_unstable();

        if !indexes.is_empty() {
            Some(indexes)
        } else {
            None
        }
    }

    /// Time complexity: O(n)
    pub fn contains(&self, value: impl AsRef<[u8]>) -> bool {
        self.indicies_of(value).is_some()
    }

    fn leaf_index(&self, index: usize) -> usize {
        len_at_height(self.height) + index
    }

    fn path_index_at_height(&self, index: usize, height: usize) -> usize {
        let leaf_index = self.leaf_index(index);
        leaf_index / len_at_height(height)
    }

    fn bounds_check(&self, index: usize) -> Result<(), Error> {
        let len = len_at_height(self.height);
        if index >= len {
            Err(Error::IndexOutOfBounds { index, len })
        } else {
            Ok(())
        }
    }
}

fn hash(x: impl AsRef<[u8]>) -> Digest {
    let mut hasher = DefaultHasher::new();
    hasher.write(x.as_ref());
    hasher.finish().to_le_bytes()
}

fn hash_pair<T>(x: T, y: T) -> Digest
where
    T: AsRef<[u8]> + Ord,
{
    let (x, y) = if x < y { (x, y) } else { (y, x) };

    let mut hasher = DefaultHasher::new();
    hasher.write(y.as_ref());
    hasher.write(x.as_ref());
    hasher.finish().to_le_bytes()
}

fn default_nodes(default_leaf: Digest, height: usize) -> Vec<Digest> {
    let mut default_node = default_leaf;
    let mut default_nodes = vec![default_node];

    for _ in 0..height {
        default_node = hash_pair(default_node, default_node);
        default_nodes.push(default_node);
    }

    default_nodes
}

fn sibling_index(index: usize) -> usize {
    if index % 2 == 0 {
        index + 1
    } else {
        index - 1
    }
}

fn len_at_height(height: usize) -> usize {
    2usize.pow(height as u32)
}

#[derive(Debug, PartialEq)]
pub enum Error {
    IndexOutOfBounds { len: usize, index: usize },
    IncorrectProofLength { len: usize, height: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_nodes_height_0() {
        let nodes = default_nodes(DEFAULT_LEAF, 0);
        let expected_nodes = vec![DEFAULT_LEAF];
        assert_eq!(nodes, expected_nodes);
    }

    #[test]
    fn default_nodes_height_1() {
        let nodes = default_nodes(DEFAULT_LEAF, 1);
        let height_1_node = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let expected_nodes = vec![DEFAULT_LEAF, height_1_node];
        assert_eq!(nodes, expected_nodes);
    }

    #[test]
    fn default_nodes_height_2() {
        let nodes = default_nodes(DEFAULT_LEAF, 2);
        let height_1_node = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let height_2_node = hash_pair(height_1_node, height_1_node);
        let expected_nodes = vec![DEFAULT_LEAF, height_1_node, height_2_node];
        assert_eq!(nodes, expected_nodes);
    }

    #[test]
    fn empty_tree_height_0() {
        let tree = MerkleTree::new(0);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 0).pop().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_1() {
        let tree = MerkleTree::new(1);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 1).pop().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_32() {
        let tree = MerkleTree::new(32);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 32).pop().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_0() {
        let mut tree = MerkleTree::new(0);
        let root = tree.insert(0, "a").unwrap();
        let expected_root = hash("a");
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_1() {
        let mut tree = MerkleTree::new(1);

        let root = tree.insert(0, "a").unwrap();
        let expected_root = hash_pair(hash("a"), DEFAULT_LEAF);
        assert_eq!(root, expected_root);

        let root = tree.insert(1, "b").unwrap();
        let expected_root = hash_pair(hash("a"), hash("b"));
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_2() {
        let mut tree = MerkleTree::new(2);

        let root = tree.insert(0, "a").unwrap();
        let height_1_node_0 = hash_pair(hash("a"), DEFAULT_LEAF);
        let height_1_node_1 = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let expected_root = hash_pair(height_1_node_0, height_1_node_1);
        assert_eq!(root, expected_root);

        let root = tree.insert(1, "b").unwrap();
        let height_1_node_0 = hash_pair(hash("a"), hash("b"));
        let height_1_node_1 = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let expected_root = hash_pair(height_1_node_0, height_1_node_1);
        assert_eq!(root, expected_root);

        let root = tree.insert(2, "c").unwrap();
        let height_1_node_0 = hash_pair(hash("a"), hash("b"));
        let height_1_node_1 = hash_pair(hash("c"), DEFAULT_LEAF);
        let expected_root = hash_pair(height_1_node_0, height_1_node_1);
        assert_eq!(root, expected_root);

        let root = tree.insert(3, "d").unwrap();
        let height_1_node_0 = hash_pair(hash("a"), hash("b"));
        let height_1_node_1 = hash_pair(hash("c"), hash("d"));
        let expected_root = hash_pair(height_1_node_0, height_1_node_1);
        assert_eq!(root, expected_root);
    }

    #[test]
    fn tree_index_bounds_check() {
        assert_eq!(
            MerkleTree::new(1).insert(2, "should_error").unwrap_err(),
            Error::IndexOutOfBounds { len: 2, index: 2 }
        );

        assert_eq!(
            MerkleTree::new(2).insert(4, "should_error").unwrap_err(),
            Error::IndexOutOfBounds { len: 4, index: 4 }
        );
    }

    #[test]
    fn proof_height_1() {
        let mut tree = MerkleTree::new(1);

        tree.insert(0, "a").unwrap();
        let proof = tree.prove(0).unwrap();
        let expected_proof = vec![DEFAULT_LEAF];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("a", proof).unwrap());

        tree.insert(1, "b").unwrap();
        let proof = tree.prove(1).unwrap();
        let expected_proof = vec![hash("a")];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("b", proof).unwrap());
    }

    #[test]
    fn proof_height_2() {
        let mut tree = MerkleTree::new(2);

        tree.insert(3, "d").unwrap();
        let proof = tree.prove(3).unwrap();
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, DEFAULT_LEAF)];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("d", proof).unwrap());

        tree.insert(1, "b").unwrap();
        let proof = tree.prove(1).unwrap();
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, hash("d"))];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("b", proof).unwrap());

        tree.insert(2, "c").unwrap();
        let proof = tree.prove(2).unwrap();
        let expected_proof = vec![hash("d"), hash_pair(DEFAULT_LEAF, hash("b"))];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("c", proof).unwrap());

        tree.insert(0, "a").unwrap();
        let proof = tree.prove(0).unwrap();
        let expected_proof = vec![hash("b"), hash_pair(hash("c"), hash("d"))];
        assert_eq!(proof, expected_proof);
        assert!(tree.validate("a", proof).unwrap());
    }

    #[test]
    fn path_height_2() {
        // Indices:
        //    1
        //  2   3
        // 4 5 6 7
        let tree = MerkleTree::new(2);

        let proof = tree.path(0).unwrap();
        assert_eq!(proof, [5, 3]);

        let proof = tree.path(1).unwrap();
        assert_eq!(proof, [4, 3]);

        let proof = tree.path(2).unwrap();
        assert_eq!(proof, [7, 2]);

        let proof = tree.path(3).unwrap();
        assert_eq!(proof, [6, 2]);
    }

    #[test]
    fn path_height_3() {
        // Indices:
        //     1
        //   2   3
        //  4 5 6 7
        // 8 9...14 15
        let tree = MerkleTree::new(3);

        let proof = tree.path(0).unwrap();
        assert_eq!(proof, [9, 5, 3]);

        let proof = tree.path(7).unwrap();
        assert_eq!(proof, [14, 6, 2]);
    }

    #[test]
    fn proof_bounds_check() {
        assert_eq!(
            MerkleTree::new(1).prove(2).unwrap_err(),
            Error::IndexOutOfBounds { len: 2, index: 2 }
        );

        assert_eq!(
            MerkleTree::new(2).prove(4).unwrap_err(),
            Error::IndexOutOfBounds { len: 4, index: 4 }
        );
    }

    #[test]
    fn proof_length_check() {
        assert_eq!(
            MerkleTree::new(1)
                .validate("empty_proof", vec![])
                .unwrap_err(),
            Error::IncorrectProofLength { len: 0, height: 1 }
        );

        assert_eq!(
            MerkleTree::new(1)
                .validate("empty_proof", vec![DEFAULT_LEAF, DEFAULT_LEAF])
                .unwrap_err(),
            Error::IncorrectProofLength { len: 2, height: 1 }
        );
    }

    #[test]
    fn indicies_of() {
        let mut tree = MerkleTree::new(2);

        tree.insert(1, "a").unwrap();
        let indexes = tree.indicies_of("a").unwrap();
        let expected_indexes = vec![1];
        assert_eq!(indexes, expected_indexes);

        tree.insert(3, "a").unwrap();
        let indexes = tree.indicies_of("a").unwrap();
        let expected_indexes = vec![1, 3];
        assert_eq!(indexes, expected_indexes);

        assert!(tree.indicies_of("not_in").is_none());
    }

    #[test]
    fn contains() {
        let mut tree = MerkleTree::new(2);

        assert!(!tree.contains("a"));

        tree.insert(3, "a").unwrap();
        assert!(tree.contains("a"));

        tree.insert(0, "a").unwrap();
        assert!(tree.contains("a"));
    }
}
