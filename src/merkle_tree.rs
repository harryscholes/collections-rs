use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use crate::{
    binary_tree::{first_index_at_height, BinaryTree},
    vector::Vector,
};

type Digest = [u8; 8];

const DEFAULT_LEAF: Digest = [0; 8];

/// Space complexity: O(n)
pub struct MerkleTree {
    tree: BinaryTree<Digest>,
    default_nodes: Vector<Digest>,
}

impl MerkleTree {
    pub fn new(height: usize) -> Self {
        Self::with_default_leaf(DEFAULT_LEAF, height)
    }

    pub fn with_default_leaf(default_leaf: Digest, height: usize) -> Self {
        let tree = BinaryTree::with_height(height);
        let default_nodes = default_nodes(default_leaf, height);

        Self {
            tree,
            default_nodes,
        }
    }

    /// Time complexity: O(1)
    pub fn root(&self) -> Digest {
        *self.tree.get(0).unwrap_or(&self.default_nodes[0])
    }

    /// Time complexity: O(log n)
    pub fn insert(&mut self, index: usize, value: impl AsRef<[u8]>) -> Result<Digest, Error> {
        self.bounds_check(index)?;

        let tree_index = self.tree_index(index);
        let leaf_node = hash(value);
        self.tree.insert(tree_index, leaf_node);

        let mut internal_index = tree_index;
        let mut internal_node = leaf_node;
        let mut internal_height = self.tree.height();

        while internal_height > 1 {
            let sibling_index = sibling_index(internal_index);
            let sibling_node = *self
                .tree
                .get(sibling_index)
                .unwrap_or(&self.default_nodes[internal_height - 1]);

            let parent_index = (internal_index - 1) / 2;
            let parent_node = hash_pair(internal_node, sibling_node);
            self.tree.insert(parent_index, parent_node);

            internal_index = parent_index;
            internal_node = parent_node;
            internal_height -= 1;
        }

        Ok(self.root())
    }

    /// Merkle path from leaf to root.
    ///
    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn path(&self, index: usize) -> Result<Vector<usize>, Error> {
        self.bounds_check(index)?;

        let tree_index = self.tree_index(index);
        let mut path = Vector::with_capacity(self.tree.height() - 1);

        let mut internal_index = tree_index;
        let mut internal_height = self.tree.height();

        while internal_height > 1 {
            dbg!(internal_index, internal_height);
            let sibling_index = sibling_index(internal_index);
            path.push_back(sibling_index);
            internal_index = (internal_index - 1) / 2;
            internal_height -= 1;
        }

        Ok(path)
    }

    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn prove(&self, index: usize) -> Result<Vector<Digest>, Error> {
        let path = self.path(index)?;

        let mut proof = Vector::with_capacity(path.len());

        let mut internal_height = self.tree.height();

        for internal_index in path {
            let internal_node = self
                .tree
                .get(internal_index)
                .unwrap_or(&self.default_nodes[internal_height - 1]);
            proof.push_back(*internal_node);
            internal_height -= 1;
        }

        Ok(proof)
    }

    /// Time complexity: O(log n)
    pub fn validate(&self, value: impl AsRef<[u8]>, proof: Vector<Digest>) -> Result<bool, Error> {
        let len = proof.len();
        let expected_len = self.tree.height() - 1;
        if len != expected_len {
            return Err(Error::IncorrectProofLength { len, expected_len });
        }

        let mut internal_node = hash(value);
        for sibling_node in proof {
            internal_node = hash_pair(internal_node, sibling_node);
        }

        let proof_root = internal_node;

        Ok(proof_root == self.root())
    }

    /// Returns the indicies whose value is `value`.
    ///
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    pub fn indicies_of(&self, value: impl AsRef<[u8]>) -> Option<Vector<usize>> {
        let leaf_node = hash(value);

        let indices = self
            .tree
            .leaf_iter()
            .enumerate()
            .filter(|(_, v)| {
                if let Some(v) = v {
                    **v == leaf_node
                } else {
                    false
                }
            })
            .map(|(i, _)| i)
            .collect::<Vector<usize>>();

        if indices.is_empty() {
            None
        } else {
            Some(indices)
        }
    }

    /// Time complexity: O(n)
    pub fn contains(&self, value: impl AsRef<[u8]>) -> bool {
        self.indicies_of(value).is_some()
    }

    fn tree_index(&self, index: usize) -> usize {
        first_index_at_height(self.tree.height()) + index
    }

    fn len(&self) -> usize {
        if self.tree.height() == 0 {
            0
        } else {
            first_index_at_height(self.tree.height() + 1)
                - first_index_at_height(self.tree.height())
        }
    }

    fn bounds_check(&self, index: usize) -> Result<(), Error> {
        let len = self.len();
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

/// The default node at height `h` is `default_nodes[h-1]`
fn default_nodes(default_leaf: Digest, height: usize) -> Vector<Digest> {
    let mut default_nodes = vec![];
    let mut internal_node = default_leaf;

    for _ in 0..height {
        default_nodes.push(internal_node);
        internal_node = hash_pair(internal_node, internal_node);
    }

    default_nodes.reverse();

    default_nodes.into()
}

fn sibling_index(index: usize) -> usize {
    assert!(index > 0);
    if index % 2 == 1 {
        index + 1
    } else {
        index - 1
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    IndexOutOfBounds { len: usize, index: usize },
    IncorrectProofLength { len: usize, expected_len: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_nodes_height_0() {
        let nodes = default_nodes(DEFAULT_LEAF, 0);
        assert!(nodes.is_empty());
    }

    #[test]
    fn default_nodes_height_1() {
        let nodes = default_nodes(DEFAULT_LEAF, 1);
        let expected_nodes = vec![DEFAULT_LEAF];
        assert_eq!(nodes, expected_nodes.into());
    }

    #[test]
    fn default_nodes_height_2() {
        let nodes = default_nodes(DEFAULT_LEAF, 2);
        let height_1_node = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let expected_nodes = vec![height_1_node, DEFAULT_LEAF];
        assert_eq!(nodes, expected_nodes.into());
    }

    #[test]
    fn default_nodes_height_3() {
        let nodes = default_nodes(DEFAULT_LEAF, 3);
        let height_2_node = hash_pair(DEFAULT_LEAF, DEFAULT_LEAF);
        let height_1_node = hash_pair(height_2_node, height_2_node);
        let expected_nodes = vec![height_1_node, height_2_node, DEFAULT_LEAF];
        assert_eq!(nodes, expected_nodes.into());
    }

    #[test]
    #[should_panic]
    fn empty_tree_height_0() {
        let tree = MerkleTree::new(0);
        tree.root();
    }

    #[test]
    fn empty_tree_height_1() {
        let tree = MerkleTree::new(1);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 1).pop_front().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_2() {
        let tree = MerkleTree::new(2);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 2).pop_front().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_32() {
        let tree = MerkleTree::new(32);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 32).pop_front().unwrap();
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_1() {
        let mut tree = MerkleTree::new(1);
        let root = tree.insert(0, "a").unwrap();
        let expected_root = hash("a");
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_2() {
        let mut tree = MerkleTree::new(2);

        let root = tree.insert(0, "a").unwrap();
        let expected_root = hash_pair(hash("a"), DEFAULT_LEAF);
        assert_eq!(root, expected_root);

        let root = tree.insert(1, "b").unwrap();
        let expected_root = hash_pair(hash("a"), hash("b"));
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_3() {
        let mut tree = MerkleTree::new(3);

        let root = tree.insert(0, "a").unwrap();
        assert_eq!(
            *tree.tree.get(1).unwrap(),
            hash_pair(hash("a"), DEFAULT_LEAF)
        );
        assert!(tree.tree.get(2).is_none());
        assert_eq!(
            root,
            hash_pair(
                hash_pair(hash("a"), DEFAULT_LEAF),
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF),
            )
        );

        let root = tree.insert(1, "b").unwrap();
        assert_eq!(*tree.tree.get(1).unwrap(), hash_pair(hash("a"), hash("b")));
        assert!(tree.tree.get(2).is_none());
        assert_eq!(
            root,
            hash_pair(
                hash_pair(hash("a"), hash("b")),
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF),
            )
        );

        let root = tree.insert(2, "c").unwrap();
        assert_eq!(*tree.tree.get(1).unwrap(), hash_pair(hash("a"), hash("b")));
        assert_eq!(
            *tree.tree.get(2).unwrap(),
            hash_pair(hash("c"), DEFAULT_LEAF)
        );
        assert_eq!(
            root,
            hash_pair(
                hash_pair(hash("a"), hash("b")),
                hash_pair(hash("c"), DEFAULT_LEAF),
            )
        );

        let root = tree.insert(3, "d").unwrap();
        assert_eq!(*tree.tree.get(1).unwrap(), hash_pair(hash("a"), hash("b")));
        assert_eq!(*tree.tree.get(2).unwrap(), hash_pair(hash("c"), hash("d")));
        assert_eq!(
            root,
            hash_pair(
                hash_pair(hash("a"), hash("b")),
                hash_pair(hash("c"), hash("d")),
            )
        );
    }

    #[test]
    fn proof_height_2() {
        let mut tree = MerkleTree::new(2);

        tree.insert(0, "a").unwrap();
        let proof = tree.prove(0).unwrap();
        let expected_proof = vec![DEFAULT_LEAF];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("a", proof).unwrap());

        tree.insert(1, "b").unwrap();
        let proof = tree.prove(1).unwrap();
        let expected_proof = vec![hash("a")];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("b", proof).unwrap());
    }

    #[test]
    fn proof_height_3() {
        let mut tree = MerkleTree::new(3);

        tree.insert(3, "d").unwrap();
        let proof = tree.prove(3).unwrap();
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, DEFAULT_LEAF)];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("d", proof).unwrap());

        tree.insert(1, "b").unwrap();
        let proof = tree.prove(1).unwrap();
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, hash("d"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("b", proof).unwrap());

        tree.insert(2, "c").unwrap();
        let proof = tree.prove(2).unwrap();
        let expected_proof = vec![hash("d"), hash_pair(DEFAULT_LEAF, hash("b"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("c", proof).unwrap());

        tree.insert(0, "a").unwrap();
        let proof = tree.prove(0).unwrap();
        let expected_proof = vec![hash("b"), hash_pair(hash("c"), hash("d"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("a", proof).unwrap());
    }

    #[test]
    fn path_height_3() {
        // Indices:
        //    0
        //  1   2
        // 3 4 5 6
        let tree = MerkleTree::new(3);

        let proof = tree.path(0).unwrap();
        assert_eq!(proof, [4, 2].into());

        let proof = tree.path(1).unwrap();
        assert_eq!(proof, [3, 2].into());

        let proof = tree.path(2).unwrap();
        assert_eq!(proof, [6, 1].into());

        let proof = tree.path(3).unwrap();
        assert_eq!(proof, [5, 1].into());
    }

    #[test]
    fn path_height_4() {
        // Indices:
        //     0
        //   1   2
        //  3 4 5 6
        // 7 8..13 14
        let tree = MerkleTree::new(4);

        let proof = tree.path(0).unwrap();
        assert_eq!(proof, [8, 4, 2].into());

        let proof = tree.path(7).unwrap();
        assert_eq!(proof, [13, 5, 1].into());
    }

    #[test]
    fn bounds_check() {
        assert_eq!(
            MerkleTree::new(1).bounds_check(1).unwrap_err(),
            Error::IndexOutOfBounds { len: 1, index: 1 }
        );

        assert_eq!(
            MerkleTree::new(2).bounds_check(2).unwrap_err(),
            Error::IndexOutOfBounds { len: 2, index: 2 }
        );

        assert_eq!(
            MerkleTree::new(3).bounds_check(4).unwrap_err(),
            Error::IndexOutOfBounds { len: 4, index: 4 }
        );
    }

    #[test]
    fn indicies_of() {
        let mut tree = MerkleTree::new(3);

        tree.insert(0, "a").unwrap();
        assert_eq!(tree.indicies_of("a"), Some(vec![0].into()));

        tree.insert(3, "a").unwrap();
        let indexes = tree.indicies_of("a").unwrap();
        let expected_indexes = vec![0, 3];
        assert_eq!(indexes, expected_indexes.into());

        assert!(tree.indicies_of("b").is_none());
    }

    #[test]
    fn contains() {
        let mut tree = MerkleTree::new(3);

        assert!(!tree.contains("a"));

        tree.insert(3, "a").unwrap();
        assert!(tree.contains("a"));

        tree.insert(0, "a").unwrap();
        assert!(tree.contains("a"));
    }

    #[test]
    fn len() {
        assert_eq!(MerkleTree::new(0).len(), 0);
        assert_eq!(MerkleTree::new(1).len(), 1);
        assert_eq!(MerkleTree::new(2).len(), 2);
        assert_eq!(MerkleTree::new(3).len(), 4);
    }
}
