use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use crate::{
    binary_tree::{first_index_at_height, BinaryTree},
    hash_map::HashMap,
    vector::Vector,
};

type Digest = [u8; 8];

const DEFAULT_LEAF: Digest = [0; 8];

/// Space complexity: O(n)
pub struct MerkleTree {
    tree: BinaryTree<Digest>,
    default_nodes: HashMap<usize, Digest>,
}

impl MerkleTree {
    pub fn with_height(height: usize) -> Self {
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
        *self.tree.get(0).unwrap_or(&self.default_nodes[1])
    }

    /// Time complexity: O(log n)
    pub fn insert(&mut self, index: usize, value: impl AsRef<[u8]>) -> Digest {
        self.bounds_check(index).unwrap();

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
                .unwrap_or(&self.default_nodes[internal_height]);

            let parent_index = (internal_index - 1) / 2;
            let parent_node = hash_pair(internal_node, sibling_node);
            self.tree.insert(parent_index, parent_node);

            internal_index = parent_index;
            internal_node = parent_node;
            internal_height -= 1;
        }

        self.root()
    }

    /// Merkle path from leaf to root.
    ///
    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn path(&self, index: usize) -> Vector<usize> {
        self.bounds_check(index).unwrap();

        let tree_index = self.tree_index(index);
        let mut path = Vector::with_capacity(self.tree.height() - 1);

        let mut internal_index = tree_index;
        let mut internal_height = self.tree.height();

        while internal_height > 1 {
            let sibling_index = sibling_index(internal_index);
            path.push_back(sibling_index);
            internal_index = (internal_index - 1) / 2;
            internal_height -= 1;
        }

        path
    }

    /// Time complexity: O(log n)
    /// Space complexity: O(log n)
    pub fn prove(&self, index: usize) -> Vector<Digest> {
        assert!(self.tree.height() > 1);

        let path = self.path(index);

        let mut proof = Vector::with_capacity(path.len());

        let mut internal_height = self.tree.height();

        for internal_index in path {
            let internal_node = self
                .tree
                .get(internal_index)
                .unwrap_or(&self.default_nodes[internal_height]);
            proof.push_back(*internal_node);
            internal_height -= 1;
        }

        proof
    }

    /// Time complexity: O(log n)
    pub fn validate(&self, value: impl AsRef<[u8]>, proof: Vector<Digest>) -> bool {
        self.proof_check(proof.len()).unwrap();

        let mut internal_node = hash(value);
        for sibling_node in proof {
            internal_node = hash_pair(internal_node, sibling_node);
        }

        let proof_root = internal_node;

        proof_root == self.root()
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

    fn proof_check(&self, proof_len: usize) -> Result<(), Error> {
        if self.tree.height() <= 1 {
            return Err(Error::InvalidTreeHeight {
                height: self.tree.height(),
            });
        }

        let expected_len = self.tree.height() - 1;
        if proof_len != expected_len {
            Err(Error::IncorrectProofLength {
                proof_len,
                expected_len,
            })
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
fn default_nodes(default_leaf: Digest, height: usize) -> HashMap<usize, Digest> {
    let mut default_nodes = HashMap::new();
    let mut internal_node = default_leaf;

    for h in (1..=height).rev() {
        default_nodes.insert(h, internal_node);
        internal_node = hash_pair(internal_node, internal_node);
    }

    // default_nodes.reverse();

    // default_nodes.into()
    default_nodes
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
    IndexOutOfBounds {
        len: usize,
        index: usize,
    },
    IncorrectProofLength {
        proof_len: usize,
        expected_len: usize,
    },
    InvalidTreeHeight {
        height: usize,
    },
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
        // let expected_nodes = vec![(1, DEFAULT_LEAF)];
        assert_eq!(nodes[1], DEFAULT_LEAF);
    }

    #[test]
    fn default_nodes_height_2() {
        let nodes = default_nodes(DEFAULT_LEAF, 2);
        assert_eq!(nodes[2], DEFAULT_LEAF);
        assert_eq!(nodes[1], hash_pair(DEFAULT_LEAF, DEFAULT_LEAF));
    }

    #[test]
    fn default_nodes_height_3() {
        let nodes = default_nodes(DEFAULT_LEAF, 3);
        assert_eq!(nodes[3], DEFAULT_LEAF);
        assert_eq!(nodes[2], hash_pair(DEFAULT_LEAF, DEFAULT_LEAF));
        assert_eq!(
            nodes[1],
            hash_pair(
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF),
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF)
            )
        );
    }

    #[test]
    #[should_panic]
    fn empty_tree_height_0() {
        let tree = MerkleTree::with_height(0);
        tree.root();
    }

    #[test]
    fn empty_tree_height_1() {
        let tree = MerkleTree::with_height(1);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 1)[1];
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_2() {
        let tree = MerkleTree::with_height(2);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 2)[1];
        assert_eq!(root, expected_root);
    }

    #[test]
    fn empty_tree_height_32() {
        let tree = MerkleTree::with_height(32);
        let root = tree.root();
        let expected_root = default_nodes(DEFAULT_LEAF, 32)[1];
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_1() {
        let mut tree = MerkleTree::with_height(1);
        let root = tree.insert(0, "a");
        let expected_root = hash("a");
        assert_eq!(root, expected_root);
    }

    #[test]
    fn root_height_2() {
        let mut tree = MerkleTree::with_height(2);

        let root = tree.insert(0, "a");
        assert_eq!(root, hash_pair(hash("a"), DEFAULT_LEAF));

        let root = tree.insert(1, "b");
        assert_eq!(root, hash_pair(hash("a"), hash("b")));
    }

    #[test]
    fn root_height_3() {
        let mut tree = MerkleTree::with_height(3);

        let root = tree.insert(0, "a");
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

        let root = tree.insert(1, "b");
        assert_eq!(*tree.tree.get(1).unwrap(), hash_pair(hash("a"), hash("b")));
        assert!(tree.tree.get(2).is_none());
        assert_eq!(
            root,
            hash_pair(
                hash_pair(hash("a"), hash("b")),
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF),
            )
        );

        let root = tree.insert(2, "c");
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

        let root = tree.insert(3, "d");
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
    fn root_height_3_sparse() {
        let mut tree = MerkleTree::with_height(3);

        let root = tree.insert(3, "d");
        assert!(tree.tree.get(1).is_none());
        assert_eq!(
            *tree.tree.get(2).unwrap(),
            hash_pair(DEFAULT_LEAF, hash("d"))
        );
        assert_eq!(
            root,
            hash_pair(
                hash_pair(DEFAULT_LEAF, DEFAULT_LEAF),
                hash_pair(DEFAULT_LEAF, hash("d")),
            )
        );

        let root = tree.insert(1, "b");
        assert_eq!(
            *tree.tree.get(1).unwrap(),
            hash_pair(DEFAULT_LEAF, hash("b"))
        );
        assert_eq!(
            *tree.tree.get(2).unwrap(),
            hash_pair(DEFAULT_LEAF, hash("d"))
        );
        assert_eq!(
            root,
            hash_pair(
                hash_pair(DEFAULT_LEAF, hash("b")),
                hash_pair(DEFAULT_LEAF, hash("d")),
            )
        );

        let root = tree.insert(2, "c");
        assert_eq!(
            *tree.tree.get(1).unwrap(),
            hash_pair(DEFAULT_LEAF, hash("b"))
        );
        assert_eq!(*tree.tree.get(2).unwrap(), hash_pair(hash("c"), hash("d")));
        assert_eq!(
            root,
            hash_pair(
                hash_pair(DEFAULT_LEAF, hash("b")),
                hash_pair(hash("c"), hash("d")),
            )
        );

        let root = tree.insert(0, "a");
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
        let mut tree = MerkleTree::with_height(2);

        tree.insert(0, "a");
        let proof = tree.prove(0);
        let expected_proof = vec![DEFAULT_LEAF];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("a", proof));

        tree.insert(1, "b");
        let proof = tree.prove(1);
        let expected_proof = vec![hash("a")];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("b", proof));
    }

    #[test]
    fn proof_height_3() {
        let mut tree = MerkleTree::with_height(3);

        tree.insert(3, "d");
        let proof = tree.prove(3);
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, DEFAULT_LEAF)];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("d", proof));

        tree.insert(1, "b");
        let proof = tree.prove(1);
        let expected_proof = vec![DEFAULT_LEAF, hash_pair(DEFAULT_LEAF, hash("d"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("b", proof));

        tree.insert(2, "c");
        let proof = tree.prove(2);
        let expected_proof = vec![hash("d"), hash_pair(DEFAULT_LEAF, hash("b"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("c", proof));

        tree.insert(0, "a");
        let proof = tree.prove(0);
        let expected_proof = vec![hash("b"), hash_pair(hash("c"), hash("d"))];
        assert_eq!(proof, expected_proof.into());
        assert!(tree.validate("a", proof));
    }

    #[test]
    fn path_height_3() {
        // Indices:
        //    0
        //  1   2
        // 3 4 5 6
        let tree = MerkleTree::with_height(3);

        let proof = tree.path(0);
        assert_eq!(proof, [4, 2].into());

        let proof = tree.path(1);
        assert_eq!(proof, [3, 2].into());

        let proof = tree.path(2);
        assert_eq!(proof, [6, 1].into());

        let proof = tree.path(3);
        assert_eq!(proof, [5, 1].into());
    }

    #[test]
    fn path_height_4() {
        // Indices:
        //     0
        //   1   2
        //  3 4 5 6
        // 7 8..13 14
        let tree = MerkleTree::with_height(4);

        let proof = tree.path(0);
        assert_eq!(proof, [8, 4, 2].into());

        let proof = tree.path(7);
        assert_eq!(proof, [13, 5, 1].into());
    }

    #[test]
    fn indicies_of() {
        let mut tree = MerkleTree::with_height(3);

        tree.insert(0, "a");
        assert_eq!(tree.indicies_of("a"), Some(vec![0].into()));

        tree.insert(3, "a");
        let indexes = tree.indicies_of("a").unwrap();
        let expected_indexes = vec![0, 3];
        assert_eq!(indexes, expected_indexes.into());

        assert!(tree.indicies_of("b").is_none());
    }

    #[test]
    fn contains() {
        let mut tree = MerkleTree::with_height(3);

        assert!(!tree.contains("a"));

        tree.insert(3, "a");
        assert!(tree.contains("a"));

        tree.insert(0, "a");
        assert!(tree.contains("a"));
    }

    #[test]
    fn len() {
        assert_eq!(MerkleTree::with_height(0).len(), 0);
        assert_eq!(MerkleTree::with_height(1).len(), 1);
        assert_eq!(MerkleTree::with_height(2).len(), 2);
        assert_eq!(MerkleTree::with_height(3).len(), 4);
        assert_eq!(MerkleTree::with_height(4).len(), 8);
    }

    #[test]
    fn bounds_check() {
        assert_eq!(
            MerkleTree::with_height(1).bounds_check(1).unwrap_err(),
            Error::IndexOutOfBounds { len: 1, index: 1 }
        );

        assert_eq!(
            MerkleTree::with_height(2).bounds_check(2).unwrap_err(),
            Error::IndexOutOfBounds { len: 2, index: 2 }
        );

        assert_eq!(
            MerkleTree::with_height(3).bounds_check(4).unwrap_err(),
            Error::IndexOutOfBounds { len: 4, index: 4 }
        );
    }

    #[test]
    fn proof_check() {
        assert_eq!(
            MerkleTree::with_height(0).proof_check(0).unwrap_err(),
            Error::InvalidTreeHeight { height: 0 }
        );

        assert_eq!(
            MerkleTree::with_height(1).proof_check(0).unwrap_err(),
            Error::InvalidTreeHeight { height: 1 }
        );

        assert_eq!(
            MerkleTree::with_height(2).proof_check(2).unwrap_err(),
            Error::IncorrectProofLength {
                proof_len: 2,
                expected_len: 1
            }
        );

        assert_eq!(
            MerkleTree::with_height(3).proof_check(3).unwrap_err(),
            Error::IncorrectProofLength {
                proof_len: 3,
                expected_len: 2
            }
        );
    }
}
