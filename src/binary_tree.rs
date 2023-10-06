use crate::hash_map::HashMap;
use std::iter::FromIterator;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BinaryTree<T> {
    values: HashMap<usize, T>,
    height: usize,
}

impl<T> BinaryTree<T> {
    pub fn new() -> Self {
        Self::with_height(0)
    }

    pub fn with_height(height: usize) -> Self {
        Self {
            values: HashMap::new(),
            height,
        }
    }

    pub fn insert(&mut self, index: usize, value: T) {
        assert!(index < self.len());
        self.values.insert(index, value);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(&index)
    }

    pub fn len(&self) -> usize {
        2usize.pow(self.height as u32) - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn new_height(&mut self, new_height: usize) {
        let old_height = self.height;
        self.height = new_height;
        if self.height < old_height {
            let new_len = self.len();
            self.values.retain(|&index, _| index < new_len);
        }
    }

    pub fn path_to_root(&self, index: usize) -> Vec<usize> {
        let mut index = index;
        let mut path = vec![index];
        while index > 0 {
            index = parent_index(index);
            path.push(index);
        }
        path
    }

    pub fn iter(&self) -> BinaryTreeIterator<T> {
        BinaryTreeIterator::new(self)
    }

    pub fn leaf_iter(&self) -> BinaryTreeLeafIterator<T> {
        BinaryTreeLeafIterator::new(self)
    }
}

pub struct BinaryTreeIterator<'a, T> {
    bt: &'a BinaryTree<T>,
    index: usize,
}

impl<'a, T> BinaryTreeIterator<'a, T> {
    pub fn new(bt: &'a BinaryTree<T>) -> Self {
        Self { bt, index: 0 }
    }
}

impl<'a, T> Iterator for BinaryTreeIterator<'a, T> {
    type Item = Option<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.bt.len() {
            None
        } else {
            let item = self.bt.get(self.index);
            self.index += 1;
            Some(item)
        }
    }
}

pub struct BinaryTreeLeafIterator<'a, T>(BinaryTreeIterator<'a, T>);

impl<'a, T> BinaryTreeLeafIterator<'a, T> {
    pub fn new(bt: &'a BinaryTree<T>) -> Self {
        let mut iter = BinaryTreeIterator::new(bt);
        iter.index = first_index_at_height(bt.height());
        Self(iter)
    }
}

impl<'a, T> Iterator for BinaryTreeLeafIterator<'a, T> {
    type Item = Option<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> FromIterator<T> for BinaryTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let values = HashMap::from_iter(iter.into_iter().enumerate());
        let height = values.len().ilog2() as usize + 1;
        Self { values, height }
    }
}

pub fn parent_index(index: usize) -> usize {
    (index - 1) / 2
}

pub fn sibling_index(index: usize) -> usize {
    assert!(index > 0);
    if index % 2 == 0 {
        index - 1
    } else {
        index + 1
    }
}

pub fn child_indices(index: usize) -> (usize, usize) {
    let left = index * 2 + 1;
    let right = left + 1;
    (left, right)
}

pub fn first_index_at_height(height: usize) -> usize {
    assert!(height > 0);
    2usize.pow(height as u32 - 1) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sibling_index() {
        assert_eq!(sibling_index(1), 2);
        assert_eq!(sibling_index(2), 1);
        assert_eq!(sibling_index(3), 4);
        assert_eq!(sibling_index(4), 3);
        assert_eq!(sibling_index(6), 5);
    }

    #[test]
    #[should_panic]
    fn test_sibling_index_0() {
        sibling_index(0);
    }

    #[test]
    fn test_insert() {
        let mut bt = BinaryTree::with_height(1);
        bt.insert(0, 0);
        assert_eq!(bt, BinaryTree::from_iter([0]));

        let mut bt = BinaryTree::with_height(2);
        bt.insert(0, 0);
        bt.insert(1, 1);
        bt.insert(2, 2);
        assert_eq!(bt, BinaryTree::from_iter(0..=2));
    }

    #[test]
    #[should_panic]
    fn test_insert_out_of_bounds_panics_height_0() {
        let mut bt = BinaryTree::with_height(0);
        bt.insert(0, 0);
    }

    #[test]
    #[should_panic]
    fn test_insert_out_of_bounds_panics_height_2() {
        let mut bt = BinaryTree::with_height(2);
        bt.insert(3, 3);
    }

    #[test]
    fn test_height() {
        let bt = BinaryTree::<usize>::new();
        assert_eq!(bt.height(), 0);

        let bt = BinaryTree::from_iter([0]);
        assert_eq!(bt.height(), 1);

        let bt = BinaryTree::from_iter(0..=1);
        assert_eq!(bt.height(), 2);
        let bt = BinaryTree::from_iter(0..=2);
        assert_eq!(bt.height(), 2);

        let bt = BinaryTree::from_iter(0..=3);
        assert_eq!(bt.height(), 3);
        let bt = BinaryTree::from_iter(0..=6);
        assert_eq!(bt.height(), 3);
    }

    #[test]
    fn test_path_to_root() {
        let bt = BinaryTree::from_iter(0..=6);

        let path = bt.path_to_root(3);
        assert_eq!(path, vec![3, 1, 0]);
        let path = bt.path_to_root(4);
        assert_eq!(path, vec![4, 1, 0]);

        let path = bt.path_to_root(5);
        assert_eq!(path, vec![5, 2, 0]);
        let path = bt.path_to_root(6);
        assert_eq!(path, vec![6, 2, 0]);
    }

    #[test]
    fn test_iter() {
        let bt = BinaryTree::from_iter(0..=6);
        let mut iter = bt.iter();

        assert_eq!(iter.next(), Some(Some(&0)));
        assert_eq!(iter.next(), Some(Some(&1)));
        assert_eq!(iter.next(), Some(Some(&2)));
        assert_eq!(iter.next(), Some(Some(&3)));
        assert_eq!(iter.next(), Some(Some(&4)));
        assert_eq!(iter.next(), Some(Some(&5)));
        assert_eq!(iter.next(), Some(Some(&6)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_leaf_iter() {
        let bt = BinaryTree::from_iter(0..=6);
        let mut iter = bt.leaf_iter();

        assert_eq!(iter.next(), Some(Some(&3)));
        assert_eq!(iter.next(), Some(Some(&4)));
        assert_eq!(iter.next(), Some(Some(&5)));
        assert_eq!(iter.next(), Some(Some(&6)));
        assert_eq!(iter.next(), None);
    }

    #[test]

    fn test_first_index_at_height() {
        assert_eq!(first_index_at_height(1), 0);
        assert_eq!(first_index_at_height(2), 1);
        assert_eq!(first_index_at_height(3), 3);
        assert_eq!(first_index_at_height(4), 7);
    }

    #[test]
    fn test_len() {
        assert_eq!(BinaryTree::<()>::with_height(1).len(), 1);
        assert_eq!(BinaryTree::<()>::with_height(2).len(), 3);
        assert_eq!(BinaryTree::<()>::with_height(3).len(), 7);
        assert_eq!(BinaryTree::<()>::with_height(4).len(), 15);
    }

    #[test]
    fn test_from_iter() {
        let bt = BinaryTree::from_iter(0..=0);
        assert_eq!(bt.height, 1);

        let bt = BinaryTree::from_iter(0..=2);
        assert_eq!(bt.height, 2);

        let bt = BinaryTree::from_iter(0..=6);
        assert_eq!(bt.height, 3);
    }

    #[test]
    fn test_new_height() {
        let mut bt = BinaryTree::with_height(1);
        bt.insert(0, 0);

        bt.new_height(2);
        bt.insert(1, 1);
        bt.insert(2, 2);
        assert_eq!(bt.get(1), Some(&1));
        assert_eq!(bt.get(2), Some(&2));

        bt.new_height(1);
        assert_eq!(bt.get(1), None);
        assert_eq!(bt.get(2), None);
    }
}
