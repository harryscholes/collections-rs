#[derive(Debug, Clone, PartialEq)]
pub struct BinaryTree<T> {
    values: Vec<Option<T>>,
}

impl<T> BinaryTree<T> {
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn with_height(height: usize) -> Self {
        let capacity = 2usize.pow(height as u32) - 1;
        let mut bt = Self::with_capacity(capacity);
        for _ in 0..capacity {
            bt.values.push(None);
        }
        bt
    }

    pub fn insert(&mut self, index: usize, value: T) {
        self.values[index] = Some(value);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.values[index].as_ref()
    }

    pub fn grow(&mut self, new_height: usize) {
        if new_height <= self.height() {
            return;
        }
        let new_len = 2usize.pow(new_height as u32) - 1;
        let additional = new_len - self.len();
        self.values.reserve(additional);
        for _ in 0..additional {
            self.values.push(None);
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn height(&self) -> usize {
        if self.len() == 0 {
            0
        } else {
            self.len().ilog2() as usize + 1
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
            return None;
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
        let mut bt = Self::new();
        bt.values.extend(iter.into_iter().map(Some));
        bt
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
    fn test_grow() {
        let mut bt = BinaryTree::<usize>::new();
        assert_eq!(bt.height(), 0);

        bt.grow(1);
        assert_eq!(bt.height(), 1);

        bt.grow(2);
        assert_eq!(bt.height(), 2);
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
}
