use std::cmp::Reverse;

pub trait Heap<T> {
    fn push(&mut self, el: T);

    fn pop(&mut self) -> Option<T>;

    fn delete<P: FnMut(&T) -> bool>(&mut self, predicate: P) -> Option<T>;

    fn peek(&self) -> Option<&T>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct MaxHeap<T>(Vec<T>);

impl<T> MaxHeap<T>
where
    T: Ord,
{
    pub fn new() -> MaxHeap<T> {
        MaxHeap::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> MaxHeap<T> {
        MaxHeap(Vec::with_capacity(capacity))
    }

    /// Time complexity: O(log(n))
    fn sift_down(&mut self, index: usize) {
        let mut parent_index = index;
        let (mut left_sibling_index, mut right_sibling_index) = sibling_indices(parent_index);

        while let Some(parent) = self.0.get(parent_index) {
            match self.0.get(left_sibling_index) {
                Some(left_sibling) => match self.0.get(right_sibling_index) {
                    Some(right_sibling) => {
                        if parent < left_sibling || parent < right_sibling {
                            // Swap `parent` and the largest sibling
                            if left_sibling > right_sibling {
                                self.0.swap(parent_index, left_sibling_index);
                                parent_index = left_sibling_index;
                            } else {
                                self.0.swap(parent_index, right_sibling_index);
                                parent_index = right_sibling_index;
                            }
                        } else {
                            // Heap property is satisfied
                            break;
                        }
                    }
                    None => {
                        if parent < left_sibling {
                            self.0.swap(parent_index, left_sibling_index);
                            parent_index = left_sibling_index;
                        } else {
                            // Heap property is satisfied
                            break;
                        }
                    }
                },
                // `parent` is a leaf node
                None => break,
            }

            (left_sibling_index, right_sibling_index) = sibling_indices(parent_index);
        }
    }

    /// Time complexity: O(log(n))
    fn sift_up(&mut self, index: usize) {
        let mut sibling_index = index;
        let mut _parent_index = parent_index(sibling_index);

        loop {
            if let Some(sibling) = self.0.get(sibling_index) {
                if let Some(parent) = self.0.get(_parent_index) {
                    if sibling > parent {
                        self.0.swap(_parent_index, sibling_index);
                        sibling_index = _parent_index;
                        _parent_index = parent_index(sibling_index)
                    } else {
                        // Heap property is satisfied
                        break;
                    }
                }
            }
        }
    }

    /// Time complexity: O(log(n))
    fn delete_at_index(&mut self, index: usize) -> Option<T> {
        let last_index = self.len() - 1;
        self.0.swap(index, last_index);
        let el = self.0.pop();

        // Heapify
        if let Some(x) = self.0.get(index) {
            if let Some(parent) = self.0.get(parent_index(index)) {
                if x > parent {
                    self.sift_up(index);
                } else {
                    self.sift_down(index);
                }
            }
        }
        debug_assert!(is_heap(&self.0));

        el
    }
}

impl<T> Heap<T> for MaxHeap<T>
where
    T: Ord,
{
    /// Time complexity: O(log(n))
    fn push(&mut self, el: T) {
        self.0.push(el);
        self.sift_up(self.len() - 1);
        debug_assert!(is_heap(&self.0));
    }

    /// Time complexity: O(log(n))
    fn pop(&mut self) -> Option<T> {
        match self.peek() {
            Some(_root) => {
                let len = self.len();
                self.0.swap(0, len - 1);
                let root = self.0.pop();
                self.sift_down(0);
                debug_assert!(is_heap(&self.0));
                root
            }
            None => None,
        }
    }

    // /// Time complexity: O(n)
    // fn delete(&mut self, el: &T) -> Option<T> {
    //     let el = self.map_delete(|x| x == el);
    //     debug_assert!(is_heap(&self.0));
    //     el
    // }

    /// Delete the largest item in the heap that matches `predicate`.
    ///
    /// Time complexity: O(n)
    fn delete<P: FnMut(&T) -> bool>(&mut self, predicate: P) -> Option<T> {
        match self.0.iter().position(predicate) {
            Some(index) => self.delete_at_index(index),
            None => None,
        }
    }

    /// Time complexity: O(1)
    fn peek(&self) -> Option<&T> {
        self.0.first()
    }

    /// Time complexity: O(1)
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Default for MaxHeap<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct MinHeap<T>(MaxHeap<Reverse<T>>);

impl<T> MinHeap<T>
where
    T: Ord,
{
    pub fn new() -> MinHeap<T> {
        MinHeap::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> MinHeap<T> {
        MinHeap(MaxHeap::with_capacity(capacity))
    }
}

impl<T> Heap<T> for MinHeap<T>
where
    T: Ord + Copy,
{
    /// Time complexity: O(log(n))
    fn push(&mut self, el: T) {
        self.0.push(Reverse(el))
    }

    /// Time complexity: O(log(n))
    fn pop(&mut self) -> Option<T> {
        self.0.pop().map(|el| el.0)
    }

    /// Time complexity: O(n)
    fn delete<P: FnMut(&T) -> bool>(&mut self, predicate: P) -> Option<T> {
        match self.0 .0.iter().map(|x| &x.0).position(predicate) {
            Some(index) => self.0.delete_at_index(index).map(|rev| rev.0),
            None => None,
        }
    }

    /// Time complexity: O(1)
    fn peek(&self) -> Option<&T> {
        self.0.peek().map(|el| &el.0)
    }

    /// Time complexity: O(1)
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Default for MinHeap<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Time complexity: O(n*log(n))
pub fn is_heap<T: PartialOrd>(v: &[T]) -> bool {
    /// Time complexity: O(log(n))
    fn _is_heap<T: PartialOrd>(v: &[T], parent_index: usize) -> bool {
        let (left_sibling_index, right_sibling_index) = sibling_indices(parent_index);

        match v.get(parent_index) {
            Some(parent) => {
                match v.get(left_sibling_index) {
                    Some(left_sibling) => {
                        if parent < left_sibling {
                            return false;
                        }
                    }
                    None => return true,
                };

                match v.get(right_sibling_index) {
                    Some(right_sibling) => {
                        if parent < right_sibling {
                            return false;
                        }
                    }
                    None => return true,
                };
            }
            None => return true,
        };

        _is_heap(v, left_sibling_index) && _is_heap(v, right_sibling_index)
    }

    _is_heap(v, 0)
}

/// Time complexity: O(n*log(n))
pub fn heapify<T: Ord>(arr: &mut [T]) {
    /// Time complexity: O(log(n))
    fn _heapify<T: Ord>(arr: &mut [T], index: usize) {
        let n = arr.len();
        let mut largest = index;
        let (l, r) = sibling_indices(index);

        if l < n && arr[l] > arr[largest] {
            largest = l;
        }
        if r < n && arr[r] > arr[largest] {
            largest = r;
        }

        if largest != index {
            arr.swap(largest, index);
            // Recursively heapify the affected sub-tree
            _heapify(arr, largest);
        }
    }

    let last_non_leaf_node_index = (arr.len() / 2).saturating_sub(1);
    for non_leaf_node_index in (0..=last_non_leaf_node_index).rev() {
        _heapify(arr, non_leaf_node_index);
    }

    debug_assert!(is_heap(arr));
}

fn sibling_indices(parent_index: usize) -> (usize, usize) {
    let left_sibling_index = 2 * parent_index + 1;
    let right_sibling_index = left_sibling_index + 1;
    (left_sibling_index, right_sibling_index)
}

fn parent_index(sibling_index: usize) -> usize {
    sibling_index.saturating_sub(1) / 2
}

#[cfg(test)]
mod test_max_heap {
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_push() {
        let mut mh = MaxHeap::new();
        for i in 1..10 {
            mh.push(i);
            assert!(mh.0.contains(&i));
        }
    }

    #[test]
    fn test_pop() {
        let mut mh = MaxHeap::new();
        for i in 1..=10 {
            mh.push(i);
        }
        for i in (1..=10).rev() {
            let root = mh.pop();
            assert_eq!(root, Some(i));
        }
    }

    #[test]
    fn test_root_n0() {
        let mh: MaxHeap<usize> = MaxHeap::new();
        assert!(mh.peek().is_none());
    }

    #[test]
    fn test_root_n1() {
        let mut mh = MaxHeap::new();
        mh.push(1);
        assert_eq!(mh.peek(), Some(&1));
    }

    #[test]
    fn test_root_new_max_each_push() {
        let mut mh = MaxHeap::new();
        for i in 1..10 {
            mh.push(i);
            assert_eq!(mh.peek(), Some(&i));
        }
    }

    #[test]
    fn test_root_same_max_each_push() {
        let mut mh = MaxHeap::new();
        for i in (1..=10).rev() {
            mh.push(i);
            assert_eq!(mh.peek(), Some(&10));
        }
    }

    #[test]
    fn test_len() {
        let mut mh = MaxHeap::new();
        assert_eq!(mh.len(), 0);
        for i in 1..=10 {
            mh.push(i);
            assert_eq!(mh.len(), i);
        }
    }

    #[test]
    fn test_is_empty() {
        let mut mh = MaxHeap::new();
        assert!(mh.is_empty());
        mh.push(1);
        assert!(!mh.is_empty());
        mh.pop();
        assert!(mh.is_empty());
    }

    #[test]
    fn test_delete() {
        let mut mh = MaxHeap::new();
        let elements = vec![2, 6, 5, 1, 3, 7, 4];
        for el in &elements {
            mh.push(el.clone());
        }
        for el in elements {
            assert!(mh.0.contains(&el));
            assert_eq!(mh.delete(|x| x == &el), Some(el));
            assert!(!mh.0.contains(&el));
        }
    }

    #[test]
    fn test_repeated_elements() {
        let mut mh = MaxHeap::new();
        for _ in 1..=10 {
            mh.push(1);
            mh.push(2);
        }
        for _ in 1..=10 {
            assert_eq!(mh.pop(), Some(2));
        }
        for _ in 1..=10 {
            assert_eq!(mh.pop(), Some(1));
        }
    }

    #[test]
    fn test_fuzz() {
        let n = 1_000;
        let mut mh = MaxHeap::with_capacity(n);
        let mut rng = thread_rng();
        let mut expected_root = usize::MIN;
        for _ in 0..n {
            let el = rng.gen_range(0..usize::MAX);
            mh.push(el);
            if el > expected_root {
                expected_root = el
            }
            assert_eq!(mh.peek(), Some(&expected_root));
        }
    }

    #[test]
    fn test_is_max_heap() {
        let v: Vec<usize> = vec![];
        assert!(is_heap(&v));

        let v = vec![10];
        assert!(is_heap(&v));

        let v = vec![10, 9, 8];
        assert!(is_heap(&v));

        let v = vec![10, 7, 8, 9];
        assert!(!is_heap(&v));

        let v = vec![10, 8, 9, 4, 5, 6, 7];
        assert!(is_heap(&v));

        let v = vec![10, 8, 9, 4, 9, 6, 7];
        assert!(!is_heap(&v));

        let v = vec![10, 8, 9, 4, 5, 6, 10];
        assert!(!is_heap(&v));
    }
}

#[cfg(test)]
mod test_min_heap {
    use super::*;

    use rand::prelude::*;

    #[test]
    fn test_push() {
        let mut mh = MinHeap::new();
        for i in 1..10 {
            mh.push(i);
            assert!(mh.0 .0.contains(&Reverse(i)));
        }
    }

    #[test]
    fn test_pop() {
        let mut mh = MinHeap::new();
        for i in (1..=10).rev() {
            mh.push(i);
            assert_eq!(mh.peek(), Some(&i));
        }
        for i in 1..=10 {
            let el = mh.pop();
            dbg!(el, &mh.0 .0);
            assert_eq!(el, Some(i));
        }
    }

    #[test]
    fn test_root_n0() {
        let mh: MinHeap<usize> = MinHeap::new();
        assert!(mh.peek().is_none());
    }

    #[test]
    fn test_root_n1() {
        let mut mh = MinHeap::new();
        mh.push(1);
        assert_eq!(mh.peek(), Some(&1));
    }

    #[test]
    fn test_root_new_min_each_push() {
        let mut mh = MinHeap::new();
        for i in (1..=10).rev() {
            mh.push(i);
            assert_eq!(mh.peek(), Some(&i));
        }
    }

    #[test]
    fn test_root_same_min_each_push() {
        let mut mh = MinHeap::new();
        for i in 1..=10 {
            mh.push(i);
            assert_eq!(mh.peek(), Some(&1));
        }
    }

    #[test]
    fn test_len() {
        let mut mh = MinHeap::new();
        assert_eq!(mh.len(), 0);
        for i in 1..=10 {
            mh.push(i);
            assert_eq!(mh.len(), i);
        }
    }

    #[test]
    fn test_is_empty() {
        let mut mh = MinHeap::new();
        assert!(mh.is_empty());
        mh.push(1);
        assert!(!mh.is_empty());
        mh.pop();
        assert!(mh.is_empty());
    }

    #[test]
    fn test_fuzz() {
        let n = 1_000;
        let mut mh = MinHeap::with_capacity(n);
        let mut rng = thread_rng();
        let mut expected_root = usize::MAX;
        for _ in 0..n {
            let el = rng.gen_range(0..usize::MAX);
            mh.push(el);
            if el < expected_root {
                expected_root = el
            }
            assert_eq!(mh.peek(), Some(&expected_root));
        }
    }

    #[test]
    fn test_delete() {
        let mut mh = MinHeap::new();
        let elements = vec![2, 6, 5, 1, 3, 7, 4];
        for el in &elements {
            mh.push(el.clone());
        }
        for el in elements {
            assert!(mh.0 .0.contains(&Reverse(el)));
            assert_eq!(mh.delete(|x| x == &el), Some(el));
            assert!(!mh.0 .0.contains(&Reverse(el)));
        }
    }

    #[test]
    fn test_repeated_elements() {
        let mut mh = MinHeap::new();
        for _ in 1..=10 {
            mh.push(1);
            mh.push(2);
        }
        for _ in 1..=10 {
            assert_eq!(mh.pop(), Some(1));
        }
        for _ in 1..=10 {
            assert_eq!(mh.pop(), Some(2));
        }
    }

    #[test]
    fn test_is_min_heap() {
        let v: Vec<Reverse<usize>> = vec![];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(is_heap(&reversed));

        let v = vec![1];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(is_heap(&reversed));

        let v = vec![1, 2, 3];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(is_heap(&reversed));

        let v = vec![1, 4, 3, 2];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(!is_heap(&reversed));

        let v = vec![1, 2, 3, 4, 5, 6, 7];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(is_heap(&reversed));

        let v = vec![1, 2, 3, 4, 1, 6, 7];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(!is_heap(&reversed));

        let v = vec![1, 2, 3, 4, 5, 6, 1];
        let reversed = v.into_iter().map(|x| Reverse(x)).collect::<Vec<_>>();
        assert!(!is_heap(&reversed));
    }
}
