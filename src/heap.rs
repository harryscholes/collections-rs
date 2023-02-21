use std::cmp::Reverse;

pub trait Heap<T> {
    fn push(&mut self, el: T);

    fn pop(&mut self) -> Option<T>;

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
        MaxHeap(vec![])
    }

    pub fn with_capacity(capacity: usize) -> MaxHeap<T> {
        MaxHeap(Vec::with_capacity(capacity))
    }

    fn sift_down(&mut self) {
        let mut parent_index = 0;
        let (mut left_sibling_index, mut right_sibling_index) = sibling_indices(parent_index);

        while let Some(parent) = self.0.get(parent_index) {
            match self.0.get(left_sibling_index) {
                Some(left_sibling) => match self.0.get(right_sibling_index) {
                    Some(right_sibling) => {
                        if parent < left_sibling || parent < right_sibling {
                            if left_sibling > right_sibling {
                                self.0.swap(parent_index, left_sibling_index);
                                parent_index = left_sibling_index;
                            } else {
                                self.0.swap(parent_index, right_sibling_index);
                                parent_index = right_sibling_index;
                            }
                        }
                    }
                    None => break,
                },
                None => break,
            }

            (left_sibling_index, right_sibling_index) = sibling_indices(parent_index);
        }

        debug_assert!(is_heap(&self.0));
    }

    fn sift_up(&mut self) {
        let mut sibling_index = self.len() - 1;
        let mut _parent_index = parent_index(sibling_index);

        loop {
            if let Some(parent) = self.0.get(_parent_index) {
                if let Some(sibling) = self.0.get(sibling_index) {
                    if sibling > parent {
                        self.0.swap(_parent_index, sibling_index);
                        sibling_index = _parent_index;
                        _parent_index = parent_index(sibling_index)
                    } else {
                        break;
                    }
                }
            }
        }

        debug_assert!(is_heap(&self.0));
    }
}

impl<T> Heap<T> for MaxHeap<T>
where
    T: Ord,
{
    fn push(&mut self, el: T) {
        self.0.push(el);
        self.sift_up();
    }

    fn pop(&mut self) -> Option<T> {
        match self.peek() {
            Some(_root) => {
                let len = self.len();
                self.0.swap(0, len - 1);
                let root = self.0.pop();
                self.sift_down();
                root
            }
            None => None,
        }
    }

    fn peek(&self) -> Option<&T> {
        self.0.first()
    }

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
        MinHeap(MaxHeap::new())
    }

    pub fn with_capacity(capacity: usize) -> MinHeap<T> {
        MinHeap(MaxHeap::with_capacity(capacity))
    }
}

impl<T> Heap<T> for MinHeap<T>
where
    T: Ord,
{
    fn push(&mut self, el: T) {
        self.0.push(Reverse(el));
    }

    fn pop(&mut self) -> Option<T> {
        self.0.pop().map(|el| el.0)
    }

    fn peek(&self) -> Option<&T> {
        self.0.peek().map(|el| &el.0)
    }

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

pub fn is_heap<T: PartialOrd>(v: &[T]) -> bool {
    recursive_is_heap(v, 0)
}

fn recursive_is_heap<T: PartialOrd>(v: &[T], parent_index: usize) -> bool {
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

    recursive_is_heap(v, left_sibling_index) && recursive_is_heap(v, right_sibling_index)
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
    use super::*;

    use rand::prelude::*;

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
            // dbg!(i, root, &mh.0);
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
