use crate::heap::{Heap, MaxHeap};

#[derive(Eq, Default)]
struct Item<T> {
    value: T,
    priority: usize,
    age: usize, // FIFO ordering
}

impl<T> PartialEq for Item<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> PartialOrd for Item<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match self.priority.partial_cmp(&other.priority) {
            Some(core::cmp::Ordering::Equal) => {
                self.age.partial_cmp(&other.age).map(|ord| ord.reverse())
            }
            ord => ord,
        }
    }
}

impl<T> Ord for Item<T>
where
    T: Eq + PartialOrd,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.priority.cmp(&other.priority) {
            core::cmp::Ordering::Equal => self.age.cmp(&other.age).reverse(),
            ord => ord,
        }
    }
}

#[derive(Default)]
pub struct PriorityQueue<T> {
    heap: MaxHeap<Item<T>>,
    age: usize, // FIFO ordering
}

impl<T> PriorityQueue<T>
where
    T: Ord,
{
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: MaxHeap::with_capacity(capacity),
            age: 0,
        }
    }

    /// Enqueue `value` with `priority`.
    ///
    /// Time complexity: O(log(n))
    pub fn enqueue(&mut self, value: T, priority: usize) {
        self.age += 1;
        self.heap.push(Item {
            value,
            priority,
            age: self.age,
        });
    }

    /// Dequeue the highest priority item.
    ///
    /// Time complexity: O(log(n))
    pub fn dequeue(&mut self) -> Option<T> {
        self.heap.pop().map(|item| item.value)
    }

    /// Delete `value` from the queue.
    ///
    /// If `value` occurs multiple times, the one with highest priority will be deleted.
    ///
    /// Time complexity: O(n)
    pub fn delete(&mut self, value: &T) -> Option<T> {
        self.heap
            .delete_match(|item| item.value == *value)
            .map(|item| item.value)
    }

    /// Update the `priority` of `value`.
    ///
    /// If `value` occurs multiple times, the one with highest priority will be updated.
    ///
    /// Time complexity: O(n)
    pub fn update_priority(&mut self, value: T, priority: usize) -> Option<T> {
        self.delete(&value).map(|deleted| {
            self.enqueue(value, priority);
            deleted
        })
    }

    /// Time complexity: O(1)
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|item| &item.value)
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

pub struct IntoIter<T>(PriorityQueue<T>);

impl<T> Iterator for IntoIter<T>
where
    T: Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.dequeue()
    }
}

impl<T> IntoIterator for PriorityQueue<T>
where
    T: Ord,
{
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self)
    }
}

#[cfg(test)]
mod test {
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_item_priority_trumps_age() {
        let item_1 = Item {
            value: 1,
            priority: 1,
            age: 1,
        };
        let item_2 = Item {
            value: 1,
            priority: 2,
            age: 2,
        };

        assert!(item_1 == item_1);
        assert!(item_1 == item_2);
        assert!(item_1 < item_2);
        assert!(item_2 > item_1);
    }

    #[test]
    fn test_item_ordered_by_age_when_same_priority() {
        let item_1 = Item {
            value: 1,
            priority: 0,
            age: 1,
        };
        let item_2 = Item {
            value: 1,
            priority: 0,
            age: 2,
        };

        assert!(item_1 == item_1);
        assert!(item_1 == item_2);
        assert!(item_1 > item_2);
        assert!(item_2 < item_1);
    }

    #[test]
    fn test_enqueue() {
        let mut pq = PriorityQueue::new();
        pq.enqueue('a', 1);
        assert_eq!(pq.peek(), Some(&'a'));
        pq.enqueue('b', 1);
        assert_eq!(pq.peek(), Some(&'a'));
        pq.enqueue('c', 3);
        assert_eq!(pq.peek(), Some(&'c'));
        pq.enqueue('d', 2);
        assert_eq!(pq.peek(), Some(&'c'));
        pq.enqueue('e', 4);
        assert_eq!(pq.peek(), Some(&'e'));
    }

    #[test]
    fn test_dequeue() {
        let mut pq = PriorityQueue::new();
        pq.enqueue('a', 1);
        pq.enqueue('b', 1);
        pq.enqueue('c', 3);
        pq.enqueue('d', 2);
        pq.enqueue('e', 4);

        assert_eq!(pq.dequeue(), Some('e'));
        assert_eq!(pq.dequeue(), Some('c'));
        assert_eq!(pq.dequeue(), Some('d'));
        assert_eq!(pq.dequeue(), Some('a'));
        assert_eq!(pq.dequeue(), Some('b'));
    }

    #[test]
    fn test_delete() {
        let mut pq = PriorityQueue::new();
        let elements = vec![2, 6, 5, 1, 3, 7, 4];
        for el in elements.clone() {
            pq.enqueue(el, el as usize);
        }
        for el in elements {
            assert_eq!(pq.delete(&el), Some(el));
        }
    }

    #[test]
    fn test_update_priority() {
        let mut pq = PriorityQueue::new();
        pq.enqueue('a', 1);
        assert_eq!(pq.peek(), Some(&'a'));
        pq.enqueue('b', 2);
        assert_eq!(pq.peek(), Some(&'b'));
        pq.update_priority('a', 3);
        assert_eq!(pq.peek(), Some(&'a'));
        assert!(pq.update_priority('z', 1).is_none());
    }

    #[test]
    fn test_equal_priority_behave_as_lifo_queue() {
        let mut pq = PriorityQueue::new();
        for el in 1..=10 {
            pq.enqueue(el, 1);
        }
        for el in 1..=10 {
            assert_eq!(pq.dequeue(), Some(el));
        }
    }

    #[test]
    fn test_len() {
        let mut pq = PriorityQueue::new();
        assert_eq!(pq.len(), 0);
        pq.enqueue('a', 1);
        assert_eq!(pq.len(), 1);
        pq.enqueue('b', 2);
        assert_eq!(pq.len(), 2);
        pq.enqueue('c', 3);
        assert_eq!(pq.len(), 3);
        pq.dequeue();
        assert_eq!(pq.len(), 2);
        pq.dequeue();
        assert_eq!(pq.len(), 1);
        pq.dequeue();
        assert_eq!(pq.len(), 0);
    }

    #[test]
    fn test_is_empty() {
        let mut pq = PriorityQueue::new();
        assert!(pq.is_empty());
        pq.enqueue('a', 1);
        assert!(!pq.is_empty());
        pq.dequeue();
        assert!(pq.is_empty());
    }

    #[test]
    fn test_into_iter() {
        let mut pq = PriorityQueue::new();
        pq.enqueue('a', 1);
        pq.enqueue('b', 1);
        pq.enqueue('c', 3);
        pq.enqueue('d', 2);
        pq.enqueue('e', 4);

        let mut iter = pq.into_iter();
        assert_eq!(iter.next(), Some('e'));
        assert_eq!(iter.next(), Some('c'));
        assert_eq!(iter.next(), Some('d'));
        assert_eq!(iter.next(), Some('a'));
        assert_eq!(iter.next(), Some('b'));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_fuzz() {
        let n = 1_000;
        let mut pq = PriorityQueue::with_capacity(n);
        let mut rng = thread_rng();
        let mut expected_root = usize::MIN;
        let mut elements = vec![];
        for _ in 0..n {
            let el = rng.gen_range(0..usize::MAX);
            elements.push(el);
            pq.enqueue(el, el);
            if el > expected_root {
                expected_root = el
            }
            assert_eq!(pq.peek(), Some(&expected_root));
        }

        elements.sort();
        elements.reverse();

        for el in elements {
            assert_eq!(pq.dequeue(), Some(el));
        }
    }
}
