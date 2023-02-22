use crate::heap::{Heap, MaxHeap};

#[derive(Eq, PartialEq)]
struct PriorityQueueElement<T, P> {
    el: T,
    priority: P,
}

impl<T, P> PartialOrd for PriorityQueueElement<T, P>
where
    T: PartialEq,
    P: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.priority.cmp(&other.priority))
    }
}

impl<T, P> Ord for PriorityQueueElement<T, P>
where
    T: Eq,
    P: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

pub struct PriorityQueue<T, P>(MaxHeap<PriorityQueueElement<T, P>>);

impl<T, P> PriorityQueue<T, P>
where
    T: Eq,
    P: Ord,
{
    pub fn new() -> PriorityQueue<T, P> {
        PriorityQueue(MaxHeap::new())
    }

    pub fn with_capacity(capacity: usize) -> PriorityQueue<T, P> {
        PriorityQueue(MaxHeap::with_capacity(capacity))
    }

    pub fn enqueue(&mut self, el: T, priority: P) {
        self.0.push(PriorityQueueElement { el, priority });
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.0.pop().map(|item| item.el)
    }

    pub fn peek(&self) -> Option<&T> {
        self.0.peek().map(|item| &item.el)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod test_max_heap {
    use std::cmp::Ordering;

    use super::*;

    #[test]
    fn test_priority_queue_element() {
        let el_1 = PriorityQueueElement {
            el: 1,
            priority: 10,
        };
        let el_2 = PriorityQueueElement {
            el: 1,
            priority: 10,
        };
        let el_3 = PriorityQueueElement { el: 2, priority: 9 };

        // Eq and PartialEq
        assert!(el_1 == el_1);
        assert!(el_1 == el_2);
        assert!(el_1 != el_3);

        // Ord
        assert_eq!(el_1.cmp(&el_2), Ordering::Equal);
        assert_eq!(el_1.cmp(&el_3), Ordering::Greater);
        assert_eq!(el_3.cmp(&el_1), Ordering::Less);
    }

    #[test]
    fn test_enqueue() {
        let mut p = PriorityQueue::new();
        p.enqueue('a', 1);
        assert_eq!(p.peek(), Some(&'a'));
        p.enqueue('b', 1);
        assert_eq!(p.peek(), Some(&'a'));
        p.enqueue('c', 3);
        assert_eq!(p.peek(), Some(&'c'));
        p.enqueue('d', 2);
        assert_eq!(p.peek(), Some(&'c'));
        p.enqueue('e', 4);
        assert_eq!(p.peek(), Some(&'e'));
    }

    #[test]
    fn test_dequeue() {
        let mut p = PriorityQueue::new();
        p.enqueue('a', 1);
        p.enqueue('b', 1);
        p.enqueue('c', 3);
        p.enqueue('d', 2);
        p.enqueue('e', 4);

        assert_eq!(p.dequeue(), Some('e'));
        assert_eq!(p.dequeue(), Some('c'));
        assert_eq!(p.dequeue(), Some('d'));
        assert_eq!(p.dequeue(), Some('a'));
        assert_eq!(p.dequeue(), Some('b'));
    }

    #[test]
    fn test_len() {
        let mut p = PriorityQueue::new();
        assert_eq!(p.len(), 0);
        p.enqueue('a', 1);
        assert_eq!(p.len(), 1);
        p.enqueue('b', 2);
        assert_eq!(p.len(), 2);
        p.enqueue('c', 3);
        assert_eq!(p.len(), 3);
        p.dequeue();
        assert_eq!(p.len(), 2);
        p.dequeue();
        assert_eq!(p.len(), 1);
        p.dequeue();
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_is_empty() {
        let mut p = PriorityQueue::new();
        assert!(p.is_empty());
        p.enqueue('a', 1);
        assert!(!p.is_empty());
        p.dequeue();
        assert!(p.is_empty());
    }
}
