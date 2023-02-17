use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

pub struct Node<T> {
    pub element: T,
    next: Option<Rc<RefCell<Node<T>>>>,
    prev: Option<Rc<RefCell<Node<T>>>>,
}

impl<T> Node<T> {
    fn new(element: T) -> Self {
        Node {
            element,
            next: None,
            prev: None,
        }
    }
}

/// Space complexity: O(n)
pub struct LinkedList<T> {
    head: Option<Rc<RefCell<Node<T>>>>,
    tail: Option<Rc<RefCell<Node<T>>>>,
    len: usize,
}

impl<T> LinkedList<T>
where
    T: Clone,
{
    pub fn new() -> Self {
        LinkedList {
            head: None,
            tail: None,
            len: 0,
        }
    }

    /// Time complexity: O(1)
    pub fn push_back(&mut self, elt: T) -> Weak<RefCell<Node<T>>> {
        let node = Rc::new(RefCell::new(Node::new(elt)));
        node.borrow_mut().next = None;
        match &self.tail {
            None => {
                self.head = Some(node.clone());
                node.borrow_mut().prev = None;
            }
            Some(tail) => {
                tail.borrow_mut().next = Some(node.clone());
                node.borrow_mut().prev = Some(tail.clone());
            }
        }
        self.tail = Some(node.clone());
        self.len += 1;
        Rc::downgrade(&node)
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, elt: T) -> Weak<RefCell<Node<T>>> {
        let node = Rc::new(RefCell::new(Node::new(elt)));
        node.borrow_mut().prev = None;
        match &self.head {
            None => {
                self.tail = Some(node.clone());
                node.borrow_mut().next = None;
            }
            Some(head) => {
                head.borrow_mut().prev = Some(node.clone());
                node.borrow_mut().next = Some(head.clone());
            }
        }
        self.head = Some(node.clone());
        self.len += 1;
        Rc::downgrade(&node)
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.clone().map(|node| self.unlink(node))
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.clone().map(|node| self.unlink(node))
    }

    /// Time complexity: O(1)
    pub fn unlink(&mut self, node: Rc<RefCell<Node<T>>>) -> T {
        match node.borrow().prev.clone() {
            Some(prev) => prev.borrow_mut().next = node.borrow().next.clone(),
            None => self.head = node.borrow().next.clone(),
        };
        match node.borrow().next.clone() {
            Some(next) => next.borrow_mut().prev = node.borrow().prev.clone(),
            None => self.tail = node.borrow().prev.clone(),
        };
        self.len -= 1;
        match Rc::try_unwrap(node) {
            Ok(node) => node.into_inner().element,
            Err(_) => panic!("Unwrapping `Rc` failed because more than one reference exists"),
        }
    }

    /// Time complexity: O(1)
    pub fn first(&self) -> Option<T> {
        self.head.clone().map(|node| node.borrow().element.clone())
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<T> {
        self.tail.clone().map(|node| node.borrow().element.clone())
    }

    /// Time complexity: O(1)
    pub fn peek_front(&self) -> Option<Weak<RefCell<Node<T>>>> {
        self.head.clone().map(|head| Rc::downgrade(&head))
    }

    /// Time complexity: O(1)
    pub fn peek_back(&self) -> Option<Weak<RefCell<Node<T>>>> {
        self.tail.clone().map(|tail| Rc::downgrade(&tail))
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }
}

impl<T> Default for LinkedList<T>
where
    T: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<T> {
    head: Option<Rc<RefCell<Node<T>>>>,
    tail: Option<Rc<RefCell<Node<T>>>>,
}

impl<T> Iter<T>
where
    T: Clone,
{
    pub fn new(l: &LinkedList<T>) -> Iter<T> {
        Iter {
            head: l.head.clone(),
            tail: l.tail.clone(),
        }
    }
}

impl<T> Iterator for Iter<T>
where
    T: Clone + std::cmp::PartialEq,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.head.clone() {
            None => None,
            Some(head) => {
                let elt = head.borrow().element.clone();
                if let Some(tail) = self.tail.clone() {
                    if Rc::ptr_eq(&head, &tail) {
                        self.head = None;
                        self.tail = None;
                        return Some(elt);
                    }
                }
                self.head = head.borrow().next.clone();
                Some(elt)
            }
        }
    }
}

impl<T> IntoIterator for LinkedList<T>
where
    T: Clone + std::cmp::PartialEq,
{
    type Item = T;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(&self)
    }
}

impl<T> DoubleEndedIterator for Iter<T>
where
    T: Clone + std::cmp::PartialEq,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.tail.clone() {
            None => None,
            Some(tail) => {
                let elt = tail.borrow().element.clone();
                if let Some(head) = self.head.clone() {
                    if Rc::ptr_eq(&head, &tail) {
                        self.head = None;
                        self.tail = None;
                        return Some(elt);
                    }
                }
                self.tail = tail.borrow().prev.clone();
                Some(elt)
            }
        }
    }
}

impl<T> FromIterator<T> for LinkedList<T>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut l = LinkedList::new();
        for elt in iter {
            l.push_back(elt);
        }
        l
    }
}

impl<T, const N: usize> From<[T; N]> for LinkedList<T>
where
    T: Clone,
{
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first() {
        let mut l = LinkedList::new();
        assert!(l.first().is_none());

        l.push_front(1);
        assert_eq!(l.first(), Some(1));

        l.push_front(2);
        assert_eq!(l.first(), Some(2));
    }

    #[test]
    fn test_last() {
        let mut l = LinkedList::new();
        assert!(l.last().is_none());

        l.push_back(1);
        assert_eq!(l.last(), Some(1));

        l.push_back(2);
        assert_eq!(l.last(), Some(2));
    }

    #[test]
    fn test_push_back() {
        let mut l = LinkedList::new();
        assert!(l.first().is_none());
        assert!(l.last().is_none());

        l.push_back(1);
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(1));

        l.push_back(2);
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(2));

        l.push_back(3);
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(3));
    }

    #[test]
    fn test_push_front() {
        let mut l = LinkedList::new();
        assert!(l.first().is_none());
        assert!(l.last().is_none());

        l.push_front(1);
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(1));

        l.push_front(2);
        assert_eq!(l.first(), Some(2));
        assert_eq!(l.last(), Some(1));

        l.push_front(3);
        assert_eq!(l.first(), Some(3));
        assert_eq!(l.last(), Some(1));
    }

    #[test]
    fn test_pop_back() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        assert_eq!(l.pop_back(), Some(3));
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(2));

        assert_eq!(l.pop_back(), Some(2));
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(1));

        assert_eq!(l.pop_back(), Some(1));
        assert!(l.first().is_none());
        assert!(l.last().is_none());
    }

    #[test]
    fn test_pop_front() {
        let mut l = LinkedList::new();
        l.push_front(1);
        l.push_front(2);
        l.push_front(3);

        assert_eq!(l.pop_front(), Some(3));
        assert_eq!(l.first(), Some(2));
        assert_eq!(l.last(), Some(1));

        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(1));

        assert_eq!(l.pop_front(), Some(1));
        assert!(l.first().is_none());
        assert!(l.last().is_none());
    }

    #[test]
    fn test_unlink() {
        let mut l = LinkedList::new();
        l.push_front(4);
        let node_3 = l.push_front(3);
        let node_2 = l.push_front(2);
        l.push_front(1);
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(4));

        l.unlink(node_2.upgrade().unwrap());
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(4));
        assert_eq!(l.iter().nth(1), Some(3));
        assert_eq!(l.iter().rev().nth(1), Some(3));

        l.unlink(node_3.upgrade().unwrap());
        assert_eq!(l.first(), Some(1));
        assert_eq!(l.last(), Some(4));
        assert_eq!(l.iter().nth(1), Some(4));
        assert_eq!(l.iter().rev().nth(1), Some(1));
    }

    #[test]
    fn test_node_weak_pointer() {
        let mut l = LinkedList::new();
        l.push_front(3);
        let weak_node_2 = l.push_front(2);
        l.push_front(1);

        let node_2 = weak_node_2.upgrade().unwrap();
        l.unlink(node_2);
        assert!(weak_node_2.upgrade().is_none());
    }

    #[test]
    #[should_panic]
    fn test_unlink_panic() {
        let mut l = LinkedList::new();
        l.push_front(3);
        let weak_node_2 = l.push_front(2);
        l.push_front(1);

        let node_2 = weak_node_2.upgrade().unwrap();
        let _another_node_2_ref = node_2.clone();
        l.unlink(node_2);
    }

    #[test]
    fn test_len() {
        let mut l = LinkedList::new();
        assert!(l.is_empty());

        l.push_back(1);
        assert_eq!(l.len(), 1);

        l.push_front(2);
        assert_eq!(l.len(), 2);

        l.pop_back();
        assert_eq!(l.len(), 1);

        l.pop_front();
        assert!(l.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        let mut iter = l.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_for_loop() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        let mut expected = 1;
        for el in l {
            assert_eq!(el, expected);
            expected += 1;
        }
    }

    #[test]
    fn test_iter_rev() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        let mut iter = l.iter().rev();
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_double_ended_iterator() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);
        l.push_back(4);
        l.push_back(5);
        l.push_back(6);

        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(6));
        assert_eq!(iter.next_back(), Some(5));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_from_iter() {
        let iter = 1..=3;
        let mut l = LinkedList::from_iter(iter);
        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![1, 2, 3];
        let mut l = LinkedList::from_iter(vec.into_iter());
        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));
    }

    #[test]
    fn test_from_arr() {
        let arr = [1, 2, 3];
        let mut l = LinkedList::from(arr);
        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));
    }
}
