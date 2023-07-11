use std::{
    marker::PhantomData,
    ops::{Index, IndexMut},
    ptr::NonNull,
};

/// Space complexity: O(n)
#[derive(Debug, Eq)]
pub struct LinkedList<T> {
    head: Link<T>,
    tail: Link<T>,
    len: usize,
}

type Link<T> = Option<NonNull<Node<T>>>;

/// Space complexity: O(1)
#[derive(Debug)]
pub struct Node<T> {
    element: T,
    next: Link<T>,
    prev: Link<T>,
}

impl<T> Node<T> {
    fn new(element: T) -> Self {
        Self {
            element,
            next: None,
            prev: None,
        }
    }
}

impl<T> From<Node<T>> for NonNull<Node<T>> {
    fn from(node: Node<T>) -> Self {
        unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(node))) }
    }
}

impl<T> From<NonNull<Node<T>>> for Node<T> {
    fn from(ptr: NonNull<Node<T>>) -> Self {
        unsafe { *Box::from_raw(ptr.as_ptr()) }
    }
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
        }
    }

    /// Time complexity: O(1)
    pub fn push_back(&mut self, elt: T) {
        unsafe {
            let mut new_tail = Node::new(elt).into();
            match self.tail {
                Some(mut old_tail) => {
                    old_tail.as_mut().next = Some(new_tail);
                    new_tail.as_mut().prev = Some(old_tail);
                }
                None => self.head = Some(new_tail),
            }
            self.tail = Some(new_tail);
            self.len += 1;
        }
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, elt: T) {
        unsafe {
            let mut new_head = Node::new(elt).into();
            match self.head {
                Some(mut old_head) => {
                    old_head.as_mut().prev = Some(new_head);
                    new_head.as_mut().next = Some(old_head);
                }
                None => self.tail = Some(new_head),
            }
            self.head = Some(new_head);
            self.len += 1;
        }
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        unsafe {
            self.tail.map(|old_tail| {
                let old_tail: Node<_> = old_tail.into();
                match old_tail.prev {
                    Some(mut new_tail) => {
                        new_tail.as_mut().next = None;
                        self.tail = Some(new_tail);
                    }
                    None => {
                        self.head = None;
                        self.tail = None;
                    }
                }
                self.len -= 1;
                old_tail.element
            })
        }
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        unsafe {
            self.head.map(|old_head| {
                let old_head: Node<_> = old_head.into();
                match old_head.next {
                    Some(mut new_head) => {
                        new_head.as_mut().prev = None;
                        self.head = Some(new_head);
                    }
                    None => {
                        self.head = None;
                        self.tail = None;
                    }
                }
                self.len -= 1;
                old_head.element
            })
        }
    }

    /// Time complexity: O(n)
    pub fn remove(&mut self, index: usize) -> Option<T> {
        unsafe {
            if index >= self.len() || self.is_empty() {
                None
            } else if index == 0 {
                self.pop_front()
            } else if index == self.len() - 1 {
                self.pop_back()
            } else {
                let el = self.node_iter_mut().nth(index).map(|node| {
                    let mut prev = node.prev.expect("`Node` has no `prev` `Node`");
                    prev.as_mut().next = node.next;

                    let mut next = node.next.expect("`Node` has no `next` `Node`");
                    next.as_mut().prev = node.prev;

                    // Hack to get hold of `Node<T>`.
                    // Safety: using `node_iter_mut` guarantees that we only have one reference to `self` and `node`,
                    // so it is safe to convert `&mut Node<T>` to `Node<T>`.
                    let ptr: NonNull<Node<_>> = node.into();
                    let node: Node<_> = ptr.into();
                    node.element
                });
                // Hack to get around the borrow checeker
                if el.is_some() {
                    self.len -= 1;
                }
                el
            }
        }
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.len
    }

    /// Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Time complexity: O(1)
    pub fn first(&self) -> Option<&T> {
        self.head.map(|head| unsafe { &head.as_ref().element })
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<&T> {
        self.tail.map(|tail| unsafe { &tail.as_ref().element })
    }

    /// Time complexity: O(n)
    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }

    /// Time complexity: O(n)
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
    }

    /// Time complexity: O(n)
    fn node_iter(&self) -> NodeIter<'_, T> {
        NodeIter::new(self)
    }

    /// Time complexity: O(n)
    fn node_iter_mut(&mut self) -> NodeIterMut<'_, T> {
        NodeIterMut::new(self)
    }
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<usize> for LinkedList<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.into_iter().nth(index).expect("index out of bounds")
    }
}

impl<T> IndexMut<usize> for LinkedList<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.into_iter().nth(index).expect("index out of bounds")
    }
}

struct NodeIter<'a, T> {
    head: Link<T>,
    tail: Link<T>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> NodeIter<'a, T> {
    pub fn new(l: &'a LinkedList<T>) -> Self {
        Self {
            head: l.head,
            tail: l.tail,
            marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for NodeIter<'a, T> {
    type Item = &'a Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.head.map(|old_head| unsafe {
            if self.head == self.tail {
                self.head = None;
                self.tail = None;
            } else {
                let new_head = old_head.as_ref().next;
                self.head = new_head;
            }
            old_head.as_ref()
        })
    }
}

impl<'a, T> DoubleEndedIterator for NodeIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.tail.map(|old_tail| unsafe {
            if self.head == self.tail {
                self.head = None;
                self.tail = None;
            } else {
                let new_tail = old_tail.as_ref().prev;
                self.tail = new_tail;
            }
            old_tail.as_ref()
        })
    }
}

struct NodeIterMut<'a, T> {
    head: Link<T>,
    tail: Link<T>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> NodeIterMut<'a, T> {
    pub fn new(l: &'a mut LinkedList<T>) -> Self {
        Self {
            head: l.head,
            tail: l.tail,
            marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for NodeIterMut<'a, T> {
    type Item = &'a mut Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.head.map(|mut old_head| unsafe {
            if self.head == self.tail {
                self.head = None;
                self.tail = None;
            } else {
                let new_head = old_head.as_ref().next;
                self.head = new_head;
            }
            old_head.as_mut()
        })
    }
}

impl<'a, T> DoubleEndedIterator for NodeIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.tail.map(|mut old_tail| unsafe {
            if self.head == self.tail {
                self.head = None;
                self.tail = None;
            } else {
                let new_tail = old_tail.as_ref().prev;
                self.tail = new_tail;
            }
            old_tail.as_mut()
        })
    }
}

pub struct Iter<'a, T>(NodeIter<'a, T>);

impl<'a, T> Iter<'a, T> {
    pub fn new(l: &'a LinkedList<T>) -> Self {
        Self(l.node_iter())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|node| &node.element)
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|node| &node.element)
    }
}

impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IterMut<'a, T>(NodeIterMut<'a, T>);

impl<'a, T> IterMut<'a, T> {
    pub fn new(l: &'a mut LinkedList<T>) -> Self {
        Self(l.node_iter_mut())
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|node| &mut node.element)
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|node| &mut node.element)
    }
}

impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
    }
}

pub struct IntoIter<T>(LinkedList<T>);

impl<T> IntoIter<T> {
    pub fn new(l: LinkedList<T>) -> Self {
        Self(l)
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back()
    }
}

impl<T> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<T> FromIterator<T> for LinkedList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut l = LinkedList::new();
        for elt in iter {
            l.push_back(elt);
        }
        l
    }
}

impl<T, const N: usize> From<[T; N]> for LinkedList<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr.into_iter())
    }
}

impl<T> PartialEq for LinkedList<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<T> Clone for LinkedList<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut ll = Self::new();
        for el in self {
            ll.push_back(el.clone())
        }
        ll
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        while self.pop_front().is_some() {}
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
        assert_eq!(l.first(), Some(&1));

        l.push_front(2);
        assert_eq!(l.first(), Some(&2));
    }

    #[test]
    fn test_last() {
        let mut l = LinkedList::new();
        assert!(l.last().is_none());

        l.push_back(1);
        assert_eq!(l.last(), Some(&1));

        l.push_back(2);
        assert_eq!(l.last(), Some(&2));
    }

    #[test]
    fn test_push_back() {
        let mut l = LinkedList::new();
        assert!(l.first().is_none());
        assert!(l.last().is_none());

        l.push_back(1);
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&1));

        l.push_back(2);
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&2));

        l.push_back(3);
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&3));
    }

    #[test]
    fn test_push_front() {
        let mut l = LinkedList::new();
        assert!(l.first().is_none());
        assert!(l.last().is_none());

        l.push_front(1);
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&1));

        l.push_front(2);
        assert_eq!(l.first(), Some(&2));
        assert_eq!(l.last(), Some(&1));

        l.push_front(3);
        assert_eq!(l.first(), Some(&3));
        assert_eq!(l.last(), Some(&1));
    }

    #[test]
    fn test_pop_back() {
        let mut l = LinkedList::from([1, 2, 3]);

        assert_eq!(l.pop_back(), Some(3));
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&2));

        assert_eq!(l.pop_back(), Some(2));
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&1));

        assert_eq!(l.pop_back(), Some(1));
        assert!(l.first().is_none());
        assert!(l.last().is_none());
    }

    #[test]
    fn test_pop_front() {
        let mut l = LinkedList::from([3, 2, 1]);

        assert_eq!(l.pop_front(), Some(3));
        assert_eq!(l.first(), Some(&2));
        assert_eq!(l.last(), Some(&1));

        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.first(), Some(&1));
        assert_eq!(l.last(), Some(&1));

        assert_eq!(l.pop_front(), Some(1));
        assert!(l.first().is_none());
        assert!(l.last().is_none());
    }

    #[test]
    fn test_len() {
        let mut l = LinkedList::new();
        assert!(l.is_empty());

        l.push_back(1);
        assert_eq!(l.len(), 1);

        l.push_front(2);
        assert_eq!(l.len(), 2);

        assert_eq!(l.pop_back(), Some(1));
        assert_eq!(l.len(), 1);

        assert_eq!(l.pop_front(), Some(2));
        assert!(l.is_empty());
    }

    #[test]
    fn test_iter() {
        let l: LinkedList<usize> = LinkedList::new();
        let mut iter = l.iter();
        assert_eq!(iter.next(), None);

        let l = LinkedList::from([1, 2, 3]);
        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let l: LinkedList<usize> = LinkedList::new();
        let mut iter = (&l).into_iter();
        assert_eq!(iter.next(), None);

        let l = LinkedList::from([1, 2, 3]);
        let mut iter = (&l).into_iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_node_iter_double_ended_iterator() {
        let l = LinkedList::from_iter(1..=6);
        let mut iter = l.node_iter();
        assert_eq!(iter.next().unwrap().element, 1);
        assert_eq!(iter.next_back().unwrap().element, 6);
        assert_eq!(iter.next_back().unwrap().element, 5);
        assert_eq!(iter.next().unwrap().element, 2);
        assert_eq!(iter.next().unwrap().element, 3);
        assert_eq!(iter.next().unwrap().element, 4);
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_node_iter_mut() {
        let mut l = LinkedList::from([1, 2, 3]);
        l.node_iter_mut().for_each(|node| node.element += 1);
        assert_eq!(l, LinkedList::from([2, 3, 4]));
    }

    #[test]
    fn test_iter_mut() {
        let mut l: LinkedList<usize> = LinkedList::new();
        let mut iter = l.iter_mut();
        assert_eq!(iter.next(), None);

        let mut l = LinkedList::from([1, 2, 3]);
        let mut iter = l.iter_mut();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut_rev() {
        let mut l = LinkedList::from([1, 2, 3]);
        let mut iter = l.iter_mut().rev();
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_mut_borrowed_into_iter() {
        let mut l = LinkedList::from([1, 2, 3]);
        let mut iter = (&mut l).into_iter();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let l: LinkedList<usize> = LinkedList::new();
        let mut iter = l.into_iter();
        assert_eq!(iter.next(), None);

        let l = LinkedList::from([1, 2, 3]);
        let mut iter = l.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_for_loop() {
        let l = LinkedList::from([1, 2, 3]);
        let mut expected = 1;
        for el in l {
            assert_eq!(el, expected);
            expected += 1;
        }
    }

    #[test]
    fn test_iter_rev() {
        let l = LinkedList::from([1, 2, 3]);
        let mut iter = l.iter().rev();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_rev() {
        let l = LinkedList::from([1, 2, 3]);
        let mut iter = l.into_iter().rev();
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_double_ended_iterator() {
        let l = LinkedList::from_iter(1..=6);
        let mut iter = l.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&6));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_double_ended_iterator() {
        let l = LinkedList::from_iter(1..=6);
        let mut iter = l.into_iter();
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
        assert_eq!(l.pop_front(), None);
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![1, 2, 3];
        let mut l = LinkedList::from_iter(vec.into_iter());
        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));
        assert_eq!(l.pop_front(), None);
    }

    #[test]
    fn test_from_arr() {
        let arr = [1, 2, 3];
        let mut l = LinkedList::from(arr);
        assert_eq!(l.pop_front(), Some(1));
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.pop_front(), Some(3));
        assert_eq!(l.pop_front(), None);
    }

    #[test]
    fn test_remove() {
        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(0), Some(1));
        assert_eq!(l.len(), 2);

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(2), Some(3));
        assert_eq!(l.len(), 2);

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(3), None);
        assert_eq!(l.len(), 3);

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(1), Some(2));
        assert_eq!(l.len(), 2);

        let mut l = LinkedList::from([1, 2, 3, 4]);
        assert_eq!(l.remove(2), Some(3));
        assert_eq!(l.len(), 3);
    }

    #[test]
    fn test_eq() {
        assert_eq!(LinkedList::<i32>::new(), LinkedList::<i32>::new());
        assert_ne!(LinkedList::new(), LinkedList::from([1, 2, 3]));
        assert_eq!(LinkedList::from([1, 2, 3]), LinkedList::from([1, 2, 3]));
        assert_ne!(LinkedList::from([1, 2, 3]), LinkedList::from([1, 2, 3, 4]));
    }

    #[test]
    fn test_clone() {
        let l: LinkedList<i32> = LinkedList::new();
        let c = l.clone();
        assert_eq!(l, c);

        let l = LinkedList::from([1, 2, 3]);
        let c = l.clone();
        assert_eq!(l, c);
    }

    #[test]
    fn test_index() {
        let mut l = LinkedList::new();
        l.push_back(0);
        l.push_back(1);
        assert_eq!(l[0], 0);
        assert_eq!(l[1], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let mut l = LinkedList::new();
        l.push_back(0);
        l.push_back(1);
        l[2];
    }

    #[test]

    fn test_index_mut() {
        let mut l = LinkedList::new();
        l.push_back(0);
        l.push_back(1);
        assert_eq!(l.last(), Some(&1));
        l[1] = 2;
        assert_eq!(l.last(), Some(&2));
    }

    #[test]
    #[should_panic]
    fn test_index_mut_out_of_bounds() {
        let mut l = LinkedList::new();
        l.push_back(0);
        l.push_back(1);
        l[2] = 2;
    }
}
