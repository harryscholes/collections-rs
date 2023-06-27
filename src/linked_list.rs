use std::marker::PhantomData;

/// Space complexity: O(n)
#[derive(Debug)]
pub struct LinkedList<T> {
    head: Link<T>,
    tail: Link<T>,
    len: usize,
}

/// Space complexity: O(1)
#[derive(Debug)]
pub struct Node<T> {
    element: T,
    next: Link<T>,
    prev: Link<T>,
}

type Link<T> = Option<*mut Node<T>>;

impl<T> Node<T> {
    fn new(element: T) -> Self {
        Self {
            element,
            next: None,
            prev: None,
        }
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
        let new_tail = Box::into_raw(Box::new(Node::new(elt)));
        match self.tail {
            Some(old_tail) => unsafe {
                (*old_tail).next = Some(new_tail);
                (*new_tail).prev = Some(old_tail);
            },
            None => self.head = Some(new_tail),
        }
        self.tail = Some(new_tail);
        self.len += 1;
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, elt: T) {
        let new_head = Box::into_raw(Box::new(Node::new(elt)));
        match self.head {
            Some(old_head) => unsafe {
                (*old_head).prev = Some(new_head);
                (*new_head).next = Some(old_head);
            },
            None => self.tail = Some(new_head),
        }
        self.head = Some(new_head);
        self.len += 1;
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.map(|old_tail| unsafe {
            match (*old_tail).prev {
                Some(new_tail) => {
                    self.tail = Some(new_tail);
                    (*new_tail).next = None;
                }
                None => {
                    self.head = None;
                    self.tail = None;
                }
            }
            let el = Box::from_raw(old_tail).element;
            self.len -= 1;
            el
        })
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.map(|old_head| unsafe {
            match (*old_head).next {
                Some(new_head) => {
                    self.head = Some(new_head);
                    (*new_head).prev = None;
                }
                None => {
                    self.head = None;
                    self.tail = None;
                }
            }
            let el = Box::from_raw(old_head).element;
            self.len -= 1;
            el
        })
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
        self.head.map(|head| unsafe { &(*head).element })
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<&T> {
        self.tail.map(|tail| unsafe { &(*tail).element })
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
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() || self.is_empty() {
            None
        } else if index == 0 {
            self.pop_front()
        } else if index == self.len() - 1 {
            self.pop_back()
        } else {
            self.cursor_mut().nth(index).map(|node| unsafe {
                if let Some(prev) = (*node).prev {
                    (*prev).next = (*node).next;
                }
                if let Some(next) = (*node).next {
                    (*next).prev = (*node).prev;
                }
                self.len -= 1;
                Box::from_raw(node).element
            })
        }
    }

    fn cursor(&self) -> Cursor<'_, T> {
        Cursor::new(self)
    }

    fn cursor_mut(&mut self) -> Cursor<'_, T> {
        Cursor::new(self)
    }
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A private `Cursor` type for iteration over `Node`s in a `LinkedList`.
/// Note that a `Cursor` is constructed using immutable borrows of `LinkedList`s,
/// so that the `Iter` and `IterMut` API can be built using this type.
struct Cursor<'a, T> {
    head: Link<T>,
    tail: Link<T>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> Cursor<'a, T> {
    pub fn new(l: &LinkedList<T>) -> Self {
        Self {
            head: l.head,
            tail: l.tail,
            marker: PhantomData,
        }
    }
}

impl<'a, T> Iterator for Cursor<'a, T> {
    type Item = *mut Node<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.head {
            Some(old_head) => unsafe {
                if self.head == self.tail {
                    self.head = None;
                    self.tail = None;
                } else {
                    let new_head = (*old_head).next;
                    self.head = new_head;
                }
                Some(old_head)
            },
            None => None,
        }
    }
}

impl<'a, T> DoubleEndedIterator for Cursor<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.tail {
            Some(old_tail) => unsafe {
                if self.head == self.tail {
                    self.head = None;
                    self.tail = None;
                } else {
                    let new_tail = (*old_tail).prev;
                    self.tail = new_tail;
                }
                Some(old_tail)
            },
            None => None,
        }
    }
}

pub struct Iter<'a, T>(Cursor<'a, T>);

impl<'a, T> Iter<'a, T> {
    pub fn new(l: &'a LinkedList<T>) -> Self {
        Self(l.cursor())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|node| unsafe { &(*node).element })
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|node| unsafe { &(*node).element })
    }
}

impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IterMut<'a, T>(Cursor<'a, T>);

impl<'a, T> IterMut<'a, T> {
    pub fn new(l: &'a mut LinkedList<T>) -> Self {
        Self(l.cursor_mut())
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|node| unsafe { &mut (*node).element })
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0
            .next_back()
            .map(|node| unsafe { &mut (*node).element })
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

        l.pop_back();
        assert_eq!(l.len(), 1);

        l.pop_front();
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
    fn test_cursor_double_ended_iterator() {
        let l = LinkedList::from_iter(1..=6);
        let mut cursor = l.cursor();
        assert_eq!(unsafe { (*cursor.next().unwrap()).element }, 1);
        assert_eq!(unsafe { (*cursor.next_back().unwrap()).element }, 6);
        assert_eq!(unsafe { (*cursor.next_back().unwrap()).element }, 5);
        assert_eq!(unsafe { (*cursor.next().unwrap()).element }, 2);
        assert_eq!(unsafe { (*cursor.next().unwrap()).element }, 3);
        assert_eq!(unsafe { (*cursor.next().unwrap()).element }, 4);
        assert_eq!(cursor.next(), None);
        assert_eq!(cursor.next_back(), None);
    }

    #[test]
    fn test_cursor_iter_mut() {
        let mut l = LinkedList::from([1, 2, 3]);
        let cursor = l.cursor_mut();
        cursor.for_each(|node| unsafe { (*node).element += 1 });
        assert_eq!(l, LinkedList::from([2, 3, 4]));

        let mut l = LinkedList::from([1, 2, 3]);
        let cursor = l.cursor_mut();
        cursor.for_each(|node| unsafe { (*node).element += 1 });
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

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(2), Some(3));

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(3), None);

        let mut l = LinkedList::from([1, 2, 3]);
        assert_eq!(l.remove(1), Some(2));

        let mut l = LinkedList::from([1, 2, 3, 4]);
        assert_eq!(l.remove(2), Some(3));
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
}
