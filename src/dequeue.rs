use crate::circular_buffer::{self, CircularBuffer};

/// Space complexity: O(n)
#[derive(Clone, Debug)]
pub struct Dequeue<T>(CircularBuffer<T>);

impl<T> Dequeue<T> {
    pub fn new() -> Self {
        Self::with_capacity(1)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(CircularBuffer::with_capacity(capacity))
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn push_back(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.capacity())
        }
        self.0.push_back(el);
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn push_front(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.capacity())
        }
        self.0.push_front(el);
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        self.0.pop_back()
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        self.0.pop_front()
    }

    /// Time complexity: O(1)
    pub fn peek_back(&self) -> Option<&T> {
        self.0.last()
    }

    /// Time complexity: O(1)
    pub fn peek_front(&self) -> Option<&T> {
        self.0.first()
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }
}

impl<T> Default for Dequeue<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, T>(circular_buffer::Iter<'a, T>);

impl<'a, T> Iter<'a, T> {
    fn new(q: &'a Dequeue<T>) -> Self {
        Self(q.0.iter())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<'a, T> IntoIterator for &'a Dequeue<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IntoIter<T>(circular_buffer::IntoIter<T>);

impl<T> IntoIter<T> {
    fn new(q: Dequeue<T>) -> Self {
        Self(q.0.into_iter())
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<T> IntoIterator for Dequeue<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<T> FromIterator<T> for Dequeue<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut d = Dequeue::new();
        for el in iter {
            d.push_back(el)
        }
        d
    }
}

impl<T, const N: usize> From<[T; N]> for Dequeue<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut d = Dequeue::new();
        assert_eq!(d.peek_back(), None);
        assert_eq!(d.peek_front(), None);
        assert_eq!(d.pop_back(), None);
        assert_eq!(d.pop_front(), None);
        d.push_back(0);
        assert_eq!(d.peek_back(), Some(&0));
        assert_eq!(d.peek_front(), Some(&0));
        d.push_front(1);
        assert_eq!(d.peek_back(), Some(&0));
        assert_eq!(d.peek_front(), Some(&1));
        d.push_back(2);
        assert_eq!(d.peek_back(), Some(&2));
        assert_eq!(d.peek_front(), Some(&1));
        d.push_front(3);
        assert_eq!(d.peek_back(), Some(&2));
        assert_eq!(d.peek_front(), Some(&3));
        assert_eq!(d.pop_back(), Some(2));
        assert_eq!(d.peek_back(), Some(&0));
        assert_eq!(d.peek_front(), Some(&3));
        assert_eq!(d.pop_front(), Some(3));
        assert_eq!(d.peek_back(), Some(&0));
        assert_eq!(d.peek_front(), Some(&1));
        assert_eq!(d.pop_back(), Some(0));
        assert_eq!(d.peek_back(), Some(&1));
        assert_eq!(d.peek_front(), Some(&1));
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(d.peek_back(), None);
        assert_eq!(d.peek_front(), None);
        assert_eq!(d.pop_front(), None);
        assert_eq!(d.pop_front(), None);
    }

    #[test]
    fn test_iter() {
        let d = Dequeue::from_iter(0..=2);
        let mut iter = d.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_rev() {
        let d = Dequeue::from_iter(0..=2);
        let mut iter = d.iter().rev();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_double_ended_iterator() {
        let d = Dequeue::from_iter(1..=6);
        let mut iter = d.iter();
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
    fn test_into_iter() {
        let d = Dequeue::from_iter(0..=1);
        let mut iter = d.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let d = Dequeue::from_iter(0..=1);
        let mut iter = (&d).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
    }
}
