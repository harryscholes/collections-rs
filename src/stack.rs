use std::iter::Rev;

use crate::circular_buffer::{self, CircularBuffer};

/// Space complexity: O(n)
#[derive(Clone, Debug)]
pub struct Stack<T>(CircularBuffer<T>);

impl<T> Stack<T> {
    pub fn new() -> Stack<T> {
        Stack(CircularBuffer::new(1))
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn push(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.capacity())
        }
        self.0.push_back(el);
    }

    /// Time complexity: O(1)
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop_back()
    }

    /// Time complexity: O(1)
    pub fn peek(&self) -> Option<&T> {
        self.0.last()
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> From<[T; N]> for Stack<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T> FromIterator<T> for Stack<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Stack<T> {
        let mut s = Stack::new();
        for el in iter {
            s.push(el);
        }
        s
    }
}

pub struct IntoIter<T>(Rev<circular_buffer::IntoIter<T>>);

impl<T> IntoIter<T> {
    fn new(s: Stack<T>) -> IntoIter<T> {
        IntoIter(s.0.into_iter().rev())
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> IntoIterator for Stack<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut s = Stack::new();
        assert_eq!(s.peek(), None);
        s.push(0);
        assert_eq!(s.peek(), Some(&0));
        s.push(1);
        assert_eq!(s.peek(), Some(&1));
        s.push(2);
        assert_eq!(s.peek(), Some(&2));
        s.push(3);
        assert_eq!(s.peek(), Some(&3));
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.peek(), Some(&2));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.peek(), Some(&1));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.peek(), Some(&0));
        assert_eq!(s.pop(), Some(0));
        assert_eq!(s.peek(), None);
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn test_loop() {
        let mut s = Stack::new();

        let min = 0;
        let max = 100_000;

        for i in min..=max {
            s.push(i);
            assert_eq!(s.peek(), Some(&i));
        }

        for i in max..=min {
            assert_eq!(s.peek(), Some(&i));
            assert_eq!(s.pop(), Some(i))
        }
    }

    #[test]
    fn test_into_iter() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);

        let mut iter = s.into_iter();
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }
}
