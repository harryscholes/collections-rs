use std::ops::Index;

use crate::vector::Vector;

/// Space complexity: O(n)
#[derive(Clone, Debug, Eq)]
pub struct CircularBuffer<T> {
    buf: Vector<Option<T>>,
    start: usize,
    end: usize,
}

impl<T> CircularBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self::from_vec((0..capacity).map(|_| None).collect())
    }

    fn from_vec(buf: Vector<Option<T>>) -> Self {
        Self {
            buf,
            // `start` is the index of the first element in the buffer
            start: 0,
            // `end` is the index after the last element in the buffer
            end: 0,
        }
    }

    /// Time complexity: O(1)
    pub fn push_back(&mut self, el: T) {
        if self.is_full() {
            self.increment_start();
        }
        self.buf[self.end] = Some(el);
        self.increment_end();
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, el: T) {
        if self.is_full() {
            self.decrement_end();
        }
        self.decrement_start();
        self.buf[self.start] = Some(el);
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.decrement_end();
            self.buf[self.end].take()
        }
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let el = self.buf[self.start].take();
            self.increment_start();
            el
        }
    }

    /// Time complexity: O(1)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            let circular_index = add_mod(self.start, index, self.buf.len());
            self.buf[circular_index].as_ref()
        }
    }

    /// Time complexity: O(1)
    pub fn first(&self) -> Option<&T> {
        match self.buf.get(self.start).as_ref() {
            Some(Some(el)) => Some(el),
            _ => None,
        }
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<&T> {
        match self.buf.get(self.decrement(self.end)).as_ref() {
            Some(Some(el)) => Some(el),
            _ => None,
        }
    }

    /// Time complexity: O(n)
    pub fn grow(&mut self, by: usize) {
        if !self.is_empty() {
            self.buf.rotate_left(self.start);
            self.end = if self.end == self.start {
                self.buf.len()
            } else {
                sub_mod(self.end, self.start, self.buf.len())
            };
            self.start = 0;
        }
        self.buf
            .extend((0..by).map(|_| None).collect::<Vector<_>>());
    }

    /// Time complexity: O(1)
    fn decrement(&self, index: usize) -> usize {
        sub_mod(index, 1, self.buf.len())
    }

    /// Time complexity: O(1)
    fn increment(&self, index: usize) -> usize {
        add_mod(index, 1, self.buf.len())
    }

    /// Time complexity: O(1)
    fn increment_start(&mut self) {
        self.start = self.increment(self.start)
    }

    /// Time complexity: O(1)
    fn decrement_start(&mut self) {
        self.start = self.decrement(self.start)
    }

    /// Time complexity: O(1)
    fn increment_end(&mut self) {
        self.end = self.increment(self.end)
    }

    /// Time complexity: O(1)
    fn decrement_end(&mut self) {
        self.end = self.decrement(self.end)
    }

    /// Time complexity: O(n)
    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }

    /// Time complexity: O(1)
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Time complexity: O(1)
    pub fn free(&self) -> usize {
        if self.start == self.end {
            if self.first().is_none() {
                self.buf.len()
            } else {
                0
            }
        } else {
            let diff = self.end.abs_diff(self.start);
            if self.end > self.start {
                self.buf.len() - diff
            } else {
                diff
            }
        }
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.capacity() - self.free()
    }

    /// Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Time complexity: O(1)
    pub fn is_full(&self) -> bool {
        self.free() == 0
    }
}

/// Time complexity: O(1)
fn add_mod(x: usize, y: usize, modulus: usize) -> usize {
    (x + y) % modulus
}

/// Time complexity: O(1)
fn sub_mod(x: usize, y: usize, modulus: usize) -> usize {
    ((modulus + x).saturating_sub(y)) % modulus
}

impl<T> FromIterator<T> for CircularBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_vec(iter.into_iter().map(|el| Some(el)).collect())
    }
}

impl<T, const N: usize> From<[T; N]> for CircularBuffer<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T> PartialEq for CircularBuffer<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }
}

impl<T> Index<usize> for CircularBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of range")
    }
}

pub struct Iter<'a, T> {
    cb: &'a CircularBuffer<T>,
    forward_index: usize,
    back_index: usize,
    finished: bool,
}

impl<'a, T> Iter<'a, T> {
    fn new(cb: &'a CircularBuffer<T>) -> Self {
        Self {
            forward_index: 0,
            back_index: cb.len() - 1,
            cb,
            finished: false,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = self.cb.get(self.forward_index);
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.forward_index += 1;
            }
            el
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = self.cb.get(self.back_index);
            if self.back_index == self.forward_index {
                self.finished = true;
            } else {
                self.back_index -= 1;
            }
            el
        }
    }
}

impl<'a, T> IntoIterator for &'a CircularBuffer<T> {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IntoIter<T>(CircularBuffer<T>);

impl<T> IntoIter<T> {
    fn new(cb: CircularBuffer<T>) -> Self {
        Self(cb)
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

impl<T> IntoIterator for CircularBuffer<T> {
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
    fn test_add_mod() {
        assert_eq!(add_mod(0, 1, 3), 1);
        assert_eq!(add_mod(2, 1, 3), 0);
    }

    #[test]
    fn test_sub_mod() {
        assert_eq!(sub_mod(1, 1, 3), 0);
        assert_eq!(sub_mod(0, 1, 3), 2);
    }

    #[test]
    #[should_panic]
    fn test_with_capacity_0() {
        CircularBuffer::<usize>::with_capacity(0);
    }

    #[test]
    fn test_from_iter() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_from_vec() {
        let cb = CircularBuffer::from_iter(vec![0, 1, 2]);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_back_n1() {
        let mut cb = CircularBuffer::with_capacity(1);
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(1);
        assert_eq!(cb.buf, vec![Some(1)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_back_n2() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert_eq!(cb.buf, vec![None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0), None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        cb.push_back(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        cb.push_back(3);
        assert_eq!(cb.buf, vec![Some(2), Some(3)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_back_n3() {
        let mut cb = CircularBuffer::with_capacity(3);
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0), None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        cb.push_back(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1), None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 2);
        cb.push_back(2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_back(3);
        assert_eq!(cb.buf, vec![Some(3), Some(1), Some(2)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
    }

    #[test]
    fn test_push_front_n1() {
        let mut cb = CircularBuffer::with_capacity(1);
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(0);
        assert_eq!(cb.buf, vec![Some(0)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(1)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_front_n3() {
        let mut cb = CircularBuffer::with_capacity(3);
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(0);
        assert_eq!(cb.buf, vec![None, None, Some(0)].into());
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 0);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![None, Some(1), Some(0)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        cb.push_front(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1), Some(0)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(3);
        assert_eq!(cb.buf, vec![Some(2), Some(1), Some(3)].into());
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 2);
        cb.push_front(4);
        assert_eq!(cb.buf, vec![Some(2), Some(4), Some(3)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
    }

    #[test]
    fn test_pop_back_n1() {
        let mut cb = CircularBuffer::from_iter(0..=0);
        assert_eq!(cb.buf, vec![Some(0)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_back(), Some(0));
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_back(), None);
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_back_n3() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_back(), Some(2));
        assert_eq!(cb.buf, vec![Some(0), Some(1), None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 2);
        assert_eq!(cb.pop_back(), Some(1));
        assert_eq!(cb.buf, vec![Some(0), None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.pop_back(), Some(0));
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_back(), None);
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_front_n1() {
        let mut cb = CircularBuffer::from_iter(0..=0);
        assert_eq!(cb.buf, vec![Some(0)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(0));
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), None);
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_front_n3() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(0));
        assert_eq!(cb.buf, vec![None, Some(1), Some(2)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(1));
        assert_eq!(cb.buf, vec![None, None, Some(2)].into());
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(2));
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_back(), None);
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_first() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert!(cb.first().is_none());
        for el in 0..=10 {
            cb.push_front(el);
            assert_eq!(cb.first(), Some(&el));
        }
    }

    #[test]
    fn test_last() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert!(cb.last().is_none());
        for el in 0..=10 {
            cb.push_back(el);
            assert_eq!(cb.last(), Some(&el));
        }
    }

    #[test]
    fn test_iter() {
        let cb = CircularBuffer::from_iter(0..=2);
        let mut iter = cb.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_buffer_not_full() {
        let mut cb = CircularBuffer::from_iter(0..=4);
        cb.pop_front();
        cb.pop_back();
        let mut iter = cb.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_rev() {
        let cb = CircularBuffer::from_iter(0..=2);
        let mut iter = cb.iter().rev();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_rev_buffer_not_full() {
        let mut cb = CircularBuffer::from_iter(0..=4);
        cb.pop_front();
        cb.pop_back();
        let mut iter = cb.iter().rev();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_double_ended_iterator() {
        let cb = CircularBuffer::from_iter(1..=6);
        let mut iter = cb.iter();
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
    fn test_iter_double_ended_iterator_buffer_not_full() {
        let mut cb = CircularBuffer::from_iter(0..=7);
        cb.pop_front();
        cb.pop_back();
        let mut iter = cb.iter();
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
        let cb = CircularBuffer::from_iter(0..=1);
        let mut iter = cb.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let cb = CircularBuffer::from_iter(0..=1);
        let mut iter = (&cb).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_rev() {
        let cb = CircularBuffer::from_iter(0..=1);
        let mut iter = cb.into_iter().rev();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_double_ended_iterator() {
        let cb = CircularBuffer::from_iter(1..=6);
        let mut iter = cb.into_iter();
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
    fn test_grow() {
        let mut cb = CircularBuffer::with_capacity(1);
        assert_eq!(cb.buf, vec![None].into());
        cb.grow(1);
        assert_eq!(cb.buf, vec![None, None].into());
        cb.push_front(0);
        assert_eq!(cb.buf, vec![None, Some(0)].into());
        cb.grow(1);
        assert_eq!(cb.buf, vec![Some(0), None, None].into());
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(0), None, Some(1)].into());
        cb.grow(1);
        assert_eq!(cb.buf, vec![Some(1), Some(0), None, None].into());
    }

    #[test]
    fn test_grow_start_eq_end() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        cb.push_back(3);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        cb.grow(1);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 3);
        cb.pop_front();
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 3);
        cb.grow(2);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 2);
    }

    #[test]
    fn test_free_n1() {
        let mut cb: CircularBuffer<usize> = CircularBuffer::with_capacity(1);
        assert_eq!(cb.free(), 1);
        cb.push_back(0);
        assert_eq!(cb.free(), 0);
        cb.push_back(1);
        assert_eq!(cb.free(), 0);
        cb.pop_back();
        assert_eq!(cb.free(), 1);
    }

    #[test]
    fn test_free_n2() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert_eq!(cb.buf, vec![None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 2);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0), None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.push_back(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1)].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 0);
        cb.push_back(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 0);
        cb.pop_front();
        assert_eq!(cb.buf, vec![Some(2), None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.pop_front();
        assert_eq!(cb.buf, vec![None, None].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 2);
    }

    #[test]
    fn test_free_n3() {
        let mut cb = CircularBuffer::with_capacity(3);
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.free(), 3);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0), None, None].into());
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 2);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(0), None, Some(1)].into());
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.push_front(2);
        assert_eq!(cb.buf, vec![Some(0), Some(2), Some(1)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 0);
        cb.pop_back();
        assert_eq!(cb.buf, vec![None, Some(2), Some(1)].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 1);
        cb.pop_back();
        assert_eq!(cb.buf, vec![None, Some(2), None].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 2);
        assert_eq!(cb.free(), 2);
        cb.pop_back();
        assert_eq!(cb.buf, vec![None, None, None].into());
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 3);
    }

    #[test]
    fn test_grow_and_free() {
        let mut cb = CircularBuffer::with_capacity(1);
        assert_eq!(cb.buf, vec![None].into());
        assert_eq!(cb.free(), 1);
        cb.push_back(0);
        assert_eq!(cb.buf, vec![Some(0)].into());
        assert_eq!(cb.free(), 0);
        cb.grow(cb.capacity());
        assert_eq!(cb.buf, vec![Some(0), None].into());
        assert_eq!(cb.capacity(), 2);
        assert_eq!(cb.free(), 1);
    }

    #[test]
    fn test_len() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert!(cb.is_empty());
        cb.push_back(0);
        assert_eq!(cb.len(), 1);
        cb.push_back(1);
        assert_eq!(cb.len(), 2);
        cb.push_back(2);
        assert_eq!(cb.len(), 2);
    }

    #[test]
    fn test_is_full() {
        let mut cb = CircularBuffer::with_capacity(2);
        assert!(!cb.is_full());
        cb.push_back(0);
        assert!(!cb.is_full());
        cb.push_back(1);
        assert!(cb.is_full());
        cb.push_back(2);
        assert!(cb.is_full());
        cb.pop_front();
        assert!(!cb.is_full());
    }

    #[test]
    fn test_eq() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb, CircularBuffer::from_iter(0..=2));

        let mut cb = CircularBuffer::from_iter(0..=3);
        cb.pop_back();
        assert_eq!(cb, CircularBuffer::from_iter(0..=2));

        let mut cb = CircularBuffer::from_iter(0..=2);
        cb.pop_front();
        assert_eq!(cb, CircularBuffer::from_iter(1..=2));
    }

    #[test]
    fn test_get() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.get(0), Some(&0));
        assert_eq!(cb.get(1), Some(&1));
        assert_eq!(cb.get(2), Some(&2));
    }

    #[test]
    fn test_get_start_gt_end() {
        let cb = CircularBuffer {
            buf: Vector::from([Some(1), Some(2), Some(0)]),
            start: 2,
            end: 2,
        };
        assert_eq!(cb.get(0), Some(&0));
        assert_eq!(cb.get(1), Some(&1));
        assert_eq!(cb.get(2), Some(&2));
    }

    #[test]
    fn test_get_buffer_not_full() {
        let mut cb = CircularBuffer::with_capacity(2);
        cb.push_back(0);
        assert_eq!(cb.get(0), Some(&0));
        assert_eq!(cb.get(1), None);
    }

    #[test]
    fn test_get_start_gt_end_buffer_not_full() {
        let cb = CircularBuffer {
            buf: Vector::from([Some(1), None, Some(0)]),
            start: 2,
            end: 2,
        };
        assert_eq!(cb.get(0), Some(&0));
        assert_eq!(cb.get(1), Some(&1));
        assert_eq!(cb.get(2), None);
    }

    #[test]
    fn test_index() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb[0], 0);
        assert_eq!(cb[1], 1);
        assert_eq!(cb[2], 2);
    }

    #[test]
    fn test_index_start_gt_end() {
        let cb = CircularBuffer {
            buf: Vector::from([Some(1), Some(2), Some(0)]),
            start: 2,
            end: 2,
        };
        assert_eq!(cb[0], 0);
        assert_eq!(cb[1], 1);
        assert_eq!(cb[2], 2);
    }

    #[test]
    fn test_index_buffer_not_full() {
        let mut cb = CircularBuffer::with_capacity(2);
        cb.push_back(0);
        assert_eq!(cb[0], 0);
    }

    #[test]
    #[should_panic]
    fn test_index_buffer_not_full_index_out_of_bounds() {
        let mut cb = CircularBuffer::with_capacity(2);
        cb.push_back(0);
        assert_eq!(cb[1], 1);
    }
}
