use crate::circular_buffer::{self, CircularBuffer};

/// Space complexity: O(n)
#[derive(Clone, Debug)]
pub struct Queue<T>(CircularBuffer<T>);

impl<T> Queue<T> {
    /// Time complexity: O(1)
    pub fn new() -> Queue<T> {
        Queue(CircularBuffer::new(1))
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn enqueue(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.capacity())
        }
        self.0.push_back(el);
    }

    /// Time complexity: O(1)
    pub fn dequeue(&mut self) -> Option<T> {
        self.0.pop_front()
    }

    /// Time complexity: O(1)
    pub fn peek(&self) -> Option<&T> {
        self.0.first()
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> From<[T; N]> for Queue<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T> FromIterator<T> for Queue<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Queue<T> {
        let mut q = Queue::new();
        for el in iter {
            q.enqueue(el);
        }
        q
    }
}

pub struct Iter<'a, T>(circular_buffer::Iter<'a, T>);

impl<'a, T> Iter<'a, T> {
    fn new(q: &'a Queue<T>) -> Self {
        Iter(q.0.iter())
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T> IntoIterator for &'a Queue<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IntoIter<T>(circular_buffer::IntoIter<T>);

impl<T> IntoIter<T> {
    fn new(q: Queue<T>) -> IntoIter<T> {
        IntoIter(q.0.into_iter())
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<T> IntoIterator for Queue<T> {
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
        let mut q = Queue::new();
        assert_eq!(q.dequeue(), None);
        q.enqueue(0);
        assert_eq!(q.peek(), Some(&0));
        q.enqueue(1);
        assert_eq!(q.peek(), Some(&0));
        q.enqueue(2);
        assert_eq!(q.peek(), Some(&0));
        assert_eq!(q.dequeue(), Some(0));
        assert_eq!(q.peek(), Some(&1));
        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.peek(), Some(&2));
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.peek(), None);
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let q = Queue::from([0, 1, 2]);
        let mut iter = (&q).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_loop() {
        let mut q = Queue::new();

        let min = 0;
        let max = 100_000;

        for i in min..=max {
            q.enqueue(i);
            assert_eq!(q.peek(), Some(&min));
        }

        for i in min..=max {
            assert_eq!(q.peek(), Some(&i));
            assert_eq!(q.dequeue(), Some(i))
        }

        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut q = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);

        let mut iter = q.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }
}
