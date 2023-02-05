use crate::circular_buffer::CircularBuffer;

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
}
