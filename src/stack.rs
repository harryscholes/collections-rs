use crate::circular_buffer::CircularBuffer;

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
}
