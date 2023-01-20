use crate::circular_buffer::CircularBuffer;

/// Space complexity: O(n)
#[derive(Clone, Debug)]
pub struct Dequeue<T>(CircularBuffer<T>);

impl<T> Dequeue<T> {
    /// Time complexity: O(1)
    pub fn new() -> Dequeue<T> {
        Dequeue(CircularBuffer::new(1))
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn push_back(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.len())
        }
        self.0.push_back(el);
    }

    /// Time complexity: amortised O(1), O(n) worst case
    pub fn push_front(&mut self, el: T) {
        if self.0.free() == 0 {
            self.0.grow(self.0.len())
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

    // #[test]
    // fn test_loop() {
    //     let mut q = Dequeue::new();

    //     let min = 0;
    //     let max = 100_000;

    //     for i in min..=max {
    //         d.enqueue(i);
    //         assert_eq!(d.peek(), Some(&min));
    //     }

    //     for i in min..=max {
    //         assert_eq!(d.peek(), Some(&i));
    //         assert_eq!(d.dequeue(), Some(i))
    //     }

    //     assert_eq!(d.dequeue(), None);
    // }
}
