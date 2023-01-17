#[derive(Clone)]
pub struct CircularBuffer<T> {
    buf: Vec<Option<T>>,
    index: usize,
}

impl<T> CircularBuffer<T> {
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    pub fn new(size: usize) -> CircularBuffer<T> {
        Self::from_buffer((0..size).map(|_| None).collect())
    }

    fn from_buffer(buf: Vec<Option<T>>) -> CircularBuffer<T> {
        CircularBuffer {
            index: buf.len() - 1,
            buf,
        }
    }

    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn push(&mut self, el: T) {
        self.index = next_index(self.index, self.buf.len());
        self.buf[self.index] = Some(el);
    }

    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn get(&self) -> Option<&T> {
        self.buf[self.index].as_ref()
    }

    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn pop(&mut self) -> Option<T> {
        let el = self.buf[self.index].take();
        self.index = previous_index(self.index, self.buf.len());
        el
    }

    /// Time complexity: O(1)
    /// Space complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        let l = self.buf.len();
        self.buf[previous_index(self.index, l)].take()
    }

    // TODO
    pub fn resize(&mut self, size: usize) {
        self.buf.rotate_righ
    }
}

fn next_index(index: usize, len: usize) -> usize {
    (index + 1) % len
}

fn previous_index(index: usize, len: usize) -> usize {
    (len + index - 1) % len
}

impl<T> FromIterator<T> for CircularBuffer<T> {
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> CircularBuffer<T> {
        let buf = iter.into_iter().map(|el| Some(el)).collect();
        CircularBuffer::from_buffer(buf)
    }
}

pub struct IntoIter<T> {
    cb: CircularBuffer<T>,
    index: usize,
}

impl<T> IntoIter<T> {
    fn new(cb: CircularBuffer<T>) -> IntoIter<T> {
        IntoIter {
            index: cb.index,
            cb,
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index = next_index(self.index, self.cb.buf.len());
        self.cb.buf[self.index].take()
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
    fn test_next_index() {
        assert_eq!(next_index(0, 3), 1);
        assert_eq!(next_index(2, 3), 0);
    }

    #[test]
    fn test_previous_index() {
        assert_eq!(previous_index(1, 3), 0);
        assert_eq!(previous_index(0, 3), 2);
    }

    #[test]
    fn test_from_iter() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_from_vec() {
        let cb = CircularBuffer::from_iter(vec![0, 1, 2]);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn test_push_n2() {
        let mut cb = CircularBuffer::new(2);
        assert_eq!(cb.buf, vec![None, None]);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None]);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1)]);
        cb.push(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1)]);
        cb.push(3);
        assert_eq!(cb.buf, vec![Some(2), Some(3)]);
    }

    #[test]
    fn test_push_n3() {
        let mut cb = CircularBuffer::new(3);
        assert_eq!(cb.buf, vec![None, None, None]);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None, None]);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1), None]);
        cb.push(2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        cb.push(3);
        assert_eq!(cb.buf, vec![Some(3), Some(1), Some(2)]);
    }

    #[test]
    fn test_pop() {
        let mut cb = CircularBuffer::from_iter(0..=1);
        assert_eq!(cb.pop(), Some(1));
        assert_eq!(cb.pop(), Some(0));
        assert_eq!(cb.pop(), None);
    }

    #[test]
    fn test_pop_front() {
        let mut cb = CircularBuffer::from_iter(0..=1);
        assert_eq!(cb.pop_front(), Some(0));
        assert_eq!(cb.pop_front(), Some(1));
        assert_eq!(cb.pop(), None);
    }

    #[test]
    fn test_pop_wrapping() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        cb.push(3);
        assert_eq!(cb.buf, vec![Some(3), Some(1), Some(2)]);
        assert_eq!(cb.pop(), Some(3));
        assert_eq!(cb.buf, vec![None, Some(1), Some(2)]);
        assert_eq!(cb.pop(), Some(2));
        assert_eq!(cb.buf, vec![None, Some(1), None]);
        assert_eq!(cb.pop(), Some(1));
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.pop(), None);
        assert_eq!(cb.buf, vec![None, None, None]);
    }

    #[test]
    fn test_get() {
        let mut cb = CircularBuffer::new(2);
        assert!(cb.get().is_none());
        for el in 0..=2 {
            cb.push(el);
            assert_eq!(cb.get(), Some(&el));
        }
    }

    #[test]
    fn test_into_iter() {
        let cb = CircularBuffer::from_iter(0..=1);
        let mut iter = cb.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }
}
