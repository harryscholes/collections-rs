/// Space complexity: O(n)
#[derive(Clone, Debug)]
pub struct CircularBuffer<T> {
    buf: Vec<Option<T>>,
    start: usize,
    end: usize,
}

impl<T> CircularBuffer<T> {
    pub fn new(capacity: usize) -> CircularBuffer<T> {
        Self::from_vec((0..capacity).map(|_| None).collect())
    }

    fn from_vec(buf: Vec<Option<T>>) -> CircularBuffer<T> {
        CircularBuffer {
            start: 0,
            end: 0,
            buf,
        }
    }

    /// Time complexity: O(1)
    pub fn push(&mut self, el: T) {
        let old = self.buf[self.end].replace(el);
        self.end = self.increment(self.end);
        if old.is_some() {
            self.start = self.increment(self.start);
        }
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, el: T) {
        self.start = self.decrement(self.start);
        let old = self.buf[self.start].replace(el);
        if old.is_some() {
            self.end = self.decrement(self.end);
        }
    }

    /// Time complexity: O(1)
    pub fn pop(&mut self) -> Option<T> {
        if self.buf.is_empty() {
            None
        } else {
            let end = self.decrement(self.end);
            let el = self.buf[end].take();
            if el.is_some() {
                self.end = end;
            }
            el
        }
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        if self.buf.is_empty() {
            None
        } else {
            let el = self.buf[self.start].take();
            if el.is_some() {
                self.start = self.increment(self.start);
            }
            el
        }
    }

    /// Time complexity: O(1)
    pub fn first(&self) -> Option<&T> {
        if self.buf.is_empty() {
            None
        } else {
            self.buf[self.start].as_ref()
        }
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<&T> {
        if self.buf.is_empty() {
            None
        } else {
            self.buf[self.decrement(self.end)].as_ref()
        }
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
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

    /// Time complexity: O(n)
    pub fn grow(&mut self, n: usize) {
        if !self.buf.is_empty() {
            self.buf.rotate_left(self.start);
            if self.end == self.start {
                self.end = self.buf.len()
            } else {
                self.end = sub_mod(self.end, self.start, self.buf.len());
            }
            self.start = 0;
        }
        self.buf.reserve(n);
        for _ in 0..n {
            self.buf.push(None);
        }
    }

    fn decrement(&self, index: usize) -> usize {
        sub_mod(index, 1, self.buf.len())
    }

    fn increment(&self, index: usize) -> usize {
        add_mod(index, 1, self.buf.len())
    }
}

fn add_mod(x: usize, y: usize, modulus: usize) -> usize {
    (x + y) % modulus
}

fn sub_mod(x: usize, y: usize, modulus: usize) -> usize {
    (modulus + x - y) % modulus
}

impl<T> FromIterator<T> for CircularBuffer<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> CircularBuffer<T> {
        let buf = iter.into_iter().map(|el| Some(el)).collect();
        CircularBuffer::from_vec(buf)
    }
}

pub struct IntoIter<T> {
    cb: CircularBuffer<T>,
    index: usize,
}

impl<T> IntoIter<T> {
    fn new(cb: CircularBuffer<T>) -> IntoIter<T> {
        IntoIter {
            index: cb.decrement(cb.start),
            cb,
        }
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.index = self.cb.increment(self.index);
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
    fn test_from_iter() {
        let cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_from_vec() {
        let cb = CircularBuffer::from_iter(vec![0, 1, 2]);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    #[should_panic]
    fn test_push_n0() {
        let mut cb = CircularBuffer::new(0);
        assert_eq!(cb.buf, vec![]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(0);
    }

    #[test]
    fn test_push_n1() {
        let mut cb = CircularBuffer::new(1);
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(1)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_n2() {
        let mut cb = CircularBuffer::new(2);
        assert_eq!(cb.buf, vec![None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        cb.push(3);
        assert_eq!(cb.buf, vec![Some(2), Some(3)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_n3() {
        let mut cb = CircularBuffer::new(3);
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1), None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 2);
        cb.push(2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push(3);
        assert_eq!(cb.buf, vec![Some(3), Some(1), Some(2)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
    }

    #[test]
    #[should_panic]
    fn test_push_front_n0() {
        let mut cb = CircularBuffer::new(0);
        cb.push_front(0);
    }

    #[test]
    fn test_push_front_n1() {
        let mut cb = CircularBuffer::new(1);
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(0);
        assert_eq!(cb.buf, vec![Some(0)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(1)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_push_front_n3() {
        let mut cb = CircularBuffer::new(3);
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(0);
        assert_eq!(cb.buf, vec![None, None, Some(0)]);
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 0);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![None, Some(1), Some(0)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        cb.push_front(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1), Some(0)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        cb.push_front(3);
        assert_eq!(cb.buf, vec![Some(2), Some(1), Some(3)]);
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 2);
        cb.push_front(4);
        assert_eq!(cb.buf, vec![Some(2), Some(4), Some(3)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
    }

    #[test]
    fn test_pop_n0() {
        let mut cb: CircularBuffer<usize> = CircularBuffer::new(0);
        assert_eq!(cb.pop(), None);
    }

    #[test]
    fn test_pop_n1() {
        let mut cb = CircularBuffer::from_iter(0..=0);
        assert_eq!(cb.buf, vec![Some(0)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop(), Some(0));
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop(), None);
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_n3() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop(), Some(2));
        assert_eq!(cb.buf, vec![Some(0), Some(1), None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 2);
        assert_eq!(cb.pop(), Some(1));
        assert_eq!(cb.buf, vec![Some(0), None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.pop(), Some(0));
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop(), None);
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_front_n0() {
        let mut cb: CircularBuffer<usize> = CircularBuffer::new(0);
        assert_eq!(cb.pop_front(), None);
    }

    #[test]
    fn test_pop_front_n1() {
        let mut cb = CircularBuffer::from_iter(0..=0);
        assert_eq!(cb.buf, vec![Some(0)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(0));
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), None);
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_pop_front_n3() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        assert_eq!(cb.buf, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(0));
        assert_eq!(cb.buf, vec![None, Some(1), Some(2)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(1));
        assert_eq!(cb.buf, vec![None, None, Some(2)]);
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop_front(), Some(2));
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.pop(), None);
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
    }

    #[test]
    fn test_first() {
        let cb: CircularBuffer<usize> = CircularBuffer::new(0);
        assert_eq!(cb.first(), None);

        let mut cb = CircularBuffer::new(2);
        assert!(cb.first().is_none());
        for el in 0..=10 {
            cb.push_front(el);
            assert_eq!(cb.first(), Some(&el));
        }
    }

    #[test]
    fn test_last() {
        let cb: CircularBuffer<usize> = CircularBuffer::new(0);
        assert_eq!(cb.last(), None);

        let mut cb = CircularBuffer::new(2);
        assert!(cb.last().is_none());
        for el in 0..=10 {
            cb.push(el);
            assert_eq!(cb.last(), Some(&el));
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

    #[test]
    fn test_grow() {
        let mut cb = CircularBuffer::new(0);
        assert_eq!(cb.buf, vec![]);
        cb.grow(1);
        assert_eq!(cb.buf, vec![None]);
        cb.grow(1);
        assert_eq!(cb.buf, vec![None, None]);
        cb.push_front(0);
        assert_eq!(cb.buf, vec![None, Some(0)]);
        cb.grow(1);
        assert_eq!(cb.buf, vec![Some(0), None, None]);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(0), None, Some(1)]);
        cb.grow(1);
        assert_eq!(cb.buf, vec![Some(1), Some(0), None, None]);
    }

    #[test]
    fn test_grow_start_eq_end() {
        let mut cb = CircularBuffer::from_iter(0..=2);
        cb.push(3);
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
    fn test_free_n0() {
        let cb: CircularBuffer<usize> = CircularBuffer::new(0);
        assert_eq!(cb.free(), 0);
    }

    #[test]
    fn test_free_n1() {
        let mut cb: CircularBuffer<usize> = CircularBuffer::new(1);
        assert_eq!(cb.free(), 1);
        cb.push(0);
        assert_eq!(cb.free(), 0);
        cb.push(1);
        assert_eq!(cb.free(), 0);
        cb.pop();
        assert_eq!(cb.free(), 1);
    }

    #[test]
    fn test_free_n2() {
        let mut cb = CircularBuffer::new(2);
        assert_eq!(cb.buf, vec![None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 2);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.push(1);
        assert_eq!(cb.buf, vec![Some(0), Some(1)]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 0);
        cb.push(2);
        assert_eq!(cb.buf, vec![Some(2), Some(1)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 0);
        cb.pop_front();
        assert_eq!(cb.buf, vec![Some(2), None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.pop_front();
        assert_eq!(cb.buf, vec![None, None]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 2);
    }

    #[test]
    fn test_free_n3() {
        let mut cb = CircularBuffer::new(3);
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.free(), 3);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0), None, None]);
        assert_eq!(cb.start, 0);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 2);
        cb.push_front(1);
        assert_eq!(cb.buf, vec![Some(0), None, Some(1)]);
        assert_eq!(cb.start, 2);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 1);
        cb.push_front(2);
        assert_eq!(cb.buf, vec![Some(0), Some(2), Some(1)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 0);
        cb.pop();
        assert_eq!(cb.buf, vec![None, Some(2), Some(1)]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 0);
        assert_eq!(cb.free(), 1);
        cb.pop();
        assert_eq!(cb.buf, vec![None, Some(2), None]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 2);
        assert_eq!(cb.free(), 2);
        cb.pop();
        assert_eq!(cb.buf, vec![None, None, None]);
        assert_eq!(cb.start, 1);
        assert_eq!(cb.end, 1);
        assert_eq!(cb.free(), 3);
    }

    #[test]
    fn test_grow_and_free() {
        let mut cb = CircularBuffer::new(1);
        assert_eq!(cb.buf, vec![None]);
        assert_eq!(cb.free(), 1);
        cb.push(0);
        assert_eq!(cb.buf, vec![Some(0)]);
        assert_eq!(cb.free(), 0);
        cb.grow(cb.len());
        assert_eq!(cb.buf, vec![Some(0), None]);
        assert_eq!(cb.len(), 2);
        assert_eq!(cb.free(), 1);
    }
}
