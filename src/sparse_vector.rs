use crate::hash_map::HashMap;
use std::iter::FusedIterator;

// Space complexity: O(d)
#[derive(Debug)]
pub struct SparseVector<T> {
    data: HashMap<usize, T>,
    len: usize,
    default: T,
}

impl<T> SparseVector<T>
where
    T: Default,
{
    pub fn with_capacity(len: usize) -> Self {
        Self {
            data: HashMap::with_capacity(len),
            len,
            default: T::default(),
        }
    }
}

impl<T> SparseVector<T>
where
    T: Clone,
{
    pub fn with_default(default: T, len: usize) -> Self {
        Self {
            data: HashMap::with_capacity(len),
            len,
            default,
        }
    }

    /// Time complexity: O(n)
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len > 0 {
            let el = self.data.remove(&0).or(Some(self.default.clone()));
            for index in 1..self.len {
                self.data
                    .remove(&index)
                    .map(|el| self.data.insert(index - 1, el));
            }
            self.len -= 1;
            el
        } else {
            None
        }
    }

    /// Time complexity: O(1)
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            self.data.remove(&self.len).or(Some(self.default.clone()))
        } else {
            None
        }
    }
}

impl<T> SparseVector<T>
where
    T: PartialEq,
{
    // Time complexity: O(1)
    pub fn insert_unchecked(&mut self, index: usize, value: T) {
        if value != self.default {
            self.data.insert(index, value);
        }
    }

    // Time complexity: O(1)
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), Error> {
        self.bounds_check(index)?;
        self.insert_unchecked(index, value);
        Ok(())
    }
}

impl<T> SparseVector<T> {
    // Time complexity: O(1)
    pub fn get(&self, index: usize) -> Result<&T, Error> {
        self.bounds_check(index)?;
        Ok(self.data.get(&index).unwrap_or(&self.default))
    }

    // Time complexity: O(d)
    pub fn resize(&mut self, new_len: usize) {
        self.len = new_len;
        self.data.retain(|&index, _| index < self.len);
    }

    // Time complexity: O(1)
    pub fn remove(&mut self, index: usize) -> Result<Option<T>, Error> {
        self.bounds_check(index)?;
        Ok(self.data.remove(&index))
    }

    // Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.len
    }

    // Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // Time complexity: O(n)
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    fn bounds_check(&self, index: usize) -> Result<(), Error> {
        if index >= self.len {
            Err(Error::IndexOutOfBounds {
                len: self.len,
                index,
            })
        } else {
            Ok(())
        }
    }
}

impl<T> PartialEq for SparseVector<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.default == other.default && self.iter().eq(other)
    }
}

impl<T> Eq for SparseVector<T> where T: Eq {}

pub struct Iter<'a, T> {
    sv: Option<&'a SparseVector<T>>,
    head: usize,
    tail: usize,
}

impl<'a, T> Iter<'a, T> {
    fn new(sv: &'a SparseVector<T>) -> Self {
        Self {
            sv: Some(sv),
            head: 0,
            tail: sv.len,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sv {
            Some(sv) => {
                let el = sv.get(self.head).ok();
                self.head += 1;
                if self.head == self.tail {
                    self.sv = None;
                }
                el
            }
            None => None,
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.sv {
            Some(sv) => {
                self.tail -= 1;
                if self.tail == self.head {
                    self.sv = None;
                }
                sv.get(self.tail).ok()
            }
            None => None,
        }
    }
}

impl<'a, T> IntoIterator for &'a SparseVector<T> {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IntoIter<T>(SparseVector<T>);

impl<T> IntoIter<T> {
    fn new(sv: SparseVector<T>) -> Self {
        Self(sv)
    }
}

impl<T> Iterator for IntoIter<T>
where
    T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T>
where
    T: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back()
    }
}

impl<T> IntoIterator for SparseVector<T>
where
    T: Clone,
{
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}
impl<T> FusedIterator for IntoIter<T> where T: Clone {}

impl<T> FromIterator<T> for SparseVector<T>
where
    T: Default + PartialEq,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (_, upper) = iter.size_hint();
        let mut sv = SparseVector::with_capacity(upper.unwrap_or(0));
        for (index, value) in iter.enumerate() {
            sv.insert_unchecked(index, value);
            sv.len = index + 1;
        }
        sv
    }
}

#[derive(Debug, PartialEq)]
pub enum Error {
    IndexOutOfBounds { len: usize, index: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut sv = SparseVector::with_capacity(1);
        assert_eq!(
            sv.insert(1, 1).unwrap_err(),
            Error::IndexOutOfBounds { len: 1, index: 1 }
        );
        sv.insert(0, 0).unwrap();
        assert!(!sv.data.contains_key(&0));
        sv.insert(0, 1).unwrap();
        assert!(sv.data.contains_key(&0));
    }

    #[test]
    fn test_remove() {
        let mut sv = SparseVector::with_capacity(3);
        assert_eq!(
            sv.remove(3).unwrap_err(),
            Error::IndexOutOfBounds { len: 3, index: 3 }
        );
        sv.insert(1, 2).unwrap();
        assert_eq!(sv.get(1).unwrap(), &2);
        assert_eq!(sv.remove(1), Ok(Some(2)));
    }

    #[test]
    fn test_get() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(1, 2).unwrap();
        assert_eq!(sv.get(0).unwrap(), &0);
        assert_eq!(sv.get(1).unwrap(), &2);
        assert_eq!(sv.get(2).unwrap(), &0);
        assert_eq!(
            sv.get(3).unwrap_err(),
            Error::IndexOutOfBounds { len: 3, index: 3 }
        );
    }

    #[test]
    fn test_pop_back() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(0, 1).unwrap();
        sv.insert(2, 2).unwrap();
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.pop_back(), Some(2));
        assert_eq!(sv.len(), 2);
        assert_eq!(sv.pop_back(), Some(0));
        assert_eq!(sv.len(), 1);
        assert_eq!(sv.pop_back(), Some(1));
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.pop_back(), None);
    }

    #[test]
    fn test_pop_front() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(0, 1).unwrap();
        sv.insert(2, 2).unwrap();
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.pop_front(), Some(1));
        assert_eq!(sv.len(), 2);
        assert_eq!(sv.pop_front(), Some(0));
        assert_eq!(sv.len(), 1);
        assert_eq!(sv.pop_front(), Some(2));
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.pop_front(), None);
    }

    #[test]
    fn test_resize() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(2, 2).unwrap();
        assert_eq!(sv.get(2).unwrap(), &2);
        sv.resize(2);
        assert_eq!(
            sv.get(2).unwrap_err(),
            Error::IndexOutOfBounds { len: 2, index: 2 }
        );
        sv.resize(3);
        assert_eq!(sv.get(2).unwrap(), &0);
    }

    #[test]
    fn test_iter() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = sv.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::with_capacity(5);
        sv.insert(4, 4usize).unwrap();
        let dense = sv.iter().collect::<Vec<&usize>>();
        assert_eq!(dense, vec![&0, &0, &0, &0, &4]);
    }

    #[test]
    fn test_iter_double_ended_iterator() {
        let mut sv = SparseVector::with_capacity(7);
        sv.insert(1, 1).unwrap();
        sv.insert(3, 3).unwrap();
        sv.insert(5, 5).unwrap();

        let mut iter = sv.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next_back(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next_back(), Some(&0));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = sv.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::with_capacity(5);
        sv.insert(4, 4usize).unwrap();
        let dense = sv.into_iter().collect::<Vec<usize>>();
        assert_eq!(dense, vec![0, 0, 0, 0, 4]);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let mut sv = SparseVector::with_capacity(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = (&sv).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::with_capacity(5);
        sv.insert(4, 4usize).unwrap();
        let dense = sv.into_iter().collect::<Vec<usize>>();
        assert_eq!(dense, vec![0, 0, 0, 0, 4]);
    }

    #[test]
    fn test_from_iter() {
        let dv = vec![0, 1, 0, 3, 0, 5, 0];
        let sv = SparseVector::from_iter(dv.clone());
        assert_eq!(sv.len(), dv.len());
        let mut iter = sv.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_eq() {
        let x = SparseVector::from_iter(vec![0, 1, 0, 3, 0, 5, 0]);
        let mut y = SparseVector::with_capacity(7);
        assert_ne!(x, y);
        y.insert(5, 5).unwrap();
        assert_ne!(x, y);
        y.insert(1, 1).unwrap();
        assert_ne!(x, y);
        y.insert(3, 3).unwrap();
        assert_eq!(x, y);
    }
}
