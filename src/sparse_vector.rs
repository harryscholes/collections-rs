use std::{collections::HashMap, iter::FusedIterator};

// Space complexity: O(d)
pub struct SparseVector<T> {
    data: HashMap<usize, T>,
    len: usize,
    default: T,
}

impl<T> SparseVector<T>
where
    T: Default,
{
    pub fn new(len: usize) -> Self {
        Self {
            data: HashMap::new(),
            len,
            default: T::default(),
        }
    }
}

impl<T> SparseVector<T>
where
    T: std::cmp::PartialEq,
{
    pub fn with_default(len: usize, default: T) -> Self {
        Self {
            data: HashMap::new(),
            len,
            default,
        }
    }

    // Time complexity: O(d)
    pub fn resize(&mut self, new_len: usize) {
        self.len = new_len;
        self.data.retain(|&index, _| index < self.len);
    }

    // Time complexity: O(1)
    pub fn insert(&mut self, index: usize, value: T) -> Result<(), Error> {
        self.bounds_check(index)?;
        if value != self.default {
            self.data.insert(index, value);
        }
        Ok(())
    }

    // Time complexity: O(1)
    pub fn remove(&mut self, index: usize) -> Result<Option<T>, Error> {
        self.bounds_check(index)?;
        Ok(self.data.remove(&index))
    }

    // Time complexity: O(1)
    pub fn get(&self, index: usize) -> Result<&T, Error> {
        self.bounds_check(index)?;
        Ok(self.data.get(&index).unwrap_or(&self.default))
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

pub struct Iter<'a, T> {
    sv: &'a SparseVector<T>,
    index: usize,
}

impl<'a, T> Iter<'a, T> {
    fn new(sv: &'a SparseVector<T>) -> Self {
        Self { sv, index: 0 }
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: std::cmp::PartialEq,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.sv.get(self.index) {
            Ok(v) => {
                self.index += 1;
                Some(v)
            }
            Err(_) => None,
        }
    }
}

impl<'a, T> IntoIterator for &'a SparseVector<T>
where
    T: Clone + std::cmp::PartialEq,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

pub struct IntoIter<T> {
    sv: SparseVector<T>,
    index: usize,
}

impl<T> IntoIter<T> {
    fn new(sv: SparseVector<T>) -> Self {
        Self { sv, index: 0 }
    }
}

impl<T> Iterator for IntoIter<T>
where
    T: Clone + std::cmp::PartialEq,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sv.get(self.index) {
            Ok(v) => {
                self.index += 1;
                Some(v.clone())
            }
            Err(_) => None,
        }
    }
}

impl<T> IntoIterator for SparseVector<T>
where
    T: Clone + std::cmp::PartialEq,
{
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> where T: std::cmp::PartialEq {}
impl<T> FusedIterator for IntoIter<T> where T: Clone + std::cmp::PartialEq {}

impl<T> FromIterator<T> for SparseVector<T>
where
    T: Default + std::cmp::PartialEq,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let dv = iter.into_iter().collect::<Vec<T>>();
        let mut sv = SparseVector::new(dv.len());
        for (index, value) in dv.into_iter().enumerate() {
            sv.insert(index, value).unwrap();
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
        let mut sv = SparseVector::new(1);
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
        let mut sv = SparseVector::new(3);
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
        let mut sv = SparseVector::new(3);
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
    fn test_resize() {
        let mut sv = SparseVector::new(3);
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
        let mut sv = SparseVector::new(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = sv.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::new(5);
        sv.insert(4, 4usize).unwrap();
        let dense = sv.iter().collect::<Vec<&usize>>();
        assert_eq!(dense, vec![&0, &0, &0, &0, &4]);
    }

    #[test]
    fn test_into_iter() {
        let mut sv = SparseVector::new(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = sv.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::new(5);
        sv.insert(4, 4usize).unwrap();
        let dense = sv.into_iter().collect::<Vec<usize>>();
        assert_eq!(dense, vec![0, 0, 0, 0, 4]);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let mut sv = SparseVector::new(3);
        sv.insert(0, 0).unwrap();
        sv.insert(2, 2).unwrap();
        let mut iter = (&sv).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);

        let mut sv = SparseVector::new(5);
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
        assert_eq!(iter.next(), None);
    }
}
