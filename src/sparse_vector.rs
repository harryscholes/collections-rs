use std::{collections::HashMap, iter::FusedIterator};

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
        SparseVector {
            data: HashMap::new(),
            len,
            default: T::default(),
        }
    }
}

impl<T> SparseVector<T> {
    pub fn new_with_default(len: usize, default: T) -> Self {
        SparseVector {
            data: HashMap::new(),
            len,
            default,
        }
    }

    pub fn resize(&mut self, new_len: usize) {
        self.len = new_len;
        self.data.retain(|&index, _| index < self.len);
    }

    pub fn insert(&mut self, index: usize, value: T) -> Result<(), Error> {
        self.bounds_check(index)?;
        self.data.insert(index, value);
        Ok(())
    }

    pub fn remove(&mut self, index: usize) -> Result<Option<T>, Error> {
        self.bounds_check(index)?;
        Ok(self.data.remove(&index))
    }

    pub fn get(&self, index: usize) -> Result<&T, Error> {
        self.bounds_check(index)?;
        let v = match self.data.get(&index) {
            Some(v) => v,
            None => &self.default,
        };
        Ok(v)
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

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }
}

impl<T> SparseVector<T> where T: Clone {}

pub struct Iter<'a, T> {
    sv: &'a SparseVector<T>,
    index: usize,
}

impl<'a, T> Iter<'a, T> {
    fn new(sv: &'a SparseVector<T>) -> Self {
        Iter { sv, index: 0 }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
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

pub struct IntoIter<T> {
    sv: SparseVector<T>,
    index: usize,
}

impl<T> IntoIter<T> {
    fn new(sv: SparseVector<T>) -> Self {
        IntoIter { sv, index: 0 }
    }
}

impl<T> Iterator for IntoIter<T>
where
    T: Clone,
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
}
