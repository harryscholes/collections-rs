#![allow(clippy::missing_safety_doc)]

use crate::hash_map::HashMap;
use std::{
    iter::FusedIterator,
    ops::{Index, IndexMut},
};

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

    // Time complexity: O(1)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            None
        } else if self.data.get(&index).is_some() {
            self.data.get_mut(&index)
        } else {
            self.data.insert(index, self.default.clone());
            self.data.get_mut(&index)
        }
    }

    // Time complexity: O(1)
    pub unsafe fn get_mut_ptr(&mut self, index: usize) -> Option<*mut T> {
        if index >= self.len {
            None
        } else if self.data.get(&index).is_some() {
            self.data.get_mut_ptr(&index)
        } else {
            self.data.insert(index, self.default.clone());
            self.data.get_mut_ptr(&index)
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
    pub fn insert(&mut self, index: usize, value: T) {
        if index >= self.len {
            panic!("index out of bounds")
        } else {
            self.insert_unchecked(index, value);
        }
    }
}

impl<T> SparseVector<T> {
    // Time complexity: O(1)
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            None
        } else {
            self.data.get(&index).or(Some(&self.default))
        }
    }

    // Time complexity: O(d)
    pub fn resize(&mut self, new_len: usize) {
        self.len = new_len;
        self.data.retain(|&index, _| index < self.len);
    }

    // Time complexity: O(1)
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            panic!("index out of bounds")
        } else {
            self.data.remove(&index)
        }
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
    pub fn iter(&self) -> Iter<T> {
        Iter::new(self)
    }

    // Time complexity: O(n)
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self)
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

impl<T> Index<usize> for SparseVector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of range")
    }
}

impl<T> IndexMut<usize> for SparseVector<T>
where
    T: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("index out of range")
    }
}

pub struct Iter<'a, T> {
    sv: &'a SparseVector<T>,
    forward_index: usize,
    back_index: usize,
    finished: bool,
}

impl<'a, T> Iter<'a, T> {
    fn new(sv: &'a SparseVector<T>) -> Self {
        Self {
            forward_index: 0,
            back_index: sv.len - 1,
            sv,
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
            let el = self.sv.get(self.forward_index);
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
            let el = self.sv.get(self.back_index);
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.back_index -= 1;
            }
            el
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

pub struct IterMut<'a, T> {
    sv: &'a mut SparseVector<T>,
    forward_index: usize,
    back_index: usize,
    finished: bool,
}

impl<'a, T> IterMut<'a, T> {
    fn new(sv: &'a mut SparseVector<T>) -> Self {
        Self {
            forward_index: 0,
            back_index: sv.len - 1,
            sv,
            finished: false,
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T>
where
    T: Clone,
{
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = unsafe { self.sv.get_mut_ptr(self.forward_index).map(|el| &mut *el) };
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.forward_index += 1;
            }
            el
        }
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T>
where
    T: Clone,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = unsafe { self.sv.get_mut_ptr(self.back_index).map(|el| &mut *el) };
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.back_index -= 1;
            }
            el
        }
    }
}

impl<'a, T> IntoIterator for &'a mut SparseVector<T>
where
    T: Clone,
{
    type Item = &'a mut T;

    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut::new(self)
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

impl<T, const N: usize> From<[T; N]> for SparseVector<T>
where
    T: Default + PartialEq,
{
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut sv = SparseVector::with_capacity(1);
        sv.insert(0, 0);
        assert!(!sv.data.contains_key(&0));
        sv.insert(0, 1);
        assert!(sv.data.contains_key(&0));
    }

    #[test]
    #[should_panic]
    fn test_insert_at_out_of_bounds_index() {
        let mut sv = SparseVector::with_capacity(1);
        sv.insert(1, 1);
    }

    #[test]
    fn test_remove() {
        let mut sv = SparseVector::from([0, 1]);
        assert_eq!(sv.get(1).unwrap(), &1);
        assert_eq!(sv.remove(1), Some(1));
        assert_eq!(sv.remove(1), None);
    }

    #[test]
    #[should_panic]
    fn test_remove_at_out_of_bounds_index() {
        let mut sv = SparseVector::from([0, 1]);
        sv.remove(2);
    }

    #[test]
    fn test_get() {
        let sv = SparseVector::from([0, 2, 0]);
        assert_eq!(sv.get(0), Some(&0));
        assert_eq!(sv.get(1), Some(&2));
        assert_eq!(sv.get(2), Some(&0));
        assert_eq!(sv.get(3), None);
    }

    #[test]
    fn test_get_mut() {
        let mut sv = SparseVector::from([0, 2, 0]);
        assert_eq!(sv.get_mut(0), Some(&mut 0));
        assert_eq!(sv.get_mut(1), Some(&mut 2));
        assert_eq!(sv.get_mut(2), Some(&mut 0));
        assert_eq!(sv.get_mut(3), None);

        if let Some(v) = sv.get_mut(0) {
            *v = 1;
        }
        assert_eq!(sv.get(0), Some(&1));
    }

    #[test]
    fn test_get_mut_ptr() {
        let mut sv = SparseVector::from([0, 2, 0]);
        assert_eq!(unsafe { *sv.get_mut_ptr(0).unwrap() }, 0);
        assert_eq!(unsafe { *sv.get_mut_ptr(1).unwrap() }, 2);
        assert_eq!(unsafe { *sv.get_mut_ptr(2).unwrap() }, 0);
        assert!(unsafe { sv.get_mut_ptr(3).is_none() });

        if let Some(v) = sv.get_mut(0) {
            *v = 1;
        }
        assert_eq!(sv.get(0), Some(&1));
    }

    #[test]
    fn test_pop_back() {
        let mut sv = SparseVector::from([1, 0, 2]);
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
        let mut sv = SparseVector::from([1, 0, 2]);
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
        sv.insert(2, 2);
        assert_eq!(sv.get(2), Some(&2));
        sv.resize(2);
        assert_eq!(sv.get(2), None);
        sv.resize(3);
        assert_eq!(sv.get(2), Some(&0));
    }

    #[test]
    fn test_iter() {
        let sv = SparseVector::from([0, 0, 2]);
        let mut iter = sv.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_rev() {
        let sv = SparseVector::from([0, 0, 2]);
        let mut iter = sv.iter().rev();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_double_ended_iterator() {
        let sv = SparseVector::from([0, 1, 0, 3, 0, 5, 0]);
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
        let sv = SparseVector::from([0, 0, 2]);
        let mut iter = sv.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_rev() {
        let sv = SparseVector::from([0, 0, 2]);
        let mut iter = sv.into_iter().rev();
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let sv = SparseVector::from([0, 0, 2]);
        let mut iter = (&sv).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter_mut() {
        let mut sv = SparseVector::from([0, 0, 2]);
        let mut iter = (&mut sv).into_iter();
        assert_eq!(iter.next(), Some(&mut 0));
        assert_eq!(iter.next(), Some(&mut 0));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), None);
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
        y.insert(5, 5);
        assert_ne!(x, y);
        y.insert(1, 1);
        assert_ne!(x, y);
        y.insert(3, 3);
        assert_eq!(x, y);
    }

    #[test]
    fn test_index() {
        let sv = SparseVector::from([0, 1, 0, 3]);
        assert_eq!(sv[0], 0);
        assert_eq!(sv[1], 1);
        assert_eq!(sv[2], 0);
        assert_eq!(sv[3], 3);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let sv = SparseVector::from([0, 1, 0, 3]);
        sv[4];
    }

    #[test]
    fn test_index_mut() {
        let mut sv = SparseVector::from([0, 1, 0, 3]);
        assert_eq!(sv[0], 0);
        sv[0] = 1;
        assert_eq!(sv[0], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_mut_index_out_of_bounds() {
        let mut sv = SparseVector::from([0, 1, 0, 3]);
        sv[4] = 0;
    }
}
