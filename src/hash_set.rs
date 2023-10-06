use std::hash::Hash;

use crate::hash_map::{self, HashMap};

/// Space complexity: O(n)
#[derive(Debug, Clone)]
pub struct HashSet<T>(HashMap<T, ()>);

impl<T> HashSet<T>
where
    T: Hash + PartialEq,
{
    /// Time complexity: O(1)
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Time complexity: O(1)
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    /// Time complexity: O(1)
    pub fn insert(&mut self, el: T) {
        self.0.insert(el, ());
    }

    /// Time complexity: O(1)
    pub fn remove(&mut self, el: &T) {
        self.0.remove(el);
    }

    /// Time complexity: O(1)
    pub fn contains(&self, el: &T) -> bool {
        self.0.contains_key(el)
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Time complexity: O(m)
    pub fn union(&mut self, other: impl IntoIterator<Item = T>) {
        for el in other {
            self.insert(el);
        }
    }

    /// Time complexity: O(m)
    pub fn intersection(&mut self, other: impl IntoIterator<Item = T>) {
        let mut s = HashSet::new();
        for el in other {
            if self.contains(&el) {
                s.insert(el)
            }
        }
        let _ = std::mem::replace(&mut self.0, s.0);
    }

    pub fn difference(&mut self, other: &HashSet<T>) {
        for el in other {
            if self.contains(el) {
                self.remove(el);
            }
        }
    }
}

impl<T> HashSet<T> {
    /// Time complexity: O(n)
    pub fn iter(&self) -> Iter<'_, T> {
        Iter(self.0.iter())
    }
}

impl<T> Default for HashSet<T>
where
    T: PartialEq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PartialEq for HashSet<T>
where
    T: PartialEq + Hash,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for HashSet<T> where T: Eq + Hash {}

pub struct Iter<'a, T>(hash_map::Iter<'a, T, ()>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(el, _)| el)
    }
}

pub struct IntoIter<T>(hash_map::IntoIter<T, ()>);

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(el, _)| el)
    }
}

impl<T> IntoIterator for HashSet<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter())
    }
}

impl<'a, T> IntoIterator for &'a HashSet<T> {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> FromIterator<T> for HashSet<T>
where
    T: Hash + PartialEq,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut hs = HashSet::new();
        for el in iter {
            hs.insert(el)
        }
        hs
    }
}

impl<T, const N: usize> From<[T; N]> for HashSet<T>
where
    T: Hash + PartialEq,
{
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let mut hs = HashSet::new();

        assert_eq!(hs.len(), 0);
        assert!(hs.is_empty());

        assert!(!hs.contains(&0));
        hs.insert(0);
        assert!(hs.contains(&0));
        assert_eq!(hs.len(), 1);
        assert!(!hs.is_empty());

        assert!(!hs.contains(&1));
        hs.insert(1);
        assert!(hs.contains(&1));
        assert_eq!(hs.len(), 2);

        hs.remove(&0);
        assert!(!hs.contains(&0));
        assert_eq!(hs.len(), 1);

        hs.remove(&1);
        assert!(!hs.contains(&1));
        assert_eq!(hs.len(), 0);
        assert!(hs.is_empty());
    }

    #[test]
    fn test_iter() {
        let hs = HashSet::from_iter(0..=2);
        let mut iter = hs.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let hs = HashSet::from_iter(0..=2);
        let mut iter = hs.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let hs = HashSet::from_iter(0..=2);
        let mut iter = (&hs).into_iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_from_iter() {
        let hs = HashSet::from_iter(0..5);
        for i in 0..5 {
            assert!(hs.contains(&i))
        }
    }

    #[test]
    fn test_union_empty_set() {
        let mut a: HashSet<usize> = HashSet::new();
        let b = HashSet::new();
        a.union(b);
        assert_eq!(a, HashSet::new());
    }

    #[test]
    fn test_union_disjoint() {
        let mut a = HashSet::from([0]);
        let b = HashSet::from([1]);
        a.intersection(b);
        assert_eq!(a, HashSet::new());
    }

    #[test]
    fn test_union_intersecting() {
        let mut a = HashSet::from([0, 1]);
        let b = HashSet::from([1, 2]);
        a.union(b);
        assert_eq!(a, HashSet::from([0, 1, 2]));
    }

    #[test]
    fn test_intersection_empty_set() {
        let mut a: HashSet<usize> = HashSet::new();
        let b = HashSet::new();
        a.intersection(b);
        assert_eq!(a, HashSet::new());
    }
    #[test]
    fn test_intersection_disjoint() {
        let mut a = HashSet::from([0]);
        let b = HashSet::from([1]);
        a.intersection(b);
        assert_eq!(a, HashSet::new());
    }

    #[test]
    fn test_intersection_intersecting() {
        let mut a = HashSet::from([0, 1]);
        let b = HashSet::from([1, 2]);
        a.intersection(b);
        assert_eq!(a, HashSet::from([1]));
    }

    #[test]
    fn test_difference_empty_set() {
        let mut a: HashSet<usize> = HashSet::new();
        let b = HashSet::new();
        a.difference(&b);
        assert_eq!(a, HashSet::new());
    }

    #[test]
    fn test_difference_disjoint() {
        let mut a = HashSet::from([0]);
        let b = HashSet::from([1]);
        a.difference(&b);
        assert_eq!(a, HashSet::from([0]));
    }

    #[test]
    fn test_difference_intersecting() {
        let mut a = HashSet::from([0, 1]);
        let b = HashSet::from([1, 2]);
        a.difference(&b);
        assert_eq!(a, HashSet::from([0]));
    }

    #[test]
    fn test_clone() {
        let hs = HashSet::from_iter(0..=5);
        let cloned = hs.clone();
        assert_eq!(hs, cloned);
    }
}
