use std::hash::Hash;

use crate::hash_map::{self, HashMap};

/// Space complexity: O(n)
pub struct HashSet<'a, T>(HashMap<'a, T, ()>);

impl<'a, T> HashSet<'a, T>
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
    pub fn delete(&mut self, el: &T) {
        self.0.delete(el);
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

    /// Time complexity: O(n)
    pub fn iter(&'a self) -> Iter<'a, T> {
        Iter(self.0.iter())
    }
}

impl<'a, T> Default for HashSet<'a, T>
where
    T: PartialEq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, T>(hash_map::Iter<'a, T, ()>);

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(el, _)| el)
    }
}

pub struct IntoIter<'a, T>(hash_map::IntoIter<'a, T, ()>);

impl<'a, T> Iterator for IntoIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(el, _)| el)
    }
}

impl<'a, T> IntoIterator for HashSet<'a, T> {
    type Item = T;

    type IntoIter = IntoIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter())
    }
}

impl<'a, T> FromIterator<T> for HashSet<'a, T>
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

        hs.delete(&0);
        assert!(!hs.contains(&0));
        assert_eq!(hs.len(), 1);

        hs.delete(&1);
        assert!(!hs.contains(&1));
        assert_eq!(hs.len(), 0);
        assert!(hs.is_empty());
    }

    #[test]
    fn test_iter() {
        let mut hs = HashSet::new();
        for i in 0..=2 {
            hs.insert(i);
        }

        let mut iter = hs.iter();
        assert_eq!(iter.next(), Some(&0));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter() {
        let mut hs = HashSet::new();
        for i in 0..=2 {
            hs.insert(i);
        }

        let mut iter = hs.into_iter();
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
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
}
