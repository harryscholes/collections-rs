use std::hash::Hash;

use crate::hash_map::HashMap;

pub struct HashSet<'a, T>(HashMap<'a, T, ()>);

impl<'a, T> HashSet<'a, T>
where
    T: Hash + PartialEq,
{
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    pub fn insert(&mut self, el: T) {
        self.0.insert(el, ());
    }

    pub fn delete(&mut self, el: &T) {
        self.0.delete(el);
    }

    pub fn contains(&self, el: &T) -> bool {
        self.0.contains_key(el)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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
}
