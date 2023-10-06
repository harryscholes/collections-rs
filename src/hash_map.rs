#![allow(clippy::missing_safety_doc)]

use crate::{linked_list::LinkedList, vector::Vector};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::Index,
};

const DEFAULT_CAPACITY: usize = 4;
const BUCKET_CAPACITY: usize = 4;

/// Space complexity: O(n)
#[derive(Debug, Default)]
pub struct HashMap<K, V> {
    buckets: Vector<Option<Bucket<K, V>>>,
    len: usize,
}

type Bucket<K, V> = LinkedList<Node<K, V>>;

#[derive(Debug, PartialEq, Eq, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
}

impl<K, V> HashMap<K, V> {
    /// Time complexity: O(1)
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Time complexity: O(1)
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            buckets: (0..capacity).map(|_| None).collect(),
            len: 0,
        }
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.len
    }

    /// Time complexity: O(1)
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Time complexity: O(n)
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self)
    }
}

impl<K, V> HashMap<K, V>
where
    K: PartialEq + Hash,
{
    /// Time complexity: O(1) amortised, O(n) worst case
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (old_value, rehash) = self.insert_impl(key, value);
        if rehash {
            self.rehash()
        }
        old_value
    }

    /// Time complexity: O(1)
    fn insert_impl(&mut self, key: K, value: V) -> (Option<V>, bool) {
        let bucket_index = self.bucket_index(&key);
        match self.buckets[bucket_index] {
            Some(ref mut ll) => {
                // Check if key exists
                for node in ll.iter_mut() {
                    if node.key == key {
                        // Update value
                        let old_value = std::mem::replace(&mut node.value, value);
                        return (Some(old_value), false);
                    }
                }
                // Otherwise add the key/value pair
                ll.push_back(Node { key, value });
                self.len += 1;
                if ll.len() >= BUCKET_CAPACITY {
                    // Map needs to be rehashed
                    (None, true)
                } else {
                    (None, false)
                }
            }
            None => {
                let mut ll = LinkedList::new();
                ll.push_back(Node { key, value });
                self.buckets[bucket_index] = Some(ll);
                self.len += 1;
                (None, false)
            }
        }
    }

    /// Time complexity: O(n)
    fn rehash(&mut self) {
        // Create a new `HashMap` with `capacity` equal to the length of `self`
        let mut tmp = HashMap::with_capacity(self.len());
        // Swap the `buckets` between `tmp` and `self`
        std::mem::swap(&mut tmp.buckets, &mut self.buckets);
        // Reset `len`
        self.len = 0;
        // Insert the key/value pairs into the new buckets
        for (key, value) in tmp {
            self.insert_impl(key, value);
        }
    }

    /// Time complexity: O(1)
    pub fn get(&self, key: &K) -> Option<&V> {
        match &self.buckets[self.bucket_index(key)] {
            Some(ll) => {
                for node in ll {
                    if node.key == *key {
                        return Some(&node.value);
                    }
                }
                None
            }
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn get_ptr(&self, key: &K) -> Option<*const V> {
        self.get(key).map(|value| value as *const V)
    }

    /// Time complexity: O(1)
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let bucket_index = self.bucket_index(key);
        match &mut self.buckets[bucket_index] {
            Some(ll) => {
                for node in ll {
                    if node.key == *key {
                        return Some(&mut node.value);
                    }
                }
                None
            }
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn get_mut_ptr(&mut self, key: &K) -> Option<*mut V> {
        self.get_mut(key).map(|value| value as *mut V)
    }

    /// Time complexity: O(1)
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let bucket_index = self.bucket_index(key);
        match &mut self.buckets[bucket_index] {
            Some(ref mut ll) => {
                for (index, node) in ll.iter().enumerate() {
                    if node.key == *key {
                        let node = ll.remove(index);
                        if ll.is_empty() {
                            self.buckets[bucket_index] = None;
                        }
                        self.len -= 1;
                        return node.map(|node| node.value);
                    }
                }
                None
            }
            None => None,
        }
    }

    /// Time complexity: O(n)
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut tmp = HashMap::new();
        std::mem::swap(&mut tmp.buckets, &mut self.buckets);
        self.len = 0;
        for (key, mut value) in tmp {
            if f(&key, &mut value) {
                self.insert(key, value);
            }
        }
    }

    fn hash(&self, k: &K) -> u64 {
        let mut hasher = DefaultHasher::new();
        k.hash(&mut hasher);
        hasher.finish()
    }

    fn bucket_index(&self, k: &K) -> usize {
        (self.hash(k) as usize) % self.buckets.len()
    }

    /// Time complexity: O(1)
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K, V> Clone for HashMap<K, V>
where
    K: Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            buckets: self.buckets.iter().cloned().collect(),
            len: self.len,
        }
    }
}

impl<K, V> Index<K> for HashMap<K, V>
where
    K: Hash + PartialEq,
{
    type Output = V;

    fn index(&self, key: K) -> &Self::Output {
        self.get(&key).expect("key does not exist")
    }
}

impl<K, V> PartialEq for HashMap<K, V>
where
    K: PartialEq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        for (k, v) in self.iter() {
            if other.get(k) != Some(v) {
                return false;
            }
        }

        true
    }
}

impl<K, V> Eq for HashMap<K, V>
where
    K: Eq + Hash,
    V: Eq,
{
}

pub struct Iter<'a, K, V> {
    map: &'a HashMap<K, V>,
    bucket_index: usize,
    bucket_iter: Option<crate::linked_list::Iter<'a, Node<K, V>>>,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn new(map: &'a HashMap<K, V>) -> Self {
        Self {
            map,
            bucket_index: 0,
            bucket_iter: None,
        }
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut iter) = &mut self.bucket_iter {
            match iter.next() {
                Some(node) => return Some((&node.key, &node.value)),
                None => self.bucket_index += 1,
            }
        }

        loop {
            if self.bucket_index >= self.map.buckets.len() {
                return None;
            }

            if let Some(ll) = &self.map.buckets[self.bucket_index] {
                let mut iter = ll.iter();
                if let Some(node) = iter.next() {
                    self.bucket_iter = Some(iter);
                    return Some((&node.key, &node.value));
                }
            }

            self.bucket_index += 1;
        }
    }
}

pub struct IntoIter<K, V> {
    map: HashMap<K, V>,
    bucket_index: usize,
    bucket_iter: Option<crate::linked_list::IntoIter<Node<K, V>>>,
}

impl<K, V> IntoIter<K, V> {
    fn new(map: HashMap<K, V>) -> Self {
        Self {
            map,
            bucket_index: 0,
            bucket_iter: None,
        }
    }
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ref mut iter) = &mut self.bucket_iter {
            match iter.next() {
                Some(node) => return Some((node.key, node.value)),
                None => self.bucket_index += 1,
            }
        }

        loop {
            if self.bucket_index >= self.map.buckets.len() {
                return None;
            }

            if let Some(ll) = self.map.buckets[self.bucket_index].take() {
                let mut iter = ll.into_iter();
                if let Some(node) = iter.next() {
                    self.bucket_iter = Some(iter);
                    return Some((node.key, node.value));
                }
            }

            self.bucket_index += 1;
        }
    }
}

impl<K, V> IntoIterator for HashMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, K, V> IntoIterator for &'a HashMap<K, V> {
    type Item = (&'a K, &'a V);

    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> FromIterator<(K, V)> for HashMap<K, V>
where
    K: PartialEq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = HashMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for HashMap<K, V>
where
    K: PartialEq + Hash,
{
    fn from(arr: [(K, V); N]) -> Self {
        Self::from_iter(arr.into_iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_with_capacity_0() {
        HashMap::<usize, usize>::with_capacity(0);
    }

    #[test]
    fn test_insert() {
        let mut hm = HashMap::new();
        let k = 0;
        let v = 1;
        let i = hm.bucket_index(&k);
        hm.insert(k, v);
        for (b, _) in hm.buckets.iter().enumerate() {
            if b == i {
                assert!(hm.buckets[b].is_some());
            } else {
                assert!(hm.buckets[b].is_none());
            }
        }
    }

    #[test]
    fn test_get() {
        let mut hm = HashMap::new();
        let k = 0;
        let v = 1;
        assert!(hm.get(&k).is_none());
        hm.insert(k, v);
        assert_eq!(hm.get(&k), Some(&v));
    }

    #[test]
    fn test_get_mut() {
        let mut hm = HashMap::new();
        let k = 0;
        let mut v = 1;
        assert!(hm.get_mut(&k).is_none());
        hm.insert(k, v);
        assert_eq!(hm.get_mut(&k), Some(&mut v));
    }

    #[test]
    fn test_get_mut_ptr() {
        let mut hm = HashMap::new();
        let k = 0;
        let v = 1;
        assert!(hm.get_mut_ptr(&k).is_none());
        hm.insert(k, v);
        assert_eq!(unsafe { *hm.get_mut_ptr(&k).unwrap() }, v);
    }

    #[test]
    fn test_overwrite_value() {
        let mut hm = HashMap::new();
        let k = 0;
        let v = 1;
        hm.insert(k, v);
        assert_eq!(hm.get(&k), Some(&v));
        let v2 = 2;
        hm.insert(k, v2);
        assert_eq!(hm.get(&k), Some(&v2));
    }

    #[test]
    fn test_remove() {
        let key_1 = 0;
        let value_1 = 1;
        let key_2 = 2;
        let value_2 = 3;
        let mut hm = HashMap::new();
        assert!(hm.remove(&key_1).is_none());
        hm.insert(key_1, value_1);
        hm.insert(key_2, value_2);
        assert_eq!(hm.remove(&key_1), Some(value_1));
        assert!(hm.get(&key_1).is_none());
        assert!(hm.remove(&key_1).is_none());
        assert_eq!(hm.remove(&key_2), Some(value_2));
        assert!(hm.get(&key_2).is_none());
        assert!(hm.remove(&key_2).is_none());
        for b in hm.buckets {
            assert!(b.is_none());
        }
    }

    #[test]
    fn test_iter() {
        let mut hm = HashMap::new();

        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);

        let key_1 = "key_1";
        let value_1 = "value_1";
        hm.insert(key_1, value_1);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_1, &value_1)]);

        let key_2 = "key_2";
        let value_2 = "value_2";
        hm.insert(key_2, value_2);
        let mut vec = hm.iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &value_1), (&key_2, &value_2)]);

        let new_value_1 = "new_value_1";
        hm.insert(key_1, new_value_1);
        let mut vec = hm.iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &new_value_1), (&key_2, &value_2),]);

        hm.remove(&key_1);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_2, &value_2)]);

        hm.remove(&key_2);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);
    }

    #[test]
    fn test_iter_multiple_items_in_a_bucket() {
        let mut hm = HashMap::new();

        let mut first_bucket = LinkedList::new();
        first_bucket.push_back(Node { key: 0, value: 0 });
        first_bucket.push_back(Node { key: 1, value: 1 });

        let mut last_bucket = LinkedList::new();
        last_bucket.push_back(Node { key: 2, value: 2 });
        last_bucket.push_back(Node { key: 3, value: 3 });

        hm.buckets[0] = Some(first_bucket);
        hm.buckets[DEFAULT_CAPACITY - 1] = Some(last_bucket);

        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&0, &0), (&1, &1), (&2, &2), (&3, &3)]);
    }

    #[test]
    fn test_into_iter() {
        let mut hm: HashMap<&str, &str> = HashMap::new();

        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);

        let key_1 = "key_1";
        let value_1 = "value_1";
        hm.insert(key_1, value_1);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_1, &value_1)]);

        let key_2 = "key_2";
        let value_2 = "value_2";
        hm.insert(key_2, value_2);
        let mut vec = hm.iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &value_1), (&key_2, &value_2)]);

        let new_value_1 = "new_value_1";
        hm.insert(key_1, new_value_1);
        let mut vec = hm.iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &new_value_1), (&key_2, &value_2),]);

        hm.remove(&key_1);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_2, &value_2)]);

        hm.remove(&key_2);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);
    }

    #[test]
    fn test_into_iter_multiple_items_in_a_bucket() {
        let mut hm = HashMap::new();

        let mut first_bucket = LinkedList::new();
        first_bucket.push_back(Node { key: 0, value: 0 });
        first_bucket.push_back(Node { key: 1, value: 1 });

        let mut last_bucket = LinkedList::new();
        last_bucket.push_back(Node { key: 2, value: 2 });
        last_bucket.push_back(Node { key: 3, value: 3 });

        hm.buckets[0] = Some(first_bucket);
        hm.buckets[DEFAULT_CAPACITY - 1] = Some(last_bucket);

        let vec = hm.into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn test_borrowed_into_iter() {
        let mut hm = HashMap::new();

        let vec = (&hm).into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);

        let key_1 = "key_1";
        let value_1 = "value_1";
        hm.insert(key_1, value_1);
        let vec = (&hm).into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_1, &value_1)]);

        let key_2 = "key_2";
        let value_2 = "value_2";
        hm.insert(key_2, value_2);
        let mut vec = (&hm).into_iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &value_1), (&key_2, &value_2)]);

        let new_value_1 = "new_value_1";
        hm.insert(key_1, new_value_1);
        let mut vec = (&hm).into_iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(&key_1, &new_value_1), (&key_2, &value_2),]);

        hm.remove(&key_1);
        let vec = (&hm).into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_2, &value_2)]);

        hm.remove(&key_2);
        let vec = (&hm).into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);
    }

    #[test]
    fn test_from_iter() {
        let hm = HashMap::from_iter(vec![(1, 2), (3, 4)]);
        assert_eq!(hm.get(&1), Some(&2));
        assert_eq!(hm.get(&3), Some(&4));
    }

    #[test]
    fn test_from() {
        let hm = HashMap::from([(1, 2), (3, 4)]);
        assert_eq!(hm.get(&1), Some(&2));
        assert_eq!(hm.get(&3), Some(&4));
    }

    #[test]
    fn test_rehash() {
        let mut hm = HashMap::new();
        let target_bucket = 0;
        let mut i = 0;

        loop {
            let bucket = hm.bucket_index(&i);
            if bucket == target_bucket {
                hm.insert(i, i);
                if let Some(ll) = &hm.buckets[0] {
                    if ll.len() == BUCKET_CAPACITY {
                        break;
                    }
                }
            }

            i += 1;
        }

        assert_eq!(hm.buckets.len(), DEFAULT_CAPACITY);
        assert_eq!(
            hm.buckets[target_bucket].as_ref().unwrap().len(),
            BUCKET_CAPACITY
        );

        loop {
            i += 1;
            let bucket = hm.bucket_index(&i);
            if bucket == target_bucket {
                hm.insert(i, i);
                break;
            }
        }

        assert_eq!(hm.buckets.len(), BUCKET_CAPACITY + 1);
        assert_eq!(hm.buckets[1].as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_len() {
        let mut hm = HashMap::new();
        assert!(hm.is_empty());
        assert_eq!(hm.len(), 0);
        hm.insert(0, 0);
        assert_eq!(hm.len(), 1);
        assert!(!hm.is_empty());
        hm.insert(0, 1);
        assert_eq!(hm.len(), 1);
        hm.insert(2, 2);
        assert_eq!(hm.len(), 2);
        hm.remove(&2);
        assert_eq!(hm.len(), 1);
        hm.remove(&0);
        assert_eq!(hm.len(), 0);
        assert!(hm.is_empty());
        hm.remove(&0);
        assert_eq!(hm.len(), 0);
    }

    #[test]
    fn test_contains_key() {
        let mut hm = HashMap::new();
        assert!(!hm.contains_key(&0));
        hm.insert(0, 0);
        assert!(hm.contains_key(&0));
        hm.insert(0, 1);
        assert!(hm.contains_key(&0));
        hm.remove(&0);
        assert!(!hm.contains_key(&0));
    }

    #[test]
    fn test_retain() {
        let mut hm = HashMap::new();
        for i in 0..=10 {
            hm.insert(i, i);
        }
        hm.retain(|k, _v| *k <= 10);
        assert_eq!(hm.len(), 11);
        hm.retain(|k, _v| *k <= 5);
        assert_eq!(hm.len(), 6);
        for i in 0..=5 {
            assert!(hm.contains_key(&i));
        }
        for i in 6..=10 {
            assert!(!hm.contains_key(&i));
        }

        let mut hm = HashMap::new();
        for i in 0..=10 {
            hm.insert(i, i);
        }
        hm.retain(|k, _v| *k >= 0);
        assert_eq!(hm.len(), 11);
        hm.retain(|k, _v| *k >= 5);
        assert_eq!(hm.len(), 6);
        for i in 0..=4 {
            assert!(!hm.contains_key(&i));
        }
        for i in 5..=10 {
            assert!(hm.contains_key(&i));
        }

        let mut hm = HashMap::new();
        for i in 0..=10 {
            hm.insert(i, i);
        }
        hm.retain(|_k, v| *v <= 10);
        assert_eq!(hm.len(), 11);
        hm.retain(|_k, v| *v <= 5);
        assert_eq!(hm.len(), 6);
        for i in 0..=5 {
            assert!(hm.contains_key(&i));
        }
        for i in 6..=10 {
            assert!(!hm.contains_key(&i));
        }

        let mut hm = HashMap::new();
        for i in 0..=10 {
            hm.insert(i, i);
        }
        hm.retain(|_k, v| *v >= 0);
        assert_eq!(hm.len(), 11);
        hm.retain(|_k, v| *v >= 5);
        assert_eq!(hm.len(), 6);
        for i in 0..=4 {
            assert!(!hm.contains_key(&i));
        }
        for i in 5..=10 {
            assert!(hm.contains_key(&i));
        }

        let mut hm = HashMap::new();
        for i in 0..=10 {
            hm.insert(i, i);
        }
        hm.retain(|_k, v| {
            *v += 1;
            *v >= 10
        });
        assert_eq!(hm.len(), 2);
        for i in 0..=8 {
            assert!(!hm.contains_key(&i));
        }
        for i in 9..=10 {
            assert!(hm.contains_key(&i));
        }
    }

    #[test]
    fn test_fuzz() {
        let mut hm = HashMap::new();
        let n = 100;

        for i in 0..=n {
            assert_eq!(hm.get(&i), None);
            assert!(!hm.contains_key(&i));

            hm.insert(i, i);

            for j in 0..=i {
                assert_eq!(hm.get(&j), Some(&j));
                assert!(hm.contains_key(&j));
            }
        }

        for i in 0..=n {
            assert_eq!(hm.remove(&i), Some(i));

            for j in 0..=i {
                assert_eq!(hm.get(&j), None);
                assert!(!hm.contains_key(&j));
            }

            for j in i + 1..=n {
                assert_eq!(hm.get(&j), Some(&j));
                assert!(hm.contains_key(&j));
            }
        }
    }

    #[test]
    fn test_clone() {
        let mut hm = HashMap::new();
        for i in 0..=5 {
            hm.insert(i, i);
        }
        let cloned = hm.clone();
        assert_eq!(hm, cloned);
    }

    #[test]
    fn test_index() {
        let mut hm = HashMap::new();
        hm.insert(0, 0);
        hm.insert(2, 2);
        assert_eq!(hm[0], 0);
        assert_eq!(hm[2], 2);
    }

    #[test]
    #[should_panic]
    fn test_index_key_does_not_exist() {
        let mut hm = HashMap::new();
        hm.insert(0, 0);
        hm.insert(2, 2);
        hm[1];
    }
}
