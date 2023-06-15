use std::collections::LinkedList;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

const DEFAULT_CAPACITY: usize = 4;
const BUCKET_CAPACITY: usize = 4;

#[derive(Debug, Clone)]
pub struct HashMap<'a, K, V> {
    buckets: Vec<Option<Bucket<K, V>>>,
    len: usize,
    phantom_data: PhantomData<&'a V>,
}

type Bucket<K, V> = LinkedList<Node<K, V>>;

#[derive(Debug, Clone)]
struct Node<K, V> {
    key: K,
    value: V,
}

impl<'a, K, V> HashMap<'a, K, V>
where
    K: PartialEq + Hash,
{
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buckets: (0..capacity).map(|_| None).collect(),
            len: 0,
            phantom_data: PhantomData,
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (ret, resize) = self.insert_without_resizing(key, value);
        if resize {
            self.resize()
        }
        ret
    }

    fn insert_without_resizing(&mut self, key: K, value: V) -> (Option<V>, bool) {
        let bucket_index = self.bucket_index(&key);
        let mut resize = false;
        match self.buckets[bucket_index] {
            Some(ref mut ll) => {
                for node in ll.iter_mut() {
                    if node.key == key {
                        let old_value = std::mem::replace(&mut node.value, value);
                        return (Some(old_value), resize);
                    }
                }

                ll.push_back(Node { key, value });
                self.len += 1;

                if ll.len() >= BUCKET_CAPACITY {
                    resize = true;
                }

                (None, resize)
            }
            None => {
                let mut ll = LinkedList::new();
                ll.push_back(Node { key, value });
                self.buckets[bucket_index] = Some(ll);
                self.len += 1;

                (None, resize)
            }
        }
    }

    fn resize(&mut self) {
        let new_buckets = (0..self.len()).map(|_| None).collect();
        let old_buckets = std::mem::replace(&mut self.buckets, new_buckets);

        let old_map = HashMap {
            buckets: old_buckets,
            len: 0,
            phantom_data: PhantomData,
        };
        self.len = 0;
        for (key, value) in old_map {
            self.insert_without_resizing(key, value);
        }
    }

    pub fn get(&'a self, key: &K) -> Option<&'a V> {
        match &self.buckets[self.bucket_index(key)] {
            Some(ll) => {
                for node in ll {
                    if &node.key == key {
                        return Some(&node.value);
                    }
                }
                None
            }
            None => None,
        }
    }

    pub fn delete(&mut self, key: &K) -> Option<V> {
        let bucket_index = self.bucket_index(key);
        match &mut self.buckets[bucket_index] {
            Some(ref mut ll) => {
                for (index, node) in ll.iter().enumerate() {
                    if node.key == *key {
                        let node = ll.remove(index);
                        self.len -= 1;
                        return Some(node.value);
                    }
                }
                None
            }
            None => None,
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

    pub fn iter(&'a self) -> Iter<'a, K, V> {
        Iter::new(self)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains_key(&self, key: &K) -> bool {
        match self.get(key) {
            Some(_) => true,
            None => false,
        }
    }
}

impl<'a, K, V> Default for HashMap<'a, K, V>
where
    K: PartialEq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct Iter<'a, K, V> {
    map: &'a HashMap<'a, K, V>,
    bucket_index: usize,
    bucket_iter: Option<std::collections::linked_list::Iter<'a, Node<K, V>>>,
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

pub struct IntoIter<'a, K, V> {
    map: HashMap<'a, K, V>,
    bucket_index: usize,
    bucket_iter: Option<std::collections::linked_list::IntoIter<Node<K, V>>>,
}

impl<'a, K, V> IntoIter<'a, K, V> {
    fn new(map: HashMap<'a, K, V>) -> Self {
        Self {
            map,
            bucket_index: 0,
            bucket_iter: None,
        }
    }
}

impl<'a, K, V> Iterator for IntoIter<'a, K, V> {
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

impl<'a, K, V> IntoIterator for HashMap<'a, K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

impl<'a, K, V> FromIterator<(K, V)> for HashMap<'a, K, V>
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

impl<'a, K, V, const N: usize> From<[(K, V); N]> for HashMap<'a, K, V>
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
    fn test_delete() {
        let key_1 = 0;
        let value_1 = 1;
        let key_2 = 2;
        let value_2 = 3;
        let mut hm = HashMap::new();
        assert!(hm.delete(&key_1).is_none());
        hm.insert(key_1, value_1);
        hm.insert(key_2, value_2);
        assert_eq!(hm.delete(&key_1), Some(value_1));
        assert!(hm.get(&key_1).is_none());
        assert!(hm.delete(&key_1).is_none());
        assert_eq!(hm.delete(&key_2), Some(value_2));
        assert!(hm.get(&key_2).is_none());
        assert!(hm.delete(&key_2).is_none());
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

        hm.delete(&key_1);
        let vec = hm.iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(&key_2, &value_2)]);

        hm.delete(&key_2);
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
        let mut hm = HashMap::new();

        let vec = hm.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![]);

        let key_1 = "key_1";
        let value_1 = "value_1";
        hm.insert(key_1, value_1);
        let vec = hm.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(key_1, value_1)]);

        let key_2 = "key_2";
        let value_2 = "value_2";
        hm.insert(key_2, value_2);
        let mut vec = hm.clone().into_iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(key_1, value_1), (key_2, value_2)]);

        let new_value_1 = "new_value_1";
        hm.insert(key_1, new_value_1);
        let mut vec = hm.clone().into_iter().collect::<Vec<_>>();
        vec.sort();
        assert_eq!(vec, vec![(key_1, new_value_1), (key_2, value_2),]);

        hm.delete(&key_1);
        let vec = hm.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(key_2, value_2)]);

        hm.delete(&key_2);
        let vec = hm.clone().into_iter().collect::<Vec<_>>();
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

        let vec = hm.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(vec, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
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
    fn test_resize() {
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
        hm.delete(&2);
        assert_eq!(hm.len(), 1);
        hm.delete(&0);
        assert_eq!(hm.len(), 0);
        assert!(hm.is_empty());
        hm.delete(&0);
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
        hm.delete(&0);
        assert!(!hm.contains_key(&0));
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
            assert_eq!(hm.delete(&i), Some(i));

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
}
