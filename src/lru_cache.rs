use std::{cell::RefCell, collections::HashMap, hash::Hash, rc::Weak};

use crate::linked_list::{LinkedList, Node};

/// Space complexity: O(n)
pub struct LRUCache<K, V> {
    list: LinkedList<Item<K, V>>,
    map: NodeMap<K, V>,
    capacity: usize,
}

type NodeMap<K, V> = HashMap<K, Weak<RefCell<Node<Item<K, V>>>>>;

#[derive(Clone)]
struct Item<K, V> {
    key: K,
    value: V,
}

#[derive(Debug)]
pub enum Error {
    KeyAlreadyExists,
}

impl<K, V> LRUCache<K, V>
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    pub fn with_capacity(capacity: usize) -> Self {
        LRUCache {
            list: LinkedList::new(),
            map: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Time complexity: O(1)
    pub fn insert(&mut self, key: K, value: V) -> Result<(), Error> {
        if self.contains_key(&key) {
            return Err(Error::KeyAlreadyExists);
        }
        if self.list.len() == self.capacity {
            if let Some(item) = self.list.pop_back() {
                self.map.remove(&item.key);
            }
        }
        let item = Item { key, value };
        self.insert_item(item);
        Ok(())
    }

    /// Time complexity: O(1)
    pub fn get(&mut self, key: &K) -> Option<V> {
        match self.map.get(key) {
            None => None,
            Some(node) => match node.upgrade() {
                Some(node) => {
                    let item = self.list.unlink(node);
                    self.insert_item(item.clone());
                    Some(item.value)
                }
                None => panic!("`Weak` pointer to `Node` could not be upgraded to `Rc`"),
            },
        }
    }

    /// Time complexity: O(1)
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.map.get(key) {
            None => None,
            Some(node) => match node.upgrade() {
                Some(node) => {
                    let item = self.list.unlink(node);
                    self.map.remove(key);
                    Some(item.value)
                }
                None => panic!("`Weak` pointer to `Node` could not be upgraded to `Rc`"),
            },
        }
    }

    /// Time complexity: O(1)
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Time complexity: O(1)
    fn insert_item(&mut self, item: Item<K, V>) {
        let node = self.list.push_front(item.clone());
        self.map.insert(item.key, node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut c = LRUCache::with_capacity(2);
        c.insert(1, 'a').unwrap();
        assert!(c.map.contains_key(&1));
        c.insert(1, 'b').unwrap_err();
        assert!(c.map.contains_key(&1));
    }

    #[test]
    fn test_get() {
        let mut c = LRUCache::with_capacity(2);
        assert!(c.get(&1).is_none());

        c.insert(1, 'a').unwrap();
        // LinkedList: head-1-None-tail
        assert_eq!(c.get(&1), Some('a'));

        c.insert(2, 'b').unwrap();
        // LinkedList: head-2-1-tail
        assert_eq!(c.get(&2), Some('b'));

        c.insert(3, 'c').unwrap();
        // LinkedList: head-3-2-tail
        assert!(c.get(&1).is_none());
        // LinkedList: head-2-3-tail
        assert_eq!(c.get(&2), Some('b'));
        // LinkedList: head-3-2-tail
        assert_eq!(c.get(&3), Some('c'));

        c.insert(4, 'd').unwrap();
        // LinkedList: head-4-3-tail
        assert!(c.get(&2).is_none());
        // LinkedList: head-3-4-tail
        assert_eq!(c.get(&3), Some('c'));
        // LinkedList: head-4-3-tail
        assert_eq!(c.get(&4), Some('d'));
    }

    #[test]
    fn test_evict_least_recently_used() {
        let mut c = LRUCache::with_capacity(3);
        c.insert(1, 'a').unwrap();
        c.insert(2, 'b').unwrap();
        c.insert(3, 'c').unwrap();
        // LinkedList: head-3-2-1-tail
        assert_eq!(c.list.first().unwrap().key, 3);
        assert_eq!(c.list.last().unwrap().key, 1);

        c.get(&1);
        // LinkedList: head-1-3-2-tail
        assert_eq!(c.list.first().unwrap().key, 1);
        assert_eq!(c.list.last().unwrap().key, 2);

        c.insert(4, 'd').unwrap();
        // LinkedList: head-4-1-3-tail
        assert_eq!(c.list.first().unwrap().key, 4);
        assert_eq!(c.list.last().unwrap().key, 3);

        c.get(&3);
        // LinkedList: head-3-4-1-tail
        assert_eq!(c.list.first().unwrap().key, 3);
        assert_eq!(c.list.last().unwrap().key, 1);

        c.insert(5, 'e').unwrap();
        // LinkedList: head-5-3-4-tail
        assert_eq!(c.list.first().unwrap().key, 5);
        assert_eq!(c.list.last().unwrap().key, 4);
    }

    #[test]
    fn test_contains_key() {
        let mut c = LRUCache::with_capacity(2);
        assert!(!c.contains_key(&1));
        c.insert(1, 'a').unwrap();
        assert!(c.contains_key(&1));
        c.remove(&1);
        assert!(!c.contains_key(&1));
    }

    #[test]
    fn test_remove() {
        let mut c = LRUCache::with_capacity(2);
        assert!(c.remove(&1).is_none());
        c.insert(1, 'a').unwrap();
        c.insert(2, 'b').unwrap();
        assert_eq!(c.remove(&1), Some('a'));
        assert!(c.remove(&1).is_none());
        assert_eq!(c.remove(&2), Some('b'));
        assert!(c.remove(&2).is_none());
    }
}
