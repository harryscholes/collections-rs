use std::{cell::RefCell, collections::HashMap, hash::Hash, rc::Weak};

use crate::linked_list::{LinkedList, Node};

type NodeMap<K, V> = HashMap<K, Weak<RefCell<Node<Item<K, V>>>>>;

/// Space complexity: O(n)
pub struct LRUCache<K, V> {
    list: LinkedList<Item<K, V>>,
    map: NodeMap<K, V>,
    capacity: usize,
}

struct Item<K, V> {
    key: K,
    value: V,
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
    pub fn insert(&mut self, key: K, value: V) {
        if self.list.len() == self.capacity {
            if let Some(item) = self.list.pop_back() {
                self.map.remove(&item.key);
            }
        }
        let item = Item {
            key: key.clone(),
            value,
        };
        let node = self.list.push_front(item);
        self.map.insert(key, node);
    }

    /// Time complexity: O(1)
    pub fn get(&mut self, key: &K) -> Option<V> {
        match self.map.get(key) {
            None => None,
            Some(node) => match node.upgrade() {
                Some(node) => {
                    self.list.unlink(node.clone());
                    self.list.push_node_front(node.clone());
                    Some(node.borrow().element.value.clone())
                }
                None => panic!("`Weak` pointer to `Node` could not be upgraded to `Rc`"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_get() {
        let mut c = LRUCache::with_capacity(2);
        assert!(c.get(&1).is_none());

        c.insert(1, 'a');
        // LinkedList: head-1-None-tail
        assert_eq!(c.get(&1), Some('a'));

        c.insert(2, 'b');
        // LinkedList: head-2-1-tail
        assert_eq!(c.get(&2), Some('b'));

        c.insert(3, 'c');
        // LinkedList: head-3-2-tail
        assert!(c.get(&1).is_none());
        // LinkedList: head-2-3-tail
        assert_eq!(c.get(&2), Some('b'));
        // LinkedList: head-3-2-tail
        assert_eq!(c.get(&3), Some('c'));

        c.insert(4, 'd');
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
        c.insert(1, 'a');
        c.insert(2, 'b');
        c.insert(3, 'c');
        // LinkedList: head-3-2-1-tail
        assert_eq!(c.list.first().unwrap().borrow().element.key, 3);
        assert_eq!(c.list.last().unwrap().borrow().element.key, 1);

        c.get(&1);
        // LinkedList: head-1-3-2-tail
        assert_eq!(c.list.first().unwrap().borrow().element.key, 1);
        assert_eq!(c.list.last().unwrap().borrow().element.key, 2);

        c.insert(4, 'd');
        // LinkedList: head-4-1-3-tail
        assert_eq!(c.list.first().unwrap().borrow().element.key, 4);
        assert_eq!(c.list.last().unwrap().borrow().element.key, 3);

        c.get(&3);
        // LinkedList: head-3-4-1-tail
        assert_eq!(c.list.first().unwrap().borrow().element.key, 3);
        assert_eq!(c.list.last().unwrap().borrow().element.key, 1);

        c.insert(5, 'e');
        // LinkedList: head-5-3-4-tail
        assert_eq!(c.list.first().unwrap().borrow().element.key, 5);
        assert_eq!(c.list.last().unwrap().borrow().element.key, 4);
    }
}
