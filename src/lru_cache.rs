use std::{collections::HashMap, hash::Hash};

use self::linked_list::{LinkedList, WeakLink};

/// Space complexity: O(n)
pub struct LRUCache<K, V> {
    list: LinkedList<Item<K, V>>,
    map: NodeMap<K, V>,
    capacity: usize,
}

type NodeMap<K, V> = HashMap<K, WeakLink<Item<K, V>>>;

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

#[allow(dead_code)]
mod linked_list {
    use std::{
        cell::RefCell,
        rc::{Rc, Weak},
    };

    pub struct Node<T> {
        pub element: T,
        next: Option<Link<T>>,
        prev: Option<Link<T>>,
    }

    type Link<T> = Rc<RefCell<Node<T>>>;
    pub type WeakLink<T> = Weak<RefCell<Node<T>>>;

    impl<T> Node<T> {
        fn new(element: T) -> Self {
            Node {
                element,
                next: None,
                prev: None,
            }
        }
    }

    /// Space complexity: O(n)
    pub struct LinkedList<T> {
        head: Option<Link<T>>,
        tail: Option<Link<T>>,
        len: usize,
    }

    impl<T> LinkedList<T> {
        pub fn new() -> Self {
            LinkedList {
                head: None,
                tail: None,
                len: 0,
            }
        }

        /// Time complexity: O(1)
        pub fn push_front(&mut self, elt: T) -> WeakLink<T> {
            let node = Rc::new(RefCell::new(Node::new(elt)));
            node.borrow_mut().prev = None;
            match &self.head {
                None => {
                    self.tail = Some(node.clone());
                    node.borrow_mut().next = None;
                }
                Some(head) => {
                    head.borrow_mut().prev = Some(node.clone());
                    node.borrow_mut().next = Some(head.clone());
                }
            }
            self.head = Some(node.clone());
            self.len += 1;
            Rc::downgrade(&node)
        }

        /// Time complexity: O(1)
        pub fn pop_back(&mut self) -> Option<T> {
            self.tail.clone().map(|node| self.unlink(node))
        }

        /// Time complexity: O(1)
        pub fn unlink(&mut self, node: Link<T>) -> T {
            match node.borrow().prev.clone() {
                Some(prev) => prev.borrow_mut().next = node.borrow().next.clone(),
                None => self.head = node.borrow().next.clone(),
            };
            match node.borrow().next.clone() {
                Some(next) => next.borrow_mut().prev = node.borrow().prev.clone(),
                None => self.tail = node.borrow().prev.clone(),
            };
            self.len -= 1;
            match Rc::try_unwrap(node) {
                Ok(node) => node.into_inner().element,
                Err(_) => panic!("Unwrapping `Rc` failed because more than one reference exists"),
            }
        }

        /// Time complexity: O(1)
        pub fn len(&self) -> usize {
            self.len
        }

        pub fn is_empty(&self) -> bool {
            self.len == 0
        }
    }

    impl<T> LinkedList<T>
    where
        T: Clone,
    {
        /// Time complexity: O(1)
        pub fn first(&self) -> Option<T> {
            self.head.clone().map(|node| node.borrow().element.clone())
        }

        /// Time complexity: O(1)
        pub fn last(&self) -> Option<T> {
            self.tail.clone().map(|node| node.borrow().element.clone())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_first() {
            let mut l = LinkedList::new();
            assert!(l.first().is_none());

            l.push_front(1);
            assert_eq!(l.first(), Some(1));

            l.push_front(2);
            assert_eq!(l.first(), Some(2));
        }

        #[test]
        fn test_last() {
            let mut l = LinkedList::new();
            assert!(l.last().is_none());

            l.push_front(1);
            assert_eq!(l.last(), Some(1));

            l.push_front(2);
            assert_eq!(l.last(), Some(1));
        }

        #[test]
        fn test_push_front() {
            let mut l = LinkedList::new();
            assert!(l.first().is_none());
            assert!(l.last().is_none());

            l.push_front(1);
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(1));

            l.push_front(2);
            assert_eq!(l.first(), Some(2));
            assert_eq!(l.last(), Some(1));

            l.push_front(3);
            assert_eq!(l.first(), Some(3));
            assert_eq!(l.last(), Some(1));
        }

        #[test]
        fn test_pop_back() {
            let mut l = LinkedList::new();
            l.push_front(3);
            l.push_front(2);
            l.push_front(1);

            assert_eq!(l.pop_back(), Some(3));
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(2));

            assert_eq!(l.pop_back(), Some(2));
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(1));

            assert_eq!(l.pop_back(), Some(1));
            assert!(l.first().is_none());
            assert!(l.last().is_none());
        }

        #[test]
        fn test_unlink() {
            let mut l = LinkedList::new();
            l.push_front(4);
            let node_3 = l.push_front(3);
            let node_2 = l.push_front(2);
            l.push_front(1);
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(4));

            l.unlink(node_2.upgrade().unwrap());
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(4));

            l.unlink(node_3.upgrade().unwrap());
            assert_eq!(l.first(), Some(1));
            assert_eq!(l.last(), Some(4));
        }

        #[test]
        fn test_node_weak_pointer() {
            let mut l = LinkedList::new();
            l.push_front(3);
            let weak_node_2 = l.push_front(2);
            l.push_front(1);

            let node_2 = weak_node_2.upgrade().unwrap();
            l.unlink(node_2);
            assert!(weak_node_2.upgrade().is_none());
        }

        #[test]
        #[should_panic]
        fn test_unlink_panic() {
            let mut l = LinkedList::new();
            l.push_front(3);
            let weak_node_2 = l.push_front(2);
            l.push_front(1);

            let node_2 = weak_node_2.upgrade().unwrap();
            let _another_node_2_ref = node_2.clone();
            l.unlink(node_2);
        }

        #[test]
        fn test_len() {
            let mut l = LinkedList::new();
            assert!(l.is_empty());

            l.push_front(1);
            assert_eq!(l.len(), 1);

            l.push_front(2);
            assert_eq!(l.len(), 2);

            l.pop_back();
            assert_eq!(l.len(), 1);

            l.pop_back();
            assert!(l.is_empty());
        }
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
