use std::{cell::Ref, collections::HashMap, hash::Hash};

use self::linked_list::{LinkedList, WeakLink};

/// Space complexity: O(n)
pub struct LRUCache<K, V> {
    list: LinkedList<Item<K, V>>,
    map: NodeMap<K, V>,
    capacity: usize,
}

type NodeMap<K, V> = HashMap<K, WeakLink<Item<K, V>>>;

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
{
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
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
    pub fn get(&mut self, key: &K) -> Option<Ref<V>> {
        match self.map.get(key) {
            Some(node) => {
                self.list.move_to_front(node);
                self.list
                    .first()
                    .map(|item| Ref::map(item, |item| &item.value))
            }
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn remove(&mut self, key: &K) -> Option<V> {
        match self.map.get(key) {
            Some(node) => match node.upgrade() {
                Some(node) => {
                    let item = self.list.remove(node);
                    self.map.remove(key);
                    Some(item.value)
                }
                None => panic!("`Weak` pointer to `Node` could not be upgraded to `Rc`"),
            },
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Time complexity: O(1)
    fn insert_item(&mut self, item: Item<K, V>) {
        let key = item.key.clone();
        let node = self.list.push_front(item);
        self.map.insert(key, node);
    }
}

mod linked_list {
    use std::{
        cell::{Ref, RefCell},
        rc::{Rc, Weak},
    };

    pub struct Node<T> {
        element: T,
        next: Option<Link<T>>,
        prev: Option<Link<T>>,
    }

    type Link<T> = Rc<RefCell<Node<T>>>;
    pub type WeakLink<T> = Weak<RefCell<Node<T>>>;

    impl<T> Node<T> {
        fn new(element: T) -> Self {
            Self {
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
            Self {
                head: None,
                tail: None,
                len: 0,
            }
        }

        /// Time complexity: O(1)
        pub fn push_front(&mut self, elt: T) -> WeakLink<T> {
            let node = Rc::new(RefCell::new(Node::new(elt)));
            self.push_node_front(node.clone());
            Rc::downgrade(&node)
        }

        /// Time complexity: O(1)
        fn push_node_front(&mut self, node: Link<T>) {
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
            self.head = Some(node);
            self.len += 1;
        }

        /// Time complexity: O(1)
        pub fn pop_back(&mut self) -> Option<T> {
            self.tail.clone().map(|node| self.remove(node))
        }

        /// Time complexity: O(1)
        pub fn move_to_front(&mut self, node: &WeakLink<T>) {
            match node.upgrade() {
                Some(node) => {
                    self.unlink(node.clone());
                    self.push_node_front(node);
                }
                None => panic!("`Weak` pointer to `Node` could not be upgraded to `Rc`"),
            };
        }

        /// Time complexity: O(1)
        fn unlink(&mut self, node: Link<T>) {
            match node.borrow().prev.clone() {
                Some(prev) => prev.borrow_mut().next = node.borrow().next.clone(),
                None => self.head = node.borrow().next.clone(),
            };
            match node.borrow().next.clone() {
                Some(next) => next.borrow_mut().prev = node.borrow().prev.clone(),
                None => self.tail = node.borrow().prev.clone(),
            };
            self.len -= 1;
        }

        pub fn remove(&mut self, node: Link<T>) -> T {
            self.unlink(node.clone());
            match Rc::try_unwrap(node) {
                Ok(node) => node.into_inner().element,
                Err(_) => panic!("Unwrapping `Rc` failed because more than one reference exists"),
            }
        }

        /// Time complexity: O(1)
        pub fn first(&self) -> Option<Ref<T>> {
            self.head
                .as_ref()
                .map(|node| Ref::map(node.borrow(), |n| &n.element))
        }

        /// Time complexity: O(1)
        #[cfg(test)]
        pub fn last(&self) -> Option<Ref<T>> {
            self.tail
                .as_ref()
                .map(|node| Ref::map(node.borrow(), |n| &n.element))
        }

        /// Time complexity: O(1)
        pub fn len(&self) -> usize {
            self.len
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
            assert_eq!(*l.first().unwrap(), 1);

            l.push_front(2);
            assert_eq!(*l.first().unwrap(), 2);
        }

        #[test]
        fn test_last() {
            let mut l = LinkedList::new();
            assert!(l.last().is_none());

            l.push_front(1);
            assert_eq!(*l.last().unwrap(), 1);

            l.push_front(2);
            assert_eq!(*l.last().unwrap(), 1);
        }

        #[test]
        fn test_push_front() {
            let mut l = LinkedList::new();
            assert!(l.first().is_none());
            assert!(l.last().is_none());

            l.push_front(1);
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 1);

            l.push_front(2);
            assert_eq!(*l.first().unwrap(), 2);
            assert_eq!(*l.last().unwrap(), 1);

            l.push_front(3);
            assert_eq!(*l.first().unwrap(), 3);
            assert_eq!(*l.last().unwrap(), 1);
        }

        #[test]
        fn test_pop_back() {
            let mut l = LinkedList::new();
            l.push_front(3);
            l.push_front(2);
            l.push_front(1);

            assert_eq!(l.pop_back(), Some(3));
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 2);

            assert_eq!(l.pop_back(), Some(2));
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 1);

            assert_eq!(l.pop_back(), Some(1));
            assert!(l.first().is_none());
            assert!(l.last().is_none());
        }

        #[test]
        fn test_unlink() {
            let mut l = LinkedList::new();
            let node_3 = l.push_front(3);
            let node_2 = l.push_front(2);
            l.push_front(1);

            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 3);

            l.unlink(node_2.upgrade().unwrap());
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 3);

            l.unlink(node_3.upgrade().unwrap());
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 1);
        }

        #[test]
        fn test_unlink_node_weak_pointer() {
            let mut l = LinkedList::new();
            l.push_front(3);
            let weak_node_2 = l.push_front(2);
            l.push_front(1);

            let node_2 = weak_node_2.upgrade().unwrap();
            l.unlink(node_2);
            assert!(weak_node_2.upgrade().is_none());
        }

        #[test]
        fn test_remove() {
            let mut l = LinkedList::new();
            let node_3 = l.push_front(3);
            let node_2 = l.push_front(2);
            l.push_front(1);
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 3);

            l.remove(node_2.upgrade().unwrap());
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 3);

            l.remove(node_3.upgrade().unwrap());
            assert_eq!(*l.first().unwrap(), 1);
            assert_eq!(*l.last().unwrap(), 1);
        }

        #[test]
        fn test_remove_node_weak_pointer() {
            let mut l = LinkedList::new();
            l.push_front(3);
            let weak_node_2 = l.push_front(2);
            l.push_front(1);

            let node_2 = weak_node_2.upgrade().unwrap();
            l.remove(node_2);
            assert!(weak_node_2.upgrade().is_none());
        }

        #[test]
        #[should_panic]
        fn test_remove_panic() {
            let mut l = LinkedList::new();
            l.push_front(3);
            let weak_node_2 = l.push_front(2);
            l.push_front(1);

            let node_2 = weak_node_2.upgrade().unwrap();
            let _another_node_2_ref = node_2.clone();
            l.remove(node_2);
        }

        #[test]
        fn test_len() {
            let mut l = LinkedList::new();
            assert_eq!(l.len(), 0);

            l.push_front(1);
            assert_eq!(l.len(), 1);

            l.push_front(2);
            assert_eq!(l.len(), 2);

            l.pop_back().unwrap();
            assert_eq!(l.len(), 1);

            l.pop_back().unwrap();
            assert_eq!(l.len(), 0);
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
        assert_eq!(*c.get(&1).unwrap(), 'a');

        c.insert(2, 'b').unwrap();
        // LinkedList: head-2-1-tail
        assert_eq!(*c.get(&2).unwrap(), 'b');

        c.insert(3, 'c').unwrap();
        // LinkedList: head-3-2-tail
        assert!(c.get(&1).is_none());
        // LinkedList: head-2-3-tail
        assert_eq!(*c.get(&2).unwrap(), 'b');
        // LinkedList: head-3-2-tail
        assert_eq!(*c.get(&3).unwrap(), 'c');

        c.insert(4, 'd').unwrap();
        // LinkedList: head-4-3-tail
        assert!(c.get(&2).is_none());
        // LinkedList: head-3-4-tail
        assert_eq!(*c.get(&3).unwrap(), 'c');
        // LinkedList: head-4-3-tail
        assert_eq!(*c.get(&4).unwrap(), 'd');
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
