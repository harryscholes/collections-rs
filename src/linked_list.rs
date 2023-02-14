// TODO impl Iterator

use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

pub struct Node<T> {
    pub element: T,
    next: Option<Rc<RefCell<Node<T>>>>,
    prev: Option<Rc<RefCell<Node<T>>>>,
}

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
    head: Option<Rc<RefCell<Node<T>>>>,
    tail: Option<Rc<RefCell<Node<T>>>>,
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
    pub fn push_back(&mut self, elt: T) -> Weak<RefCell<Node<T>>> {
        let node = Rc::new(RefCell::new(Node::new(elt)));
        node.borrow_mut().next = None;
        match &self.tail {
            None => {
                self.head = Some(node.clone());
                node.borrow_mut().prev = None;
            }
            Some(tail) => {
                tail.borrow_mut().next = Some(node.clone());
                node.borrow_mut().prev = Some(tail.clone());
            }
        }
        self.tail = Some(node.clone());
        self.len += 1;
        Rc::downgrade(&node)
    }

    /// Time complexity: O(1)
    pub fn push_front(&mut self, elt: T) -> Weak<RefCell<Node<T>>> {
        let node = Rc::new(RefCell::new(Node::new(elt)));
        self.push_node_front(node)
    }

    /// Time complexity: O(1)
    pub(crate) fn push_node_front(&mut self, node: Rc<RefCell<Node<T>>>) -> Weak<RefCell<Node<T>>> {
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
        match self.tail.clone() {
            None => None,
            Some(node) => {
                self.unlink(node.clone());
                Some(self.try_unwrap(node))
            }
        }
    }

    /// Time complexity: O(1)
    pub fn pop_front(&mut self) -> Option<T> {
        match self.head.clone() {
            None => None,
            Some(node) => {
                self.unlink(node.clone());
                Some(self.try_unwrap(node))
            }
        }
    }

    /// Time complexity: O(1)
    pub fn unlink(&mut self, node: Rc<RefCell<Node<T>>>) {
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

    /// Time complexity: O(1)
    pub fn first(&self) -> Option<Rc<RefCell<Node<T>>>> {
        match self.head.clone() {
            Some(node) => Some(node),
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn last(&self) -> Option<Rc<RefCell<Node<T>>>> {
        match self.tail.clone() {
            Some(node) => Some(node),
            None => None,
        }
    }

    /// Time complexity: O(1)
    pub fn len(&self) -> usize {
        self.len
    }

    fn try_unwrap(&self, node: Rc<RefCell<Node<T>>>) -> T {
        match Rc::try_unwrap(node) {
            Ok(node) => node.into_inner().element,
            Err(_) => panic!("Unwrapping `Rc` failed because more than one reference exists"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_back() {
        let mut l = LinkedList::new();
        assert!(l.head.is_none());
        assert!(l.tail.is_none());

        l.push_back(1);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        l.push_back(2);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 2);

        l.push_back(3);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 3);
    }

    #[test]
    fn test_push_front() {
        let mut l = LinkedList::new();
        assert!(l.head.is_none());
        assert!(l.tail.is_none());

        l.push_front(1);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        l.push_front(2);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 2);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        l.push_front(3);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 3);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);
    }

    #[test]
    fn test_pop_back() {
        let mut l = LinkedList::new();
        l.push_back(1);
        l.push_back(2);
        l.push_back(3);

        assert_eq!(l.pop_back(), Some(3));
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 2);

        assert_eq!(l.pop_back(), Some(2));
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        assert_eq!(l.pop_back(), Some(1));
        assert!(l.head.is_none());
        assert!(l.tail.is_none());
    }

    #[test]
    fn test_pop_front() {
        let mut l = LinkedList::new();
        l.push_front(1);
        l.push_front(2);
        l.push_front(3);

        assert_eq!(l.pop_front(), Some(3));
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 2);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 1);

        assert_eq!(l.pop_front(), Some(1));
        assert!(l.head.is_none());
        assert!(l.tail.is_none());
    }

    #[test]
    fn test_unlink() {
        let mut l = LinkedList::new();
        l.push_front(4);
        let node_3 = l.push_front(3);
        let node_2 = l.push_front(2);
        l.push_front(1);
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 4);

        l.unlink(node_2.upgrade().unwrap());
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 4);
        assert_eq!(
            l.head
                .as_ref()
                .unwrap()
                .borrow()
                .next
                .as_ref()
                .unwrap()
                .borrow()
                .element,
            3
        );
        assert_eq!(
            l.tail
                .as_ref()
                .unwrap()
                .borrow()
                .prev
                .as_ref()
                .unwrap()
                .borrow()
                .element,
            3
        );

        l.unlink(node_3.upgrade().unwrap());
        assert_eq!(l.head.as_ref().unwrap().borrow().element, 1);
        assert_eq!(l.tail.as_ref().unwrap().borrow().element, 4);
        assert_eq!(
            l.head
                .as_ref()
                .unwrap()
                .borrow()
                .next
                .as_ref()
                .unwrap()
                .borrow()
                .element,
            4
        );
        assert_eq!(
            l.tail
                .as_ref()
                .unwrap()
                .borrow()
                .prev
                .as_ref()
                .unwrap()
                .borrow()
                .element,
            1
        );
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
}
