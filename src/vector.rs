#![allow(clippy::missing_safety_doc)]

use std::{
    alloc::Layout,
    ops::{Deref, Index, IndexMut},
};

#[derive(Debug, Eq)]
pub struct Vector<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T> Vector<T> {
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut vec = Self::new();
        vec.reserve(capacity);
        vec
    }

    pub fn push_front(&mut self, el: T) {
        if self.full() {
            self.grow();
        }
        unsafe {
            std::ptr::copy(self.ptr, self.ptr.add(1), self.len);
            self.ptr.write(el);
        }
        self.len += 1;
    }

    pub fn push_back(&mut self, el: T) {
        if self.full() {
            self.grow();
        }
        unsafe { self.ptr.add(self.len).write(el) };
        self.len += 1;
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        for el in iter {
            self.push_back(el)
        }
    }

    pub unsafe fn get_ptr_unchecked(&self, index: usize) -> *const T {
        self.ptr.add(index)
    }

    pub unsafe fn get_ptr_mut_unchecked(&mut self, index: usize) -> *mut T {
        self.ptr.add(index)
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.get_ptr_unchecked(index)
    }

    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.get_ptr_mut_unchecked(index)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if self.index_is_in_bounds(index) {
            Some(unsafe { self.get_unchecked(index) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.index_is_in_bounds(index) {
            Some(unsafe { self.get_unchecked_mut(index) })
        } else {
            None
        }
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            unsafe {
                let el = self.ptr.read();
                std::ptr::copy(self.ptr.add(1), self.ptr, self.len);
                self.len -= 1;
                Some(el)
            }
        }
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            self.len -= 1;
            Some(unsafe { self.ptr.add(self.len).read() })
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        if self.capacity < self.len + additional {
            let new_capacity = self.len + additional;
            let new_ptr = self.alloc(new_capacity);

            if !self.ptr.is_null() {
                unsafe {
                    std::ptr::copy(self.ptr, new_ptr, self.len);
                    std::alloc::dealloc(
                        self.ptr as *mut u8,
                        Layout::array::<T>(self.capacity).unwrap(),
                    );
                }
            }

            self.capacity = new_capacity;
            self.ptr = new_ptr;
        }
    }

    fn grow(&mut self) {
        self.reserve((self.capacity * 2).max(1))
    }

    fn alloc(&mut self, size: usize) -> *mut T {
        let size = size.min(isize::MAX as usize);
        unsafe { std::alloc::alloc(Layout::array::<T>(size).unwrap()) as *mut T }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn available(&self) -> usize {
        self.capacity - self.len
    }

    pub fn full(&self) -> bool {
        self.available() == 0
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut::new(self)
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.assert_in_bounds(i);
        self.assert_in_bounds(j);
        unsafe {
            let ptr_i = self.ptr.add(i);
            let ptr_j = self.ptr.add(j);
            let el_i = ptr_i.read();
            let el_j = ptr_j.read();
            ptr_i.write(el_j);
            ptr_j.write(el_i);
        }
    }

    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    pub fn last(&self) -> Option<&T> {
        self.get(self.len.saturating_sub(1))
    }

    pub fn rotate_left(&mut self, n: usize) {
        if !self.is_empty() {
            let n = n % self.len;

            if n > 0 {
                // Reverse the first portion of the data
                self.reverse_between(0, n - 1);
                // Reverse the second portion of the data
                self.reverse_between(n, self.len - 1);
                // Reverse the data
                self.reverse();
            }
        }
    }

    pub fn rotate_right(&mut self, n: usize) {
        if !self.is_empty() {
            let n = n % self.len;

            if n > 0 {
                // Reverse the data
                self.reverse();
                // Reverse the first portion of the data
                self.reverse_between(0, n - 1);
                // Reverse the second portion of the data
                self.reverse_between(n, self.len - 1);
            }
        }
    }

    pub fn reverse_between(&mut self, start: usize, end: usize) {
        let mut i = start;
        let mut j = end;
        while i < j {
            self.swap(i, j);
            i += 1;
            j -= 1;
        }
    }

    pub fn reverse(&mut self) {
        self.reverse_between(0, self.len - 1);
    }

    pub fn to_vec(self) -> Vec<T> {
        self.into_iter().collect()
    }

    fn index_is_in_bounds(&self, index: usize) -> bool {
        index < self.len
    }

    fn assert_in_bounds(&self, index: usize) {
        if !self.index_is_in_bounds(index) {
            panic!("`index` {index} is out of bounds");
        }
    }
}

impl<T> Clone for Vector<T> {
    fn clone(&self) -> Self {
        let mut vec = Self::with_capacity(self.capacity);
        unsafe {
            std::ptr::copy(self.ptr, vec.ptr, self.len);
        }
        vec.len = self.len;
        vec
    }
}

impl<T> Vector<T>
where
    T: Clone + PartialOrd,
{
    pub fn sort(&mut self) {
        if !self.is_empty() {
            quick_sort(self, 0, self.len - 1);
        }
    }
}

fn partition<T>(arr: &mut Vector<T>, low: usize, high: usize) -> usize
where
    T: Clone + PartialOrd,
{
    let pivot = arr[high].clone();
    let mut i = low;
    for j in low..high {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1
        }
    }
    arr.swap(i, high);
    i
}

fn quick_sort<T>(arr: &mut Vector<T>, low: usize, high: usize)
where
    T: Clone + PartialOrd,
{
    if low < high {
        let pivot = partition(arr, low, high);
        quick_sort(arr, low, pivot.saturating_sub(1));
        quick_sort(arr, pivot + 1, high);
    }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = Self::new();
        v.extend(iter);
        v
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T> {
    fn from(arr: [T; N]) -> Self {
        Self::from_iter(arr)
    }
}

impl<T> Drop for Vector<T> {
    fn drop(&mut self) {
        unsafe {
            std::ptr::drop_in_place(self.as_mut_slice());
            std::alloc::dealloc(
                self.ptr as *mut u8,
                Layout::array::<T>(self.capacity).unwrap(),
            );
        }
    }
}

impl<T> PartialEq for Vector<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(vec: Vec<T>) -> Self {
        let mut vector = Self::with_capacity(vec.len());
        vector.extend(vec);
        vector
    }
}

impl<T> From<Vector<T>> for Vec<T> {
    fn from(vector: Vector<T>) -> Self {
        vector.into_iter().collect()
    }
}

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for Vector<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.assert_in_bounds(index);
        unsafe { self.get_unchecked(index) }
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.assert_in_bounds(index);
        unsafe { self.get_unchecked_mut(index) }
    }
}

pub struct Iter<'a, T> {
    vec: &'a Vector<T>,
    forward_index: usize,
    back_index: usize,
    finished: bool,
}

impl<'a, T> Iter<'a, T> {
    fn new(vec: &'a Vector<T>) -> Self {
        Self {
            forward_index: 0,
            back_index: vec.len.saturating_sub(1),
            vec,
            finished: false,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = Some(unsafe { self.vec.get_unchecked(self.forward_index) });
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.forward_index += 1;
            }
            el
        }
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            None
        } else {
            let el = Some(unsafe { self.vec.get_unchecked(self.back_index) });
            if self.forward_index == self.back_index {
                self.finished = true;
            } else {
                self.back_index -= 1;
            }
            el
        }
    }
}

impl<'a, T> IntoIterator for &'a Vector<T> {
    type Item = &'a T;

    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct IterMut<'a, T> {
    vec: &'a mut Vector<T>,
    head: usize,
    tail: usize,
}

impl<'a, T> IterMut<'a, T> {
    fn new(vec: &'a mut Vector<T>) -> Self {
        Self {
            head: 0,
            tail: vec.len.saturating_sub(1),
            vec,
        }
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.head > self.tail {
            None
        } else {
            let el = unsafe { self.vec.get_ptr_mut_unchecked(self.head) };
            self.head += 1;
            Some(unsafe { &mut *el })
        }
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.tail < self.head {
            None
        } else {
            let el = unsafe { self.vec.get_ptr_mut_unchecked(self.tail) };
            self.tail -= 1;
            Some(unsafe { &mut *el })
        }
    }
}

impl<'a, T> IntoIterator for &'a mut Vector<T> {
    type Item = &'a mut T;

    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

pub struct IntoIter<T>(Vector<T>);

impl<T> IntoIter<T> {
    fn new(vec: Vector<T>) -> Self {
        Self(vec)
    }
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back()
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instantiate() {
        let _i32_elems: Vector<i32> = Vector::new();

        let _i64_elems: Vector<i64> = Vector::new();

        let _f32_elems: Vector<f32> = Vector::new();

        let _f64_elems: Vector<f64> = Vector::new();

        let _usize_elems: Vector<usize> = Vector::new();

        let _string_elems: Vector<String> = Vector::new();

        let _str_elems: Vector<&str> = Vector::new();

        struct Foo(usize, i32, f64);
        let _foo_elems: Vector<Foo> = Vector::new();
    }

    #[test]
    fn test_with_capacity() {
        let v: Vector<usize> = Vector::with_capacity(42);
        assert_eq!(v.capacity, 42);
    }

    #[test]
    fn test_new_null_ptr() {
        let v: Vector<i32> = Vector::new();
        assert!(v.ptr.is_null());
    }

    #[test]
    fn test_push_front() {
        let mut v = Vector::new();
        v.push_front(1);
        assert_eq!(unsafe { v.ptr.offset(0).read() }, 1);
        v.push_front(2);
        assert_eq!(unsafe { v.ptr.offset(0).read() }, 2);
        assert_eq!(unsafe { v.ptr.offset(1).read() }, 1);
    }

    #[test]
    fn test_push_back() {
        let mut v = Vector::new();
        v.push_back(1);
        assert_eq!(unsafe { v.ptr.offset(0).read() }, 1);
        v.push_back(2);
        assert_eq!(unsafe { v.ptr.offset(0).read() }, 1);
        assert_eq!(unsafe { v.ptr.offset(1).read() }, 2);
    }

    #[test]
    fn test_extend() {
        let mut v = Vector::from_iter(0..=2);
        v.extend(3..=6);
        assert_eq!(v, Vector::from_iter(0..=6));
    }

    #[test]
    fn test_get_ptr_unchecked() {
        let v = Vector::from([0, 1]);
        assert_eq!(unsafe { *v.get_ptr_unchecked(0) }, 0);
        assert_eq!(unsafe { *v.get_ptr_unchecked(1) }, 1);
    }

    #[test]
    fn test_get_ptr_mut_unchecked() {
        let mut v = Vector::from([0, 1]);
        assert_eq!(unsafe { *v.get_ptr_mut_unchecked(0) }, 0);
        assert_eq!(unsafe { *v.get_ptr_mut_unchecked(1) }, 1);
    }

    #[test]
    fn test_get_unchecked() {
        let v = Vector::from([0, 1]);
        assert_eq!(unsafe { v.get_unchecked(0) }, &0);
        assert_eq!(unsafe { v.get_unchecked(1) }, &1);
    }

    #[test]
    fn test_get_unchecked_mut() {
        let mut v = Vector::from([0, 1]);
        assert_eq!(unsafe { v.get_unchecked_mut(0) }, &mut 0);
        assert_eq!(unsafe { v.get_unchecked_mut(1) }, &mut 1);
    }

    #[test]
    fn test_get() {
        let mut v = Vector::new();
        assert_eq!(v.get(0), None);
        v.push_back(0);
        assert_eq!(v.get(0), Some(&0));
        assert_eq!(v.get(1), None);
        v.push_back(1);
        assert_eq!(v.get(0), Some(&0));
        assert_eq!(v.get(1), Some(&1));
        assert_eq!(v.get(2), None);
    }

    #[test]
    fn test_get_mut() {
        let mut v = Vector::new();
        assert_eq!(v.get_mut(0), None);
        v.push_back(0);
        assert_eq!(v.get_mut(0), Some(&mut 0));
        assert_eq!(v.get_mut(1), None);
        v.push_back(1);
        assert_eq!(v.get_mut(0), Some(&mut 0));
        assert_eq!(v.get_mut(1), Some(&mut 1));
        assert_eq!(v.get_mut(2), None);
    }

    #[test]
    fn test_pop_front() {
        let mut v = Vector::new();
        assert_eq!(v.pop_front(), None);
        v.push_back(0);
        v.push_back(1);
        assert_eq!(v.len(), 2);
        assert_eq!(v.pop_front(), Some(0));
        assert_eq!(v.len(), 1);
        assert_eq!(v.pop_front(), Some(1));
        assert_eq!(v.len(), 0);
        assert_eq!(v.pop_front(), None);
    }

    #[test]
    fn test_pop_back() {
        let mut v = Vector::new();
        assert_eq!(v.pop_back(), None);
        v.push_back(0);
        v.push_back(1);
        assert_eq!(v.pop_back(), Some(1));
        assert_eq!(v.pop_back(), Some(0));
        assert_eq!(v.pop_back(), None);
    }

    #[test]
    fn test_reserve() {
        let mut v: Vector<usize> = Vector::new();
        assert_eq!(v.capacity, 0);
        v.reserve(1);
        assert_eq!(v.capacity, 1);
        v.reserve(2);
        assert_eq!(v.capacity, 2);
        v.push_back(0);
        v.reserve(2);
        assert_eq!(v.capacity, 3);
    }

    #[test]
    fn test_grow() {
        let mut v: Vector<usize> = Vector::new();
        assert_eq!(v.capacity, 0);
        v.reserve(1);
        assert_eq!(v.capacity, 1);
        v.reserve(2);
        assert_eq!(v.capacity, 2);
        v.push_back(0);
        v.reserve(2);
        assert_eq!(v.capacity, 3);
    }

    #[test]
    fn test_len_is_empty() {
        let mut v = Vector::new();
        assert!(v.is_empty());
        v.push_back(0);
        assert_eq!(v.len(), 1);
        v.push_back(0);
        assert_eq!(v.len(), 2);
        v.push_back(0);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_capacity() {
        let mut v = Vector::new();
        assert_eq!(v.capacity(), 0);
        v.push_back(0);
        assert_eq!(v.capacity(), 1);
        v.push_back(0);
        assert_eq!(v.capacity(), 3);
        v.push_back(0);
        v.push_back(0);
        v.push_back(0);
        assert_eq!(v.capacity(), 9);
    }

    #[test]
    fn test_available() {
        let mut v = Vector::new();
        assert_eq!(v.capacity(), 0);
        assert_eq!(v.available(), 0);
        v.push_back(0);
        assert_eq!(v.capacity(), 1);
        assert_eq!(v.available(), 0);
        v.push_back(0);
        assert_eq!(v.capacity(), 3);
        assert_eq!(v.available(), 1);
        v.push_back(0);
        assert_eq!(v.capacity(), 3);
        assert_eq!(v.available(), 0);
        v.push_back(0);
        assert_eq!(v.capacity(), 9);
        assert_eq!(v.available(), 5);
        v.push_back(0);
        assert_eq!(v.capacity(), 9);
        assert_eq!(v.available(), 4);
    }

    #[test]
    fn test_full() {
        let mut v = Vector::new();
        assert_eq!(v.capacity(), 0);
        assert!(v.full());
        v.push_back(0);
        assert_eq!(v.capacity(), 1);
        assert!(v.full());
        v.push_back(0);
        assert_eq!(v.capacity(), 3);
        assert!(!v.full());
        v.push_back(0);
        assert_eq!(v.capacity(), 3);
        assert!(v.full());
        v.push_back(0);
        assert_eq!(v.capacity(), 9);
        assert!(!v.full());
    }

    #[test]
    fn test_as_slice() {
        let v = Vector::from_iter(0..=2);
        let s = v.as_slice();
        assert_eq!(s, &vec![0, 1, 2]);
    }

    #[test]
    fn test_deref_coercion() {
        let v = Vector::from_iter(0..=2);
        fn f(_slice: &[usize]) -> bool {
            true
        }
        assert!(f(&v));
    }

    #[test]
    fn test_collect() {
        let iter = 0..=2;
        let v = iter.collect::<Vector<_>>();
        assert_eq!(v.get(0), Some(&0));
        assert_eq!(v.get(1), Some(&1));
        assert_eq!(v.get(2), Some(&2));
    }

    #[test]
    fn test_eq_empty() {
        let x: Vector<usize> = Vector::new();
        let y: Vector<usize> = Vector::new();
        assert_eq!(x, y);
    }

    #[test]
    fn test_eq_same_capacity() {
        let x = Vector::from([0, 1, 2]);
        let y = Vector::from_iter(0..=2);
        assert_eq!(x, y);
    }

    #[test]
    fn test_eq_different_capacity() {
        let x = Vector::from([0, 1, 2]);
        let mut y = Vector::with_capacity(4);
        y.extend(0..=2);
        assert_eq!(x, y);
    }

    #[test]
    fn test_iter() {
        let v = Vector::from_iter(1..=6);
        let mut iter = v.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), Some(&6));
        assert_eq!(iter.next_back(), Some(&5));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_rev() {
        let v = Vector::from_iter(1..=3);
        let mut iter = v.iter().rev();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_mut() {
        let mut v = Vector::from_iter(1..=6);
        let mut iter = v.iter_mut();
        assert_eq!(iter.next(), Some(&mut 1));
        assert_eq!(iter.next_back(), Some(&mut 6));
        assert_eq!(iter.next_back(), Some(&mut 5));
        assert_eq!(iter.next(), Some(&mut 2));
        assert_eq!(iter.next(), Some(&mut 3));
        assert_eq!(iter.next(), Some(&mut 4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter() {
        let v = Vector::from_iter(1..=6);
        let mut iter = v.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(6));
        assert_eq!(iter.next_back(), Some(5));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_rev() {
        let v = Vector::from_iter(1..=3);
        let mut iter = v.into_iter().rev();
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_first() {
        let mut v = Vector::new();
        assert_eq!(v.first(), None);
        v.push_front(0);
        assert_eq!(v.first(), Some(&0));
        v.push_front(1);
        assert_eq!(v.first(), Some(&1));
    }

    #[test]
    fn test_last() {
        let mut v = Vector::new();
        assert_eq!(v.last(), None);
        v.push_back(0);
        assert_eq!(v.last(), Some(&0));
        v.push_back(1);
        assert_eq!(v.last(), Some(&1));
    }

    #[test]
    fn test_swap() {
        let mut v = Vector::from_iter(0..=3);
        v.swap(0, 2);
        assert_eq!(v, Vector::from([2, 1, 0, 3]));
    }

    #[test]
    #[should_panic]
    fn test_swap_index_i_out_of_bounds() {
        let mut v = Vector::from_iter(0..=2);
        v.swap(3, 2);
    }

    #[test]
    #[should_panic]
    fn test_swap_index_j_out_of_bounds() {
        let mut v = Vector::from_iter(0..=2);
        v.swap(0, 3);
    }

    #[test]
    fn test_index() {
        let v = Vector::from_iter(0..=2);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 1);
        assert_eq!(v[2], 2);
    }

    #[test]
    fn test_to_vec() {
        let v = Vector::from_iter(0..=2);
        assert_eq!(v.to_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_from_vec() {
        let v: Vector<_> = vec![0, 1, 2].into();
        assert_eq!(v, Vector::from_iter(0..=2));
    }

    #[test]
    fn test_into_vec() {
        let v: Vec<_> = Vector::from_iter(0..=2).into();
        assert_eq!(v, vec![0, 1, 2]);
    }

    #[test]
    fn test_rotate_left() {
        let mut v = Vector::from_iter(0..=5);
        v.rotate_left(0);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
        v.rotate_left(12);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
        v.rotate_left(2);
        assert_eq!(v, vec![2, 3, 4, 5, 0, 1].into());
        v.rotate_left(3);
        assert_eq!(v, vec![5, 0, 1, 2, 3, 4].into());
        v.rotate_left(1);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
    }

    #[test]
    fn test_rotate_right() {
        let mut v = Vector::from_iter(0..=5);
        v.rotate_right(0);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
        v.rotate_right(12);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
        v.rotate_right(2);
        assert_eq!(v, vec![4, 5, 0, 1, 2, 3].into());
        v.rotate_right(3);
        assert_eq!(v, vec![1, 2, 3, 4, 5, 0].into());
        v.rotate_right(1);
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5].into());
    }

    #[test]
    fn test_sort_empty_vector() {
        let mut v: Vector<usize> = Vector::new();
        v.sort();
        assert_eq!(v, Vector::new());
    }

    #[test]
    fn test_sort_vector_unsorted() {
        let mut v = Vector::from([1, 4, 0, 3, 5, 2]);
        v.sort();
        assert_eq!(v, Vector::from_iter(0..=5));
    }

    #[test]
    fn test_sort_vector_sorted() {
        let mut v = Vector::from_iter(0..=5);
        v.sort();
        assert_eq!(v, Vector::from_iter(0..=5));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::{
        collection::{SizeRange, VecStrategy, VecValueTree},
        prelude::*,
        strategy::{statics::MapFn, NewTree, ValueTree},
        test_runner::TestRunner,
    };

    // Generate arbitrary `Vector`s of arbitrary `T`s.
    //
    // This is very complicated and the same behaviour could be achieved with:
    //
    // ```
    // #[test]
    // fn f(v in any::<Vec<usize>>().prop_map(Vector::from)) { }
    // ```

    #[derive(Debug, Clone)]
    pub struct VectorStrategy<T>(proptest::strategy::statics::Map<VecStrategy<T>, VecToVector>)
    where
        T: Strategy;

    impl<T> Strategy for VectorStrategy<T>
    where
        T: Strategy,
    {
        type Tree = VectorValueTree<T::Tree>;

        type Value = Vector<T::Value>;

        fn new_tree(&self, runner: &mut TestRunner) -> NewTree<Self> {
            self.0.new_tree(runner).map(VectorValueTree)
        }
    }

    pub struct VectorValueTree<T>(proptest::strategy::statics::Map<VecValueTree<T>, VecToVector>)
    where
        T: ValueTree;

    impl<T> ValueTree for VectorValueTree<T>
    where
        T: ValueTree,
    {
        type Value = Vector<T::Value>;

        fn current(&self) -> Self::Value {
            self.0.current().into()
        }

        fn simplify(&mut self) -> bool {
            self.0.simplify()
        }

        fn complicate(&mut self) -> bool {
            self.0.complicate()
        }
    }

    #[derive(Clone)]
    struct VecToVector;

    impl<T> MapFn<Vec<T>> for VecToVector
    where
        T: core::fmt::Debug,
    {
        type Output = Vector<T>;

        fn apply(&self, vec: Vec<T>) -> Vector<T> {
            vec.into()
        }
    }

    impl<T> Arbitrary for Vector<T>
    where
        T: Arbitrary,
    {
        type Parameters = (SizeRange, T::Parameters);

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let (range, a) = args;
            vector(any_with::<T>(a), range)
        }

        type Strategy = VectorStrategy<T::Strategy>;
    }

    pub fn vector<T>(element: T, size: impl Into<SizeRange>) -> VectorStrategy<T>
    where
        T: Strategy,
    {
        VectorStrategy(proptest::strategy::statics::Map::new(
            proptest::collection::vec(element, size),
            VecToVector,
        ))
    }

    fn vec_and_vector<T>() -> impl Strategy<Value = (Vec<T>, Vector<T>)>
    where
        T: Arbitrary + Clone,
    {
        any::<Vec<T>>().prop_map(|v| (v.clone(), v.into()))
    }

    fn vector_and_index<T>() -> impl Strategy<Value = (Vector<T>, usize)>
    where
        T: Arbitrary,
    {
        any::<Vector<T>>().prop_flat_map(|v| {
            let len = v.len();
            // Hack to satisfy `low < high` requirement in `rand::distributions::uniform::Uniform`
            if len == 0 {
                (Just(v), 0..=0)
            } else {
                (Just(v), 0..=(len - 1))
            }
        })
    }

    fn vector_and_swap_indices<T>() -> impl Strategy<Value = (Vector<T>, usize, usize)>
    where
        T: Arbitrary,
    {
        any::<Vector<T>>().prop_flat_map(|v| {
            let len = v.len();
            // Hack to satisfy `low < high` requirement in `rand::distributions::uniform::Uniform`
            if len == 0 {
                (Just(v), 0..=0, 0..=0)
            } else {
                (Just(v), 0..=(len - 1), 0..=(len - 1))
            }
        })
    }

    proptest! {
        #[test]
        fn proptest_push_back(vec in any::<Vec<usize>>()) {
            let mut vector = Vector::new();
            prop_assert_eq!(vector.first(), None);
            prop_assert_eq!(vector.last(), None);
            for el in vec.clone() {
                vector.push_back(el);
                prop_assert_eq!(vector.last(), Some(&el));
            }
            prop_assert_eq!(vector.first(), vec.first());
            prop_assert_eq!(vector.last(), vec.last());
            prop_assert_eq!(vector, vec.into());
        }

        #[test]
        fn proptest_push_front(vec in any::<Vec<usize>>()) {
            let mut vector = Vector::new();
            prop_assert_eq!(vector.first(), None);
            prop_assert_eq!(vector.last(), None);
            for el in vec.clone() {
                vector.push_front(el);
                prop_assert_eq!(vector.first(), Some(&el));
            }
            prop_assert_eq!(vector.first(), vec.last());
            prop_assert_eq!(vector.last(), vec.first());
            let vec_rev = vec.into_iter().rev().collect::<Vec<_>>();
            prop_assert_eq!(vector, vec_rev.into());
        }

        #[test]
        fn proptest_pop_back((mut vec, mut vector) in vec_and_vector::<usize>()) {
            let l = vector.len();
            while let Some(el) = vector.pop_back() {
                prop_assert_eq!(el, vec.pop().unwrap());
            }
            prop_assert_eq!(vector.len(), 0);
            if l > 0 {
                prop_assert!(vector.capacity() > 0);
            } else {
                prop_assert_eq!(vector.capacity(), 0);
            }
            prop_assert_eq!(vector.first(), None);
            prop_assert_eq!(vector.last(), None);
            prop_assert_eq!(vector, Vector::new());
        }

        #[test]
        fn proptest_pop_front((mut vec, mut vector) in vec_and_vector::<usize>()) {
            let l = vector.len();
            vec.reverse();
            while let Some(el) = vector.pop_front() {
                prop_assert_eq!(el, vec.pop().unwrap());
            }
            prop_assert_eq!(vector.len(), 0);
            if l > 0 {
                prop_assert!(vector.capacity() > 0);
            } else {
                prop_assert_eq!(vector.capacity(), 0);
            }
            prop_assert_eq!(vector.first(), None);
            prop_assert_eq!(vector.last(), None);
            prop_assert_eq!(vector, Vector::new());
        }

        #[test]
        fn proptest_index((mut vec, mut vector) in vec_and_vector::<usize>()) {
            for i in 0..vec.len() {
                prop_assert_eq!(unsafe {vector.get_unchecked(i)}, &vec[i]);
                prop_assert_eq!(vector.get(i), Some(&vec[i]));
                prop_assert_eq!(unsafe {vector.get_unchecked_mut(i)}, &mut vec[i]);
                prop_assert_eq!(vector.get_mut(i), Some(&mut vec[i]));
                prop_assert_eq!(vector[i], vec[i]);
                prop_assert_eq!(&mut vector[i], &mut vec[i]);
            }
        }

        #[test]
        fn proptest_len_capacity((vec, vector) in vec_and_vector::<usize>()) {
            prop_assert_eq!(vector.len(), vec.len());
            prop_assert!(vector.capacity() >= vec.len());
        }

        #[test]
        fn proptest_eq(vector in any::<usize>()) {
            prop_assert_eq!(&vector, &vector);
            prop_assert_eq!(&vector, &vector.clone());
        }

        #[test]
        fn proptest_extend(vec in any::<Vec<usize>>()) {
            let mut vector = Vector::new();
            vector.extend(vec.clone());
            prop_assert_eq!(vector.len(), vec.len());
            prop_assert!(vector.capacity() >= vec.len());
            prop_assert_eq!(vector, vec.into());
        }

        #[test]
        fn proptest_capacity(vec in any::<Vec<usize>>(), additional in 0usize..1000) {
            let mut vector = Vector::with_capacity(vec.len());
            prop_assert_eq!(vector.capacity(), vec.len());
            prop_assert_eq!(vector.available(), vec.len());
            prop_assert_eq!(vector.len(), 0);
            if vector.capacity() > 0 {
                prop_assert!(!vector.full());
            } else {
                prop_assert!(vector.full());
            }

            vector.extend(vec.clone());
            prop_assert_eq!(vector.len(), vec.len());
            prop_assert_eq!(vector.capacity(), vec.len());
            prop_assert_eq!(vector.available(), 0);
            prop_assert!(vector.full());

            vector.reserve(additional);
            prop_assert_eq!(vector.capacity(), vec.len() + additional);
            prop_assert_eq!(vector.available(), additional);
            prop_assert_eq!(vector.len(), vec.len());
            if vector.capacity() > 0 && additional > 0 {
                prop_assert!(!vector.full());
            } else {
                prop_assert!(vector.full());
            }
        }

        #[test]
        fn proptest_clone(vector in any::<Vector<usize>>()) {
            prop_assert_eq!(vector.clone(), vector);
        }

        #[test]
        fn proptest_iter((vec, vector) in vec_and_vector::<usize>()) {
            prop_assert!(vector.iter().eq(vec.iter()));
        }

        #[test]
        fn proptest_into_iter((vec, vector) in vec_and_vector::<usize>()) {
            prop_assert!(vector.into_iter().eq(vec.into_iter()));
        }

        #[test]
        fn proptest_iter_mut((mut vec, mut vector) in vec_and_vector::<usize>()) {
            prop_assert!(vector.iter_mut().eq(vec.iter_mut()));
        }

        #[test]
        fn proptest_to_vec((vec, vector) in vec_and_vector::<usize>()) {
            prop_assert_eq!(vector.to_vec(), vec);
        }

        #[test]
        fn proptest_swap((mut v, i, j) in vector_and_swap_indices::<usize>()) {
            if v.len() > 0 {
                let el_i = v[i];
                let el_j = v[j];
                v.swap(i, j);
                prop_assert_eq!(v[j], el_i);
                prop_assert_eq!(v[i], el_j);
                v.swap(j, i);
                prop_assert_eq!(v[i], el_i);
                prop_assert_eq!(v[j], el_j);
            }
        }

        #[test]
        fn proptest_sort((mut vec, mut vector) in vec_and_vector::<usize>()) {
            prop_assert!(vector.iter().eq(vec.iter()));
            vec.sort();
            vector.sort();
            prop_assert!(vector.iter().eq(vec.iter()));
        }

        #[test]
        fn proptest_rotate_right((mut vector, index) in vector_and_index::<usize>()) {
            let original = vector.clone();
            vector.rotate_right(index);
            vector.rotate_right(vector.len() - index);
            prop_assert_eq!(&vector, &original);
        }

        #[test]
        fn proptest_rotate_left((mut vector, index) in vector_and_index::<usize>()) {
            let original = vector.clone();
            vector.rotate_left(index);
            vector.rotate_left(vector.len() - index);
            prop_assert_eq!(&vector, &original);
        }
    }
}
