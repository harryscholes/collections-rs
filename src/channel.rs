use async_trait::async_trait;

use crate::queue::Queue;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

trait Channel<T> {
    fn send(&self, value: T) -> Result<(), Error>;

    fn receive(&self) -> Result<Option<T>, Error>;
}

trait Sender<T> {
    fn send(&self, value: T) -> Result<(), Error>;
}

trait Receiver<T> {
    fn receive(&self) -> Result<Option<T>, Error>;
}

#[async_trait]
trait AsyncChannel<T> {
    async fn send(&self, value: T) -> Result<(), Error>;

    async fn receive(&self) -> Result<Option<T>, Error>;
}

#[async_trait]
trait AsyncSender<T> {
    async fn send(&self, value: T) -> Result<(), Error>;
}

#[async_trait]
trait AsyncReceiver<T> {
    async fn receive(&self) -> Result<Option<T>, Error>;
}

// Synchronization between channel senders and receivers

#[derive(Clone)]
struct Synchronizer {
    sender_count: Arc<AtomicUsize>,
    sender_waker: Arc<Condvar>,
    receiver_count: Arc<AtomicUsize>,
    receiver_waker: Arc<Condvar>,
}

impl Synchronizer {
    fn new() -> Self {
        Self {
            sender_count: Arc::new(AtomicUsize::new(0)),
            sender_waker: Arc::new(Condvar::new()),
            receiver_count: Arc::new(AtomicUsize::new(0)),
            receiver_waker: Arc::new(Condvar::new()),
        }
    }
}

// Buffered channel

pub fn buffered_channel<T>(size: usize) -> (BufferedSender<T>, BufferedReceiver<T>) {
    let chan = BufferedChannel::with_capacity(size);

    (
        BufferedSender::new(chan.clone()),
        BufferedReceiver::new(chan),
    )
}

struct BufferedChannel<T> {
    inner: Arc<Mutex<Queue<T>>>,
    synchronizer: Synchronizer,
}

impl<T> BufferedChannel<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Queue::with_capacity(capacity))),
            synchronizer: Synchronizer::new(),
        }
    }
}

impl<T> Channel<T> for BufferedChannel<T> {
    fn send(&self, value: T) -> Result<(), Error> {
        if self.synchronizer.receiver_count.load(Ordering::SeqCst) == 0 {
            Err(Error::SendError)
        } else {
            let mut q = self.inner.lock()?;

            while q.len() == q.capacity() {
                q = self.synchronizer.sender_waker.wait(q)?;
            }

            q.enqueue(value);

            drop(q);

            self.synchronizer.receiver_waker.notify_all();

            Ok(())
        }
    }

    fn receive(&self) -> Result<Option<T>, Error> {
        let mut q = self.inner.lock()?;

        while q.is_empty() && self.synchronizer.sender_count.load(Ordering::SeqCst) > 0 {
            q = self.synchronizer.receiver_waker.wait(q)?;
        }

        let value = q.dequeue();

        drop(q);

        self.synchronizer.sender_waker.notify_all();

        Ok(value)
    }
}

impl<T> Clone for BufferedChannel<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            synchronizer: self.synchronizer.clone(),
        }
    }
}

pub struct BufferedSender<T> {
    chan: BufferedChannel<T>,
}

impl<T> BufferedSender<T> {
    fn new(chan: BufferedChannel<T>) -> Self {
        chan.synchronizer
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

impl<T> Sender<T> for BufferedSender<T> {
    fn send(&self, value: T) -> Result<(), Error> {
        self.chan.send(value)
    }
}

unsafe impl<T> Send for BufferedSender<T> where T: Send {}

unsafe impl<T> Sync for BufferedSender<T> where T: Send {}

impl<T> Clone for BufferedSender<T> {
    fn clone(&self) -> Self {
        self.chan
            .synchronizer
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self {
            chan: self.chan.clone(),
        }
    }
}

impl<T> Drop for BufferedSender<T> {
    fn drop(&mut self) {
        self.chan
            .synchronizer
            .sender_count
            .fetch_sub(1, Ordering::SeqCst);

        self.chan.synchronizer.receiver_waker.notify_all();
    }
}

pub struct BufferedReceiver<T> {
    chan: BufferedChannel<T>,
}

impl<T> BufferedReceiver<T> {
    fn new(chan: BufferedChannel<T>) -> Self {
        chan.synchronizer
            .receiver_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

impl<T> Receiver<T> for BufferedReceiver<T> {
    fn receive(&self) -> Result<Option<T>, Error> {
        self.chan.receive()
    }
}

unsafe impl<T> Send for BufferedReceiver<T> where T: Send {}

impl<T> Drop for BufferedReceiver<T> {
    fn drop(&mut self) {
        self.chan
            .synchronizer
            .receiver_count
            .fetch_sub(1, Ordering::SeqCst);

        self.chan.synchronizer.sender_waker.notify_all();
    }
}

// Unbuffered channel

pub fn unbuffered_channel<T>() -> (UnbufferedSender<T>, UnbufferedReceiver<T>) {
    let chan = UnbufferedChannel::new();

    (
        UnbufferedSender::new(chan.clone()),
        UnbufferedReceiver::new(chan),
    )
}

struct UnbufferedChannel<T> {
    inner: Arc<Mutex<Option<T>>>,
    synchronizer: Synchronizer,
}

impl<T> UnbufferedChannel<T> {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
            synchronizer: Synchronizer::new(),
        }
    }
}

impl<T> Channel<T> for UnbufferedChannel<T> {
    fn send(&self, value: T) -> Result<(), Error> {
        if self.synchronizer.receiver_count.load(Ordering::SeqCst) == 0 {
            Err(Error::SendError)
        } else {
            let mut opt = self.inner.lock()?;

            while opt.is_some() {
                opt = self.synchronizer.sender_waker.wait(opt)?;
            }

            *opt = Some(value);

            drop(opt);

            self.synchronizer.receiver_waker.notify_all();

            Ok(())
        }
    }

    fn receive(&self) -> Result<Option<T>, Error> {
        let mut opt = self.inner.lock()?;

        while opt.is_none() && self.synchronizer.sender_count.load(Ordering::SeqCst) > 0 {
            opt = self.synchronizer.receiver_waker.wait(opt)?;
        }

        let value = opt.take();

        drop(opt);

        self.synchronizer.sender_waker.notify_all();

        Ok(value)
    }
}

impl<T> Clone for UnbufferedChannel<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            synchronizer: self.synchronizer.clone(),
        }
    }
}

pub struct UnbufferedSender<T> {
    chan: UnbufferedChannel<T>,
}

impl<T> UnbufferedSender<T> {
    fn new(chan: UnbufferedChannel<T>) -> Self {
        chan.synchronizer
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

impl<T> Sender<T> for UnbufferedSender<T> {
    fn send(&self, value: T) -> Result<(), Error> {
        self.chan.send(value)
    }
}

unsafe impl<T> Send for UnbufferedSender<T> where T: Send {}

unsafe impl<T> Sync for UnbufferedSender<T> where T: Send {}

impl<T> Clone for UnbufferedSender<T> {
    fn clone(&self) -> Self {
        self.chan
            .synchronizer
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self {
            chan: self.chan.clone(),
        }
    }
}

impl<T> Drop for UnbufferedSender<T> {
    fn drop(&mut self) {
        self.chan
            .synchronizer
            .sender_count
            .fetch_sub(1, Ordering::SeqCst);

        self.chan.synchronizer.receiver_waker.notify_all();
    }
}

pub struct UnbufferedReceiver<T> {
    chan: UnbufferedChannel<T>,
}

impl<T> UnbufferedReceiver<T> {
    fn new(chan: UnbufferedChannel<T>) -> Self {
        chan.synchronizer
            .receiver_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

impl<T> Receiver<T> for UnbufferedReceiver<T> {
    fn receive(&self) -> Result<Option<T>, Error> {
        self.chan.receive()
    }
}

unsafe impl<T> Send for UnbufferedReceiver<T> where T: Send {}

impl<T> Drop for UnbufferedReceiver<T> {
    fn drop(&mut self) {
        self.chan
            .synchronizer
            .receiver_count
            .fetch_sub(1, Ordering::SeqCst);

        self.chan.synchronizer.sender_waker.notify_all();
    }
}

// Oneshot channel

pub fn oneshot_channel<T>() -> (OneshotSender<T>, UnbufferedReceiver<T>) {
    let chan = UnbufferedChannel::new();

    (
        OneshotSender::new(chan.clone()),
        UnbufferedReceiver::new(chan),
    )
}

pub struct OneshotSender<T> {
    sender: Arc<Mutex<Option<UnbufferedSender<T>>>>,
}

impl<T> OneshotSender<T> {
    fn new(chan: UnbufferedChannel<T>) -> Self {
        let sender = UnbufferedSender::new(chan.clone());

        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }
}

impl<T> Sender<T> for OneshotSender<T> {
    fn send(&self, value: T) -> Result<(), Error> {
        if let Some(sender) = self.sender.lock().unwrap().take() {
            sender.send(value)
        } else {
            Err(Error::SendError)
        }
    }
}

// Async channel

pub fn async_buffered_channel<T>(
    size: usize,
) -> (AsyncBufferedSender<T>, AsyncBufferedReceiver<T>) {
    let chan = AsyncBufferedChannel::with_capacity(size);

    (
        AsyncBufferedSender::new(chan.clone()),
        AsyncBufferedReceiver::new(chan),
    )
}

#[derive(Clone)]
struct SharedState {
    sender_count: Arc<AtomicUsize>,
    receiver_count: Arc<AtomicUsize>,
    wakers: Arc<Mutex<Queue<Waker>>>,
}

impl SharedState {
    fn new() -> Self {
        Self {
            sender_count: Arc::new(AtomicUsize::new(0)),
            receiver_count: Arc::new(AtomicUsize::new(0)),
            wakers: Arc::new(Mutex::new(Queue::new())),
        }
    }
}

struct AsyncBufferedChannel<T> {
    inner: Arc<Mutex<Queue<T>>>,
    shared_state: SharedState,
}

impl<T> AsyncBufferedChannel<T> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Queue::with_capacity(capacity))),
            shared_state: SharedState::new(),
        }
    }
}

#[async_trait]
impl<T> AsyncChannel<T> for AsyncBufferedChannel<T>
where
    T: Send,
{
    async fn send(&self, value: T) -> Result<(), Error> {
        AsyncBufferedSenderTask {
            chan: self.clone(),
            value: Arc::new(Mutex::new(Some(value))),
        }
        .await
    }

    async fn receive(&self) -> Result<Option<T>, Error> {
        AsyncBufferedReceiverTask { chan: self.clone() }.await
    }
}

unsafe impl<T> Send for AsyncBufferedChannel<T> where T: Send {}

unsafe impl<T> Sync for AsyncBufferedChannel<T> where T: Send {}

impl<T> Clone for AsyncBufferedChannel<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            shared_state: self.shared_state.clone(),
        }
    }
}

pub struct AsyncBufferedSenderTask<T> {
    chan: AsyncBufferedChannel<T>,
    value: Arc<Mutex<Option<T>>>,
}

impl<T> Future for AsyncBufferedSenderTask<T> {
    type Output = Result<(), Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut q = self.chan.inner.lock()?;

        if self.chan.shared_state.receiver_count.load(Ordering::SeqCst) == 0 {
            Poll::Ready(Err(Error::SendError))
        } else {
            if q.len() == q.capacity() {
                self.chan
                    .shared_state
                    .wakers
                    .lock()?
                    .enqueue(cx.waker().clone());

                Poll::Pending
            } else {
                let mut opt = self.value.lock()?;
                let value = opt.take().unwrap();
                q.enqueue(value);

                if let Some(waker) = self.chan.shared_state.wakers.lock()?.dequeue() {
                    waker.wake()
                };

                Poll::Ready(Ok(()))
            }
        }
    }
}

pub struct AsyncBufferedReceiverTask<T> {
    chan: AsyncBufferedChannel<T>,
}

impl<T> Future for AsyncBufferedReceiverTask<T> {
    type Output = Result<Option<T>, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut q = self.chan.inner.lock()?;

        match (
            q.is_empty(),
            self.chan.shared_state.sender_count.load(Ordering::SeqCst),
        ) {
            (true, 0) => Poll::Ready(Ok(None)),
            (true, _) => {
                self.chan
                    .shared_state
                    .wakers
                    .lock()?
                    .enqueue(cx.waker().clone());

                Poll::Pending
            }
            _ => {
                let value = q.dequeue();

                if let Some(waker) = self.chan.shared_state.wakers.lock()?.dequeue() {
                    waker.wake()
                };

                Poll::Ready(Ok(value))
            }
        }
    }
}

pub struct AsyncBufferedSender<T> {
    chan: AsyncBufferedChannel<T>,
}

impl<T> AsyncBufferedSender<T> {
    fn new(chan: AsyncBufferedChannel<T>) -> Self {
        chan.shared_state
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

#[async_trait]
impl<T> AsyncSender<T> for AsyncBufferedSender<T>
where
    T: Send,
{
    async fn send(&self, value: T) -> Result<(), Error> {
        self.chan.send(value).await
    }
}

unsafe impl<T> Send for AsyncBufferedSender<T> where T: Send {}

unsafe impl<T> Sync for AsyncBufferedSender<T> where T: Send {}

impl<T> Clone for AsyncBufferedSender<T> {
    fn clone(&self) -> Self {
        let chan = self.chan.clone();
        chan.shared_state
            .sender_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

impl<T> Drop for AsyncBufferedSender<T> {
    fn drop(&mut self) {
        self.chan
            .shared_state
            .sender_count
            .fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct AsyncBufferedReceiver<T> {
    chan: AsyncBufferedChannel<T>,
}

impl<T> AsyncBufferedReceiver<T> {
    fn new(chan: AsyncBufferedChannel<T>) -> Self {
        chan.shared_state
            .receiver_count
            .fetch_add(1, Ordering::SeqCst);

        Self { chan }
    }
}

#[async_trait]
impl<T> AsyncReceiver<T> for AsyncBufferedReceiver<T>
where
    T: Send,
{
    async fn receive(&self) -> Result<Option<T>, Error> {
        self.chan.receive().await
    }
}

unsafe impl<T> Send for AsyncBufferedReceiver<T> where T: Send {}

impl<T> Drop for AsyncBufferedReceiver<T> {
    fn drop(&mut self) {
        self.chan
            .shared_state
            .receiver_count
            .fetch_sub(1, Ordering::SeqCst);
    }
}

// Error

#[derive(Debug, PartialEq)]
pub enum Error {
    SendError,
    ReceiveError,
    PoisonError,
}

impl std::error::Error for Error {}

impl<T> From<PoisonError<T>> for Error {
    fn from(_: PoisonError<T>) -> Self {
        Error::PoisonError
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, panic, thread};

    #[derive(Clone)]
    struct Synchronizer {
        inner: Arc<(Mutex<bool>, Condvar)>,
    }

    impl Synchronizer {
        fn new() -> Self {
            Self {
                inner: Arc::new((Mutex::new(false), Condvar::new())),
            }
        }

        fn wait(&self) {
            let (done, cvar) = &*self.inner;
            let mut done = done.lock().unwrap();
            while !*done {
                done = cvar.wait(done).unwrap();
            }
        }

        fn notify(&self) {
            let (done, cvar) = &*self.inner;
            let mut done = done.lock().unwrap();
            *done = true;
            cvar.notify_all();
        }
    }

    fn synchronizer() -> (Synchronizer, Synchronizer) {
        let s = Synchronizer::new();
        (s.clone(), s)
    }

    #[test]
    fn test_buffered_channel_single_sender() {
        let (s, r) = buffered_channel(2);

        let n = 1_000;

        thread::spawn(move || {
            for i in 0..n {
                s.send(i).unwrap();
            }
        });

        for i in 0..n {
            assert_eq!(r.receive().unwrap(), Some(i));
            dbg!("received single", i);
        }

        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_buffered_channel_buffering() {
        let (s, r) = buffered_channel(2);

        let (notifier, waiter) = synchronizer();

        thread::spawn(move || {
            s.send(1).unwrap();
            s.send(2).unwrap();

            // Notify that the channel is full
            notifier.notify();

            s.send(3).unwrap();
            s.send(4).unwrap();
        });

        // Wait for the channel to be full
        waiter.wait();

        // Assert that the channel is full
        {
            let q = r.chan.inner.lock().unwrap();
            assert_eq!(q.capacity(), 2);
            assert_eq!(q.len(), 2);
        }

        // Start receiving from the channel to make space for more values to be sent
        assert_eq!(r.receive().unwrap(), Some(1));
        assert_eq!(r.receive().unwrap(), Some(2));
        assert_eq!(r.receive().unwrap(), Some(3));
        assert_eq!(r.receive().unwrap(), Some(4));
        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_buffered_channel_multiple_senders() {
        let (s, r) = buffered_channel(2);

        let s_2 = s.clone();
        let s_3 = s.clone();

        assert_eq!(s.chan.synchronizer.sender_count.load(Ordering::SeqCst), 3);

        let n = 1_000;

        thread::spawn(move || {
            for i in 0..n {
                s.send(i).unwrap();
            }
        });

        thread::spawn(move || {
            for i in n..2 * n {
                s_2.send(i).unwrap();
            }
        });

        thread::spawn(move || {
            for i in 2 * n..3 * n {
                s_3.send(i).unwrap();
            }
        });

        let mut received = HashSet::new();

        while let Some(value) = r.receive().unwrap() {
            received.insert(value);
        }

        assert_eq!(received, HashSet::from_iter(0..3 * n));
    }

    #[test]
    fn test_buffered_channel_receiver_dropped() {
        let (s, r) = buffered_channel(2);
        drop(r);
        assert_eq!(s.send(1), Err(Error::SendError));
    }

    #[test]
    fn test_buffered_channel_sender_dropped() {
        let (s, r): (BufferedSender<usize>, BufferedReceiver<usize>) = buffered_channel(2);
        drop(s);
        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_unbuffered_channel() {
        let (s, r) = unbuffered_channel();

        let n = 1_000;

        thread::spawn(move || {
            for i in 0..n {
                s.send(i).unwrap();
            }
        });

        for i in 0..n {
            assert_eq!(r.receive().unwrap(), Some(i));
        }

        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_unbuffered_channel_multiple_senders() {
        let (s, r) = unbuffered_channel();

        let s_2 = s.clone();
        let s_3 = s.clone();

        assert_eq!(s.chan.synchronizer.sender_count.load(Ordering::SeqCst), 3);

        let n = 1_000;

        thread::spawn(move || {
            for i in 0..n {
                s.send(i).unwrap();
            }
        });

        thread::spawn(move || {
            for i in n..2 * n {
                s_2.send(i).unwrap();
            }
        });

        thread::spawn(move || {
            for i in 2 * n..3 * n {
                s_3.send(i).unwrap();
            }
        });

        let mut received = HashSet::new();

        while let Some(value) = r.receive().unwrap() {
            received.insert(value);
        }

        assert_eq!(received, HashSet::from_iter(0..3 * n));
    }

    #[test]
    fn test_unbuffered_channel_receiver_dropped() {
        let (s, r) = unbuffered_channel();
        drop(r);
        assert_eq!(s.send(1), Err(Error::SendError));
    }

    #[test]
    fn test_unbuffered_channel_sender_dropped() {
        let (s, r): (UnbufferedSender<usize>, UnbufferedReceiver<usize>) = unbuffered_channel();
        drop(s);
        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_oneshot_channel() {
        let (s, r) = oneshot_channel();

        thread::spawn(move || {
            s.send(1).unwrap();

            assert_eq!(s.send(2), Err(Error::SendError));
        });

        assert_eq!(r.receive().unwrap(), Some(1));
        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_oneshot_channel_receiver_dropped() {
        let (s, r) = oneshot_channel();
        drop(r);
        assert_eq!(s.send(1), Err(Error::SendError));
    }

    #[test]
    fn test_oneshot_channel_sender_dropped() {
        let (s, r): (OneshotSender<usize>, UnbufferedReceiver<usize>) = oneshot_channel();
        drop(s);
        assert_eq!(r.receive().unwrap(), None);
    }

    #[test]
    fn test_channel_poison_error() {
        let (s, r): (UnbufferedSender<usize>, UnbufferedReceiver<usize>) = unbuffered_channel();

        _ = thread::spawn(move || {
            // Poison the mutex by panicing whilst holding the lock
            let _guard = s.chan.inner.lock().unwrap();
            panic!("Poison the mutex");
        })
        // Join the thread immediately to ensure the panic occurs before the receiver is called
        .join();

        assert_eq!(r.receive(), Err(Error::PoisonError));
    }

    #[tokio::test]
    async fn test_async_buffered_channel_single_sender() {
        let (s, r) = async_buffered_channel(2);

        let n = 1_000;

        tokio::task::spawn(async move {
            for i in 0..n {
                s.send(i).await.unwrap();
            }
        });

        for i in 0..n {
            assert_eq!(r.receive().await.unwrap(), Some(i));
        }

        assert_eq!(r.receive().await.unwrap(), None);
    }

    #[tokio::test]
    async fn test_async_buffered_channel_buffering() {
        let (s, r) = async_buffered_channel(2);

        // Use a oneshot channel for synchronization between threads
        let (notifier, waiter) = oneshot_channel();

        let handle_1 = tokio::task::spawn(async move {
            s.send(1).await.unwrap();
            s.send(2).await.unwrap();

            // Notify that the channel is full
            notifier.send(()).unwrap();

            s.send(3).await.unwrap();
            s.send(4).await.unwrap();
        });

        let handle_2 = tokio::task::spawn(async move {
            // Wait for the channel to be full
            waiter.receive().unwrap();

            // Assert that the channel is full
            {
                let q = r.chan.inner.lock().unwrap();
                assert_eq!(q.capacity(), 2);
                assert_eq!(q.len(), 2);
            }

            // Start receiving from the channel to make space for more values to be sent
            assert_eq!(r.receive().await.unwrap(), Some(1));
            assert_eq!(r.receive().await.unwrap(), Some(2));
            assert_eq!(r.receive().await.unwrap(), Some(3));
            assert_eq!(r.receive().await.unwrap(), Some(4));
            assert_eq!(r.receive().await.unwrap(), None);
        });

        tokio::try_join!(handle_1, handle_2).unwrap();
    }

    #[tokio::test]
    async fn test_async_buffered_channel_multiple_senders() {
        let (s, r) = async_buffered_channel(2);

        let s_2 = s.clone();
        let s_3 = s.clone();

        assert_eq!(s.chan.shared_state.sender_count.load(Ordering::SeqCst), 3);

        let n = 1_000;

        let handle_1 = tokio::task::spawn(async move {
            for i in 0..n {
                s.send(i).await.unwrap();
            }
        });

        let handle_2 = tokio::task::spawn(async move {
            for i in n..2 * n {
                s_2.send(i).await.unwrap();
            }
        });

        let handle_3 = tokio::task::spawn(async move {
            for i in 2 * n..3 * n {
                s_3.send(i).await.unwrap();
            }
        });

        let handle_4 = tokio::task::spawn(async move {
            let mut received = HashSet::new();

            while let Some(value) = r.receive().await.unwrap() {
                received.insert(value);
            }

            assert_eq!(received, HashSet::from_iter(0..3 * n));
        });

        tokio::try_join!(handle_1, handle_2, handle_3, handle_4).unwrap();
    }

    #[tokio::test]
    async fn test_async_buffered_channel_receiver_dropped() {
        let (s, r) = async_buffered_channel(2);
        drop(r);
        assert_eq!(s.send(1).await, Err(Error::SendError));
    }

    #[tokio::test]
    async fn test_async_buffered_channel_sender_dropped() {
        let (s, r): (AsyncBufferedSender<usize>, AsyncBufferedReceiver<usize>) =
            async_buffered_channel(2);
        drop(s);
        assert_eq!(r.receive().await.unwrap(), None);
    }
}
