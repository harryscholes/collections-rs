# collections-rs

Rust data structures, methods and traits:

- `BinaryTree<T>`
- `CircularBuffer<T>`
- `Dequeue<T>`
- `Graph<T>`
- `HashMap<K, V>`
- `HashSet<T>`
- `LinkedList<T>`
- `LRUCache<T>`
- `MerkleTree`
- `MaxHeap<T>`
- `MinHeap<T>`
- `PriorityQueue<T>`
- `Queue<T>`
- `SparseVector<T>`
- `Stack<T>`
- `Vector<T>`

All data structures are implemented from scratch and no data structures from the standard library are used, not even `Vec`.
The dependency graph between the various data structures is:

```
┌─────────────┐
│   Vector    │
└┬─┬─┬───────┬┘
 │ │ │      ┌▽─────────────┐
 │ │ │      │CircularBuffer│
 │ │ │      └┬──┬──────────┘
 │ │ │┌─────┐│  │
 │ │ ││Queue││  │
 │ │ │└△────┘│  │
 │ │ │┌┴─────▽─┐│
 │ │ ││Dequeue ││
 │ │ │└────────┘│
 │ │ │┌─────────▽─┐
 │ │ ││   Stack   │
 │ │ │└───────────┘
 │ │┌▽────────────────────────┐
 │ ││         HashMap         │
 │ │└┬────────┬─────┬──┬──┬──△┘
 │ │┌▽───────┐│     │  │  │  │
 │ ││LRUCache││     │  │  │  │
 │ │└△───────┘│     │  │  │  │
 │ │ │┌───────▽────┐│  │  │  │
 │ │ ││SparseVector││  │  │  │
 │ │ │└────────────┘│  │  │  │
 │ │ │┌─────────────▽─┐│  │  │
 │ │ ││     Graph     ││  │  │
 │ │ │└───────────────┘│  │  │
 │ │ │┌────────────────▽─┐│  │
 │ │ ││     HashSet      ││  │
 │ │ │└──────────────────┘│  │
 │ │ │┌───────────────────▽─┐│
 │ │ ││     BinaryTree      ││
 │ │ │└─────────────────────┘│
 │ │┌┴───────────────────────┴─┐
 │ ││        LinkedList        │
 │ │└──────────────────────────┘
 │┌▽─────────┐
 ││MerkleTree│
 │└──────────┘
┌▽──────────────┐
│    MaxHeap    │
└┬─────────────┬┘
┌▽────────────┐│
│PriorityQueue││
└─────────────┘│
┌──────────────▽─┐
│    MinHeap     │
└────────────────┘

```
