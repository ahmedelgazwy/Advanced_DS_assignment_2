# Phase 1 — Theory Notes: Binomial, Fibonacci and Pairing Heaps

> AI-assisted: generated with Claude Code; reviewed and tested by Ahmed Elgazwy (see README, "AI usage").
> These notes are the source for the "Complexity Analysis" section of the report.
> Every bound and bibliographic detail below was checked against the cited source
> (publisher or Crossref metadata, the paper's abstract, or the paper itself).

All three structures implement the **mergeable min-heap** abstract data type.
Each item is a pair (key, value), and the caller keeps a **handle** to it:

| Operation | Meaning |
|---|---|
| `insert(k, v)` | add an item, return its handle |
| `find_min()` | handle of an item with the smallest key |
| `extract_min()` | remove and return an item with the smallest key |
| `decrease_key(h, k')` | lower the key of item $h$ to $k' \le h.\text{key}$ |
| `delete(h)` | remove item $h$ |
| `merge(H_2)` | move every item of heap $H_2$ into this heap (destructive union) |

`decrease_key` and `merge` are the two operations that separate these
structures from an ordinary binary heap. Dijkstra's and Prim's algorithms use
`decrease_key` heavily, which is why Fibonacci heaps were invented
[Fredman & Tarjan 1987].

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of items in the heap |
| $B_k$ | binomial tree of order $k$ |
| $\deg(x)$ | number of children of node $x$ |
| $D(n)$ | maximum degree of any node in an $n$-item Fibonacci heap |
| $t(H)$ | number of trees in the root list of heap $H$ |
| $m(H)$ | number of marked nodes in a Fibonacci heap $H$ |
| $\Phi(H)$ | potential function used in an amortized analysis |
| $\varphi$ | golden ratio $(1+\sqrt5)/2 \approx 1.618$ |
| $F_k$ | $k$-th Fibonacci number ($F_0=0$, $F_1=1$, $F_{k}=F_{k-1}+F_{k-2}$) |
| $c$ | number of cascading cuts made by one Fibonacci `decrease_key` |
| $r$ | number of children of the root removed by a pairing-heap `extract_min` |
| $V$, $E$ | number of vertices and edges of a graph (Dijkstra) |

"Amortized" means the bound holds for the **total** cost of any sequence of
operations divided by its length, even though a single operation can be more
expensive. All three analyses use the potential method [CLRS 3e §17.3].

---

## 1. Binomial heap [Vuillemin 1978; Brown 1978; CLRS 2e ch. 19; CLRS 3e Problem 19-2]

### 1.1 Binomial trees
$B_0$ is a single node. $B_k$ consists of two $B_{k-1}$ trees, **linked**: one
root becomes the leftmost child of the other. By induction on $k$
[CLRS 2e Lemma 19.1], $B_k$

1. has exactly $2^k$ nodes,
2. has height $k$,
3. has exactly $\binom{k}{i}$ nodes at depth $i$,
4. has a root of degree $k$, whose children are roots of $B_{k-1}, B_{k-2}, \dots, B_0$.

Therefore the maximum degree of any node in an $n$-node binomial tree is $\log_2 n$.

### 1.2 The heap
A binomial heap is a list of heap-ordered binomial trees (each key is $\ge$ its
parent's key), **at most one tree of each order**, in increasing order. Since
$B_k$ has $2^k$ nodes, the orders present are exactly the positions of the 1-bits
in the binary representation of $n$. For example, $n = 13 = 1101_2$ gives $B_3, B_2, B_0$.
Hence

$$
t(H) \le \lfloor \log_2 n \rfloor + 1 .
$$

### 1.3 Operations
- **link** two $B_{k-1}$ into one $B_k$: $O(1)$ (one comparison and pointer updates).
- **union**: merge the two root lists by order, then link equal-order trees from
  left to right. This is exactly **binary addition** of $n_1$ and $n_2$, where a link
  is a carry. Cost $O(\log n_1 + \log n_2) = O(\log n)$.
- **insert** = union with a one-node heap $B_0$, which is a binary-counter **increment**.
- **find_min**: we keep a pointer to the minimum root, so the cost is $O(1)$.
  CLRS scans the roots, at $O(\log n)$ cost. The pointer is updated in $O(1)$ by
  insert and union, and recomputed in $O(\log n)$ after extract-min or delete.
- **extract_min**: remove the minimum root of some $B_k$. Its $k \le \log_2 n$ children,
  read in reverse, form a binomial heap; union it with the rest: $O(\log n)$.
- **decrease_key**: lower the key, then swap the item with its parent's item while
  it is smaller (bubble up). The path is at most the tree height, so the cost is $O(\log n)$.
- **delete**: bubble the item unconditionally to its root (as if its key were $-\infty$),
  then remove that root as in extract-min: $O(\log n)$.

### 1.4 Amortized insert: $O(1)$
Let $\Phi(H) = t(H)$, the number of trees. An insert that triggers $c$ carries
(links) costs $1 + c$ actual work and changes the number of trees by $1 - c$:

$$
\hat{c}_{\text{insert}} = (1 + c) + (1 - c) = 2 = O(1).
$$

Extract-min raises $\Phi$ by at most $k - 1 \le \log_2 n$ (the root's children become
trees), so it stays $O(\log n)$ amortized. A **single** insert can still cost
$\Theta(\log n)$, for example when $n = 2^j - 1$ and every tree order is occupied.

### 1.5 Space
$\Theta(n)$: one node per item. Each node holds a reference to its item, plus
`parent`, `child`, `sibling` and `degree`. The item holds `key`, `value` and a
back-pointer to its node (see §6, "Handle design").

---

## 2. Fibonacci heap [Fredman & Tarjan 1987; CLRS 3e ch. 19]

### 2.1 Structure
A collection of heap-ordered trees. The roots are kept in a **circular doubly
linked root list** with a pointer to the minimum root. Each node stores `key`,
`value`, `parent`, one `child`, `left`/`right` siblings (circular), `degree`, and a
boolean `mark` (the node has lost a child since it last became a child itself).

The structure is **lazy**: insert and union only splice lists. All tidying is
deferred to `extract_min`, which *consolidates* the root list until no two roots
have the same degree.

### 2.2 Potential function
$$
\Phi(H) = t(H) + 2\,m(H).
$$

### 2.3 Amortized costs
| Operation | Actual cost | $\Delta\Phi$ | Amortized |
|---|---|---|---|
| insert | $O(1)$ | $+1$ (one new root) | $O(1)$ |
| find_min | $O(1)$ | 0 | $O(1)$ |
| merge (union) | $O(1)$ (concatenate two circular lists) | 0 | $O(1)$ |
| extract_min | $O(D(n) + t(H))$ | $\le (D(n)+1) - t(H)$ | $O(D(n)) = O(\log n)$ |
| decrease_key | $O(c)$ | $\le (t+c) + 2(m-c+2) - (t+2m) = 4 - c$ | $O(1)$ |
| delete | decrease_key + extract_min | | $O(D(n)) = O(\log n)$ |

- **extract_min.** After the min root is removed and its children are added to
  the root list, `consolidate` links roots of equal degree. When it finishes, at
  most $D(n)+1$ roots remain. The $t(H)$ term of the actual cost is paid by the
  drop in potential [CLRS 3e §19.2].
- **decrease_key.** If the new key violates heap order, the node is **cut** and
  moved to the root list. Then its parent is cut if it was already marked, and so
  on up the tree (**cascading cut**), $c$ cuts in total. Each cut removes a mark,
  and the two potential units per mark pay for it [CLRS 3e §19.3].

### 2.4 Why $D(n) = O(\log n)$: the Fibonacci bound
- **Lemma 19.1** [CLRS 3e]: if $x$ has degree $k$ and children $y_1,\dots,y_k$ in the
  order they were linked, then $\deg(y_i) \ge i - 2$, because a node loses at most one
  child before it is cut.
- **Lemma 19.4**: every node $x$ of degree $k$ roots a subtree of size
  $\text{size}(x) \ge F_{k+2} \ge \varphi^k$.
- **Corollary 19.5**: $n \ge \varphi^{D(n)}$, therefore

$$
\boxed{\,D(n) \le \lfloor \log_\varphi n \rfloor \approx 1.44 \log_2 n\,}
$$

The name "Fibonacci heap" comes from this bound. Consolidation therefore needs a
degree table of only $\lfloor\log_\varphi n\rfloor + 1$ entries. We allocate one more for
safety; that is 35 entries at $n = 10^7$.

### 2.5 Worst case of a single operation
Amortization hides spikes. Right after $n$ inserts the root list has $t(H) = n$
singleton trees, so the **first** `extract_min` costs $\Theta(n)$ actual work. The
$n$ units of potential built up by the inserts pay for it. Our benchmark measures
this spike directly (experiment E2).

### 2.6 Space
$\Theta(n)$: one node per item, with 8 fields per node. The degree table is
$O(\log n)$ and temporary.

---

## 3. Pairing heap [Fredman, Sedgewick, Sleator & Tarjan 1986]

### 3.1 Structure
A **single** heap-ordered multiway tree. Each node stores `key`, `value`, its
leftmost `child`, its `next` sibling and `prev`. `prev` points to the left sibling,
or to the parent for a leftmost child. The `prev` pointer makes cutting a subtree $O(1)$.

The basic step is **link(a, b)**: compare the two roots, and make the one with the
larger key the leftmost child of the other. Cost $O(1)$.

### 3.2 Operations
- **insert** = link the new one-node tree with the root: $O(1)$ actual.
- **merge** = link the two roots: $O(1)$ actual.
- **find_min** = the root: $O(1)$.
- **decrease_key**: if the node is the root, just lower its key. Otherwise cut its
  subtree out (via `prev`), lower the key, and link the subtree with the root:
  $O(1)$ actual.
  The standard algorithm always cuts. Finding the parent to check heap order would
  require walking the sibling list.
- **extract_min** (**two-pass pairing**): remove the root, leaving its $r$ children
  $c_1, \dots, c_r$.
  1. *Pass 1*, left to right: link $(c_1,c_2), (c_3,c_4), \dots$.
  2. *Pass 2*, right to left: link each resulting tree into the accumulated tree.

  Actual cost $\Theta(r)$, and $r$ can be $n - 1$, e.g. right after $n$ inserts.
- **delete**: if the node is the root, extract_min. Otherwise cut the node's subtree,
  run two-pass pairing on the node's children, and link the result with the root.

### 3.3 Amortized bounds (the analysis is still open)
| Result | Bound | Source |
|---|---|---|
| All operations | $O(\log n)$ amortized | FSST 1986 (potential $\Phi = \sum_x \log_2 s(x)$, where $s(x)$ is the subtree size in the binary child–sibling view) |
| insert, merge | $O(1)$ amortized, while extract-min, delete and decrease-key stay $O(\log n)$ | Iacono 2000 |
| decrease_key **lower bound** | $\Omega(\log\log n)$ amortized, for pairing heaps and related structures whose insert and extract-min are $O(\log n)$ | Fredman 1999; generalized by Iacono & Özkan 2014 |
| decrease_key **upper bound** | $O\!\left(2^{2\sqrt{\log\log n}}\right)$ amortized, which is $o(\log n)$ (insert and merge also have this bound; extract-min stays $O(\log n)$) | Pettie 2005 |

So, unlike in a Fibonacci heap, `decrease_key` is provably **not** $O(1)$
amortized, and its exact cost is between $\Omega(\log\log n)$ and
$O(2^{2\sqrt{\log\log n}})$. These bounds come from **different analyses**, i.e.
different potential functions, so they are listed separately rather than combined.

**At our sizes, $2^{2\sqrt{\log_2\log_2 n}}$ is not smaller than $\log_2 n$:**

| $n$ | $\log_2 n$ | $2^{2\sqrt{\log_2\log_2 n}}$ |
|---|---|---|
| $10^3$ | 9.97 | 12.49 |
| $10^6$ | 19.93 | 17.82 |
| $10^7$ | 23.25 | 19.18 |

The two are equal at $\log_2\log_2 n = 4$, i.e. $n = 2^{16} = 65{,}536$. Even
beyond that, the two bounds cannot be told apart in a benchmark. The pairing
heap's appeal is practical, not asymptotic.

### 3.4 Practice
- Stasko & Vitter [1987] gave the first large experimental study and found pairing
  heaps efficient in practice.
- Larkin, Sen & Tarjan [2014] state that *"Pairing heaps dominate Fibonacci heaps
  across the board"*. They found that wall-clock time correlates most strongly with
  L1 cache misses, and that implicit d-ary heaps often win when no `decrease_key` is needed.
- The reasons: a pairing-heap node has 5 fields against 8, `decrease_key` does no
  mark bookkeeping, and there is no degree table.

### 3.5 Space
$\Theta(n)$: one node per item, with 5 fields per node.

---

## 4. Summary table (goes into the report)

| Operation | Binomial | Fibonacci (amortized) | Pairing (amortized) |
|---|---|---|---|
| insert | $O(\log n)$ worst, $O(1)$ amortized | $O(1)$ | $O(1)$ [Iacono 2000] |
| find_min | $O(1)$ (min pointer) | $O(1)$ | $O(1)$ |
| extract_min | $O(\log n)$ | $O(\log n)$; single op $O(n)$ worst | $O(\log n)$; single op $O(n)$ worst |
| decrease_key | $O(\log n)$ | $O(1)$ | $\Omega(\log\log n)$, $O(2^{2\sqrt{\log\log n}})$ |
| delete | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ |
| merge | $O(\log n)$ | $O(1)$ | $O(1)$ |
| space | $\Theta(n)$ | $\Theta(n)$ | $\Theta(n)$ |
| fields per node | 5 (+3 per item) | 8 | 5 |

The binomial-heap bounds are worst-case per operation, except insert, which is
also given amortized. Its operations are "eager": the structure is always tidy, so
it has no expensive spikes beyond $O(\log n)$.

---

## 5. Application: Dijkstra's single-source shortest paths [Dijkstra 1959]

Our version inserts a vertex when it is first reached, calls `decrease_key` when
a shorter path to a vertex still in the heap is found, and finalizes a vertex
when it is extracted. On a graph with $V$ vertices and $E$ edges:

$$
\#\text{insert} \le V,\qquad \#\text{extract\_min} \le V,\qquad \#\text{decrease\_key} \le E .
$$

| Heap | Running time |
|---|---|
| Binomial | $O(V\log V + E\log V) = O((V+E)\log V)$ |
| Fibonacci | $O(V\log V + E) = O(E + V\log V)$ [Fredman & Tarjan 1987] |
| Pairing | $O((V+E)\log V)$ from the proven $O(\log n)$ decrease-key; $O(V\log V + E\cdot 2^{2\sqrt{\log\log V}})$ with Pettie's bound |

**How many decrease-keys really happen?** The $E$ term only matters if many
relaxations actually decrease a key. Suppose, for every vertex, the lengths of
its incoming edges are drawn independently from the same distribution. Then the
**expected number of decrease-key operations is $O(V\log(1+E/V))$** [Goldberg &
Tarjan 1996].

For our benchmark graphs ($E = 10V$, uniform random weights), that is roughly
$V \log 11$, far below $E$. So the theoretical advantage of the Fibonacci heap's
$O(1)$ `decrease_key` has little room to show. We record the actual operation
counts (experiment E7).

---

## 6. Design choices that affect the analysis

| Choice | Reason |
|---|---|
| **Handle design (binomial).** Each binomial node points to an *item* (key, value, node back-pointer). `decrease_key` swaps *items* between nodes and updates the back-pointers. | CLRS's bubble-up swaps the stored keys and data, so a handle held by the caller would silently start pointing at another item. Swapping items keeps every handle valid; the cost is one extra object per item. |
| **Min pointer in the binomial heap** | `find_min` becomes $O(1)$, which matches the other two heaps and makes the comparison fair. |
| **Two-pass pairing** (not multipass or auxiliary two-pass) | The variant analysed by FSST 1986, Iacono 2000 and Pettie 2005, and the most widely used. |
| **Fibonacci degree table of $\lfloor\log_\varphi n\rfloor + 2$ entries** | §2.4 bound plus one spare. |
| **Iterative code only** (no recursion) | Python's default recursion limit is 1000. A pairing heap after $n$ sorted inserts, or a heap-order chain, has depth or width up to $n$. |
| **Native `delete`** rather than "decrease to $-\infty$, then extract" | Keys can be any comparable type, so there is no universal $-\infty$. |
| **`__slots__` on every node class** | Removes the per-object `__dict__`, roughly halving memory. A 5-field node takes 72 bytes instead of 48 + 104 (dict) = 152 bytes (§7). |

---

## 7. Back-of-envelope memory at $n = 10^7$ (CPython 3.10, 64-bit)

Measured with `sys.getsizeof` on objects with the same number of `__slots__`
fields. Keys are uniform in $[0, 2^{31})$: `int` objects take 28 bytes below
$2^{30}$ and 32 bytes from $2^{30}$ up, about 30 bytes on average. `value = None`
is shared and free.

| Heap | Per item | Bytes per item | Total at $n = 10^7$ |
|---|---|---|---|
| Binomial | node (5 slots, 72 B) + item (3 slots, 56 B) + key (≈30 B) | ≈158 | ≈1.6 GB |
| Fibonacci | node (8 slots, 96 B) + key (≈30 B) | ≈126 | ≈1.3 GB |
| Pairing | node (5 slots, 72 B) + key (≈30 B) | ≈102 | ≈1.0 GB |

All three fit in the 16 GB of the benchmark machine, but only one heap at a
time is kept alive during the benchmark. Phase 9 replaces these estimates with
measured values.

---

## 8. Hypotheses for the experiments (checked in Phase 9)

| # | Experiment | Prediction from the analysis | Possible deviation and why |
|---|---|---|---|
| H1 | E1 insert, µs per op vs $n$ | flat for all three (log-log slope ≈ 0) | binomial slightly slower (carry links); a small upward drift from cache misses as the heap outgrows the CPU caches |
| H2 | E2 first extract_min after $n$ inserts | Fibonacci and pairing $\Theta(n)$ (slope ≈ 1); binomial $O(\log n)$ (slope ≈ 0) | none expected; this is amortization made visible |
| H3 | E3 steady-state extract_min | grows like $a + b\log_2 n$ for all three | grows faster than $\log n$ at large $n$ because every node visit can be a cache miss [Larkin et al. 2014] |
| H4 | E4 decrease_key | Fibonacci $O(1)$; binomial $O(\log n)$ for a large decrease, $O(1)$ for a small one; pairing $O(1)$ **actual** | pairing and Fibonacci decrease-keys look cheap in isolation, because their cost is deferred: the next extract_min has more roots or children to process. We therefore also time the extract_min that follows a batch of decrease-keys. |
| H5 | E5 merge | Fibonacci and pairing $O(1)$; binomial $O(\log n)$ | merge times are microseconds, so measurements are noisy; we use the median of several repeats |
| H6 | E6 memory | pairing < Fibonacci < binomial, by the estimates in §7 | |
| H7 | E7 Dijkstra | Fibonacci best by the worst-case bound | **likely contradicted**: few decrease-keys (§5) and heavier nodes, so pairing should be fastest [Stasko & Vitter 1987; Larkin et al. 2014] |

---

## References

1. T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. 2009. *Introduction to Algorithms* (3rd ed.). MIT Press. Ch. 17 (amortized analysis), ch. 19 (Fibonacci heaps), Problem 19-2 (binomial heaps).
2. T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein. 2001. *Introduction to Algorithms* (2nd ed.). MIT Press. Ch. 19 (binomial heaps).
3. J. Vuillemin. 1978. A Data Structure for Manipulating Priority Queues. *Commun. ACM* 21, 4, 309–315. https://doi.org/10.1145/359460.359478
4. M. R. Brown. 1978. Implementation and Analysis of Binomial Queue Algorithms. *SIAM J. Comput.* 7, 3, 298–319. https://doi.org/10.1137/0207026
5. M. L. Fredman and R. E. Tarjan. 1987. Fibonacci Heaps and Their Uses in Improved Network Optimization Algorithms. *J. ACM* 34, 3, 596–615. https://doi.org/10.1145/28869.28874
6. M. L. Fredman, R. Sedgewick, D. D. Sleator, and R. E. Tarjan. 1986. The Pairing Heap: A New Form of Self-Adjusting Heap. *Algorithmica* 1, 1–4, 111–129. https://doi.org/10.1007/BF01840439
7. J. T. Stasko and J. S. Vitter. 1987. Pairing Heaps: Experiments and Analysis. *Commun. ACM* 30, 3, 234–249. https://doi.org/10.1145/214748.214759
8. M. L. Fredman. 1999. On the Efficiency of Pairing Heaps and Related Data Structures. *J. ACM* 46, 4, 473–501. https://doi.org/10.1145/320211.320214
9. J. Iacono. 2000. Improved Upper Bounds for Pairing Heaps. In *Algorithm Theory – SWAT 2000*, LNCS 1851, 32–45. https://doi.org/10.1007/3-540-44985-X_5
10. S. Pettie. 2005. Towards a Final Analysis of Pairing Heaps. In *46th Annual IEEE Symposium on Foundations of Computer Science (FOCS '05)*, 174–183. https://doi.org/10.1109/SFCS.2005.75
11. J. Iacono and Ö. Özkan. 2014. Why Some Heaps Support Constant-Amortized-Time Decrease-Key Operations, and Others Do Not. In *ICALP 2014*, LNCS 8572, 637–649. https://doi.org/10.1007/978-3-662-43948-7_53
12. D. H. Larkin, S. Sen, and R. E. Tarjan. 2014. A Back-to-Basics Empirical Study of Priority Queues. In *ALENEX 2014*, 61–72. https://doi.org/10.1137/1.9781611973198.7
13. E. W. Dijkstra. 1959. A Note on Two Problems in Connexion with Graphs. *Numerische Mathematik* 1, 269–271. https://doi.org/10.1007/BF01386390
14. A. V. Goldberg and R. E. Tarjan. 1996. Expected Performance of Dijkstra's Shortest Path Algorithm. Technical Report TR-530-96, Department of Computer Science, Princeton University.
