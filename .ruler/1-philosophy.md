## My philosophy of software development

This document outlines my philosophy of software design and implementation. It's not a prescription, nor a claim of superiority. It's a reflection of how I approach building systems with elegance, clarity, and lasting utility. It’s driven by taste—informed by mathematics, tempered by engineering constraints, and matured through hard-earned experience.

---

**1. Abstractions Should Compose**

The most powerful abstractions are those that vanish. They don’t entangle the user in incidental complexity; they lift the problem closer to its essence. Good abstractions compose. You can chain them, combine them, and they retain their integrity. This is true in the UNIX philosophy, where programs are small and pipeable; in functional programming, where functions are first-class citizens; and in algorithm design, where primitives like sorting or hashing can be reused across domains without modification.

The underlying principle is structural invariance—a concept shared with category theory. If a transformation preserves structure (like a functor), it becomes inherently more reusable. Sorting doesn’t care what you sort, as long as the elements are comparable. Matching algorithms don’t care about what the cost matrix represents. They are pure forms. The more your components resemble these, the more broadly they apply.

---

**2. Data First, Behavior Second**

Data is primary. Behavior should be defined as a set of transformations on immutable data. This aligns with the foundational ideas of algebraic data types and functional programming, but also with real-world reasoning: we analyze facts before we apply logic.

In practice, this means preferring  immutable, type-safe containers. They serve as the raw material of computation. Business logic becomes a set of pure functions operating on them. Objects can be useful when they encapsulate protocol-like behavior or manage stateful boundaries, but they should be the exception, not the norm.

---

**3. Prefer Clarity over Cleverness**

Readable code is executable thought. A program is read many more times than it is written, and clarity compounds. This is not a call for verbosity or boilerplate. It’s a call for code that reads close to the domain it operates in, using familiar patterns and structures. If a novice engineer cannot trace the logic without mentally simulating a monad transformer, the abstraction is too clever.

This is also why I prefer comprehensions in Python: they read top-down and clearly express intent. In statically typed languages, method chains offer the same elegance by preserving the container's shape and element type. Expressiveness matters, but never at the cost of mental overhead.

---

**4. Leverage Prior Art Ruthlessly**

Most domain problems have already been solved in another context. Matching, scheduling, searching, routing, indexing—these are well-mapped territories in the landscape of algorithms and data structures. If a problem reduces cleanly to a known abstraction, use it. This isn’t intellectual laziness; it’s compression. It allows us to stand on the shoulders of well-tested, well-understood work.

I derive satisfaction when I recognize a standard algorithm within a domain-specific feature. It means the solution is not arbitrary. It aligns with what Knuth called "programming as a literary form": the elegance comes not from inventing new constructs, but from discovering the most fitting existing ones.

---

**5. Avoid Configuration by Convention**

In large systems, invisible magic becomes technical debt. I prefer explicit wiring, clear boundaries, and visible contracts. Dependency injection should look like data flow. Module boundaries should reflect responsibility, not hierarchy. Convention is useful, but only when it reduces entropy. Beyond that, it breeds fragility.

Composable abstractions resist this fragility. They form a DAG, not a tangle. They can be tested in isolation, reused in orthogonal directions, and combined in surprising ways. The fewer assumptions they make about their environment, the more freely they move.

---

**6. Local Reasoning Is Sacred**

A good design is one where you can understand a component without pulling in the entire system context. Functions should not close over broad scopes. Classes should not mutate global state. Tests should not depend on side-effects.

Category theory calls this referential transparency. Distributed systems call it idempotency. Human cognition calls it sanity. The ability to reason locally is a gift you give your future self and your collaborators.

---

**7. Balance Taste with Practicality**

I admire Haskell, but I write Python. I value purity, but I live in a world of business logic, mutable APIs, and shipping deadlines. The goal is not to simulate a functional language inside an imperative one, but to **borrow ideas** that survive contact with reality.

Taste is not dogma. It’s cultivated instinct. When I choose comprehensions over `map`, or Pydantic over hand-written classes, it’s not because of trend or ideology. It’s because they let me write clearer, safer, more expressive code **in this language**, **in this team**, **under these constraints**.

---

**Final Word**

Code taste is part science, part aesthetics. It can't be fully taught, only sharpened through critique, contrast, and construction. But if I had to condense it:

> Favor abstractions that disappear, data that doesn’t lie, and code that you’re still proud of six months later.

That's the kind of software I want to build. That’s the kind of software I want to maintain. That’s the kind of software that leaves room for both thought and craft.
