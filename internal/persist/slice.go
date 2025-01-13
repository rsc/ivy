// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package persist implements a persistent slice data structure,
// similar to [Clojure's persistent vectors].
// The meaning of “persistent” here is that operations on the slice
// create a new slice, with the old versions continuing to be valid.
// The [Slice] type is the persistent slice, and a batch of changes
// can be made to it using the [TransientSlice] type.
//
// [Clojure's persistent vectors]: https://hypirion.com/musings/understanding-persistent-vector-pt-1
package persist

import (
	"fmt"
	"iter"
	"math/bits"
	"slices"
	"sync/atomic"
)

// A Slice is an immutable slice of values.
// It can be read but not written.
// To create a modified version of the slice,
// call [Slice.Transient] to obtain a [TransientSlice],
// make the changes in the transient,
// and then call [Transient.Persist] to obtain a new [Slice].
type Slice[T any] struct {
	// A Slice is a 32-way indexed tree (technically a trie) of values.
	// That is, each leaf node in the tree holds 32 values,
	// each interior node (inode) holds 32 pointers to the next level,
	// and the tree is perfectly balanced, since the keys are [0, tlen).
	// For cheaper appends and small slices, only full chunks
	// are stored in the tree: the final fragment is in tail.
	// The tree can have “holes”, meaning a nil where a leaf or inode
	// pointer should be; holes are treated as full of zero values.
	tree   any // tree of chunks
	height int // height of tree
	tlen   int // number of elements in tree (excludes tail)
	tail   []T // pending values (up to chunk) to flush to tree
}

// A TransientSlice is a mutable slice of values,
// typically derived from and intended to become a [Slice].
type TransientSlice[T any] struct {
	// A TransientSlice is the mutable form of a Slice.
	// It uses the same 32-way tree, but copy-on-write.
	// More specifically, the TransientSlice keeps track
	// of which nodes and leaves it owns, meaning they
	// are not shared with any Slices. Those can be modified
	// directly. Other nodes and leaves must be copied before
	// being modified.
	//
	// Each TransientSlice has a unique ID (the id field below).
	// A node or leaf with the same ID is owned by that TransientSlice
	// and writable; others are copy-on-write.
	// The [TransientSlice.Persist] operation picks a new top-level ID,
	// giving up ownership of all the previously owned nodes and leaves
	// and making them safe to publish as a Slice.
	//
	// In addition to the tree structure, a Slice has a tail of up to 32
	// elements not yet stored in the tree, cutting tree updates by 32X.
	// The wtail field tracks whether s.tail is owned by the TransientSlice,
	// meaning can be written to. If wtail is true, s.tail has capacity 32.
	s     Slice[T] // underlying slice data structure
	id    uint64   // id of this transient (see [transientID])
	wtail bool     // whether s.tail is writable
}

// transientID is the most recently used transient ID.
// To obtain a new ID, use transientID.Add(1).
var transientID atomic.Uint64

// The tree uses chunks of size 32 and 32-way branching in the interior nodes.
// A power of two is convenient, Clojure uses 32, and it seems to work fine.
// No experiments have been run to see if 16 or 64 would be better.
const (
	chunkBits = 5
	chunkMask = chunk - 1
	chunk     = 1 << chunkBits
)

// height returns the height of a tree with tlen elements.
func height(tlen int) int {
	if tlen == 0 {
		return 0
	}
	return 1 + bits.Len(uint(tlen-1))/chunkBits
}

// A leaf is a leaf node in the tree.
type leaf[T any] struct {
	val [chunk]T // values
	id  uint64   // id of TransientSlice that can write this leaf
}

// An inode is an interior node in the tree.
type inode struct {
	ptr [chunk]any // all *leaf[T] or all *node, depending on tree level
	id  uint64     // id of TransientSlice that can write this node
}

// Len returns len(s).
func (s *Slice[T]) Len() int {
	return s.tlen + len(s.tail)
}

// Len returns len(t).
func (t *TransientSlice[T]) Len() int { return t.s.Len() }

// At returns s[i].
func (s *Slice[T]) At(i int) T {
	if i < 0 || i >= s.Len() {
		panic(fmt.Sprintf("index %d out of range [0:%d]", i, s.Len()))
	}
	if i >= s.tlen {
		return s.tail[i&chunkMask]
	}
	p := s.tree
	for shift := (s.height - 1) * chunkBits; shift > 0 && p != nil; shift -= chunkBits {
		p = p.(*inode).ptr[(i>>shift)&chunkMask]
	}
	if p == nil {
		var zero T
		return zero
	}
	return p.(*leaf[T]).val[i&chunkMask]
}

// At returns t[i].
func (t *TransientSlice[T]) At(i int) T { return t.s.At(i) }

// All returns an iterator over s[0:len(s)].
func (s *Slice[T]) All() iter.Seq2[int, T] {
	return s.Slice(0, s.Len())
}

// All returns an iterator over t[0:len(t)].
func (t *TransientSlice[T]) All() iter.Seq2[int, T] { return t.s.All() }

// Slice returns an iterator over s[i:j].
func (s *Slice[T]) Slice(i, j int) iter.Seq2[int, T] {
	if i < 0 || j < i || j > s.Len() {
		panic(fmt.Sprintf("slice [%d:%d] out of range [0:%d]", i, j, s.Len()))
	}
	return func(yield func(int, T) bool) {
		if i < s.tlen && !s.yield(s.tree, s.height-1, i, min(j, s.tlen), yield) {
			return
		}
		for k := max(i, s.tlen); k < j; k++ {
			if !yield(k, s.tail[k-s.tlen]) {
				return
			}
		}
	}
}

// Slice returns an iterator over t[i:j].
func (t *TransientSlice[T]) Slice(i, j int) iter.Seq2[int, T] { return t.s.Slice(i, j) }

// yield calls yield(i, s[i]) for each element in s[start:end],
// stopping and returning false if any of the yield calls return false.
// p is a node at the given level (level 0 is leaves) and covers all of s[start:end].
func (s *Slice[T]) yield(p any, level, start, end int, yield func(int, T) bool) bool {
	if p == nil {
		var zero T
		for ; start < end; start++ {
			if !yield(start, zero) {
				return false
			}
		}
		return true
	}

	if level == 0 {
		l := p.(*leaf[T])
		for i := range end - start {
			if !yield(start+i, l.val[start&chunkMask+i]) {
				return false
			}
		}
		return true
	}

	// Interior node.
	ip := p.(*inode)

	shift := level * chunkBits
	width := 1 << shift // width of subtree of each child
	for j := (start >> shift) & chunkMask; j < chunk && start < end; j++ {
		m := min(end-start, width-start&(width-1))
		if !s.yield(ip.ptr[j], level-1, start, start+m, yield) {
			return false
		}
		start += m
	}
	if start != end {
		// unreachable
		panic("persist: internal error: invalid yield")
	}
	return true
}

// Transient returns a TransientSlice for modifying (a copy of) s.
func (s *Slice[T]) Transient() *TransientSlice[T] {
	t := &TransientSlice[T]{
		s: *s,
	}
	t.id = transientID.Add(1)
	return t
}

// Persist returns a [Slice] corresponding to the current state of t.
// Future modifications of t will not affect the returned slice.
func (t *TransientSlice[T]) Persist() *Slice[T] {
	s := t.s
	t.id = transientID.Add(1)
	if t.wtail {
		s.tail = slices.Clone(s.tail)
	}
	return &s
}

// wleaf returns a writable version of the leaf *p.
// It implements the "create on write" or "copy on write"
// logic needed when *p is missing or shared with other Slice[T].
func (t *TransientSlice[T]) wleaf(p *any) *leaf[T] {
	l, _ := (*p).(*leaf[T])       // could be nil
	if l == nil || l.id != t.id { // create or copy-on-write
		l1 := new(leaf[T])
		if l != nil {
			*l1 = *l
		}
		l1.id = t.id
		l = l1
		*p = l
	}
	return l
}

// wnode returns a writable version of the inode *p.
// It implements the "create on write" or "copy on write"
// logic needed when *p is missing or shared with other Slice[T].
func (t *TransientSlice[T]) wnode(p *any) *inode {
	ip, _ := (*p).(*inode)          // could be nil
	if ip == nil || ip.id != t.id { // create or copy-on-write
		ip1 := new(inode)
		if ip != nil {
			*ip1 = *ip
		}
		ip1.id = t.id
		ip = ip1
		*p = ip
	}
	return ip
}

// growTree grows the tree t.s.tree to size tlen,
// adding new height levels as needed.
// The newly accessible content is undefined
// and must be initialized by the caller.
func (t *TransientSlice[T]) growTree(tlen int) {
	if tlen < t.s.tlen {
		// unreachable
		panic("persist: internal error: invalid growTree")
	}
	t.s.tlen = tlen
	h := height(tlen)
	for t.s.height < h {
		ip := new(inode)
		ip.id = t.id
		ip.ptr[0] = t.s.tree
		t.s.tree = ip
		t.s.height++
	}
}

// shrinkTree shrinks the tree t.s.tree to size tlen,
// removing height levels as needed.
func (t *TransientSlice[T]) shrinkTree(tlen int) {
	t.s.tlen = tlen
	h := height(tlen)
	if h == 0 {
		t.s.tree = nil
		t.s.height = 0
		return
	}
	// Note: growTree and shrinkTree maintain the invariant that
	// the chain of inodes on the leftmost side of the tree (following ptr[0])
	// is fully allocated, so we don't need to check for t.s.tree == nil
	// during the loop.
	for t.s.height > h {
		t.s.tree = t.s.tree.(*inode).ptr[0]
		t.s.height--
	}
}

// Set sets t[i] = x.
func (t *TransientSlice[T]) Set(i int, x T) {
	if i < 0 || i >= t.s.Len() {
		panic(fmt.Sprintf("index %d out of range [0:%d]", i, t.s.Len()))
	}

	// Write into tail?
	if i >= t.s.tlen {
		if !t.wtail {
			t.writeTail()
		}
		t.s.tail[i&chunkMask] = x
		return
	}

	// Write into tree.
	p := &t.s.tree
	for b := (t.s.height - 1) * chunkBits; b > 0; b -= chunkBits {
		p = &t.wnode(p).ptr[(i>>b)&chunkMask]
	}
	t.wleaf(p).val[i&chunkMask] = x
}

// writeTail makes sure that t.s.tail is writable.
// Typically the caller has checked that it is not (t.rwtail is false),
// in which case writeTail replaces t.s.tail with a copy
// and sets t.rwtail to true.
// The copy has the same length but capacity set to chunk.
func (t *TransientSlice[T]) writeTail() {
	if t.wtail {
		return
	}
	tail := make([]T, len(t.s.tail), chunk)
	copy(tail, t.s.tail)
	t.s.tail = tail
	t.wtail = true
}

// Append appends the src elements to t.
func (t *TransientSlice[T]) Append(src ...T) {
	if len(src) == 0 {
		return
	}

	// Append fragment to complete tail.
	if len(t.s.tail) > 0 {
		t.writeTail()
		n := copy(t.s.tail[len(t.s.tail):cap(t.s.tail)], src)
		t.s.tail = t.s.tail[:len(t.s.tail)+n]
		if src = src[n:]; len(src) == 0 {
			return
		}

		// Tail is full with more to write; append tail to tree.
		t.appendTree(t.s.tail, chunk)
		t.s.tail = t.s.tail[:0]
	}

	// Flush full chunks directly from src.
	if len(src) >= chunk {
		n := len(src) >> chunkBits << chunkBits
		t.appendTree(src, n)
		if src = src[n:]; len(src) == 0 {
			return
		}
	}

	// Copy fragment to tail.
	t.writeTail()
	t.s.tail = append(t.s.tail, src...)
}

// appendTree appends xs (an integral number of chunks) to the tree.
func (t *TransientSlice[T]) appendTree(src []T, total int) {
	if total&chunkMask != 0 || total == 0 {
		// unreachable
		panic("persist: internal error: invalid appendTree")
	}

	// Update length, adding height to tree if needed.
	off := t.s.tlen
	t.growTree(off + total)

	// Copy new data into tree.
	t.copy(&t.s.tree, t.s.height-1, off, src, total)
}

// copy is like copy(t[off:], src[:total]),
// where p points to a node at the given level of the tree.
func (t *TransientSlice[T]) copy(p *any, level, off int, src []T, total int) {
	if level == 0 {
		// Leaf level.
		l := t.wleaf(p)
		l.val = [chunk]T(src[:chunk])
		return
	}

	// Interior node.
	n := t.wnode(p)

	// Copy parts of xs into the appropriate child nodes.
	shift := level * chunkBits
	width := 1 << shift // width of subtree of each child
	for j := (off >> shift) & chunkMask; j < chunk && total > 0; j++ {
		m := min(total, width-off&(width-1))
		var next []T
		next, src = src[:m], src[m:]
		t.copy(&n.ptr[j], level-1, off, next, m)
		off += m
		total -= m
	}
	if total != 0 {
		// unreachable
		panic("persist: internal error: invalid copy")
	}
}

// Resize resizes t to have n elements.
// If t is being grown, the value of new elements is undefined.
func (t *TransientSlice[T]) Resize(n int) {
	tlen, tail := n&^chunkMask, n&chunkMask
	switch {
	case n > t.s.Len():
		// Grow.
		t.writeTail()
		if tlen != t.s.tlen {
			// Flush tail into tree and then grow tree more if needed.
			t.appendTree(t.s.tail[:chunk], chunk)
			t.growTree(tlen)
		}
		t.s.tail = t.s.tail[:tail]
	case n < t.s.Len():
		// Shrink. May need to load different tail from tree before shrinking tree.
		if tlen != t.s.tlen {
			t.writeTail()
			// Note: growTree and shrinkTree maintain the invariant that
			// the chain of inodes on the leftmost side of the tree (following ptr[0])
			// is fully allocated, so we don't need to check for t.s.tree == nil
			// during the loop. The final leaf may be missing, though.
			p := t.s.tree
			for b := (t.s.height - 1) * chunkBits; b > 0; b -= chunkBits {
				p = p.(*inode).ptr[(tlen>>b)&chunkMask]
			}
			if p == nil {
				clear(t.s.tail[:tail])
			} else {
				copy(t.s.tail[:tail], p.(*leaf[T]).val[:tail])
			}
			t.shrinkTree(tlen)
		}
		t.s.tail = t.s.tail[:tail]
	}
	if n != t.s.Len() {
		// unreachable
		panic("persist: internal error: invalid Resize")
	}
	return
}
