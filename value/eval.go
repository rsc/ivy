// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package value

import (
	"strings"
)

type valueType int

const (
	intType valueType = iota
	charType
	bigIntType
	bigRatType
	bigFloatType
	vectorType
	matrixType
	numType
)

var typeName = [...]string{"int", "char", "big int", "rational", "float", "vector", "matrix"}

func (t valueType) String() string {
	return typeName[t]
}

type unaryFn func(Context, Value) Value

type unaryOp struct {
	name        string
	elementwise bool // whether the operation applies elementwise to vectors and matrices
	fn          [numType]unaryFn
}

func (op *unaryOp) EvalUnary(c Context, v Value) Value {
	which := whichType(v)
	fn := op.fn[which]
	if fn == nil {
		if op.elementwise {
			switch which {
			case vectorType:
				return unaryVectorOp(c, op.name, v)
			case matrixType:
				return unaryMatrixOp(c, op.name, v)
			}
		}
		Errorf("unary %s not implemented on type %s", op.name, which)
	}
	return fn(c, v)
}

type binaryFn func(Context, Value, Value) Value

type binaryOp struct {
	name        string
	elementwise bool // whether the operation applies elementwise to vectors and matrices
	whichType   func(a, b valueType) (valueType, valueType)
	fn          [numType]binaryFn
}

func whichType(v Value) valueType {
	switch v.Inner().(type) {
	case Int:
		return intType
	case Char:
		return charType
	case BigInt:
		return bigIntType
	case BigRat:
		return bigRatType
	case BigFloat:
		return bigFloatType
	case Vector:
		return vectorType
	case *Matrix:
		return matrixType
	}
	Errorf("unknown type %T in whichType", v)
	panic("which type")
}

func (op *binaryOp) EvalBinary(c Context, u, v Value) Value {
	if op.whichType == nil {
		// At the moment, "text" is the only operator that leaves
		// both arg types alone. Perhaps more will arrive.
		if op.name != "text" {
			Errorf("internal error: nil whichType")
		}
		return op.fn[0](c, u, v)
	}
	whichU, whichV := op.whichType(whichType(u), whichType(v))
	conf := c.Config()
	u = u.toType(op.name, conf, whichU)
	v = v.toType(op.name, conf, whichV)
	fn := op.fn[whichV]
	if fn == nil {
		if op.elementwise {
			switch whichV {
			case vectorType:
				return binaryVectorOp(c, u, op.name, v)
			case matrixType:
				return binaryMatrixOp(c, u, op.name, v)
			}
		}
		Errorf("binary %s not implemented on type %s", op.name, whichV)
	}
	return fn(c, u, v)
}

// index computes u[v].
// v is known to be an int, vector, or matrix.
// u could be anything.
// The rank of the result depends only on the rank of u and v,
// not on the actual data.
// For example, if u is a vector and v is a vector,
// then u[v] is always a vector, even if v has only one element.
// Giving each expression a rank that depends only on the
// input ranks, and not on the actual indexes, is important
// for avoiding indexing problems in user-defined operators.
func index(c Context, u, v Value) Value {
	var Astride Int
	var Adata Vector
	switch A := u.(type) {
	case Vector:
		Astride = 1
		Adata = A
	case *Matrix:
		Astride = Int(len(A.data) / A.shape[0])
		Adata = A.data
	default:
		Errorf("cannot index %v", whichType(u))
	}

	// Collect indexed values.
	origin := Int(c.Config().Origin())
	var data Vector
	switch v.(type) {
	default:
		Errorf("index must be integer")
	case Int:
		B := v.(Int)
		B -= origin
		if B < 0 || B >= Int(len(Adata))/Astride {
			Errorf("index %d out of range", B+origin)
		}
		data = append(data, Adata[B*Astride:(B+1)*Astride]...)
	case Vector, *Matrix:
		var Bdata []Value
		switch B := v.(type) {
		case Vector:
			Bdata = B
		case *Matrix:
			Bdata = B.data
		}
		data = make([]Value, 0, Int(len(Bdata))*Astride)
		for i := range Bdata {
			b, ok := Bdata[i].(Int)
			if !ok {
				Errorf("index must be integer")
			}
			b -= origin
			if b < 0 || b >= Int(len(Adata))/Astride {
				Errorf("index %d out of range", b+origin)
			}
			data = append(data, Adata[b*Astride:(b+1)*Astride]...)
		}
	}

	// Shape of result of A[B] is the shape of B + shape of A[i].
	var shape []int
	switch B := v.(type) {
	case Vector:
		shape = []int{len(B)}
	case *Matrix:
		shape = append(shape, B.shape...)
	}
	switch A := u.(type) {
	case *Matrix:
		shape = append(shape, A.shape[1:]...)
	}

	switch len(shape) {
	case 0:
		return data[0]
	case 1:
		return data // vector
	default:
		return NewMatrix(shape, data)
	}
}

// Product computes a compound product, such as an inner product
// "+.*" or outer product "o.*". The op is known to contain a
// period. The operands are all at least vectors, and for inner product
// they must both be vectors.
func Product(c Context, u Value, op string, v Value) Value {
	dot := strings.IndexByte(op, '.')
	left := op[:dot]
	right := op[dot+1:]
	which, _ := atLeastVectorType(whichType(u), whichType(v))
	u = u.toType(op, c.Config(), which)
	v = v.toType(op, c.Config(), which)
	if left == "o" {
		return outerProduct(c, u, right, v)
	}
	return innerProduct(c, u, left, right, v)
}

// inner product computes an inner product such as "+.*".
// u and v are known to be the same type and at least Vectors.
func innerProduct(c Context, u Value, left, right string, v Value) Value {
	switch u := u.(type) {
	case Vector:
		v := v.(Vector)
		u.sameLength(v)
		n := len(u)
		if n == 0 {
			Errorf("empty inner product")
		}
		x := c.EvalBinary(u[n-1], right, v[n-1])
		for k := n - 2; k >= 0; k-- {
			x = c.EvalBinary(c.EvalBinary(u[k], right, v[k]), left, x)
		}
		return x
	case *Matrix:
		// Say we're doing +.*
		// result[i,j] = +/(u[row i] * v[column j])
		// Number of columns of u must be the number of rows of v: (-1 take rho u) == (1 take rho v)
		// The result is has shape (-1 drop rho u), (1 drop rho v)
		v := v.(*Matrix)
		if u.Rank() < 1 || v.Rank() < 1 || u.shape[len(u.shape)-1] != v.shape[0] {
			Errorf("inner product: mismatched shapes %s and %s", NewIntVector(u.shape), NewIntVector(v.shape))
		}
		n := v.shape[0]
		vstride := len(v.data) / n
		data := make(Vector, len(u.data)/n*vstride)
		for i := 0; i < len(u.data); i += n {
			for j := 0; j < vstride; j++ {
				acc := c.EvalBinary(u.data[i+n-1], right, v.data[j+(n-1)*vstride])
				for k := n - 2; k >= 0; k-- {
					acc = c.EvalBinary(c.EvalBinary(u.data[i+k], right, v.data[j+k*vstride]), left, acc)
				}
				data[i/n*vstride+j] = acc
			}
		}
		rank := len(u.shape) + len(v.shape) - 2
		if rank == 1 {
			return data
		}
		shape := make([]int, rank)
		copy(shape, u.shape[:len(u.shape)-1])
		copy(shape[len(u.shape)-1:], v.shape[1:])
		return NewMatrix(shape, data)
	}
	Errorf("can't do inner product on %s", whichType(u))
	panic("not reached")
}

// outer product computes an outer product such as "o.*".
// u and v are known to be at least Vectors.
func outerProduct(c Context, u Value, op string, v Value) Value {
	switch u := u.(type) {
	case Vector:
		v := v.(Vector)
		m := Matrix{
			shape: []int{len(u), len(v)},
			data:  NewVector(make(Vector, len(u)*len(v))),
		}
		index := 0
		for _, vu := range u {
			for _, vv := range v {
				m.data[index] = c.EvalBinary(vu, op, vv)
				index++
			}
		}
		return &m // TODO: Shrink?
	case *Matrix:
		v := v.(*Matrix)
		m := Matrix{
			shape: append(u.Shape(), v.Shape()...),
			data:  NewVector(make(Vector, len(u.Data())*len(v.Data()))),
		}
		index := 0
		for _, vu := range u.Data() {
			for _, vv := range v.Data() {
				m.data[index] = c.EvalBinary(vu, op, vv)
				index++
			}
		}
		return &m // TODO: Shrink?
	}
	Errorf("can't do outer product on %s", whichType(u))
	panic("not reached")
}

// Reduce computes a reduction such as +/. The slash has been removed.
func Reduce(c Context, op string, v Value) Value {
	// We must be right associative; that is the grammar.
	// -/1 2 3 == 1-2-3 is 1-(2-3) not (1-2)-3. Answer: 2.
	switch v := v.(type) {
	case Int, BigInt, BigRat:
		return v
	case Vector:
		if len(v) == 0 {
			return v
		}
		acc := v[len(v)-1]
		for i := len(v) - 2; i >= 0; i-- {
			acc = c.EvalBinary(v[i], op, acc)
		}
		return acc
	case *Matrix:
		if v.Rank() < 2 {
			Errorf("shape for matrix is degenerate: %s", NewIntVector(v.shape))
		}
		stride := v.shape[v.Rank()-1]
		if stride == 0 {
			Errorf("shape for matrix is degenerate: %s", NewIntVector(v.shape))
		}
		shape := v.shape[:v.Rank()-1]
		data := make(Vector, size(shape))
		index := 0
		for i := range data {
			pos := index + stride - 1
			acc := v.data[pos]
			pos--
			for i := 1; i < stride; i++ {
				acc = c.EvalBinary(v.data[pos], op, acc)
				pos--
			}
			data[i] = acc
			index += stride
		}
		if len(shape) == 1 { // TODO: Matrix.shrink()?
			return NewVector(data)
		}
		return NewMatrix(shape, data)
	}
	Errorf("can't do reduce on %s", whichType(v))
	panic("not reached")
}

// Scan computes a scan of the op; the \ has been removed.
// It gives the successive values of reducing op through v.
// We must be right associative; that is the grammar.
func Scan(c Context, op string, v Value) Value {
	switch v := v.(type) {
	case Int, BigInt, BigRat:
		return v
	case Vector:
		if len(v) == 0 {
			return v
		}
		values := make(Vector, len(v))
		acc := v[0]
		values[0] = acc
		// TODO: This is n^2.
		for i := 1; i < len(v); i++ {
			values[i] = Reduce(c, op, v[:i+1])
		}
		return NewVector(values)
	case *Matrix:
		if v.Rank() < 2 {
			Errorf("shape for matrix is degenerate: %s", NewIntVector(v.shape))
		}
		stride := v.shape[v.Rank()-1]
		if stride == 0 {
			Errorf("shape for matrix is degenerate: %s", NewIntVector(v.shape))
		}
		data := make(Vector, len(v.data))
		index := 0
		nrows := 1
		for i := 0; i < v.Rank()-1; i++ {
			// Guaranteed by NewMatrix not to overflow.
			nrows *= v.shape[i]
		}
		for i := 0; i < nrows; i++ {
			acc := v.data[index]
			data[index] = acc
			// TODO: This is n^2.
			for j := 1; j < stride; j++ {
				data[index+j] = Reduce(c, op, v.data[index:index+j+1])
			}
			index += stride
		}
		return NewMatrix(v.shape, data)
	}
	Errorf("can't do scan on %s", whichType(v))
	panic("not reached")
}

// dataShape returns the data shape of v.
// The data shape of a scalar is []int{}, to distinguish from a vector of length 1.
func dataShape(v Value) []int {
	switch v := v.(type) {
	case Vector:
		return []int{len(v)}
	case *Matrix:
		return v.Shape()
	}
	return []int{}
}

// appendData appends the underlying data of v to list.
func appendData(list Vector, v Value) Vector {
	switch v := v.(type) {
	case Vector:
		return append(list, v...)
	case *Matrix:
		return append(list, v.data...)
	}
	return append(list, v)
}

// BinaryMap computes the result of mapping op between lv and rv.
func BinaryMap(c Context, lv Value, op string, rv Value) Value {
	l := lv.toType(op, c.Config(), matrixType).(*Matrix)
	r := rv.toType(op, c.Config(), matrixType).(*Matrix)
	lsize := len(l.data)
	rsize := len(r.data)
	ln := 0
	for ln < len(op) && op[ln] == '@' {
		if l.shape[ln] > 0 {
			lsize /= l.shape[ln]
		}
		ln++
	}
	rn := 0
	for rn < len(op) && op[len(op)-1-rn] == '@' {
		if r.shape[rn] > 0 {
			rsize /= r.shape[rn]
		}
		rn++
	}
	var shape []int
	var result Vector
	for i := 0; i < len(l.data); i += lsize {
		for j := 0; j < len(r.data); j += rsize {
			var x Value = l.data[i : i+lsize]
			var y Value = r.data[j : j+rsize]
			if len(l.shape)-ln > 1 {
				x = NewMatrix(l.shape[ln:], x.(Vector))
			} else if len(l.shape)-ln == 0 {
				x = x.(Vector).shrink()
			}
			if len(r.shape)-rn > 1 {
				y = NewMatrix(r.shape[rn:], y.(Vector))
			} else if len(r.shape)-rn == 0 {
				y = y.(Vector).shrink()
			}
			a := c.EvalBinary(x, op[ln:len(op)-rn], y)
			if shape == nil {
				shape = dataShape(a)
			} else {
				if !sameShape(shape, dataShape(a)) {
					Errorf("map %s: conflicting result shapes %s and %s",
						op, NewIntVector(shape), NewIntVector(dataShape(a)))
				}
			}
			result = appendData(result, a)
		}
	}
	var mshape []int
	mshape = append(mshape, l.shape[:ln]...)
	mshape = append(mshape, r.shape[:rn]...)
	if len(shape) > 1 || len(shape) == 1 && shape[0] > 1 {
		mshape = append(mshape, shape...)
	}
	if len(mshape) == 1 {
		return result
	}
	return NewMatrix(mshape, result)
}

// Map computes the result of mapping op onto v.
// The trailing @ has been removed.
func Map(c Context, op string, v Value) Value {
	var shape []int
	var result Vector
	var n int
	switch v := v.(type) {
	default:
		Errorf("can't map %s| on %s", op, whichType(v))

	case Vector:
		n = len(v)
		for _, x := range v {
			a := c.EvalUnary(op, x)
			if shape == nil {
				shape = dataShape(a)
			} else {
				if !sameShape(shape, dataShape(a)) {
					Errorf("map %s|: conflicting result shapes %s and %s",
						op, NewIntVector(shape), NewIntVector(dataShape(a)))
				}
			}
			result = appendData(result, a)
		}

	case *Matrix:
		elem := len(v.data) / v.shape[0]
		n = v.shape[0]
		for i := 0; i < len(v.data); i += elem {
			var x Value
			if len(v.shape) == 2 {
				x = v.data[i : i+elem]
			} else {
				x = NewMatrix(v.shape[1:], v.data[i:i+elem])
			}
			a := c.EvalUnary(op, x)
			if shape == nil {
				shape = dataShape(a)
			} else {
				if !sameShape(shape, dataShape(a)) {
					Errorf("map %s@: conflicting result shapes %s and %s",
						op, NewIntVector(shape), NewIntVector(dataShape(a)))
				}
			}
			result = appendData(result, a)
		}
	}

	if len(shape) == 0 || len(shape) == 1 && shape[0] == 1 {
		return result
	}
	return NewMatrix(append([]int{n}, shape...), result)
}

// unaryVectorOp applies op elementwise to i.
func unaryVectorOp(c Context, op string, i Value) Value {
	u := i.(Vector)
	n := make([]Value, len(u))
	for k := range u {
		n[k] = c.EvalUnary(op, u[k])
	}
	return NewVector(n)
}

// unaryMatrixOp applies op elementwise to i.
func unaryMatrixOp(c Context, op string, i Value) Value {
	u := i.(*Matrix)
	n := make([]Value, len(u.data))
	for k := range u.data {
		n[k] = c.EvalUnary(op, u.data[k])
	}
	return NewMatrix(u.shape, NewVector(n))
}

// binaryVectorOp applies op elementwise to i and j.
func binaryVectorOp(c Context, i Value, op string, j Value) Value {
	u, v := i.(Vector), j.(Vector)
	if len(u) == 1 {
		n := make([]Value, len(v))
		for k := range v {
			n[k] = c.EvalBinary(u[0], op, v[k])
		}
		return NewVector(n)
	}
	if len(v) == 1 {
		n := make([]Value, len(u))
		for k := range u {
			n[k] = c.EvalBinary(u[k], op, v[0])
		}
		return NewVector(n)
	}
	u.sameLength(v)
	n := make([]Value, len(u))
	for k := range u {
		n[k] = c.EvalBinary(u[k], op, v[k])
	}
	return NewVector(n)
}

// binaryMatrixOp applies op elementwise to i and j.
func binaryMatrixOp(c Context, i Value, op string, j Value) Value {
	u, v := i.(*Matrix), j.(*Matrix)
	shape := u.shape
	var n []Value
	// One or the other may be a scalar in disguise.
	switch {
	case isScalar(u):
		// Scalar op Matrix.
		shape = v.shape
		n = make([]Value, len(v.data))
		for k := range v.data {
			n[k] = c.EvalBinary(u.data[0], op, v.data[k])
		}
	case isScalar(v):
		// Matrix op Scalar.
		n = make([]Value, len(u.data))
		for k := range u.data {
			n[k] = c.EvalBinary(u.data[k], op, v.data[0])
		}
	case isVector(u, v.shape):
		// Vector op Matrix.
		shape = v.shape
		n = make([]Value, len(v.data))
		dim := u.shape[0]
		index := 0
		for k := range v.data {
			n[k] = c.EvalBinary(u.data[index], op, v.data[k])
			index++
			if index >= dim {
				index = 0
			}
		}
	case isVector(v, u.shape):
		// Matrix op Vector.
		n = make([]Value, len(u.data))
		dim := v.shape[0]
		index := 0
		for k := range u.data {
			n[k] = c.EvalBinary(u.data[k], op, v.data[index])
			index++
			if index >= dim {
				index = 0
			}
		}
	default:
		// Matrix op Matrix.
		u.sameShape(v)
		n = make([]Value, len(u.data))
		for k := range u.data {
			n[k] = c.EvalBinary(u.data[k], op, v.data[k])
		}
	}
	return NewMatrix(shape, NewVector(n))
}

// isScalar reports whether u is a 1x1x1x... item, that is, a scalar promoted to matrix.
func isScalar(u *Matrix) bool {
	for _, dim := range u.shape {
		if dim != 1 {
			return false
		}
	}
	return true
}

// isVector reports whether u is an 1x1x...xn item where n is the last dimension
// of the shape, that is, an n-vector promoted to matrix.
func isVector(u *Matrix, shape []int) bool {
	if u.Rank() == 0 || len(shape) == 0 || u.shape[0] != shape[len(shape)-1] {
		return false
	}
	for _, dim := range u.shape[1:] {
		if dim != 1 {
			return false
		}
	}
	return true
}

// EvalFunctionBody evaluates the list of expressions inside a function,
// possibly with conditionals that generate an early return.
func EvalFunctionBody(context Context, fnName string, body []Expr) Value {
	var v Value
	for _, e := range body {
		if d, ok := e.(Decomposable); ok && d.Operator() == ":" {
			left, right := d.Operands()
			if isTrue(fnName, left.Eval(context)) {
				return right.Eval(context)
			}
			continue
		}
		v = e.Eval(context)
	}
	return v
}

// isTrue reports whether v represents boolean truth. If v is not
// a scalar, an error results.
func isTrue(fnName string, v Value) bool {
	switch i := v.(type) {
	case Char:
		return i != 0
	case Int:
		return i != 0
	case BigInt:
		return true // If it's a BigInt, it can't be 0 - that's an Int.
	case BigRat:
		return true // If it's a BigRat, it can't be 0 - that's an Int.
	case BigFloat:
		return i.Float.Sign() != 0
	default:
		Errorf("invalid expression %s for conditional inside %q", v, fnName)
		return false
	}
}
