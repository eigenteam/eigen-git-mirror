/* This file is part of Eigen, a C++ template library for linear algebra
 * Copyright (C) 2007 Benoit Jacob <jacob@math.jussieu.fr>
 *
 * Based on Tvmet source code, http://tvmet.sourceforge.net,
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * lesser General Public License for more details.
 *
 * You should have received a copy of the GNU lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: Matrix.h,v 1.54 2005/03/02 12:12:51 opetzold Exp $
 */

#ifndef TVMET_MATRIX_H
#define TVMET_MATRIX_H

#include <iterator> // reverse_iterator
#include <cassert>

#include <tvmet/tvmet.h>
#include <tvmet/TypePromotion.h>
#include <tvmet/CommaInitializer.h>

#include <tvmet/xpr/Matrix.h>
#include <tvmet/xpr/MatrixRow.h>
#include <tvmet/xpr/MatrixCol.h>
#include <tvmet/xpr/MatrixDiag.h>

namespace tvmet {

/* forwards */
template<class T, int Rows, int Cols> class Matrix;

/**
 * \class MatrixConstRef Matrix.h "tvmet/Matrix.h"
 * \brief value iterator for ET
 */
template<class T, int Rows, int Cols>
class MatrixConstRef
  : public TvmetBase < MatrixConstRef<T, Rows, Cols> >
{

public:
  /** Complexity counter. */
  enum {
    ops       = Rows * Cols
  };
  typedef T value_type;

private:
  MatrixConstRef();
  MatrixConstRef& operator=(const MatrixConstRef&);

public:
  /** Constructor. */
  explicit MatrixConstRef(const Matrix<T, Rows, Cols>& rhs)
    : m_array(rhs.array())
  { }

  /** Constructor by a given memory pointer. */
  explicit MatrixConstRef(const T* data)
    : m_array(data)
  { }

  /** access by index. */
  T operator()(int i, int j) const {
    assert(i >= 0 && j >= 0 && i < Rows && j < Cols);
    return m_array[i + j * Rows];
  }

  /** debugging Xpr parse tree */
  void print_xpr(std::ostream& os, int l=0) const {
    os << IndentLevel(l)
       << "MatrixConstRef[O=" << ops << "]<"
       << "T=" << typeid(T).name() << ">,"
       << std::endl;
  }

private:
  const T* _tvmet_restrict m_array;
};


/**
 * \class Matrix Matrix.h "tvmet/Matrix.h"
 * \brief A tiny matrix class.
 *
 * The array syntax A[j][j] isn't supported here. The reason is that
 * operator[] always takes exactly one parameter, but operator() can
 * take any number of parameters (in the case of a rectangular matrix,
 * two paramters are needed). Therefore the cleanest way to do it is
 * with operator() rather than with operator[]. \see C++ FAQ Lite 13.8
 */
template<class T, int Rows, int Cols>
class Matrix
{
public:

  typedef T value_type;
  
  /** Complexity counter. */
  enum {
    ops_assign = Rows * Cols,
    ops        = ops_assign,
    use_meta   = ops < TVMET_COMPLEXITY_M_ASSIGN_TRIGGER ? true : false
  };

  /** The number of rows of the matrix. */
  static int rows() { return Rows; }

  /** The number of columns of the matrix. */
  static int cols() { return Cols; }

public:
  /** Default Destructor. Does nothing. */
  ~Matrix() {}

  /** Default Constructor. Does nothing. The matrix entries are not initialized. */
  explicit Matrix() {}

  /** Copy Constructor, not explicit! */
  Matrix(const Matrix& rhs)
  {
    *this = XprMatrix<ConstRef, Rows, Cols>(rhs.constRef());
  }
  
  explicit Matrix(const value_type* array)
  {
    for(int i = 0; i < Rows * Cols; i++) m_array[i] = array[i];
  }
  
  /** Construct a matrix by expression. */
  template<class E>
  explicit Matrix(const XprMatrix<E, Rows, Cols>& e)
  {
    *this = e;
  }

  /** assign a T on array, this can be used for a single value
      or a comma separeted list of values. */
  CommaInitializer<Matrix, Rows * Cols> operator=(T rhs) {
    return CommaInitializer<Matrix, Rows * Cols>(*this, rhs);
  }

public: // access operators
  T* _tvmet_restrict array() { return m_array; }
  const T* _tvmet_restrict array() const { return m_array; }

public: // index access operators
  T& _tvmet_restrict operator()(int i, int j) {
    // Note: g++-2.95.3 does have problems on typedef reference
    assert(i >= 0 && j >= 0 && i < Rows && j < Cols);
    return m_array[i + j * Rows];
  }

  const T& operator()(int i, int j) const {
    assert(i >= 0 && j >= 0 && i < Rows && j < Cols);
    return m_array[i + j * Rows];
  }

public: // ET interface
  typedef MatrixConstRef<T, Rows, Cols> ConstRef;

  /** Return a const Reference of the internal data */
  ConstRef constRef() const { return ConstRef(*this); }

  /** Return the matrix as const expression. */
  XprMatrix<ConstRef, Rows, Cols> expr() const {
    return XprMatrix<ConstRef, Rows, Cols>(this->constRef());
  }

private:
  /** Wrapper for meta assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<true>, Dest& dest, const Src& src, const Assign& assign_fn) {
    meta::Matrix<Rows, Cols, 0, 0>::assign(dest, src, assign_fn);
  }

  /** Wrapper for loop assign. */
  template<class Dest, class Src, class Assign>
  static inline
  void do_assign(dispatch<false>, Dest& dest, const Src& src, const Assign& assign_fn) {
    loop::Matrix<Rows, Cols>::assign(dest, src, assign_fn);
  }

  /** assign *this to a matrix of a different type T2 using
      the functional assign_fn. */
  template<class T2, class Assign>
  void assign_to(Matrix<T2, Rows, Cols>& dest, const Assign& assign_fn) const {
    do_assign(dispatch<use_meta>(), dest, *this, assign_fn);
  }

public:  // assign operations
  /** assign a given matrix of a different type T2 element wise
      to this matrix. The operator=(const Matrix&) is compiler
      generated. */
  template<class T2>
  Matrix& operator=(const Matrix<T2, Rows, Cols>& rhs) {
    rhs.assign_to(*this, Fcnl_assign<T, T2>());
    return *this;
  }

  /** assign a given XprMatrix element wise to this matrix. */
  template <class E>
  Matrix& operator=(const XprMatrix<E, Rows, Cols>& rhs) {
    rhs.assign_to(*this, Fcnl_assign<T, typename E::value_type>());
    return *this;
  }

private:
  template<class Obj, int LEN> friend class CommaInitializer;
  
  void commaWrite(int index, T rhs)
  {
    int row = index / Cols;
    int col = index % Cols;
    m_array[row + col * Rows] = rhs;
  }

  /** This is a helper for assigning a comma separated initializer
      list. It's equal to Matrix& operator=(T) which does
      replace it. */
  Matrix& assign_value(T rhs) {
    typedef XprLiteral<T> expr_type;
    *this = XprMatrix<expr_type, Rows, Cols>(expr_type(rhs));
    return *this;
  }

public: // math operators with scalars
  Matrix& operator+=(T) _tvmet_always_inline;
  Matrix& operator-=(T) _tvmet_always_inline;
  Matrix& operator*=(T) _tvmet_always_inline;
  Matrix& operator/=(T) _tvmet_always_inline;

  template <class T2> Matrix& M_add_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& M_sub_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& M_mul_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& M_div_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;

public: // math operators with expressions
  template <class E> Matrix& M_add_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& M_sub_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& M_mul_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& M_div_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;

public: // aliased math operators with expressions
  template <class T2> Matrix& alias_assign(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& alias_add_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& alias_sub_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& alias_mul_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;
  template <class T2> Matrix& alias_div_eq(const Matrix<T2, Rows, Cols>&) _tvmet_always_inline;

  template <class E> Matrix& alias_assign(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& alias_add_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& alias_sub_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& alias_mul_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;
  template <class E> Matrix& alias_div_eq(const XprMatrix<E, Rows, Cols>&) _tvmet_always_inline;

public: // io
  /** Structure for info printing as Matrix<T, Rows, Cols>. */
  struct Info : public TvmetBase<Info> {
    std::ostream& print_xpr(std::ostream& os) const {
      os << "Matrix<T=" << typeid(T).name()
	 << ", R=" << Rows << ", C=" << Cols << ">";
      return os;
    }
  };

  /** Get an info object of this matrix. */
  static Info info() { return Info(); }

  /** Member function for expression level printing. */
  std::ostream& print_xpr(std::ostream& os, int l=0) const;

  /** Member function for printing internal data. */
  std::ostream& print_on(std::ostream& os) const;

private:
  /** The data of matrix self. */
  T m_array[Rows * Cols];
};

typedef Matrix<int, 2, 2> Matrix2i;
typedef Matrix<int, 3, 3> Matrix3i;
typedef Matrix<int, 4, 4> Matrix4i;
typedef Matrix<float, 2, 2> Matrix2f;
typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<float, 4, 4> Matrix4f;
typedef Matrix<double, 2, 2> Matrix2d;
typedef Matrix<double, 3, 3> Matrix3d;
typedef Matrix<double, 4, 4> Matrix4d;

} // namespace tvmet

#include <tvmet/MatrixImpl.h>
#include <tvmet/MatrixFunctions.h>
#include <tvmet/MatrixUnaryFunctions.h>
#include <tvmet/MatrixOperators.h>
#include <tvmet/MatrixEval.h>
#include <tvmet/AliasProxy.h>

#endif // TVMET_MATRIX_H

// Local Variables:
// mode:C++
// End:
