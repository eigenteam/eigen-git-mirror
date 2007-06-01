/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
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

#include <iterator>					// reverse_iterator

#include <tvmet/tvmet.h>
#include <tvmet/TypePromotion.h>
#include <tvmet/CommaInitializer.h>
#include <tvmet/RunTimeError.h>

#include <tvmet/xpr/Matrix.h>
#include <tvmet/xpr/MatrixRow.h>
#include <tvmet/xpr/MatrixCol.h>
#include <tvmet/xpr/MatrixDiag.h>

namespace tvmet {


/* forwards */
template<class T, std::size_t Rows, std::size_t Cols> class Matrix;
template<class T,
	 std::size_t RowsBgn, std::size_t RowsEnd,
	 std::size_t ColsBgn, std::size_t ColsEnd,
	 std::size_t RowStride, std::size_t ColStride /*=1*/>
class MatrixSliceConstReference; // unused here; for me only


/**
 * \class MatrixConstReference Matrix.h "tvmet/Matrix.h"
 * \brief value iterator for ET
 */
template<class T, std::size_t NRows, std::size_t NCols>
class MatrixConstReference
  : public TvmetBase < MatrixConstReference<T, NRows, NCols> >
{
public:
  typedef T						value_type;
  typedef T*						pointer;
  typedef const T*					const_pointer;

  /** Dimensions. */
  enum {
    Rows = NRows,			/**< Number of rows. */
    Cols = NCols,			/**< Number of cols. */
    Size = Rows * Cols			/**< Complete Size of Matrix. */
  };

public:
  /** Complexity counter. */
  enum {
    ops       = Rows * Cols
  };

private:
  MatrixConstReference();
  MatrixConstReference& operator=(const MatrixConstReference&);

public:
  /** Constructor. */
  explicit MatrixConstReference(const Matrix<T, Rows, Cols>& rhs)
    : m_data(rhs.data())
  { }

  /** Constructor by a given memory pointer. */
  explicit MatrixConstReference(const_pointer data)
    : m_data(data)
  { }

public: // access operators
  /** access by index. */
  value_type operator()(std::size_t i, std::size_t j) const {
    TVMET_RT_CONDITION((i < Rows) && (j < Cols), "MatrixConstReference Bounce Violation")
    return m_data[i * Cols + j];
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l)
       << "MatrixConstReference[O=" << ops << "]<"
       << "T=" << typeid(value_type).name() << ">,"
       << std::endl;
  }

private:
  const_pointer _tvmet_restrict 			m_data;
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
template<class T, std::size_t NRows, std::size_t NCols>
class Matrix
{
public:
  /** Data type of the tvmet::Matrix. */
  typedef T						value_type;

  /** Reference type of the tvmet::Matrix data elements. */
  typedef T&     					reference;

  /** const reference type of the tvmet::Matrix data elements. */
  typedef const T&     					const_reference;

  /** STL iterator interface. */
  typedef T*     					iterator;

  /** STL const_iterator interface. */
  typedef const T*     					const_iterator;

  /** STL reverse iterator interface. */
  typedef std::reverse_iterator<iterator> 		reverse_iterator;

  /** STL const reverse iterator interface. */
  typedef std::reverse_iterator<const_iterator> 	const_reverse_iterator;

public:
  /** Dimensions. */
  enum {
    Rows = NRows,			/**< Number of rows. */
    Cols = NCols,			/**< Number of cols. */
    Size = Rows * Cols			/**< Complete Size of Matrix. */
  };

public:
  /** Complexity counter. */
  enum {
    ops_assign = Rows * Cols,
    ops        = ops_assign,
    use_meta   = ops < TVMET_COMPLEXITY_M_ASSIGN_TRIGGER ? true : false
  };

public: // STL  interface
  /** STL iterator interface. */
  iterator begin() { return m_data; }

  /** STL iterator interface. */
  iterator end() { return m_data + Size; }

  /** STL const_iterator interface. */
  const_iterator begin() const { return m_data; }

  /** STL const_iterator interface. */
  const_iterator end() const { return m_data + Size; }

  /** STL reverse iterator interface reverse begin. */
  reverse_iterator rbegin() { return reverse_iterator( end() ); }

  /** STL const reverse iterator interface reverse begin. */
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator( end() );
  }

  /** STL reverse iterator interface reverse end. */
  reverse_iterator rend() { return reverse_iterator( begin() ); }

  /** STL const reverse iterator interface reverse end. */
  const_reverse_iterator rend() const {
    return const_reverse_iterator( begin() );
  }

  /** The size of the matrix. */
  static std::size_t size() { return Size; }

  /** STL vector max_size() - returns allways rows()*cols(). */
  static std::size_t max_size() { return Size; }

  /** STL vector empty() - returns allways false. */
  static bool empty() { return false; }

public:
  /** The number of rows of matrix. */
  static std::size_t rows() { return Rows; }

  /** The number of columns of matrix. */
  static std::size_t cols() { return Cols; }

public:
  /** Default Destructor */
  ~Matrix() {}

  /** Default Constructor. The allocated memory region isn't cleared. If you want
   a clean use the constructor argument zero. */
  explicit Matrix() {}

  /** Copy Constructor, not explicit! */
  Matrix(const Matrix& rhs)
  {
    *this = XprMatrix<ConstReference, Rows, Cols>(rhs.const_ref());
  }

  /**
   * Constructor with STL iterator interface. The data will be copied into the matrix
   * self, there isn't any stored reference to the array pointer.
   */
  template<class InputIterator>
  explicit Matrix(InputIterator first, InputIterator last)
  {
    TVMET_RT_CONDITION(static_cast<std::size_t>(std::distance(first, last)) <= Size,
		       "InputIterator doesn't fits in size" )
    std::copy(first, last, m_data);
  }

  /**
   * Constructor with STL iterator interface. The data will be copied into the matrix
   * self, there isn't any stored reference to the array pointer.
   */
  template<class InputIterator>
  explicit Matrix(InputIterator first, std::size_t sz)
  {
    TVMET_RT_CONDITION(sz <= Size, "InputIterator doesn't fits in size" )
    std::copy(first, first + sz, m_data);
  }

  /** Construct the matrix by value. */
  explicit Matrix(value_type rhs)
  {
    typedef XprLiteral<value_type> expr_type;
    *this = XprMatrix<expr_type, Rows, Cols>(expr_type(rhs));
  }

  /** Construct a matrix by expression. */
  template<class E>
  explicit Matrix(const XprMatrix<E, Rows, Cols>& e)
  {
    *this = e;
  }

  /** assign a value_type on array, this can be used for a single value
      or a comma separeted list of values. */
  CommaInitializer<Matrix, Size> operator=(value_type rhs) {
    return CommaInitializer<Matrix, Size>(*this, rhs);
  }

public: // access operators
  value_type* _tvmet_restrict data() { return m_data; }
  const value_type* _tvmet_restrict data() const { return m_data; }

public: // index access operators
  value_type& _tvmet_restrict operator()(std::size_t i, std::size_t j) {
    // Note: g++-2.95.3 does have problems on typedef reference
    TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Matrix Bounce Violation")
    return m_data[i * Cols + j];
  }

  value_type operator()(std::size_t i, std::size_t j) const {
    TVMET_RT_CONDITION((i < Rows) && (j < Cols), "Matrix Bounce Violation")
    return m_data[i * Cols + j];
  }

public: // ET interface
  typedef MatrixConstReference<T, Rows, Cols>   	ConstReference;

  typedef MatrixSliceConstReference<
    T,
    0, Rows, 0, Cols,
    Rows, 1
  >							SliceConstReference;

  /** Return a const Reference of the internal data */
  ConstReference const_ref() const { return ConstReference(*this); }

  /**
   * Return a sliced const Reference of the internal data.
   * \note Doesn't work since isn't implemented, but it is in
   * progress. Therefore this is a placeholder. */
  ConstReference const_sliceref() const { return SliceConstReference(*this); }

  /** Return the vector as const expression. */
  XprMatrix<ConstReference, Rows, Cols> as_expr() const {
    return XprMatrix<ConstReference, Rows, Cols>(this->const_ref());
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

private:
  /** assign this to a matrix  of a different type T2 using
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
    rhs.assign_to(*this, Fcnl_assign<value_type, T2>());
    return *this;
  }

  /** assign a given XprMatrix element wise to this matrix. */
  template <class E>
  Matrix& operator=(const XprMatrix<E, Rows, Cols>& rhs) {
    rhs.assign_to(*this, Fcnl_assign<value_type, typename E::value_type>());
    return *this;
  }

private:
  template<class Obj, std::size_t LEN> friend class CommaInitializer;

  /** This is a helper for assigning a comma separated initializer
      list. It's equal to Matrix& operator=(value_type) which does
      replace it. */
  Matrix& assign_value(value_type rhs) {
    typedef XprLiteral<value_type> 			expr_type;
    *this = XprMatrix<expr_type, Rows, Cols>(expr_type(rhs));
    return *this;
  }

public: // math operators with scalars
  // NOTE: this meaning is clear - element wise ops even if not in ns element_wise
  Matrix& operator+=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator-=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator*=(value_type) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator/=(value_type) TVMET_CXX_ALWAYS_INLINE;

  Matrix& operator%=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator^=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator&=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator|=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator<<=(std::size_t) TVMET_CXX_ALWAYS_INLINE;
  Matrix& operator>>=(std::size_t) TVMET_CXX_ALWAYS_INLINE;

public: // math operators with matrizes
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class T2> Matrix& M_add_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_sub_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_mul_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_div_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_mod_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_xor_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_and_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_or_eq (const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_shl_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& M_shr_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;

public: // math operators with expressions
  // NOTE: access using the operators in ns element_wise, since that's what is does
  template <class E> Matrix& M_add_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_sub_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_mul_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_div_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_mod_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_xor_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_and_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_or_eq (const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_shl_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& M_shr_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;

public: // aliased math operators with expressions
  template <class T2> Matrix& alias_assign(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& alias_add_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& alias_sub_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& alias_mul_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class T2> Matrix& alias_div_eq(const Matrix<T2, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;

  template <class E> Matrix& alias_assign(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& alias_add_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& alias_sub_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& alias_mul_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;
  template <class E> Matrix& alias_div_eq(const XprMatrix<E, Rows, Cols>&) TVMET_CXX_ALWAYS_INLINE;

public: // io
  /** Structure for info printing as Matrix<T, Rows, Cols>. */
  struct Info : public TvmetBase<Info> {
    std::ostream& print_xpr(std::ostream& os) const {
      os << "Matrix<T=" << typeid(value_type).name()
	 << ", R=" << Rows << ", C=" << Cols << ">";
      return os;
    }
  };

  /** Get an info object of this matrix. */
  static Info info() { return Info(); }

  /** Member function for expression level printing. */
  std::ostream& print_xpr(std::ostream& os, std::size_t l=0) const;

  /** Member function for printing internal data. */
  std::ostream& print_on(std::ostream& os) const;

private:
  /** The data of matrix self. */
  value_type m_data[Size];
};


} // namespace tvmet

#include <tvmet/MatrixImpl.h>
#include <tvmet/MatrixFunctions.h>
#include <tvmet/MatrixBinaryFunctions.h>
#include <tvmet/MatrixUnaryFunctions.h>
#include <tvmet/MatrixOperators.h>
#include <tvmet/MatrixEval.h>
#include <tvmet/AliasProxy.h>

#endif // TVMET_MATRIX_H

// Local Variables:
// mode:C++
// End:
