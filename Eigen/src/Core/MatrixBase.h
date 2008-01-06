// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_MATRIXBASE_H
#define EIGEN_MATRIXBASE_H

#include "Util.h"

template <typename Derived>
struct DerivedTraits
{
    static const TraversalOrder Order = Derived::Order;
    
    /** The number of rows at compile-time. This is just a copy of the value provided
      * by the \a Derived type. If a value is not known at compile-time,
      * it is set to the \a Dynamic constant.
      * \sa rows(), cols(), ColsAtCompileTime, SizeAtCompileTime */
    static const int RowsAtCompileTime = Derived::RowsAtCompileTime;
    
    /** The number of columns at compile-time. This is just a copy of the value provided
      * by the \a Derived type. If a value is not known at compile-time,
      * it is set to the \a Dynamic constant.
      * \sa rows(), cols(), RowsAtCompileTime, SizeAtCompileTime */
    static const int ColsAtCompileTime = Derived::ColsAtCompileTime;
    
    /** This is equal to the number of coefficients, i.e. the number of
      * rows times the number of columns, or to \a Dynamic if this is not
      * known at compile-time. \sa RowsAtCompileTime, ColsAtCompileTime */
    static const int SizeAtCompileTime
      = Derived::RowsAtCompileTime == Dynamic || Derived::ColsAtCompileTime == Dynamic
      ? Dynamic : Derived::RowsAtCompileTime * Derived::ColsAtCompileTime;
    
    /** This is set to true if either the number of rows or the number of
      * columns is known at compile-time to be equal to 1. Indeed, in that case,
      * we are dealing with a column-vector (if there is only one column) or with
      * a row-vector (if there is only one row). */
    static const bool IsVectorAtCompileTime = Derived::RowsAtCompileTime == 1 || Derived::ColsAtCompileTime == 1;
};

/** \class MatrixBase
  *
  * \brief Base class for all matrices, vectors, and expressions
  *
  * This class is the base that is inherited by all matrix, vector, and expression
  * types. Most of the Eigen API is contained in this class.
  *
  * This class takes two template parameters:
  *
  * \a Scalar is the type of the coefficients, e.g. float, double, etc.
  *
  * \a Derived is the derived type, e.g. a matrix type, or an expression, etc.
  * Indeed, a separate MatrixBase type is generated for each derived type
  * so one knows from inside MatrixBase, at compile-time, what the derived type is.
  *
  * When writing a function taking Eigen objects as argument, if you want your function
  * to take as argument any matrix, vector, or expression, just let it take a
  * MatrixBase argument. As an example, here is a function printFirstRow which, given
  * a matrix, vector, or expression \a x, prints the first row of \a x.
  *
  * \code
    template<typename Scalar, typename Derived>
    void printFirstRow(const Eigen::MatrixBase<Scalar, Derived>& x)
    {
      cout << x.row(0) << endl;
    }
  * \endcode
  */
template<typename Scalar, typename Derived> class MatrixBase
{
  public:
    typedef DerivedTraits<Derived> Traits;
    
    /** This is the "reference type" used to pass objects of type MatrixBase as arguments
      * to functions. If this MatrixBase type represents an expression, then \a Ref
      * is just this MatrixBase type itself, i.e. expressions are just passed by value
      * and the compiler is usually clever enough to optimize that. If, on the
      * other hand, this MatrixBase type is an actual matrix or vector type, then \a Ref is
      * a typedef to MatrixRef, which works as a reference, so that matrices and vectors
      * are passed by reference, not by value. \sa ref()*/
    typedef typename ForwardDecl<Derived>::Ref Ref;
    
    /** This is the "real scalar" type; if the \a Scalar type is already real numbers
      * (e.g. int, float or double) then RealScalar is just the same as \a Scalar. If
      * \a Scalar is \a std::complex<T> then RealScalar is \a T. */
    typedef typename NumTraits<Scalar>::Real RealScalar;
    
    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    int rows() const { return static_cast<const Derived *>(this)->_rows(); }
    /** \returns the number of columns. \sa row(), ColsAtCompileTime*/
    int cols() const { return static_cast<const Derived *>(this)->_cols(); }
    /** \returns the number of coefficients, which is \a rows()*cols().
      * \sa rows(), cols(), SizeAtCompileTime. */
    int size() const { return rows() * cols(); }
    /** \returns true if either the number of rows or the number of columns is equal to 1.
      * In other words, this function returns
      * \code rows()==1 || cols()==1 \endcode
      * \sa rows(), cols(), IsVectorAtCompileTime. */
    bool isVector() const { return rows()==1 || cols()==1; }
    /** \returns a Ref to *this. \sa Ref */
    Ref ref() const
    { return static_cast<const Derived *>(this)->_ref(); }
    
    /** Copies \a other into *this. \returns a reference to *this. */
    template<typename OtherDerived>
    Derived& operator=(const MatrixBase<Scalar, OtherDerived>& other);
    
    // Special case of the above template operator=, in order to prevent the compiler
    //from generating a default operator= (issue hit with g++ 4.1)
    Derived& operator=(const MatrixBase& other)
    {
      return this->operator=<Derived>(other);
    }
    
    template<typename NewScalar> const Cast<NewScalar, Derived> cast() const;
    
    Row<Derived> row(int i);
    const Row<Derived> row(int i) const;
    
    Column<Derived> col(int i);
    const Column<Derived> col(int i) const;
    
    Minor<Derived> minor(int row, int col);
    const Minor<Derived> minor(int row, int col) const;
    
    DynBlock<Derived> dynBlock(int startRow, int startCol, int blockRows, int blockCols);
    const DynBlock<Derived>
    dynBlock(int startRow, int startCol, int blockRows, int blockCols) const;
    
    template<int BlockRows, int BlockCols>
    Block<Derived, BlockRows, BlockCols> block(int startRow, int startCol);
    template<int BlockRows, int BlockCols>
    const Block<Derived, BlockRows, BlockCols> block(int startRow, int startCol) const;
    
    Transpose<Derived> transpose();
    const Transpose<Derived> transpose() const;
    
    const Conjugate<Derived> conjugate() const;
    const Transpose<Conjugate<Derived> > adjoint() const;
    Scalar trace() const;
    
    template<typename OtherDerived>
    Scalar dot(const OtherDerived& other) const;
    RealScalar norm2() const;
    RealScalar norm()  const;
    const ScalarMultiple<RealScalar, Derived> normalized() const;
    template<typename OtherDerived>
    bool isOrtho(const OtherDerived& other,
                 const typename NumTraits<Scalar>::Real& prec) const;
    bool isOrtho(const typename NumTraits<Scalar>::Real& prec) const;
    
    static const Eval<Random<Derived> > random(int rows, int cols);
    static const Eval<Random<Derived> > random(int size);
    static const Eval<Random<Derived> > random();
    static const Zero<Derived> zero(int rows, int cols);
    static const Zero<Derived> zero(int size);
    static const Zero<Derived> zero();
    static const Ones<Derived> ones(int rows, int cols);
    static const Ones<Derived> ones(int size);
    static const Ones<Derived> ones();
    static const Identity<Derived> identity(int rows = Derived::RowsAtCompileTime);
    
    bool isZero(const typename NumTraits<Scalar>::Real& prec) const;
    bool isOnes(const typename NumTraits<Scalar>::Real& prec) const;
    bool isIdentity(const typename NumTraits<Scalar>::Real& prec) const;
    
    const DiagonalMatrix<Derived> asDiagonal() const;
    
    DiagonalCoeffs<Derived> diagonal();
    const DiagonalCoeffs<Derived> diagonal() const;
    
    template<typename OtherDerived>
    bool isApprox(
      const OtherDerived& other,
      const typename NumTraits<Scalar>::Real& prec
    ) const;
    bool isMuchSmallerThan(
      const typename NumTraits<Scalar>::Real& other,
      const typename NumTraits<Scalar>::Real& prec
    ) const;
    template<typename OtherDerived>
    bool isMuchSmallerThan(
      const MatrixBase<Scalar, OtherDerived>& other,
      const typename NumTraits<Scalar>::Real& prec
    ) const;
    
    template<typename OtherDerived>
    const Product<Derived, OtherDerived>
    lazyProduct(const MatrixBase<Scalar, OtherDerived>& other) const EIGEN_ALWAYS_INLINE;
    
    const Opposite<Derived> operator-() const;
    
    template<typename OtherDerived>
    Derived& operator+=(const MatrixBase<Scalar, OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const MatrixBase<Scalar, OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator*=(const MatrixBase<Scalar, OtherDerived>& other);
   
    Derived& operator*=(const int& other);
    Derived& operator*=(const float& other);
    Derived& operator*=(const double& other);
    Derived& operator*=(const std::complex<float>& other);
    Derived& operator*=(const std::complex<double>& other);
    
    Derived& operator/=(const int& other);
    Derived& operator/=(const float& other);
    Derived& operator/=(const double& other);
    Derived& operator/=(const std::complex<float>& other);
    Derived& operator/=(const std::complex<double>& other);

    Scalar coeff(int row, int col) const;
    Scalar operator()(int row, int col) const;
    
    Scalar& coeffRef(int row, int col);
    Scalar& operator()(int row, int col);
    
    Scalar coeff(int index) const;
    Scalar operator[](int index) const;
    
    Scalar& coeffRef(int index);
    Scalar& operator[](int index);
    
    Scalar x() const;
    Scalar y() const;
    Scalar z() const;
    Scalar w() const;
    Scalar& x();
    Scalar& y();
    Scalar& z();
    Scalar& w();

    const Eval<Derived> eval() const EIGEN_ALWAYS_INLINE;
};

#endif // EIGEN_MATRIXBASE_H
