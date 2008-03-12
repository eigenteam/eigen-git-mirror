// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either 
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of 
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public 
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_EVAL_H
#define EIGEN_EVAL_H

/** \class Eval
  *
  * \brief Evaluation of an expression
  *
  * The template parameter Expression is the type of the expression that we are evaluating.
  *
  * This class is the return
  * type of MatrixBase::eval() and most of the time this is the only way it
  * is used.
  *
  * However, if you want to write a function returning an evaluation of an expression, you
  * will need to use this class.
  *
  * Here is an example illustrating this:
  * \include class_Eval.cpp
  * Output: \verbinclude class_Eval.out
  *
  * \sa MatrixBase::eval()
  */
template<typename ExpressionType>
struct ei_traits<Eval<ExpressionType> >
{
  typedef typename ExpressionType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ExpressionType::RowsAtCompileTime,
    ColsAtCompileTime = ExpressionType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ExpressionType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ExpressionType::MaxColsAtCompileTime
  };
};

template<typename ExpressionType> class Eval : NoOperatorEquals,
  public Matrix< typename ExpressionType::Scalar,
                 ExpressionType::RowsAtCompileTime,
                 ExpressionType::ColsAtCompileTime,
                 EIGEN_DEFAULT_MATRIX_STORAGE_ORDER,
                 ExpressionType::MaxRowsAtCompileTime,
                 ExpressionType::MaxColsAtCompileTime>
{
  public:

    /** The actual matrix type to evaluate to. This type can be used independently
      * of the rest of this class to get the actual matrix type to evaluate and store
      * the value of an expression.
      *
      * Here is an example illustrating this:
      * \include Eval_MatrixType.cpp
      * Output: \verbinclude Eval_MatrixType.out
      */
    typedef Matrix<typename ExpressionType::Scalar,
                   ExpressionType::RowsAtCompileTime,
                   ExpressionType::ColsAtCompileTime,
                   EIGEN_DEFAULT_MATRIX_STORAGE_ORDER,
                   ExpressionType::MaxRowsAtCompileTime,
                   ExpressionType::MaxColsAtCompileTime> MatrixType;

    _EIGEN_BASIC_PUBLIC_INTERFACE(Eval, MatrixType)

    explicit Eval(const ExpressionType& expr) : MatrixType(expr) {}
};

/** Evaluates *this, which can be any expression, and returns the obtained matrix.
  *
  * A common use case for this is the following. In an expression-templates library
  * like Eigen, the coefficients of an expression are only computed as they are
  * accessed, they are not computed when the expression itself is constructed. This is
  * usually a good thing, as this "lazy evaluation" improves performance, but can also
  * in certain cases lead to wrong results and/or to redundant computations. In such
  * cases, one can restore the classical immediate-evaluation behavior by calling eval().
  *
  * Example: \include MatrixBase_eval.cpp
  * Output: \verbinclude MatrixBase_eval.out
  *
  * \sa class Eval */
template<typename Derived>
const Eval<Derived> MatrixBase<Derived>::eval() const
{
  return Eval<Derived>(*static_cast<const Derived*>(this));
}

#endif // EIGEN_EVAL_H
