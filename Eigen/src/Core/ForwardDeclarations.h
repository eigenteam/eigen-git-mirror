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

#ifndef EIGEN_FORWARDDECLARATIONS_H
#define EIGEN_FORWARDDECLARATIONS_H

template<typename T> struct ei_traits;
template<typename Lhs, typename Rhs> struct ei_product_eval_mode;
template<typename T> struct NumTraits;

template<typename _Scalar, int _Rows, int _Cols, unsigned int _Flags, int _MaxRows, int _MaxCols> class Matrix;
template<typename ExpressionType> class Lazy;
template<typename ExpressionType> class Temporary;
template<typename MatrixType> class Minor;
template<typename MatrixType, int BlockRows=Dynamic, int BlockCols=Dynamic> class Block;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename BinaryOp, typename Lhs, typename Rhs> class CwiseBinaryOp;
template<typename UnaryOp, typename MatrixType> class CwiseUnaryOp;
template<typename Lhs, typename Rhs, int EvalMode=ei_product_eval_mode<Lhs,Rhs>::value> class Product;
template<typename MatrixType> class Random;
template<typename MatrixType> class Zero;
template<typename MatrixType> class Ones;
template<typename CoeffsVectorType> class DiagonalMatrix;
template<typename MatrixType> class DiagonalCoeffs;
template<typename MatrixType> class Identity;
template<typename MatrixType> class Map;
template<typename Derived> class Eval;
template<typename Derived> class EvalOMP;
template<int Direction, typename UnaryOp, typename MatrixType> class PartialRedux;

template<typename Scalar> struct ei_scalar_sum_op;
template<typename Scalar> struct ei_scalar_difference_op;
template<typename Scalar> struct ei_scalar_product_op;
template<typename Scalar> struct ei_scalar_quotient_op;
template<typename Scalar> struct ei_scalar_opposite_op;
template<typename Scalar> struct ei_scalar_conjugate_op;
template<typename Scalar> struct ei_scalar_abs_op;
template<typename Scalar> struct ei_scalar_abs2_op;
template<typename Scalar> struct ei_scalar_sqrt_op;
template<typename Scalar> struct ei_scalar_exp_op;
template<typename Scalar> struct ei_scalar_log_op;
template<typename Scalar> struct ei_scalar_cos_op;
template<typename Scalar> struct ei_scalar_sin_op;
template<typename Scalar> struct ei_scalar_pow_op;
template<typename Scalar, typename NewType> struct ei_scalar_cast_op;
template<typename Scalar> struct ei_scalar_multiple_op;
template<typename Scalar> struct ei_scalar_quotient1_op;
template<typename Scalar> struct ei_scalar_min_op;
template<typename Scalar> struct ei_scalar_max_op;

template<typename T> struct ei_eval
{
  typedef Matrix<typename ei_traits<T>::Scalar,
                 ei_traits<T>::RowsAtCompileTime,
                 ei_traits<T>::ColsAtCompileTime,
                 ei_traits<T>::Flags & ~(EvalBeforeNestingBit | EvalBeforeAssigningBit),
                 ei_traits<T>::MaxRowsAtCompileTime,
                 ei_traits<T>::MaxColsAtCompileTime> type;
};

template<typename T> struct ei_unref { typedef T type; };
template<typename T> struct ei_unref<T&> { typedef T type; };

template<typename T> struct ei_xpr_copy
{
  typedef typename ei_meta_if< ei_traits<T>::Flags & EvalBeforeNestingBit,
      typename ei_eval<T>::type, const T&>::ret type;
};

template<typename T> struct ei_xpr_copy<Temporary<T> >
{
  typedef Temporary<T> type;
};

template<typename T, int n=1> struct ei_eval_if_needed_before_nesting
{
  // FIXME should we consider the additional store as well as the creation cost of the temporary ?
  enum { eval = T::Flags & EvalBeforeNestingBit
             || (n+1) * NumTraits<typename ei_traits<T>::Scalar>::ReadCost < (n-1) * T::CoeffReadCost };
  typedef typename ei_meta_if<eval, typename ei_eval<T>::type, T>::ret XprType;
  typedef typename ei_meta_if<eval, typename ei_eval<T>::type, typename T::XprCopy>::ret CopyType;
};


template<typename T> struct ei_functor_traits
{
  enum
  {
    Cost = 10,
    IsVectorizable = false
  };
};


#endif // EIGEN_FORWARDDECLARATIONS_H
