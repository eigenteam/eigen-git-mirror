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

template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols> class Matrix;
template<typename MatrixType> class MatrixRef;
template<typename MatrixType> class Minor;
template<typename MatrixType, int BlockRows=Dynamic, int BlockCols=Dynamic> class Block;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename BinaryOp, typename Lhs, typename Rhs> class CwiseBinaryOp;
template<typename UnaryOp, typename MatrixType> class CwiseUnaryOp;
template<typename Lhs, typename Rhs> class Product;
template<typename MatrixType> class Random;
template<typename MatrixType> class Zero;
template<typename MatrixType> class Ones;
template<typename CoeffsVectorType> class DiagonalMatrix;
template<typename MatrixType> class DiagonalCoeffs;
template<typename MatrixType> class Identity;
template<typename MatrixType> class Map;
template<typename Derived> class Eval;
template<typename Derived> class EvalOMP;

struct ei_scalar_sum_op;
struct ei_scalar_difference_op;
struct ei_scalar_product_op;
struct ei_scalar_quotient_op;
struct ei_scalar_opposite_op;
struct ei_scalar_conjugate_op;
struct ei_scalar_abs_op;
struct ei_scalar_abs2_op;
struct ei_scalar_sqrt_op;
struct ei_scalar_exp_op;
struct ei_scalar_log_op;
struct ei_scalar_cos_op;
struct ei_scalar_sin_op;
template<typename Scalar> struct ei_scalar_pow_op;
template<typename NewType> struct ei_scalar_cast_op;
template<typename Scalar>  struct ei_scalar_multiple_op;

template<typename T> struct ei_xpr_copy
{
  typedef T Type;
};

template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols>
struct ei_xpr_copy<Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> >
{
  typedef const Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> & Type;
};

#endif // EIGEN_FORWARDDECLARATIONS_H
