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
template<typename Scalar, int Size, unsigned int SuggestedFlags> class ei_corrected_matrix_flags;

template<int _Rows, int _Cols> struct ei_size_at_compile_time;

template<typename _Scalar, int _Rows, int _Cols,
         int _MaxRows = _Rows, int _MaxCols = _Cols,
         unsigned int _Flags = ei_corrected_matrix_flags<
                                   _Scalar,
                                   ei_size_at_compile_time<_MaxRows,_MaxCols>::ret,
                                   EIGEN_DEFAULT_MATRIX_FLAGS
                               >::ret
>
class Matrix;

template<typename ExpressionType, unsigned int Added, unsigned int Removed> class Flagged;
template<typename ExpressionType> class NestByValue;
template<typename MatrixType> class Minor;
template<typename MatrixType, int BlockRows=Dynamic, int BlockCols=Dynamic> class Block;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename NullaryOp, typename MatrixType>         class CwiseNullaryOp;
template<typename UnaryOp,   typename MatrixType>         class CwiseUnaryOp;
template<typename BinaryOp,  typename Lhs, typename Rhs>  class CwiseBinaryOp;
template<typename Lhs, typename Rhs, int EvalMode=ei_product_eval_mode<Lhs,Rhs>::value> class Product;
template<typename CoeffsVectorType> class DiagonalMatrix;
template<typename MatrixType> class DiagonalCoeffs;
template<typename MatrixType> class Map;
template<int Direction, typename UnaryOp, typename MatrixType> class PartialRedux;
template<typename MatrixType, unsigned int Mode> class Part;
template<typename MatrixType, unsigned int Mode> class Extract;
template<typename Derived, bool HasArrayFlag = int(ei_traits<Derived>::Flags) & ArrayBit> class ArrayBase {};
template<typename Lhs, typename Rhs> class Cross;
template<typename Scalar> class Quaternion;
template<typename Scalar> class Rotation2D;
template<typename Scalar> class AngleAxis;
template<typename Scalar,int Dim> class Transform;


template<typename Scalar> struct ei_scalar_sum_op;
template<typename Scalar> struct ei_scalar_difference_op;
template<typename Scalar> struct ei_scalar_product_op;
template<typename Scalar> struct ei_scalar_quotient_op;
template<typename Scalar> struct ei_scalar_opposite_op;
template<typename Scalar> struct ei_scalar_conjugate_op;
template<typename Scalar> struct ei_scalar_real_op;
template<typename Scalar> struct ei_scalar_abs_op;
template<typename Scalar> struct ei_scalar_abs2_op;
template<typename Scalar> struct ei_scalar_sqrt_op;
template<typename Scalar> struct ei_scalar_exp_op;
template<typename Scalar> struct ei_scalar_log_op;
template<typename Scalar> struct ei_scalar_cos_op;
template<typename Scalar> struct ei_scalar_sin_op;
template<typename Scalar> struct ei_scalar_pow_op;
template<typename Scalar> struct ei_scalar_inverse_op;
template<typename Scalar, typename NewType> struct ei_scalar_cast_op;
template<typename Scalar, bool IsVectorizable> struct ei_scalar_multiple_op;
template<typename Scalar> struct ei_scalar_quotient1_op;
template<typename Scalar> struct ei_scalar_min_op;
template<typename Scalar> struct ei_scalar_max_op;
template<typename Scalar> struct ei_scalar_random_op;

template<typename Scalar>
void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride);

template<typename ExpressionType, bool CheckExistence = true> class Inverse;
template<typename MatrixType> class QR;

#endif // EIGEN_FORWARDDECLARATIONS_H
