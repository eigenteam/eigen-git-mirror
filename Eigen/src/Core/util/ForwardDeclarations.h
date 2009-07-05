// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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
template<typename T> struct NumTraits;

template<typename _Scalar, int _Rows, int _Cols,
         int _Options = EIGEN_DEFAULT_MATRIX_STORAGE_ORDER_OPTION | AutoAlign,
         int _MaxRows = _Rows, int _MaxCols = _Cols> class Matrix;

template<typename ExpressionType, unsigned int Added, unsigned int Removed> class Flagged;
template<typename ExpressionType> class NestByValue;
template<typename ExpressionType> class SwapWrapper;
template<typename MatrixType> class Minor;
template<typename MatrixType, int BlockRows=Dynamic, int BlockCols=Dynamic, int PacketAccess=AsRequested,
         int _DirectAccessStatus = ei_traits<MatrixType>::Flags&DirectAccessBit ? DirectAccessBit
                                 : ei_traits<MatrixType>::Flags&SparseBit> class Block;
template<typename MatrixType, int Size=Dynamic, int PacketAccess=AsRequested> class VectorBlock;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename NullaryOp, typename MatrixType>         class CwiseNullaryOp;
template<typename UnaryOp,   typename MatrixType>         class CwiseUnaryOp;
template<typename ViewOp,    typename MatrixType>         class CwiseUnaryView;
template<typename BinaryOp,  typename Lhs, typename Rhs>  class CwiseBinaryOp;
template<typename Lhs, typename Rhs, int ProductMode> class Product;
template<typename CoeffsVectorType, typename Derived> class DiagonalMatrixBase;
template<typename CoeffsVectorType> class DiagonalMatrixWrapper;
template<typename _Scalar, int _Size> class DiagonalMatrix;
template<typename MatrixType, int Index> class Diagonal;
template<typename MatrixType, int PacketAccess = AsRequested> class Map;
template<typename MatrixType, unsigned int Mode> class Part;
template<typename MatrixType, unsigned int Mode> class Extract;
template<typename ExpressionType> class Cwise;
template<typename ExpressionType> class WithFormat;
template<typename MatrixType> struct CommaInitializer;
template<typename Functor, typename EvalType> class ReturnByValue;


template<typename Lhs, typename Rhs> struct ei_product_mode;
template<typename Lhs, typename Rhs, int ProductMode = ei_product_mode<Lhs,Rhs>::value> struct ProductReturnType;

template<typename Scalar> struct ei_scalar_sum_op;
template<typename Scalar> struct ei_scalar_difference_op;
template<typename Scalar> struct ei_scalar_product_op;
template<typename Scalar> struct ei_scalar_quotient_op;
template<typename Scalar> struct ei_scalar_opposite_op;
template<typename Scalar> struct ei_scalar_conjugate_op;
template<typename Scalar> struct ei_scalar_real_op;
template<typename Scalar> struct ei_scalar_imag_op;
template<typename Scalar> struct ei_scalar_abs_op;
template<typename Scalar> struct ei_scalar_abs2_op;
template<typename Scalar> struct ei_scalar_sqrt_op;
template<typename Scalar> struct ei_scalar_exp_op;
template<typename Scalar> struct ei_scalar_log_op;
template<typename Scalar> struct ei_scalar_cos_op;
template<typename Scalar> struct ei_scalar_sin_op;
template<typename Scalar> struct ei_scalar_pow_op;
template<typename Scalar> struct ei_scalar_inverse_op;
template<typename Scalar> struct ei_scalar_square_op;
template<typename Scalar> struct ei_scalar_cube_op;
template<typename Scalar, typename NewType> struct ei_scalar_cast_op;
template<typename Scalar> struct ei_scalar_multiple_op;
template<typename Scalar> struct ei_scalar_quotient1_op;
template<typename Scalar> struct ei_scalar_min_op;
template<typename Scalar> struct ei_scalar_max_op;
template<typename Scalar> struct ei_scalar_random_op;
template<typename Scalar> struct ei_scalar_add_op;
template<typename Scalar> struct ei_scalar_constant_op;
template<typename Scalar> struct ei_scalar_identity_op;

template<typename Scalar1,typename Scalar2> struct ei_scalar_multiple2_op;

struct IOFormat;

template<typename Scalar>
void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride);

// Array module
template<typename ConditionMatrixType, typename ThenMatrixType, typename ElseMatrixType> class Select;
template<typename MatrixType, typename BinaryOp, int Direction> class PartialReduxExpr;
template<typename ExpressionType, int Direction> class VectorwiseOp;
template<typename MatrixType,int RowFactor,int ColFactor> class Replicate;
template<typename MatrixType, int Direction = BothDirections> class Reverse;

template<typename MatrixType> class LU;
template<typename MatrixType> class PartialLU;
template<typename MatrixType> class QR;
template<typename MatrixType> class SVD;
template<typename MatrixType> class LLT;
template<typename MatrixType> class LDLT;

// Geometry module:
template<typename Derived, int _Dim> class RotationBase;
template<typename Lhs, typename Rhs> class Cross;
template<typename Scalar> class Quaternion;
template<typename Scalar> class Rotation2D;
template<typename Scalar> class AngleAxis;
template<typename Scalar,int Dim,int Mode=Affine> class Transform;
template <typename _Scalar, int _AmbientDim> class ParametrizedLine;
template <typename _Scalar, int _AmbientDim> class Hyperplane;
template<typename Scalar,int Dim> class Translation;
template<typename Scalar> class UniformScaling;
template<typename MatrixType,int Direction> class Homogeneous;

// Sparse module:
template<typename Lhs, typename Rhs, int ProductMode> class SparseProduct;

#endif // EIGEN_FORWARDDECLARATIONS_H
