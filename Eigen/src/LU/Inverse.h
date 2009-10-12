// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_INVERSE_H
#define EIGEN_INVERSE_H

/********************************************************************
*** Part 1 : optimized implementations for fixed-size 2,3,4 cases ***
********************************************************************/

template<typename XprType, typename MatrixType>
inline void ei_compute_inverse_size2_helper(
    const XprType& matrix, const typename MatrixType::Scalar& invdet,
    MatrixType* result)
{
  result->coeffRef(0,0) = matrix.coeff(1,1) * invdet;
  result->coeffRef(1,0) = -matrix.coeff(1,0) * invdet;
  result->coeffRef(0,1) = -matrix.coeff(0,1) * invdet;
  result->coeffRef(1,1) = matrix.coeff(0,0) * invdet;
}

template<typename MatrixType>
inline void ei_compute_inverse_size2(const MatrixType& matrix, MatrixType* result)
{
  typedef typename MatrixType::Scalar Scalar;
  const Scalar invdet = Scalar(1) / matrix.determinant();
  ei_compute_inverse_size2_helper( matrix, invdet, result );
}

template<typename XprType, typename MatrixType>
bool ei_compute_inverse_size2_with_check(const XprType& matrix, MatrixType* result)
{
  typedef typename MatrixType::Scalar Scalar;
  const Scalar det = matrix.determinant();
  if(ei_isMuchSmallerThan(det, matrix.cwise().abs().maxCoeff())) return false;
  const Scalar invdet = Scalar(1) / det;
  ei_compute_inverse_size2_helper( matrix, invdet, result );
  return true;
}

template<typename XprType, typename MatrixType>
void ei_compute_inverse_size3_helper(
    const XprType& matrix,
    const typename MatrixType::Scalar& invdet,
    const typename MatrixType::Scalar& det_minor00,
    const typename MatrixType::Scalar& det_minor10,
    const typename MatrixType::Scalar& det_minor20,
    MatrixType* result)
{
  result->coeffRef(0, 0) = det_minor00 * invdet;
  result->coeffRef(0, 1) = -det_minor10 * invdet;
  result->coeffRef(0, 2) = det_minor20 * invdet;
  result->coeffRef(1, 0) = -matrix.minor(0,1).determinant() * invdet;
  result->coeffRef(1, 1) = matrix.minor(1,1).determinant() * invdet;
  result->coeffRef(1, 2) = -matrix.minor(2,1).determinant() * invdet;
  result->coeffRef(2, 0) = matrix.minor(0,2).determinant() * invdet;
  result->coeffRef(2, 1) = -matrix.minor(1,2).determinant() * invdet;
  result->coeffRef(2, 2) = matrix.minor(2,2).determinant() * invdet;
}

template<bool Check, typename XprType, typename MatrixType>
bool ei_compute_inverse_size3(const XprType& matrix, MatrixType* result)
{
  typedef typename MatrixType::Scalar Scalar;
  const Scalar det_minor00 = matrix.minor(0,0).determinant();
  const Scalar det_minor10 = matrix.minor(1,0).determinant();
  const Scalar det_minor20 = matrix.minor(2,0).determinant();
  const Scalar det = ( det_minor00 * matrix.coeff(0,0)
      - det_minor10 * matrix.coeff(1,0)
      + det_minor20 * matrix.coeff(2,0) );
  if(Check) if(ei_isMuchSmallerThan(det, matrix.cwise().abs().maxCoeff())) return false;
  const Scalar invdet = Scalar(1) / det;
  ei_compute_inverse_size3_helper( matrix, invdet, det_minor00, det_minor10, det_minor20, result );
  return true;
}

template<typename MatrixType, typename ResultType>
bool ei_compute_inverse_size4_helper(const MatrixType& matrix, ResultType* result)
{
  /* Let's split M into four 2x2 blocks:
    * (P Q)
    * (R S)
    * If P is invertible, with inverse denoted by P_inverse, and if
    * (S - R*P_inverse*Q) is also invertible, then the inverse of M is
    * (P' Q')
    * (R' S')
    * where
    * S' = (S - R*P_inverse*Q)^(-1)
    * P' = P1 + (P1*Q) * S' *(R*P_inverse)
    * Q' = -(P_inverse*Q) * S'
    * R' = -S' * (R*P_inverse)
    */
  typedef Block<MatrixType,2,2> XprBlock22;
  typedef typename MatrixBase<XprBlock22>::PlainMatrixType Block22;
  Block22 P_inverse;
  if(ei_compute_inverse_size2_with_check(matrix.template block<2,2>(0,0), &P_inverse))
  {
    const Block22 Q = matrix.template block<2,2>(0,2);
    const Block22 P_inverse_times_Q = P_inverse * Q;
    const XprBlock22 R = matrix.template block<2,2>(2,0);
    const Block22 R_times_P_inverse = R * P_inverse;
    const Block22 R_times_P_inverse_times_Q = R_times_P_inverse * Q;
    const XprBlock22 S = matrix.template block<2,2>(2,2);
    const Block22 X = S - R_times_P_inverse_times_Q;
    Block22 Y;
    ei_compute_inverse_size2(X, &Y);
    result->template block<2,2>(2,2) = Y;
    result->template block<2,2>(2,0) = - Y * R_times_P_inverse;
    const Block22 Z = P_inverse_times_Q * Y;
    result->template block<2,2>(0,2) = - Z;
    result->template block<2,2>(0,0) = P_inverse + Z * R_times_P_inverse;
    return true;
  }
  else
  {
    return false;
  }
}

template<typename XprType, typename MatrixType>
bool ei_compute_inverse_size4_with_check(const XprType& matrix, MatrixType* result)
{
  if(ei_compute_inverse_size4_helper(matrix, result))
  {
    // good ! The topleft 2x2 block was invertible, so the 2x2 blocks approach is successful.
    return true;
  }
  else
  {
    // rare case: the topleft 2x2 block is not invertible (but the matrix itself is assumed to be).
    // since this is a rare case, we don't need to optimize it. We just want to handle it with little
    // additional code.
    MatrixType m(matrix);
    m.row(0).swap(m.row(2));
    m.row(1).swap(m.row(3));
    if(ei_compute_inverse_size4_helper(m, result))
    {
      // good, the topleft 2x2 block of m is invertible. Since m is different from matrix in that some
      // rows were permuted, the actual inverse of matrix is derived from the inverse of m by permuting
      // the corresponding columns.
      result->col(0).swap(result->col(2));
      result->col(1).swap(result->col(3));
    return true;
    }
    else
    {
      // first, undo the swaps previously made
      m.row(0).swap(m.row(2));
      m.row(1).swap(m.row(3));
      // swap row 0 with the the row among 0 and 1 that has the biggest 2 first coeffs
      int swap0with = ei_abs(m.coeff(0,0))+ei_abs(m.coeff(0,1))>ei_abs(m.coeff(1,0))+ei_abs(m.coeff(1,1)) ? 0 : 1;
      m.row(0).swap(m.row(swap0with));
      // swap row 1 with the the row among 2 and 3 that has the biggest 2 first coeffs
      int swap1with = ei_abs(m.coeff(2,0))+ei_abs(m.coeff(2,1))>ei_abs(m.coeff(3,0))+ei_abs(m.coeff(3,1)) ? 2 : 3;
      m.row(1).swap(m.row(swap1with));
      if( ei_compute_inverse_size4_helper(m, result) )
      {
        result->col(1).swap(result->col(swap1with));
        result->col(0).swap(result->col(swap0with));
        return true;
      }
      else
      {
        // non-invertible matrix
        return false;
      }
    }
  }
}



/***********************************************
*** Part 2 : selector and MatrixBase methods ***
***********************************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse
{
  static inline void run(const MatrixType& matrix, ResultType* result)
  {
    matrix.partialLu().computeInverse(result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 1>
{
  static inline void run(const MatrixType& matrix, ResultType* result)
  {
    typedef typename MatrixType::Scalar Scalar;
    result->coeffRef(0,0) = Scalar(1) / matrix.coeff(0,0);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 2>
{
  static inline void run(const MatrixType& matrix, ResultType* result)
  {
    ei_compute_inverse_size2(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 3>
{
  static inline void run(const MatrixType& matrix, ResultType* result)
  {
    ei_compute_inverse_size3<false, MatrixType, ResultType>(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 4>
{
  static inline void run(const MatrixType& matrix, ResultType* result)
  {
    ei_compute_inverse_size4_with_check(matrix, result);
  }
};

/** \lu_module
  *
  * Computes the matrix inverse of this matrix.
  *
  * \note This matrix must be invertible, otherwise the result is undefined. If you need an invertibility check, use
  * computeInverseWithCheck().
  *
  * \param result Pointer to the matrix in which to store the result.
  *
  * Example: \include MatrixBase_computeInverse.cpp
  * Output: \verbinclude MatrixBase_computeInverse.out
  *
  * \sa inverse(), computeInverseWithCheck()
  */
template<typename Derived>
template<typename ResultType>
inline void MatrixBase<Derived>::computeInverse(ResultType *result) const
{
  ei_assert(rows() == cols());
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::HasFloatingPoint,NUMERIC_TYPE_MUST_BE_FLOATING_POINT)
  ei_compute_inverse<PlainMatrixType, ResultType>::run(eval(), result);
}

/** \lu_module
  *
  * \returns the matrix inverse of this matrix.
  *
  * \note This matrix must be invertible, otherwise the result is undefined. If you need an invertibility check, use
  * computeInverseWithCheck().
  *
  * \note This method returns a matrix by value, which can be inefficient. To avoid that overhead,
  * use computeInverse() instead.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa computeInverse(), computeInverseWithCheck()
  */
template<typename Derived>
inline const typename MatrixBase<Derived>::PlainMatrixType MatrixBase<Derived>::inverse() const
{
  PlainMatrixType result(rows(), cols());
  computeInverse(&result);
  return result;
}


/********************************************
 * Compute inverse with invertibility check *
 *******************************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse_with_check
{
  static inline bool run(const MatrixType& matrix, ResultType* result)
  {
    typedef typename MatrixType::Scalar Scalar;
    LU<MatrixType> lu( matrix );
    if( !lu.isInvertible() ) return false;
    lu.computeInverse(result);
    return true;
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_with_check<MatrixType, ResultType, 1>
{
  static inline bool run(const MatrixType& matrix, ResultType* result)
  {
    typedef typename MatrixType::Scalar Scalar;
    if( matrix.coeff(0,0) == Scalar(0) ) return false;
    result->coeffRef(0,0) = Scalar(1) / matrix.coeff(0,0);
    return true;
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_with_check<MatrixType, ResultType, 2>
{
  static inline bool run(const MatrixType& matrix, ResultType* result)
  {
    return ei_compute_inverse_size2_with_check(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_with_check<MatrixType, ResultType, 3>
{
  static inline bool run(const MatrixType& matrix, ResultType* result)
  {
    return ei_compute_inverse_size3<true, MatrixType, ResultType>(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_with_check<MatrixType, ResultType, 4>
{
  static inline bool run(const MatrixType& matrix, ResultType* result)
  {
    return ei_compute_inverse_size4_with_check(matrix, result);
  }
};

/** \lu_module
  *
  * Computation of matrix inverse, with invertibility check.
  *
  * \returns true if the matrix is invertible, false otherwise.
  *
  * \param result Pointer to the matrix in which to store the result.
  *
  * \sa inverse(), computeInverse()
  */
template<typename Derived>
template<typename ResultType>
inline bool MatrixBase<Derived>::computeInverseWithCheck(ResultType *result) const
{
  ei_assert(rows() == cols());
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::HasFloatingPoint,NUMERIC_TYPE_MUST_BE_FLOATING_POINT)
  return ei_compute_inverse_with_check<PlainMatrixType, ResultType>::run(eval(), result);
}


#endif // EIGEN_INVERSE_H
