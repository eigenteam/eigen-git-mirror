// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

template<typename MatrixType, typename ResultType>
inline void ei_compute_inverse_size2_helper(
    const MatrixType& matrix, const typename ResultType::Scalar& invdet,
    ResultType& result)
{
  result.coeffRef(0,0) = matrix.coeff(1,1) * invdet;
  result.coeffRef(1,0) = -matrix.coeff(1,0) * invdet;
  result.coeffRef(0,1) = -matrix.coeff(0,1) * invdet;
  result.coeffRef(1,1) = matrix.coeff(0,0) * invdet;
}

template<typename MatrixType, typename ResultType>
inline void ei_compute_inverse_size2(const MatrixType& matrix, ResultType& result)
{
  typedef typename ResultType::Scalar Scalar;
  const Scalar invdet = typename MatrixType::Scalar(1) / matrix.determinant();
  ei_compute_inverse_size2_helper(matrix, invdet, result);
}

template<typename MatrixType, typename ResultType>
inline void ei_compute_inverse_and_det_size2_with_check(
  const MatrixType& matrix,
  const typename MatrixType::RealScalar& absDeterminantThreshold,
  ResultType& inverse,
  typename ResultType::Scalar& determinant,
  bool& invertible
  )
{
  typedef typename ResultType::Scalar Scalar;
  determinant = matrix.determinant();
  invertible = ei_abs(determinant) > absDeterminantThreshold;
  if(!invertible) return;
  const Scalar invdet = Scalar(1) / determinant;
  ei_compute_inverse_size2_helper(matrix, invdet, inverse);
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_size3_helper(
    const MatrixType& matrix,
    const typename ResultType::Scalar& invdet,
    const Matrix<typename ResultType::Scalar,3,1>& cofactors_col0,
    ResultType& result)
{
  result.row(0) = cofactors_col0 * invdet;
  result.coeffRef(1,0) = -matrix.minor(0,1).determinant() * invdet;
  result.coeffRef(1,1) = matrix.minor(1,1).determinant() * invdet;
  result.coeffRef(1,2) = -matrix.minor(2,1).determinant() * invdet;
  result.coeffRef(2,0) = matrix.minor(0,2).determinant() * invdet;
  result.coeffRef(2,1) = -matrix.minor(1,2).determinant() * invdet;
  result.coeffRef(2,2) = matrix.minor(2,2).determinant() * invdet;
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_size3(
  const MatrixType& matrix,
  ResultType& result)
{
  typedef typename ResultType::Scalar Scalar;
  Matrix<Scalar,3,1> cofactors_col0;
  cofactors_col0.coeffRef(0) = matrix.minor(0,0).determinant();
  cofactors_col0.coeffRef(1) = -matrix.minor(1,0).determinant();
  cofactors_col0.coeffRef(2) = matrix.minor(2,0).determinant();
  const Scalar det = (cofactors_col0.cwise()*matrix.col(0)).sum();
  const Scalar invdet = Scalar(1) / det;
  ei_compute_inverse_size3_helper(matrix, invdet, cofactors_col0, result);
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_and_det_size3_with_check(
  const MatrixType& matrix,
  const typename MatrixType::RealScalar& absDeterminantThreshold,
  ResultType& inverse,
  typename ResultType::Scalar& determinant,
  bool& invertible
  )
{
  typedef typename ResultType::Scalar Scalar;
  Matrix<Scalar,3,1> cofactors_col0;
  cofactors_col0.coeffRef(0) = matrix.minor(0,0).determinant();
  cofactors_col0.coeffRef(1) = -matrix.minor(1,0).determinant();
  cofactors_col0.coeffRef(2) = matrix.minor(2,0).determinant();
  determinant = (cofactors_col0.cwise()*matrix.col(0)).sum();
  invertible = ei_abs(determinant) > absDeterminantThreshold;
  if(!invertible) return;
  const Scalar invdet = Scalar(1) / determinant;
  ei_compute_inverse_size3_helper(matrix, invdet, cofactors_col0, inverse);
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_size4_helper(const MatrixType& matrix, ResultType& result)
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
  typedef Block<ResultType,2,2> XprBlock22;
  typedef typename MatrixBase<XprBlock22>::PlainMatrixType Block22;
  Block22 P_inverse;
  ei_compute_inverse_size2(matrix.template block<2,2>(0,0), P_inverse);
  const Block22 Q = matrix.template block<2,2>(0,2);
  const Block22 P_inverse_times_Q = P_inverse * Q;
  const XprBlock22 R = matrix.template block<2,2>(2,0);
  const Block22 R_times_P_inverse = R * P_inverse;
  const Block22 R_times_P_inverse_times_Q = R_times_P_inverse * Q;
  const XprBlock22 S = matrix.template block<2,2>(2,2);
  const Block22 X = S - R_times_P_inverse_times_Q;
  Block22 Y;
  ei_compute_inverse_size2(X, Y);
  result.template block<2,2>(2,2) = Y;
  result.template block<2,2>(2,0) = - Y * R_times_P_inverse;
  const Block22 Z = P_inverse_times_Q * Y;
  result.template block<2,2>(0,2) = - Z;
  result.template block<2,2>(0,0) = P_inverse + Z * R_times_P_inverse;
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_size4(const MatrixType& _matrix, ResultType& result)
{
  typedef typename ResultType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;

  // we will do row permutations on the matrix. This copy should have negligible cost.
  // if not, consider working in-place on the matrix (const-cast it, but then undo the permutations
  // to nevertheless honor constness)
  typename MatrixType::PlainMatrixType matrix(_matrix);

  // let's extract from the 2 first colums a 2x2 block whose determinant is as big as possible.
  int good_row0=0, good_row1=1;
  RealScalar good_absdet(-1);
  // this double for loop shouldn't be too costly: only 6 iterations
  for(int row0=0; row0<4; ++row0) {
    for(int row1=row0+1; row1<4; ++row1)
    {
      RealScalar absdet = ei_abs(matrix.coeff(row0,0)*matrix.coeff(row1,1)
                               - matrix.coeff(row0,1)*matrix.coeff(row1,0));
      if(absdet > good_absdet)
      {
        good_absdet = absdet;
        good_row0 = row0;
        good_row1 = row1;
      }
    }
  }
  // do row permutations to move this 2x2 block to the top
  matrix.row(0).swap(matrix.row(good_row0));
  matrix.row(1).swap(matrix.row(good_row1));
  // now applying our helper function is numerically stable
  ei_compute_inverse_size4_helper(matrix, result);
  // Since we did row permutations on the original matrix, we need to do column permutations
  // in the reverse order on the inverse
  result.col(1).swap(result.col(good_row1));
  result.col(0).swap(result.col(good_row0));
}

template<typename MatrixType, typename ResultType>
void ei_compute_inverse_and_det_size4_with_check(
  const MatrixType& matrix,
  const typename MatrixType::RealScalar& absDeterminantThreshold,
  ResultType& result,
  typename ResultType::Scalar& determinant,
  bool& invertible
  )
{
  determinant = matrix.determinant();
  invertible = ei_abs(determinant) > absDeterminantThreshold;
  if(invertible) ei_compute_inverse_size4(matrix, result);
}

/***********************************************
*** Part 2 : selectors and MatrixBase methods ***
***********************************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    result = matrix.partialLu().inverse();
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 1>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    typedef typename MatrixType::Scalar Scalar;
    result.coeffRef(0,0) = Scalar(1) / matrix.coeff(0,0);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 2>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    ei_compute_inverse_size2(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 3>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    ei_compute_inverse_size3(matrix, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 4>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    ei_compute_inverse_size4(matrix, result);
  }
};

/** \lu_module
  *
  * \returns the matrix inverse of this matrix.
  *
  * For small fixed sizes up to 4x4, this method uses ad-hoc methods (cofactors up to 3x3, Euler's trick for 4x4).
  * In the general case, this method uses class PartialLU.
  *
  * \note This matrix must be invertible, otherwise the result is undefined. If you need an
  * invertibility check, do the following:
  * \li for fixed sizes up to 4x4, use computeInverseAndDetWithCheck().
  * \li for the general case, use class LU.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa computeInverseAndDetWithCheck()
  */
template<typename Derived>
inline const typename MatrixBase<Derived>::PlainMatrixType MatrixBase<Derived>::inverse() const
{
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::HasFloatingPoint,NUMERIC_TYPE_MUST_BE_FLOATING_POINT)
  ei_assert(rows() == cols());
  typedef typename MatrixBase<Derived>::PlainMatrixType ResultType;
  ResultType result(rows(), cols());
  // for 2x2, it's worth giving a chance to avoid evaluating.
  // for larger sizes, evaluating has negligible cost and limits code size.
  typedef typename ei_meta_if<
    RowsAtCompileTime == 2,
    typename ei_cleantype<typename ei_nested<Derived,2>::type>::type,
    PlainMatrixType
  >::ret MatrixType;
  ei_compute_inverse<MatrixType, ResultType>::run(derived(), result);
  return result;
}


/********************************************
 * Compute inverse with invertibility check *
 *******************************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse_and_det_with_check {};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 1>
{
  static inline void run(
    const MatrixType& matrix,
    const typename MatrixType::RealScalar& absDeterminantThreshold,
    ResultType& result,
    typename ResultType::Scalar& determinant,
    bool& invertible
  )
  {
    determinant = matrix.coeff(0,0);
    invertible = ei_abs(determinant) > absDeterminantThreshold;
    if(invertible) result.coeffRef(0,0) = typename ResultType::Scalar(1) / determinant;
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 2>
{
  static inline void run(
    const MatrixType& matrix,
    const typename MatrixType::RealScalar& absDeterminantThreshold,
    ResultType& result,
    typename ResultType::Scalar& determinant,
    bool& invertible
  )
  {
    ei_compute_inverse_and_det_size2_with_check
      (matrix, absDeterminantThreshold, result, determinant, invertible);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 3>
{
  static inline void run(
    const MatrixType& matrix,
    const typename MatrixType::RealScalar& absDeterminantThreshold,
    ResultType& result,
    typename ResultType::Scalar& determinant,
    bool& invertible
  )
  {
    ei_compute_inverse_and_det_size3_with_check
      (matrix, absDeterminantThreshold, result, determinant, invertible);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 4>
{
  static inline void run(
    const MatrixType& matrix,
    const typename MatrixType::RealScalar& absDeterminantThreshold,
    ResultType& result,
    typename ResultType::Scalar& determinant,
    bool& invertible
  )
  {
    ei_compute_inverse_and_det_size4_with_check
      (matrix, absDeterminantThreshold, result, determinant, invertible);
  }
};

/** \lu_module
  *
  * Computation of matrix inverse and determinant, with invertibility check.
  *
  * This is only for fixed-size square matrices of size up to 4x4.
  *
  * \param inverse Reference to the matrix in which to store the inverse.
  * \param determinant Reference to the variable in which to store the inverse.
  * \param invertible Reference to the bool variable in which to store whether the matrix is invertible.
  * \param absDeterminantThreshold Optional parameter controlling the invertibility check.
  *                                The matrix will be declared invertible if the absolute value of its
  *                                determinant is greater than this threshold.
  *
  * \sa inverse()
  */
template<typename Derived>
template<typename ResultType>
inline void MatrixBase<Derived>::computeInverseAndDetWithCheck(
    ResultType& inverse,
    typename ResultType::Scalar& determinant,
    bool& invertible,
    const RealScalar& absDeterminantThreshold
  ) const
{
  // i'd love to put some static assertions there, but SFINAE means that they have no effect...
  ei_assert(rows() == cols());
  // for 2x2, it's worth giving a chance to avoid evaluating.
  // for larger sizes, evaluating has negligible cost and limits code size.
  typedef typename ei_meta_if<
    RowsAtCompileTime == 2,
    typename ei_cleantype<typename ei_nested<Derived, 2>::type>::type,
    PlainMatrixType
  >::ret MatrixType;
  ei_compute_inverse_and_det_with_check<MatrixType, ResultType>::run
    (derived(), absDeterminantThreshold, inverse, determinant, invertible);
}


#endif // EIGEN_INVERSE_H
