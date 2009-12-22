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

/**********************************
*** General case implementation ***
**********************************/

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    result = matrix.partialPivLu().inverse();
  }
};

template<typename MatrixType, typename ResultType, int Size = MatrixType::RowsAtCompileTime>
struct ei_compute_inverse_and_det_with_check { /* nothing! general case not supported. */ };

/****************************
*** Size 1 implementation ***
****************************/

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

/****************************
*** Size 2 implementation ***
****************************/

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
struct ei_compute_inverse<MatrixType, ResultType, 2>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    typedef typename ResultType::Scalar Scalar;
    const Scalar invdet = typename MatrixType::Scalar(1) / matrix.determinant();
    ei_compute_inverse_size2_helper(matrix, invdet, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 2>
{
  static inline void run(
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
};

/****************************
*** Size 3 implementation ***
****************************/

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
struct ei_compute_inverse<MatrixType, ResultType, 3>
{
  static inline void run(const MatrixType& matrix, ResultType& result)
  {
    typedef typename ResultType::Scalar Scalar;
    Matrix<Scalar,3,1> cofactors_col0;
    cofactors_col0.coeffRef(0) = matrix.minor(0,0).determinant();
    cofactors_col0.coeffRef(1) = -matrix.minor(1,0).determinant();
    cofactors_col0.coeffRef(2) = matrix.minor(2,0).determinant();
    const Scalar det = (cofactors_col0.cwiseProduct(matrix.col(0))).sum();
    const Scalar invdet = Scalar(1) / det;
    ei_compute_inverse_size3_helper(matrix, invdet, cofactors_col0, result);
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 3>
{
  static inline void run(
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
    determinant = (cofactors_col0.cwiseProduct(matrix.col(0))).sum();
    invertible = ei_abs(determinant) > absDeterminantThreshold;
    if(!invertible) return;
    const Scalar invdet = Scalar(1) / determinant;
    ei_compute_inverse_size3_helper(matrix, invdet, cofactors_col0, inverse);
  }
};

/****************************
*** Size 4 implementation ***
****************************/

template<int Arch, typename Scalar, typename MatrixType, typename ResultType>
struct ei_compute_inverse_size4
{
  static void run(const MatrixType& matrix, ResultType& result)
  {
    result.coeffRef(0,0) = matrix.minor(0,0).determinant();
    result.coeffRef(1,0) = -matrix.minor(0,1).determinant();
    result.coeffRef(2,0) = matrix.minor(0,2).determinant();
    result.coeffRef(3,0) = -matrix.minor(0,3).determinant();
    result.coeffRef(0,2) = matrix.minor(2,0).determinant();
    result.coeffRef(1,2) = -matrix.minor(2,1).determinant();
    result.coeffRef(2,2) = matrix.minor(2,2).determinant();
    result.coeffRef(3,2) = -matrix.minor(2,3).determinant();
    result.coeffRef(0,1) = -matrix.minor(1,0).determinant();
    result.coeffRef(1,1) = matrix.minor(1,1).determinant();
    result.coeffRef(2,1) = -matrix.minor(1,2).determinant();
    result.coeffRef(3,1) = matrix.minor(1,3).determinant();
    result.coeffRef(0,3) = -matrix.minor(3,0).determinant();
    result.coeffRef(1,3) = matrix.minor(3,1).determinant();
    result.coeffRef(2,3) = -matrix.minor(3,2).determinant();
    result.coeffRef(3,3) = matrix.minor(3,3).determinant();
    result /= (matrix.col(0).cwise()*result.row(0).transpose()).sum();
  }
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse<MatrixType, ResultType, 4>
 : ei_compute_inverse_size4<Architecture::Target, typename MatrixType::Scalar,
                            MatrixType, ResultType>
{
  // FIXME empty?
};

template<typename MatrixType, typename ResultType>
struct ei_compute_inverse_and_det_with_check<MatrixType, ResultType, 4>
{
  static inline void run(
    const MatrixType& matrix,
    const typename MatrixType::RealScalar& absDeterminantThreshold,
    ResultType& inverse,
    typename ResultType::Scalar& determinant,
    bool& invertible
  )
  {
    determinant = matrix.determinant();
    invertible = ei_abs(determinant) > absDeterminantThreshold;
    if(invertible) ei_compute_inverse<MatrixType, ResultType>::run(matrix, inverse);
  }
};

/*************************
*** MatrixBase methods ***
*************************/

template<typename MatrixType>
struct ei_traits<ei_inverse_impl<MatrixType> >
{
  typedef typename MatrixType::PlainMatrixType ReturnMatrixType;
};

template<typename MatrixType>
struct ei_inverse_impl : public ReturnByValue<ei_inverse_impl<MatrixType> >
{
  // for 2x2, it's worth giving a chance to avoid evaluating.
  // for larger sizes, evaluating has negligible cost, limits code size,
  // and allows for vectorized paths.
  typedef typename ei_meta_if<
    MatrixType::RowsAtCompileTime == 2,
    typename ei_nested<MatrixType,2>::type,
    typename ei_eval<MatrixType>::type
  >::ret MatrixTypeNested;
  typedef typename ei_cleantype<MatrixTypeNested>::type MatrixTypeNestedCleaned;
  const MatrixTypeNested m_matrix;

  ei_inverse_impl(const MatrixType& matrix)
    : m_matrix(matrix)
  {}

  inline int rows() const { return m_matrix.rows(); }
  inline int cols() const { return m_matrix.cols(); }

  template<typename Dest> inline void evalTo(Dest& dst) const
  {
    ei_compute_inverse<MatrixTypeNestedCleaned, Dest>::run(m_matrix, dst);
  }
};

/** \lu_module
  *
  * \returns the matrix inverse of this matrix.
  *
  * For small fixed sizes up to 4x4, this method uses cofactors.
  * In the general case, this method uses class PartialPivLU.
  *
  * \note This matrix must be invertible, otherwise the result is undefined. If you need an
  * invertibility check, do the following:
  * \li for fixed sizes up to 4x4, use computeInverseAndDetWithCheck().
  * \li for the general case, use class FullPivLU.
  *
  * Example: \include MatrixBase_inverse.cpp
  * Output: \verbinclude MatrixBase_inverse.out
  *
  * \sa computeInverseAndDetWithCheck()
  */
template<typename Derived>
inline const ei_inverse_impl<Derived> MatrixBase<Derived>::inverse() const
{
  EIGEN_STATIC_ASSERT(NumTraits<Scalar>::HasFloatingPoint,NUMERIC_TYPE_MUST_BE_FLOATING_POINT)
  ei_assert(rows() == cols());
  return ei_inverse_impl<Derived>(derived());
}

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
  * Example: \include MatrixBase_computeInverseAndDetWithCheck.cpp
  * Output: \verbinclude MatrixBase_computeInverseAndDetWithCheck.out
  *
  * \sa inverse(), computeInverseWithCheck()
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

/** \lu_module
  *
  * Computation of matrix inverse, with invertibility check.
  *
  * This is only for fixed-size square matrices of size up to 4x4.
  *
  * \param inverse Reference to the matrix in which to store the inverse.
  * \param invertible Reference to the bool variable in which to store whether the matrix is invertible.
  * \param absDeterminantThreshold Optional parameter controlling the invertibility check.
  *                                The matrix will be declared invertible if the absolute value of its
  *                                determinant is greater than this threshold.
  *
  * Example: \include MatrixBase_computeInverseWithCheck.cpp
  * Output: \verbinclude MatrixBase_computeInverseWithCheck.out
  *
  * \sa inverse(), computeInverseAndDetWithCheck()
  */
template<typename Derived>
template<typename ResultType>
inline void MatrixBase<Derived>::computeInverseWithCheck(
    ResultType& inverse,
    bool& invertible,
    const RealScalar& absDeterminantThreshold
  ) const
{
  RealScalar determinant;
  // i'd love to put some static assertions there, but SFINAE means that they have no effect...
  ei_assert(rows() == cols());
  computeInverseAndDetWithCheck(inverse,determinant,invertible,absDeterminantThreshold);
}

#endif // EIGEN_INVERSE_H
