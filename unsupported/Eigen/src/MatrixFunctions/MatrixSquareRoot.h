// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
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

#ifndef EIGEN_MATRIX_SQUARE_ROOT
#define EIGEN_MATRIX_SQUARE_ROOT

/** \ingroup MatrixFunctions_Module
  * \brief Class for computing matrix square roots.
  * \tparam MatrixType type of the argument of the matrix square root,
  * expected to be an instantiation of the Matrix class template.
  */
template <typename MatrixType, int IsComplex = NumTraits<typename internal::traits<MatrixType>::Scalar>::IsComplex>
class MatrixSquareRoot
{  
  public:

    /** \brief Constructor. 
      *
      * \param[in]  A  matrix whose square root is to be computed.
      *
      * The class stores a reference to \p A, so it should not be
      * changed (or destroyed) before compute() is called.
      */
    MatrixSquareRoot(const MatrixType& A);

    /** \brief Compute the matrix square root
      *
      * \param[out] result  square root of \p A, as specified in the constructor.
      *
      * See MatrixBase::sqrt() for details on how this computation
      * is implemented.
      */
    template <typename ResultType> 
    void compute(ResultType &result);    
};


// ********** Partial specialization for real matrices **********

template <typename MatrixType>
class MatrixSquareRoot<MatrixType, 0>
{
  public:
    MatrixSquareRoot(const MatrixType& A) 
      : m_A(A) 
    {
      eigen_assert(A.rows() == A.cols());
    }

    template <typename ResultType> void compute(ResultType &result);    

 private:
    const MatrixType& m_A;
};

template <typename MatrixType>
template <typename ResultType> 
void MatrixSquareRoot<MatrixType, 0>::compute(ResultType &result)
{
  eigen_assert("Square root of real matrices is not implemented!");
}


// ********** Partial specialization for complex matrices **********

template <typename MatrixType>
class MatrixSquareRoot<MatrixType, 1>
{
  public:
    MatrixSquareRoot(const MatrixType& A) 
      : m_A(A) 
    {
      eigen_assert(A.rows() == A.cols());
    }

    template <typename ResultType> void compute(ResultType &result);    

 private:
    const MatrixType& m_A;
};

template <typename MatrixType>
template <typename ResultType> 
void MatrixSquareRoot<MatrixType, 1>::compute(ResultType &result)
{
  // Compute Schur decomposition of m_A
  const ComplexSchur<MatrixType> schurOfA(m_A);  
  const MatrixType& T = schurOfA.matrixT();
  const MatrixType& U = schurOfA.matrixU();

  // Compute square root of T and store it in upper triangular part of result
  // This uses that the square root of triangular matrices can be computed directly.
  result.resize(m_A.rows(), m_A.cols());
  typedef typename MatrixType::Index Index;
  for (Index i = 0; i < m_A.rows(); i++) {
    result.coeffRef(i,i) = internal::sqrt(T.coeff(i,i));
  }
  for (Index j = 1; j < m_A.cols(); j++) {
    for (Index i = j-1; i >= 0; i--) {
      typedef typename MatrixType::Scalar Scalar;
      // if i = j-1, then segment has length 0 so tmp = 0
      Scalar tmp = result.row(i).segment(i+1,j-i-1) * result.col(j).segment(i+1,j-i-1);
      // denominator may be zero if original matrix is singular
      result.coeffRef(i,j) = (T.coeff(i,j) - tmp) / (result.coeff(i,i) + result.coeff(j,j));
    }
  }

  // Compute square root of m_A as U * result * U.adjoint()
  MatrixType tmp;
  tmp.noalias() = U * result.template triangularView<Upper>();
  result.noalias() = tmp * U.adjoint();
}

#endif // EIGEN_MATRIX_FUNCTION
