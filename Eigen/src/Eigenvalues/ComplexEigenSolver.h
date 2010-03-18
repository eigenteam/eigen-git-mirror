// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Claire Maurice
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_COMPLEX_EIGEN_SOLVER_H
#define EIGEN_COMPLEX_EIGEN_SOLVER_H

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class ComplexEigenSolver
  *
  * \brief Computes eigenvalues and eigenvectors of general complex matrices
  *
  * \tparam _MatrixType the type of the matrix of which we are
  * computing the eigendecomposition; this is expected to be an
  * instantiation of the Matrix class template.
  *
  * \sa class EigenSolver, class SelfAdjointEigenSolver
  */
template<typename _MatrixType> class ComplexEigenSolver
{
  public:
    typedef _MatrixType MatrixType;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      Options = MatrixType::Options,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    /** \brief Scalar type for matrices of type \p _MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    
    /** \brief Complex scalar type for \p _MatrixType. 
      *
      * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
      * \c float or \c double) and just \c Scalar if #Scalar is
      * complex.
      */
    typedef std::complex<RealScalar> Complex;

    /** \brief Type for vector of eigenvalues as returned by eigenvalues(). 
      *
      * This is a column vector with entries of type #Complex.
      * The length of the vector is the size of \p _MatrixType.
      */
    typedef Matrix<Complex, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> EigenvalueType;

    /** \brief Type for matrix of eigenvectors as returned by eigenvectors(). 
      *
      * This is a square matrix with entries of type #Complex. 
      * The size is the same as the size of \p _MatrixType.
      */
    typedef Matrix<Complex, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, ColsAtCompileTime> EigenvectorType;

    /** \brief Default constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute().
      */
    ComplexEigenSolver() : m_eivec(), m_eivalues(), m_isInitialized(false)
    {}

    /** \brief Constructor; computes eigendecomposition of given matrix. 
      *
      * This constructor calls compute() to compute the eigendecomposition.
      * 
      * \param[in]  matrix  %Matrix whose eigendecomposition is to be computed.
      */
    ComplexEigenSolver(const MatrixType& matrix)
            : m_eivec(matrix.rows(),matrix.cols()),
              m_eivalues(matrix.cols()),
              m_isInitialized(false)
    {
      compute(matrix);
    }

    /** \brief Returns the eigenvectors of given matrix. */
    EigenvectorType eigenvectors() const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivec;
    }

    /** \brief Returns the eigenvalues of given matrix. */
    EigenvalueType eigenvalues() const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivalues;
    }

    /** \brief Computes eigendecomposition of given matrix. 
      *
      * This function computes the eigenvalues and eigenvectors of \p
      * matrix.  The eigenvalues() and eigenvectors() functions can be
      * used to retrieve the computed eigendecomposition.
      *
      * The matrix is first reduced to Schur form using the
      * ComplexSchur class. The Schur decomposition is then used to
      * compute the eigenvalues and eigenvectors.
      *
      * The cost of the computation is dominated by the cost of the
      * Schur decomposition, which is \f$ O(n^3) \f$ where \f$ n \f$
      * is the size of the matrix.
      * 
      * \param[in]  matrix  %Matrix whose eigendecomposition is to be computed.
      */
    void compute(const MatrixType& matrix);

  protected:
    MatrixType m_eivec;
    EigenvalueType m_eivalues;
    bool m_isInitialized;
};


template<typename MatrixType>
void ComplexEigenSolver<MatrixType>::compute(const MatrixType& matrix)
{
  // this code is inspired from Jampack
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();
  m_eivalues.resize(n,1);
  m_eivec.resize(n,n);

  RealScalar eps = NumTraits<RealScalar>::epsilon();

  // Reduce to complex Schur form
  ComplexSchur<MatrixType> schur(matrix);

  m_eivalues = schur.matrixT().diagonal();

  m_eivec.setZero();

  Scalar d2, z;
  RealScalar norm = matrix.norm();

  // compute the (normalized) eigenvectors
  for(int k=n-1 ; k>=0 ; k--)
  {
    d2 = schur.matrixT().coeff(k,k);
    m_eivec.coeffRef(k,k) = Scalar(1.0,0.0);
    for(int i=k-1 ; i>=0 ; i--)
    {
      m_eivec.coeffRef(i,k) = -schur.matrixT().coeff(i,k);
      if(k-i-1>0)
        m_eivec.coeffRef(i,k) -= (schur.matrixT().row(i).segment(i+1,k-i-1) * m_eivec.col(k).segment(i+1,k-i-1)).value();
      z = schur.matrixT().coeff(i,i) - d2;
      if(z==Scalar(0))
        ei_real_ref(z) = eps * norm;
      m_eivec.coeffRef(i,k) = m_eivec.coeff(i,k) / z;

    }
    m_eivec.col(k).normalize();
  }

  m_eivec = schur.matrixU() * m_eivec;
  m_isInitialized = true;

  // sort the eigenvalues
  {
    for (int i=0; i<n; i++)
    {
      int k;
      m_eivalues.cwiseAbs().tail(n-i).minCoeff(&k);
      if (k != 0)
      {
        k += i;
        std::swap(m_eivalues[k],m_eivalues[i]);
        m_eivec.col(i).swap(m_eivec.col(k));
      }
    }
  }
}



#endif // EIGEN_COMPLEX_EIGEN_SOLVER_H
