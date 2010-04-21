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
  * The eigenvalues and eigenvectors of a matrix \f$ A \f$ are scalars
  * \f$ \lambda \f$ and vectors \f$ v \f$ such that \f$ Av = \lambda v
  * \f$.  If \f$ D \f$ is a diagonal matrix with the eigenvalues on
  * the diagonal, and \f$ V \f$ is a matrix with the eigenvectors as
  * its columns, then \f$ A V = V D \f$. The matrix \f$ V \f$ is
  * almost always invertible, in which case we have \f$ A = V D V^{-1}
  * \f$. This is called the eigendecomposition.
  *
  * The main function in this class is compute(), which computes the
  * eigenvalues and eigenvectors of a given function. The
  * documentation for that function contains an example showing the
  * main features of the class.
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
    typedef std::complex<RealScalar> ComplexScalar;

    /** \brief Type for vector of eigenvalues as returned by eigenvalues(). 
      *
      * This is a column vector with entries of type #ComplexScalar.
      * The length of the vector is the size of \p _MatrixType.
      */
    typedef Matrix<ComplexScalar, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> EigenvalueType;

    /** \brief Type for matrix of eigenvectors as returned by eigenvectors(). 
      *
      * This is a square matrix with entries of type #ComplexScalar. 
      * The size is the same as the size of \p _MatrixType.
      */
    typedef Matrix<ComplexScalar, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, ColsAtCompileTime> EigenvectorType;

    /** \brief Default constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute().
      */
    ComplexEigenSolver()
            : m_eivec(),
              m_eivalues(),
              m_schur(),
              m_isInitialized(false)
    {}
    
    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa ComplexEigenSolver()
      */
    ComplexEigenSolver(int size)
            : m_eivec(size, size),
              m_eivalues(size),
              m_schur(size),
              m_isInitialized(false)
    {}

    /** \brief Constructor; computes eigendecomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose eigendecomposition is to be computed.
      *
      * This constructor calls compute() to compute the eigendecomposition.
      */
    ComplexEigenSolver(const MatrixType& matrix)
            : m_eivec(matrix.rows(),matrix.cols()),
              m_eivalues(matrix.cols()),
              m_schur(matrix.rows()),
              m_isInitialized(false)
    {
      compute(matrix);
    }

    /** \brief Returns the eigenvectors of given matrix. 
      *
      * It is assumed that either the constructor
      * ComplexEigenSolver(const MatrixType& matrix) or the member
      * function compute(const MatrixType& matrix) has been called
      * before to compute the eigendecomposition of a matrix. This
      * function returns a matrix whose columns are the
      * eigenvectors. Column \f$ k \f$ is an eigenvector
      * corresponding to eigenvalue number \f$ k \f$ as returned by
      * eigenvalues().  The eigenvectors are normalized to have
      * (Euclidean) norm equal to one. The matrix returned by this
      * function is the matrix \f$ V \f$ in the eigendecomposition \f$
      * A = V D V^{-1} \f$, if it exists.
      *
      * Example: \include ComplexEigenSolver_eigenvectors.cpp
      * Output: \verbinclude ComplexEigenSolver_eigenvectors.out
      */
    EigenvectorType eigenvectors() const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivec;
    }

    /** \brief Returns the eigenvalues of given matrix. 
      *
      * It is assumed that either the constructor
      * ComplexEigenSolver(const MatrixType& matrix) or the member
      * function compute(const MatrixType& matrix) has been called
      * before to compute the eigendecomposition of a matrix. This
      * function returns a column vector containing the
      * eigenvalues. Eigenvalues are repeated according to their
      * algebraic multiplicity, so there are as many eigenvalues as
      * rows in the matrix.
      *
      * Example: \include ComplexEigenSolver_eigenvalues.cpp
      * Output: \verbinclude ComplexEigenSolver_eigenvalues.out
      */
    EigenvalueType eigenvalues() const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivalues;
    }

    /** \brief Computes eigendecomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose eigendecomposition is to be computed.
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
      * Example: \include ComplexEigenSolver_compute.cpp
      * Output: \verbinclude ComplexEigenSolver_compute.out
      */
    void compute(const MatrixType& matrix);

  protected:
    EigenvectorType m_eivec;
    EigenvalueType m_eivalues;
    ComplexSchur<MatrixType> m_schur;
    bool m_isInitialized;
};


template<typename MatrixType>
void ComplexEigenSolver<MatrixType>::compute(const MatrixType& matrix)
{
  // this code is inspired from Jampack
  assert(matrix.cols() == matrix.rows());
  const int n = matrix.cols();
  const RealScalar matrixnorm = matrix.norm();

  // Step 1: Do a complex Schur decomposition, A = U T U^*
  // The eigenvalues are on the diagonal of T.
  m_schur.compute(matrix);
  m_eivalues = m_schur.matrixT().diagonal();

  // Step 2: Compute X such that T = X D X^(-1), where D is the diagonal of T.
  // The matrix X is unit triangular.
  EigenvectorType X = EigenvectorType::Zero(n, n);
  for(int k=n-1 ; k>=0 ; k--)
  {
    X.coeffRef(k,k) = ComplexScalar(1.0,0.0);
    // Compute X(i,k) using the (i,k) entry of the equation X T = D X
    for(int i=k-1 ; i>=0 ; i--)
    {
      X.coeffRef(i,k) = -m_schur.matrixT().coeff(i,k);
      if(k-i-1>0)
        X.coeffRef(i,k) -= (m_schur.matrixT().row(i).segment(i+1,k-i-1) * X.col(k).segment(i+1,k-i-1)).value();
      ComplexScalar z = m_schur.matrixT().coeff(i,i) - m_schur.matrixT().coeff(k,k);
      if(z==ComplexScalar(0))
      {
	// If the i-th and k-th eigenvalue are equal, then z equals 0. 
	// Use a small value instead, to prevent division by zero.
        ei_real_ref(z) = NumTraits<RealScalar>::epsilon() * matrixnorm;
      }
      X.coeffRef(i,k) = X.coeff(i,k) / z;
    }
  }

  // Step 3: Compute V as V = U X; now A = U T U^* = U X D X^(-1) U^* = V D V^(-1)
  m_eivec = m_schur.matrixU() * X;
  // .. and normalize the eigenvectors
  for(int k=0 ; k<n ; k++)
  {
    m_eivec.col(k).normalize();
  }
  m_isInitialized = true;

  // Step 4: Sort the eigenvalues
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



#endif // EIGEN_COMPLEX_EIGEN_SOLVER_H
