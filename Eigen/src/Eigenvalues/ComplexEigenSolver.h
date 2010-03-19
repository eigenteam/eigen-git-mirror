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
  * \brief Eigen values/vectors solver for general complex matrices
  *
  * \param MatrixType the type of the matrix of which we are computing the eigen decomposition
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
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef std::complex<RealScalar> Complex;
    typedef typename ei_plain_col_type<MatrixType, Complex>::type EigenvalueType;
    typedef Matrix<Complex, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, ColsAtCompileTime> EigenvectorType;

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via ComplexEigenSolver::compute(const MatrixType&).
    */
    ComplexEigenSolver() : m_eivec(), m_eivalues(), m_isInitialized(false)
    {}

    ComplexEigenSolver(const MatrixType& matrix)
            : m_eivec(matrix.rows(),matrix.cols()),
              m_eivalues(matrix.cols()),
              m_isInitialized(false)
    {
      compute(matrix);
    }

    EigenvectorType eigenvectors(void) const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivec;
    }

    EigenvalueType eigenvalues() const
    {
      ei_assert(m_isInitialized && "ComplexEigenSolver is not initialized.");
      return m_eivalues;
    }

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
