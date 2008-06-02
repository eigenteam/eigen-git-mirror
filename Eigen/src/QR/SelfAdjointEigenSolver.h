// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SELFADJOINTEIGENSOLVER_H
#define EIGEN_SELFADJOINTEIGENSOLVER_H

/** \class SelfAdjointEigenSolver
  *
  * \brief Eigen values/vectors solver for selfadjoint matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the eigen decomposition
  *
  * \sa MatrixBase::eigenvalues(), class EigenSolver
  */
template<typename _MatrixType> class SelfAdjointEigenSolver
{
  public:

    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef std::complex<RealScalar> Complex;
//     typedef Matrix<RealScalar, MatrixType::ColsAtCompileTime, 1> EigenvalueType;
    typedef Matrix<RealScalar, MatrixType::ColsAtCompileTime, 1> RealVectorType;
    typedef Matrix<RealScalar, Dynamic, 1> RealVectorTypeX;

    SelfAdjointEigenSolver(const MatrixType& matrix)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_eivalues(matrix.cols())
    {
      compute(matrix);
    }

    void compute(const MatrixType& matrix);

    MatrixType eigenvectors(void) const { return m_eivec; }

    RealVectorType eigenvalues(void) const { return m_eivalues; }


  protected:
    MatrixType m_eivec;
    RealVectorType m_eivalues;
};

// from Golub's "Matrix Computations", algorithm 5.1.3
template<typename Scalar>
static void ei_givens_rotation(Scalar a, Scalar b, Scalar& c, Scalar& s)
{
  if (b==0)
  {
    c = 1; s = 0;
  }
  else if (ei_abs(b)>ei_abs(a))
  {
    Scalar t = -a/b;
    s = Scalar(1)/ei_sqrt(1+t*t);
    c = s * t;
  }
  else
  {
    Scalar t = -b/a;
    c = Scalar(1)/ei_sqrt(1+t*t);
    s = c * t;
  }
}

/** \internal
  * Performs a QR step on a tridiagonal symmetric matrix represented as a
  * pair of two vectors \a diag and \a subdiag.
  *
  * \param matA the input selfadjoint matrix
  * \param hCoeffs returned Householder coefficients
  *
  * For compilation efficiency reasons, this procedure does not use eigen expression
  * for its arguments.
  *
  * Implemented from Golub's "Matrix Computations", algorithm 8.3.2:
  * "implicit symmetric QR step with Wilkinson shift"
  */
template<typename Scalar>
static void ei_tridiagonal_qr_step(Scalar* diag, Scalar* subdiag, int n)
{
  Scalar td = (diag[n-2] - diag[n-1])*0.5;
  Scalar e2 = ei_abs2(subdiag[n-2]);
  Scalar mu = diag[n-1] - e2 / (td + (td>0 ? 1 : -1) * ei_sqrt(td*td + e2));
  Scalar x = diag[0] - mu;
  Scalar z = subdiag[0];

  for (int k = 0; k < n-1; ++k)
  {
    Scalar c, s;
    ei_givens_rotation(x, z, c, s);

    // do T = G' T G
    Scalar sdk = s * diag[k] + c * subdiag[k];
    Scalar dkp1 = s * subdiag[k] + c * diag[k+1];

    diag[k] = c * (c * diag[k] - s * subdiag[k]) - s * (c * subdiag[k] - s * diag[k+1]);
    diag[k+1] = s * sdk + c * dkp1;
    subdiag[k] = c * sdk - s * dkp1;

    if (k > 0)
      subdiag[k - 1] = c * subdiag[k-1] - s * z;

    x = subdiag[k];
    z = -s * subdiag[k+1];

    if (k < n - 2)
      subdiag[k + 1] = c * subdiag[k+1];
  }
}

template<typename MatrixType>
void SelfAdjointEigenSolver<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();
  m_eivalues.resize(n,1);
  m_eivec = matrix;

  Tridiagonalization<MatrixType> tridiag(m_eivec);
  RealVectorType& diag = m_eivalues;
  RealVectorTypeX subdiag(n-1);
  diag = tridiag.diagonal();
  subdiag = tridiag.subDiagonal();

  int end = n-1;
  int start = 0;
  while (end>0)
  {
    for (int i = start; i<end; ++i)
      if (ei_isMuchSmallerThan(ei_abs(subdiag[i]),(ei_abs(diag[i])+ei_abs(diag[i+1]))))
        subdiag[i] = 0;

    // find the largest unreduced block
    while (end>0 && subdiag[end-1]==0)
      end--;
    if (end<=0)
      break;
    start = end - 1;
    while (start>0 && subdiag[start-1]!=0)
      start--;

    ei_tridiagonal_qr_step(&diag.coeffRef(start), &subdiag.coeffRef(start), end-start+1);
  }

  std::cout << "ei values = " << m_eivalues.transpose() << "\n\n";
}

#endif // EIGEN_SELFADJOINTEIGENSOLVER_H
