// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2010 Jitse Niesen <jitse@maths.leeds.ac.uk>
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

#ifndef EIGEN_REAL_SCHUR_H
#define EIGEN_REAL_SCHUR_H

#include "./HessenbergDecomposition.h"

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class RealSchur
  *
  * \brief Performs a real Schur decomposition of a square matrix
  */
template<typename _MatrixType> class RealSchur
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
    typedef std::complex<typename NumTraits<Scalar>::Real> ComplexScalar;
    typedef Matrix<ComplexScalar, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> EigenvalueType;

    /** \brief Constructor; computes Schur decomposition of given matrix. */
    RealSchur(const MatrixType& matrix)
            : m_matT(matrix.rows(),matrix.cols()),
              m_matU(matrix.rows(),matrix.cols()),
              m_eivalues(matrix.rows()),
              m_isInitialized(false)
    {
      compute(matrix);
    }

    /** \brief Returns the orthogonal matrix in the Schur decomposition. */
    const MatrixType& matrixU() const
    {
      ei_assert(m_isInitialized && "RealSchur is not initialized.");
      return m_matU;
    }

    /** \brief Returns the quasi-triangular matrix in the Schur decomposition. */
    const MatrixType& matrixT() const
    {
      ei_assert(m_isInitialized && "RealSchur is not initialized.");
      return m_matT;
    }
  
    /** \brief Returns vector of eigenvalues. 
      *
      * This function will likely be removed. */
    const EigenvalueType& eigenvalues() const
    {
      ei_assert(m_isInitialized && "RealSchur is not initialized.");
      return m_eivalues;
    }
  
    /** \brief Computes Schur decomposition of given matrix. */
    void compute(const MatrixType& matrix);

  private:
    
    MatrixType m_matT;
    MatrixType m_matU;
    EigenvalueType m_eivalues;
    bool m_isInitialized;

    typedef Matrix<Scalar,3,1> Vector3s;

    Scalar computeNormOfT();
    int findSmallSubdiagEntry(int n, Scalar norm);
    void computeShift(Scalar& x, Scalar& y, Scalar& w, int iu, Scalar& exshift, int iter);
    void findTwoSmallSubdiagEntries(Scalar x, Scalar y, Scalar w, int il, int& m, int iu, Vector3s& firstHouseholderVector);
    void doFrancisStep(int il, int m, int iu, const Vector3s& firstHouseholderVector, Scalar* workspace);
    void splitOffTwoRows(int iu, Scalar exshift);
};


template<typename MatrixType>
void RealSchur<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.cols() == matrix.rows());

  // Step 1. Reduce to Hessenberg form
  // TODO skip Q if skipU = true
  HessenbergDecomposition<MatrixType> hess(matrix);
  m_matT = hess.matrixH();
  m_matU = hess.matrixQ();

  // Step 2. Reduce to real Schur form  
  typedef Matrix<Scalar, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> ColumnVectorType;
  ColumnVectorType workspaceVector(m_matU.cols());
  Scalar* workspace = &workspaceVector.coeffRef(0);

  // The matrix m_matT is divided in three parts. 
  // Rows 0,...,il-1 are decoupled from the rest because m_matT(il,il-1) is zero. 
  // Rows il,...,iu is the part we are working on (the active window).
  // Rows iu+1,...,end are already brought in triangular form.
  int iu = m_matU.cols() - 1;
  Scalar exshift = 0.0;
  Scalar norm = computeNormOfT();

  int iter = 0;
  while (iu >= 0)
  {
    int il = findSmallSubdiagEntry(iu, norm);

    // Check for convergence
    if (il == iu) // One root found
    {
      m_matT.coeffRef(iu,iu) = m_matT.coeff(iu,iu) + exshift;
      m_eivalues.coeffRef(iu) = ComplexScalar(m_matT.coeff(iu,iu), 0.0);
      iu--;
      iter = 0;
    }
    else if (il == iu-1) // Two roots found
    {
      splitOffTwoRows(iu, exshift);
      iu -= 2;
      iter = 0;
    }
    else // No convergence yet
    {
      Scalar x, y, w;
      Vector3s firstHouseholderVector;
      computeShift(x, y, w, iu, exshift, iter);
      iter = iter + 1;   // (Could check iteration count here.)
      int m;
      findTwoSmallSubdiagEntries(x, y, w, il, m, iu, firstHouseholderVector);
      doFrancisStep(il, m, iu, firstHouseholderVector, workspace);
    }  // check convergence
  }  // while (iu >= 0)

  m_isInitialized = true;
}

// Compute matrix norm
template<typename MatrixType>
inline typename MatrixType::Scalar RealSchur<MatrixType>::computeNormOfT()
{
  const int size = m_matU.cols();
  // FIXME to be efficient the following would requires a triangular reduxion code
  // Scalar norm = m_matT.upper().cwiseAbs().sum() + m_matT.corner(BottomLeft,size-1,size-1).diagonal().cwiseAbs().sum();
  Scalar norm = 0.0;
  for (int j = 0; j < size; ++j)
    norm += m_matT.row(j).segment(std::max(j-1,0), size-std::max(j-1,0)).cwiseAbs().sum();
  return norm;
}

// Look for single small sub-diagonal element
template<typename MatrixType>
inline int RealSchur<MatrixType>::findSmallSubdiagEntry(int iu, Scalar norm)
{
  int res = iu;
  while (res > 0)
  {
    Scalar s = ei_abs(m_matT.coeff(res-1,res-1)) + ei_abs(m_matT.coeff(res,res));
    if (s == 0.0)
      s = norm;
    if (ei_abs(m_matT.coeff(res,res-1)) < NumTraits<Scalar>::epsilon() * s)
      break;
    res--;
  }
  return res;
}

template<typename MatrixType>
inline void RealSchur<MatrixType>::splitOffTwoRows(int iu, Scalar exshift)
{
  const int size = m_matU.cols();
  Scalar w = m_matT.coeff(iu,iu-1) * m_matT.coeff(iu-1,iu);
  Scalar p = (m_matT.coeff(iu-1,iu-1) - m_matT.coeff(iu,iu)) * Scalar(0.5);
  Scalar q = p * p + w;
  Scalar z = ei_sqrt(ei_abs(q));
  m_matT.coeffRef(iu,iu) = m_matT.coeff(iu,iu) + exshift;
  m_matT.coeffRef(iu-1,iu-1) = m_matT.coeff(iu-1,iu-1) + exshift;
  Scalar x = m_matT.coeff(iu,iu);

  // Scalar pair
  if (q >= 0)
  {
    if (p >= 0)
      z = p + z;
    else
      z = p - z;

    m_eivalues.coeffRef(iu-1) = ComplexScalar(x + z, 0.0);
    m_eivalues.coeffRef(iu) = ComplexScalar(z!=0.0 ? x - w / z : m_eivalues.coeff(iu-1).real(), 0.0);

    PlanarRotation<Scalar> rot;
    rot.makeGivens(z, m_matT.coeff(iu, iu-1));
    m_matT.block(0, iu-1, size, size-iu+1).applyOnTheLeft(iu-1, iu, rot.adjoint());
    m_matT.block(0, 0, iu+1, size).applyOnTheRight(iu-1, iu, rot);
    m_matU.applyOnTheRight(iu-1, iu, rot);
  }
  else // Complex pair
  {
    m_eivalues.coeffRef(iu-1) = ComplexScalar(x + p, z);
    m_eivalues.coeffRef(iu)   = ComplexScalar(x + p, -z);
  }
}

// Form shift
template<typename MatrixType>
inline void RealSchur<MatrixType>::computeShift(Scalar& x, Scalar& y, Scalar& w, int iu, Scalar& exshift, int iter)
{
  x = m_matT.coeff(iu,iu);
  y = m_matT.coeff(iu-1,iu-1);
  w = m_matT.coeff(iu,iu-1) * m_matT.coeff(iu-1,iu);

  // Wilkinson's original ad hoc shift
  if (iter == 10)
  {
    exshift += x;
    for (int i = 0; i <= iu; ++i)
      m_matT.coeffRef(i,i) -= x;
    Scalar s = ei_abs(m_matT.coeff(iu,iu-1)) + ei_abs(m_matT.coeff(iu-1,iu-2));
    x = y = Scalar(0.75) * s;
    w = Scalar(-0.4375) * s * s;
  }

  // MATLAB's new ad hoc shift
  if (iter == 30)
  {
    Scalar s = Scalar((y - x) / 2.0);
    s = s * s + w;
    if (s > 0)
    {
      s = ei_sqrt(s);
      if (y < x)
        s = -s;
      s = Scalar(x - w / ((y - x) / 2.0 + s));
      for (int i = 0; i <= iu; ++i)
        m_matT.coeffRef(i,i) -= s;
      exshift += s;
      x = y = w = Scalar(0.964);
    }
  }
}

// Look for two consecutive small sub-diagonal elements
template<typename MatrixType>
inline void RealSchur<MatrixType>::findTwoSmallSubdiagEntries(Scalar x, Scalar y, Scalar w, int il, int& m, int iu, Vector3s& firstHouseholderVector)
{
  Scalar p = 0, q = 0, r = 0;

  m = iu-2;
  while (m >= il)
  {
    Scalar z = m_matT.coeff(m,m);
    r = x - z;
    Scalar s = y - z;
    p = (r * s - w) / m_matT.coeff(m+1,m) + m_matT.coeff(m,m+1);
    q = m_matT.coeff(m+1,m+1) - z - r - s;
    r = m_matT.coeff(m+2,m+1);
    s = ei_abs(p) + ei_abs(q) + ei_abs(r);
    p = p / s;
    q = q / s;
    r = r / s;
    if (m == il) {
      break;
    }
    if (ei_abs(m_matT.coeff(m,m-1)) * (ei_abs(q) + ei_abs(r)) <
      NumTraits<Scalar>::epsilon() * (ei_abs(p) * (ei_abs(m_matT.coeff(m-1,m-1)) + ei_abs(z) +
      ei_abs(m_matT.coeff(m+1,m+1)))))
    {
      break;
    }
    m--;
  }

  for (int i = m+2; i <= iu; ++i)
  {
    m_matT.coeffRef(i,i-2) = 0.0;
    if (i > m+2)
      m_matT.coeffRef(i,i-3) = 0.0;
  }

  firstHouseholderVector << p, q, r;
}

// Double QR step involving rows il:iu and columns m:iu
template<typename MatrixType>
inline void RealSchur<MatrixType>::doFrancisStep(int il, int m, int iu, const Vector3s& firstHouseholderVector, Scalar* workspace)
{
  assert(m >= il);
  assert(m <= iu-2);

  const int size = m_matU.cols();

  for (int k = m; k <= iu-2; ++k)
  {
    bool firstIteration = (k == m);

    Vector3s v;
    if (firstIteration)
      v = firstHouseholderVector;
    else
      v = m_matT.template block<3,1>(k,k-1);

    Scalar tau, beta;
    Matrix<Scalar, 2, 1> ess;
    v.makeHouseholder(ess, tau, beta);
    
    if (beta != Scalar(0)) // if v is not zero
    {
      if (firstIteration && k > il)
        m_matT.coeffRef(k,k-1) = -m_matT.coeff(k,k-1);
      else if (!firstIteration)
        m_matT.coeffRef(k,k-1) = beta;

      // These Householder transformations form the O(n^3) part of the algorithm
      m_matT.block(k, k, 3, size-k).applyHouseholderOnTheLeft(ess, tau, workspace);
      m_matT.block(0, k, std::min(iu,k+3) + 1, 3).applyHouseholderOnTheRight(ess, tau, workspace);
      m_matU.block(0, k, size, 3).applyHouseholderOnTheRight(ess, tau, workspace);
    }
  }

  Matrix<Scalar, 2, 1> v = m_matT.template block<2,1>(iu-1, iu-2);
  Scalar tau, beta;
  Matrix<Scalar, 1, 1> ess;
  v.makeHouseholder(ess, tau, beta);

  if (beta != Scalar(0)) // if v is not zero
  {
    m_matT.coeffRef(iu-1, iu-2) = beta;
    m_matT.block(iu-1, iu-1, 2, size-iu+1).applyHouseholderOnTheLeft(ess, tau, workspace);
    m_matT.block(0, iu-1, iu+1, 2).applyHouseholderOnTheRight(ess, tau, workspace);
    m_matU.block(0, iu-1, size, 2).applyHouseholderOnTheRight(ess, tau, workspace);
  }
}

#endif // EIGEN_REAL_SCHUR_H
