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

    void hqr2();
};


template<typename MatrixType>
void RealSchur<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.cols() == matrix.rows());

  // Reduce to Hessenberg form
  // TODO skip Q if skipU = true
  HessenbergDecomposition<MatrixType> hess(matrix);
  m_matT = hess.matrixH();
  m_matU = hess.matrixQ();

  // Reduce to Real Schur form
  hqr2();

  m_isInitialized = true;
}


template<typename MatrixType>
void RealSchur<MatrixType>::hqr2()
{
  //  This is derived from the Algol procedure hqr2,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  // Initialize
  const int size = m_matU.cols();
  int n = size-1;
  const int low = 0;
  const int high = size-1;
  Scalar exshift = 0.0;
  Scalar p=0,q=0,r=0,s=0,z=0,w,x,y;

  // Store roots isolated by balanc and compute matrix norm
  // FIXME to be efficient the following would requires a triangular reduxion code
  // Scalar norm = m_matT.upper().cwiseAbs().sum() + m_matT.corner(BottomLeft,n,n).diagonal().cwiseAbs().sum();
  Scalar norm = 0.0;
  for (int j = 0; j < size; ++j)
  {
    // FIXME what's the purpose of the following since the condition is always false
    if ((j < low) || (j > high))
    {
      m_eivalues.coeffRef(j) = ComplexScalar(m_matT.coeff(j,j), 0.0);
    }
    norm += m_matT.row(j).segment(std::max(j-1,0), size-std::max(j-1,0)).cwiseAbs().sum();
  }

  // Outer loop over eigenvalue index
  int iter = 0;
  while (n >= low)
  {
    // Look for single small sub-diagonal element
    int l = n;
    while (l > low)
    {
      s = ei_abs(m_matT.coeff(l-1,l-1)) + ei_abs(m_matT.coeff(l,l));
      if (s == 0.0)
          s = norm;
      if (ei_abs(m_matT.coeff(l,l-1)) < NumTraits<Scalar>::epsilon() * s)
        break;
      l--;
    }

    // Check for convergence
    // One root found
    if (l == n)
    {
      m_matT.coeffRef(n,n) = m_matT.coeff(n,n) + exshift;
      m_eivalues.coeffRef(n) = ComplexScalar(m_matT.coeff(n,n), 0.0);
      n--;
      iter = 0;
    }
    else if (l == n-1) // Two roots found
    {
      w = m_matT.coeff(n,n-1) * m_matT.coeff(n-1,n);
      p = (m_matT.coeff(n-1,n-1) - m_matT.coeff(n,n)) * Scalar(0.5);
      q = p * p + w;
      z = ei_sqrt(ei_abs(q));
      m_matT.coeffRef(n,n) = m_matT.coeff(n,n) + exshift;
      m_matT.coeffRef(n-1,n-1) = m_matT.coeff(n-1,n-1) + exshift;
      x = m_matT.coeff(n,n);

      // Scalar pair
      if (q >= 0)
      {
        if (p >= 0)
          z = p + z;
        else
          z = p - z;

        m_eivalues.coeffRef(n-1) = ComplexScalar(x + z, 0.0);
        m_eivalues.coeffRef(n) = ComplexScalar(z!=0.0 ? x - w / z : m_eivalues.coeff(n-1).real(), 0.0);

        x = m_matT.coeff(n,n-1);
        s = ei_abs(x) + ei_abs(z);
        p = x / s;
        q = z / s;
        r = ei_sqrt(p * p+q * q);
        p = p / r;
        q = q / r;

        // Row modification
        for (int j = n-1; j < size; ++j)
        {
          z = m_matT.coeff(n-1,j);
          m_matT.coeffRef(n-1,j) = q * z + p * m_matT.coeff(n,j);
          m_matT.coeffRef(n,j) = q * m_matT.coeff(n,j) - p * z;
        }

        // Column modification
        for (int i = 0; i <= n; ++i)
        {
          z = m_matT.coeff(i,n-1);
          m_matT.coeffRef(i,n-1) = q * z + p * m_matT.coeff(i,n);
          m_matT.coeffRef(i,n) = q * m_matT.coeff(i,n) - p * z;
        }

        // Accumulate transformations
        for (int i = low; i <= high; ++i)
        {
          z = m_matU.coeff(i,n-1);
          m_matU.coeffRef(i,n-1) = q * z + p * m_matU.coeff(i,n);
          m_matU.coeffRef(i,n) = q * m_matU.coeff(i,n) - p * z;
        }
      }
      else // Complex pair
      {
        m_eivalues.coeffRef(n-1) = ComplexScalar(x + p, z);
        m_eivalues.coeffRef(n)   = ComplexScalar(x + p, -z);
      }
      n = n - 2;
      iter = 0;
    }
    else // No convergence yet
    {
      // Form shift
      x = m_matT.coeff(n,n);
      y = 0.0;
      w = 0.0;
      if (l < n)
      {
          y = m_matT.coeff(n-1,n-1);
          w = m_matT.coeff(n,n-1) * m_matT.coeff(n-1,n);
      }

      // Wilkinson's original ad hoc shift
      if (iter == 10)
      {
        exshift += x;
        for (int i = low; i <= n; ++i)
          m_matT.coeffRef(i,i) -= x;
        s = ei_abs(m_matT.coeff(n,n-1)) + ei_abs(m_matT.coeff(n-1,n-2));
        x = y = Scalar(0.75) * s;
        w = Scalar(-0.4375) * s * s;
      }

      // MATLAB's new ad hoc shift
      if (iter == 30)
      {
        s = Scalar((y - x) / 2.0);
        s = s * s + w;
        if (s > 0)
        {
          s = ei_sqrt(s);
          if (y < x)
            s = -s;
          s = Scalar(x - w / ((y - x) / 2.0 + s));
          for (int i = low; i <= n; ++i)
            m_matT.coeffRef(i,i) -= s;
          exshift += s;
          x = y = w = Scalar(0.964);
        }
      }

      iter = iter + 1;   // (Could check iteration count here.)

      // Look for two consecutive small sub-diagonal elements
      int m = n-2;
      while (m >= l)
      {
        z = m_matT.coeff(m,m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / m_matT.coeff(m+1,m) + m_matT.coeff(m,m+1);
        q = m_matT.coeff(m+1,m+1) - z - r - s;
        r = m_matT.coeff(m+2,m+1);
        s = ei_abs(p) + ei_abs(q) + ei_abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) {
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

      for (int i = m+2; i <= n; ++i)
      {
        m_matT.coeffRef(i,i-2) = 0.0;
        if (i > m+2)
          m_matT.coeffRef(i,i-3) = 0.0;
      }

      // Double QR step involving rows l:n and columns m:n
      for (int k = m; k <= n-1; ++k)
      {
        int notlast = (k != n-1);
        if (k != m) {
          p = m_matT.coeff(k,k-1);
          q = m_matT.coeff(k+1,k-1);
          r = notlast ? m_matT.coeff(k+2,k-1) : Scalar(0);
          x = ei_abs(p) + ei_abs(q) + ei_abs(r);
          if (x != 0.0)
          {
            p = p / x;
            q = q / x;
            r = r / x;
          }
        }

        if (x == 0.0)
          break;

        s = ei_sqrt(p * p + q * q + r * r);

        if (p < 0)
          s = -s;

        if (s != 0)
        {
          if (k != m)
            m_matT.coeffRef(k,k-1) = -s * x;
          else if (l != m)
            m_matT.coeffRef(k,k-1) = -m_matT.coeff(k,k-1);

          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          // Row modification
          for (int j = k; j < size; ++j)
          {
            p = m_matT.coeff(k,j) + q * m_matT.coeff(k+1,j);
            if (notlast)
            {
              p = p + r * m_matT.coeff(k+2,j);
              m_matT.coeffRef(k+2,j) = m_matT.coeff(k+2,j) - p * z;
            }
            m_matT.coeffRef(k,j) = m_matT.coeff(k,j) - p * x;
            m_matT.coeffRef(k+1,j) = m_matT.coeff(k+1,j) - p * y;
          }

          // Column modification
          for (int i = 0; i <= std::min(n,k+3); ++i)
          {
            p = x * m_matT.coeff(i,k) + y * m_matT.coeff(i,k+1);
            if (notlast)
            {
              p = p + z * m_matT.coeff(i,k+2);
              m_matT.coeffRef(i,k+2) = m_matT.coeff(i,k+2) - p * r;
            }
            m_matT.coeffRef(i,k) = m_matT.coeff(i,k) - p;
            m_matT.coeffRef(i,k+1) = m_matT.coeff(i,k+1) - p * q;
          }

          // Accumulate transformations
          for (int i = low; i <= high; ++i)
          {
            p = x * m_matU.coeff(i,k) + y * m_matU.coeff(i,k+1);
            if (notlast)
            {
              p = p + z * m_matU.coeff(i,k+2);
              m_matU.coeffRef(i,k+2) = m_matU.coeff(i,k+2) - p * r;
            }
            m_matU.coeffRef(i,k) = m_matU.coeff(i,k) - p;
            m_matU.coeffRef(i,k+1) = m_matU.coeff(i,k+1) - p * q;
          }
        }  // (s != 0)
      }  // k loop
    }  // check convergence
  }  // while (n >= low)
}

#endif // EIGEN_REAL_SCHUR_H
