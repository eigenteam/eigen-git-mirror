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

#ifndef EIGEN_COMPLEX_SCHUR_H
#define EIGEN_COMPLEX_SCHUR_H

#define MAXITER 30

/** \ingroup QR
  *
  * \class ComplexShur
  *
  * \brief Performs a complex Shur decomposition of a real or complex square matrix
  *
  */
template<typename _MatrixType> class ComplexSchur
{
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef std::complex<RealScalar> Complex;
    typedef Matrix<Complex, MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime> ComplexMatrixType;

    /**
      * \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via ComplexSchur::compute(const MatrixType&).
      */
    ComplexSchur() : m_matT(), m_matU(), m_isInitialized(false)
    {}

    ComplexSchur(const MatrixType& matrix)
            : m_matT(matrix.rows(),matrix.cols()),
              m_matU(matrix.rows(),matrix.cols()),
              m_isInitialized(false)
    {
      compute(matrix);
    }

    ComplexMatrixType matrixU() const
    {
      ei_assert(m_isInitialized && "ComplexSchur is not initialized.");
      return m_matU;
    }

    ComplexMatrixType matrixT() const
    {
      ei_assert(m_isInitialized && "ComplexShur is not initialized.");
      return m_matT;
    }

    void compute(const MatrixType& matrix);

  protected:
    ComplexMatrixType m_matT, m_matU;
    bool m_isInitialized;
};

/** Computes the principal value of the square root of the complex \a z. */
template<typename RealScalar>
std::complex<RealScalar> ei_sqrt(const std::complex<RealScalar> &z)
{
  RealScalar t, tre, tim;

  t = ei_abs(z);

  if (ei_abs(ei_real(z)) <= ei_abs(ei_imag(z)))
  {
    // No cancellation in these formulas
    tre = ei_sqrt(0.5*(t + ei_real(z)));
    tim = ei_sqrt(0.5*(t - ei_real(z)));
  }
  else
  {
    // Stable computation of the above formulas
    if (z.real() > 0)
    {
      tre = t + z.real();
      tim = ei_abs(ei_imag(z))*ei_sqrt(0.5/tre);
      tre = ei_sqrt(0.5*tre);
    }
    else
    {
      tim = t - z.real();
      tre = ei_abs(ei_imag(z))*ei_sqrt(0.5/tim);
      tim = ei_sqrt(0.5*tim);
    }
  }
  if(z.imag() < 0)
    tim = -tim;

  return (std::complex<RealScalar>(tre,tim));

}

template<typename MatrixType>
void ComplexSchur<MatrixType>::compute(const MatrixType& matrix)
{
  // this code is inspired from Jampack
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();

  // Reduce to Hessenberg form
  HessenbergDecomposition<MatrixType> hess(matrix);

  m_matT = hess.matrixH();
  m_matU = hess.matrixQ();

  int iu = m_matT.cols() - 1;
  int il;
  RealScalar d,sd,sf;
  Complex c,b,disc,r1,r2,kappa;

  RealScalar eps = epsilon<RealScalar>();

  int iter = 0;
  while(true)
  {
    //locate the range in which to iterate
    while(iu > 0)
    {
      d = ei_norm1(m_matT.coeffRef(iu,iu)) + ei_norm1(m_matT.coeffRef(iu-1,iu-1));
      sd = ei_norm1(m_matT.coeffRef(iu,iu-1));

      if(sd >= eps * d) break; // FIXME : precision criterion ??

      m_matT.coeffRef(iu,iu-1) = Complex(0);
      iter = 0;
      --iu;
    }
    if(iu==0) break;
    iter++;

    if(iter >= MAXITER)
    {
      // FIXME : what to do when iter==MAXITER ??
      std::cerr << "MAXITER" << std::endl;
      return;
    }

    il = iu-1;
    while( il > 0 )
    {
      // check if the current 2x2 block on the diagonal is upper triangular
      d = ei_norm1(m_matT.coeffRef(il,il)) + ei_norm1(m_matT.coeffRef(il-1,il-1));
      sd = ei_norm1(m_matT.coeffRef(il,il-1));

      if(sd < eps * d) break; // FIXME : precision criterion ??

      --il;
    }

    if( il != 0 ) m_matT.coeffRef(il,il-1) = Complex(0);

    // compute the shift (the normalization by sf is to avoid under/overflow)
    Matrix<Scalar,2,2> t = m_matT.template block<2,2>(iu-1,iu-1);
    sf = t.cwise().abs().sum();
    t /= sf;

    c = t.determinant();
    b = t.diagonal().sum();

    disc = ei_sqrt(b*b - RealScalar(4)*c);

    r1 = (b+disc)/RealScalar(2);
    r2 = (b-disc)/RealScalar(2);

    if(ei_norm1(r1) > ei_norm1(r2))
      r2 = c/r1;
    else
      r1 = c/r2;

    if(ei_norm1(r1-t.coeff(1,1)) < ei_norm1(r2-t.coeff(1,1)))
      kappa = sf * r1;
    else
      kappa = sf * r2;

    // perform the QR step using Givens rotations
    PlanarRotation<Complex> rot;
    rot.makeGivens(m_matT.coeff(il,il) - kappa, m_matT.coeff(il+1,il));

    for(int i=il ; i<iu ; i++)
    {
      m_matT.block(0,i,n,n-i).applyOnTheLeft(i, i+1, rot.adjoint());
      m_matT.block(0,0,std::min(i+2,iu)+1,n).applyOnTheRight(i, i+1, rot);
      m_matU.applyOnTheRight(i, i+1, rot);

      if(i != iu-1)
      {
        int i1 = i+1;
        int i2 = i+2;

        rot.makeGivens(m_matT.coeffRef(i1,i), m_matT.coeffRef(i2,i), &m_matT.coeffRef(i1,i));
        m_matT.coeffRef(i2,i) = Complex(0);
      }
    }
  }

  // FIXME : is it necessary ?
  /*
  for(int i=0 ; i<n ; i++)
    for(int j=0 ; j<n ; j++)
    {
      if(ei_abs(ei_real(m_matT.coeff(i,j))) < eps)
        ei_real_ref(m_matT.coeffRef(i,j)) = 0;
      if(ei_imag(ei_abs(m_matT.coeff(i,j))) < eps)
        ei_imag_ref(m_matT.coeffRef(i,j)) = 0;
    }
  */

  m_isInitialized = true;
}

#endif // EIGEN_COMPLEX_SCHUR_H
