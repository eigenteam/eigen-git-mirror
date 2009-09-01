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

// computes the plane rotation G such that G' x |p| = | c  s' |' |p| = |z|
//                                              |q|   |-s  c' |  |q|   |0|
// and returns z if requested. Note that the returned c is real.
template<typename T> void ei_make_givens(const std::complex<T>& p, const std::complex<T>& q,
                                           JacobiRotation<std::complex<T> >& rot, std::complex<T>* z=0)
{
  typedef std::complex<T> Complex;
  T scale, absx, absxy;
  if(p==Complex(0))
  {
    // return identity
    rot.c() = Complex(1,0);
    rot.s() = Complex(0,0);
    if(z) *z = p;
  }
  else
  {
    scale = cnorm1(p);
    absx = scale * ei_sqrt(ei_abs2(p/scale));
    scale = ei_abs(scale) + cnorm1(q);
    absxy = scale * ei_sqrt((absx/scale)*(absx/scale) + ei_abs2(q/scale));
    rot.c() = Complex(absx / absxy);
    Complex np = p/absx;
    rot.s() = -ei_conj(np) * q / absxy;
    if(z) *z = np * absxy;
  }
}

template<typename MatrixType>
void ComplexSchur<MatrixType>::compute(const MatrixType& matrix)
{
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
      d = cnorm1(m_matT.coeffRef(iu,iu)) + cnorm1(m_matT.coeffRef(iu-1,iu-1));
      sd = cnorm1(m_matT.coeffRef(iu,iu-1));

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
      d = cnorm1(m_matT.coeffRef(il,il)) + cnorm1(m_matT.coeffRef(il-1,il-1));
      sd = cnorm1(m_matT.coeffRef(il,il-1));

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

    disc = csqrt(b*b - RealScalar(4)*c);

    r1 = (b+disc)/RealScalar(2);
    r2 = (b-disc)/RealScalar(2);

    if(cnorm1(r1) > cnorm1(r2))
      r2 = c/r1;
    else
      r1 = c/r2;

    if(cnorm1(r1-t.coeff(1,1)) < cnorm1(r2-t.coeff(1,1)))
      kappa = sf * r1;
    else
      kappa = sf * r2;

    // perform the QR step using Givens rotations
    JacobiRotation<Complex> rot;
    ei_make_givens(m_matT.coeff(il,il) - kappa, m_matT.coeff(il+1,il), rot);

    for(int i=il ; i<iu ; i++)
    {
      m_matT.block(0,i,n,n-i).applyJacobiOnTheLeft(i, i+1, rot.adjoint());
      m_matT.block(0,0,std::min(i+2,iu)+1,n).applyJacobiOnTheRight(i, i+1, rot);
      m_matU.applyJacobiOnTheRight(i, i+1, rot);

      if(i != iu-1)
      {
        int i1 = i+1;
        int i2 = i+2;

        ei_make_givens(m_matT.coeffRef(i1,i), m_matT.coeffRef(i2,i), rot, &m_matT.coeffRef(i1,i));
        m_matT.coeffRef(i2,i) = Complex(0);
      }
    }
  }

  // FIXME : is it necessary ?
  for(int i=0 ; i<n ; i++)
    for(int j=0 ; j<n ; j++)
    {
      if(ei_abs(m_matT.coeff(i,j).real()) < eps)
        m_matT.coeffRef(i,j).real() = 0;
      if(ei_abs(m_matT.coeff(i,j).imag()) < eps)
        m_matT.coeffRef(i,j).imag() = 0;
    }

  m_isInitialized = true;
}

// norm1 of complex numbers
template<typename T>
T cnorm1(const std::complex<T> &Z)
{
  return(ei_abs(Z.real()) + ei_abs(Z.imag()));
}

/**
  * Computes the principal value of the square root of the complex \a z.
  */
template<typename RealScalar>
std::complex<RealScalar> csqrt(const std::complex<RealScalar> &z)
{
  RealScalar t, tre, tim;

  t = ei_abs(z);

  if (ei_abs(z.real()) <= ei_abs(z.imag()))
  {
    // No cancellation in these formulas
    tre = ei_sqrt(0.5*(t+z.real()));
    tim = ei_sqrt(0.5*(t-z.real()));
  }
  else
  {
    // Stable computation of the above formulas
    if (z.real() > 0)
    {
      tre = t + z.real();
      tim = ei_abs(z.imag())*ei_sqrt(0.5/tre);
      tre = ei_sqrt(0.5*tre);
    }
    else
    {
      tim = t - z.real();
      tre = ei_abs(z.imag())*ei_sqrt(0.5/tim);
      tim = ei_sqrt(0.5*tim);
    }
  }
  if(z.imag() < 0)
    tim = -tim;

  return (std::complex<RealScalar>(tre,tim));

}


#endif // EIGEN_COMPLEX_SCHUR_H
