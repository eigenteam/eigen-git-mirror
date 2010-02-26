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

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class ComplexShur
  *
  * \brief Performs a complex Schur decomposition of a real or complex square matrix
  *
  * Given a real or complex square matrix A, this class computes the Schur decomposition:
  * \f$ A = U T U^*\f$ where U is a unitary complex matrix, and T is a complex upper
  * triangular matrix.
  *
  * The diagonal of the matrix T corresponds to the eigenvalues of the matrix A.
  *
  * \sa class RealSchur, class EigenSolver
  */
template<typename _MatrixType> class ComplexSchur
{
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef std::complex<RealScalar> Complex;
    typedef Matrix<Complex, MatrixType::RowsAtCompileTime,MatrixType::ColsAtCompileTime> ComplexMatrixType;
    enum {
      Size = MatrixType::RowsAtCompileTime
    };

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via ComplexSchur::compute().
      */
    ComplexSchur(int size = Size==Dynamic ? 0 : Size)
      : m_matT(size,size), m_matU(size,size), m_isInitialized(false), m_matUisUptodate(false)
    {}

    /** Constructor computing the Schur decomposition of the matrix \a matrix.
      * If \a skipU is true, then the matrix U is not computed. */
    ComplexSchur(const MatrixType& matrix, bool skipU = false)
            : m_matT(matrix.rows(),matrix.cols()),
              m_matU(matrix.rows(),matrix.cols()),
              m_isInitialized(false),
              m_matUisUptodate(false)
    {
      compute(matrix, skipU);
    }

    /** \returns a const reference to the matrix U of the respective Schur decomposition. */
    const ComplexMatrixType& matrixU() const
    {
      ei_assert(m_isInitialized && "ComplexSchur is not initialized.");
      ei_assert(m_matUisUptodate && "The matrix U has not been computed during the ComplexSchur decomposition.");
      return m_matU;
    }

    /** \returns a const reference to the matrix T of the respective Schur decomposition.
      * Note that this function returns a plain square matrix. If you want to reference
      * only the upper triangular part, use:
      * \code schur.matrixT().triangularView<Upper>() \endcode. */
    const ComplexMatrixType& matrixT() const
    {
      ei_assert(m_isInitialized && "ComplexShur is not initialized.");
      return m_matT;
    }

    /** Computes the Schur decomposition of the matrix \a matrix.
      * If \a skipU is true, then the matrix U is not computed. */
    void compute(const MatrixType& matrix, bool skipU = false);

  protected:
    ComplexMatrixType m_matT, m_matU;
    bool m_isInitialized;
    bool m_matUisUptodate;
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
void ComplexSchur<MatrixType>::compute(const MatrixType& matrix, bool skipU)
{
  // this code is inspired from Jampack

  m_matUisUptodate = false;
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();

  // Reduce to Hessenberg form
  // TODO skip Q if skipU = true
  HessenbergDecomposition<MatrixType> hess(matrix);

  m_matT = hess.matrixH();
  if(!skipU)  m_matU = hess.matrixQ();


  // Reduce the Hessenberg matrix m_matT to triangular form by QR iteration.

  // The matrix m_matT is divided in three parts. 
  // Rows 0,...,il-1 are decoupled from the rest because m_matT(il,il-1) is zero. 
  // Rows il,...,iu is the part we are working on (the active submatrix).
  // Rows iu+1,...,end are already brought in triangular form.

  int iu = m_matT.cols() - 1;
  int il;
  RealScalar d,sd,sf;
  Complex c,b,disc,r1,r2,kappa;

  RealScalar eps = NumTraits<RealScalar>::epsilon();

  int iter = 0;
  while(true)
  {
    // find iu, the bottom row of the active submatrix
    while(iu > 0)
    {
      d = ei_norm1(m_matT.coeff(iu,iu)) + ei_norm1(m_matT.coeff(iu-1,iu-1));
      sd = ei_norm1(m_matT.coeff(iu,iu-1));

      if(!ei_isMuchSmallerThan(sd,d,eps))
        break;

      m_matT.coeffRef(iu,iu-1) = Complex(0);
      iter = 0;
      --iu;
    }
    if(iu==0) break;
    iter++;

    if(iter >= 30)
    {
      // FIXME : what to do when iter==MAXITER ??
      std::cerr << "MAXITER" << std::endl;
      return;
    }

    // find il, the top row of the active submatrix
    il = iu-1;
    while(il > 0)
    {
      // check if the current 2x2 block on the diagonal is upper triangular
      d = ei_norm1(m_matT.coeff(il,il)) + ei_norm1(m_matT.coeff(il-1,il-1));
      sd = ei_norm1(m_matT.coeff(il,il-1));

      if(ei_isMuchSmallerThan(sd,d,eps))
        break;

      --il;
    }

    if( il != 0 ) m_matT.coeffRef(il,il-1) = Complex(0);

    // compute the shift kappa as one of the eigenvalues of the 2x2
    // diagonal block on the bottom of the active submatrix

    Matrix<Scalar,2,2> t = m_matT.template block<2,2>(iu-1,iu-1);
    sf = t.cwiseAbs().sum();
    t /= sf;     // the normalization by sf is to avoid under/overflow

    b = t.coeff(0,0) + t.coeff(1,1);
    c = t.coeff(0,0) - t.coeff(1,1);
    disc = ei_sqrt(c*c + RealScalar(4)*t.coeff(0,1)*t.coeff(1,0));

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
    
    if (iter == 10 || iter == 20) 
    {
      // exceptional shift, taken from http://www.netlib.org/eispack/comqr.f
      kappa = ei_abs(ei_real(m_matT.coeff(iu,iu-1))) + ei_abs(ei_real(m_matT.coeff(iu-1,iu-2)));
    }

    // perform the QR step using Givens rotations
    PlanarRotation<Complex> rot;
    rot.makeGivens(m_matT.coeff(il,il) - kappa, m_matT.coeff(il+1,il));

    for(int i=il ; i<iu ; i++)
    {
      m_matT.block(0,i,n,n-i).applyOnTheLeft(i, i+1, rot.adjoint());
      m_matT.block(0,0,std::min(i+2,iu)+1,n).applyOnTheRight(i, i+1, rot);
      if(!skipU) m_matU.applyOnTheRight(i, i+1, rot);

      if(i != iu-1)
      {
        int i1 = i+1;
        int i2 = i+2;

        rot.makeGivens(m_matT.coeffRef(i1,i), m_matT.coeffRef(i2,i), &m_matT.coeffRef(i1,i));
        m_matT.coeffRef(i2,i) = Complex(0);
      }
    }
  }

  m_isInitialized = true;
  m_matUisUptodate = !skipU;
}

#endif // EIGEN_COMPLEX_SCHUR_H
