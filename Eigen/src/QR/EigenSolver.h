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

#ifndef EIGEN_EIGENSOLVER_H
#define EIGEN_EIGENSOLVER_H

/** \class EigenSolver
  *
  * \brief Eigen values/vectors solver
  *
  * \param MatrixType the type of the matrix of which we are computing the eigen decomposition
  * \param IsSelfadjoint tells the input matrix is guaranteed to be selfadjoint (hermitian). In that case the
  * return type of eigenvalues() is a real vector.
  *
  * Currently it only support real matrices.
  *
  * \note this code was adapted from JAMA (public domain)
  *
  * \sa MatrixBase::eigenvalues()
  */
template<typename _MatrixType, bool IsSelfadjoint=false> class EigenSolver
{
  public:

    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef std::complex<RealScalar> Complex;
    typedef Matrix<typename ei_meta_if<IsSelfadjoint, Scalar, Complex>::ret, MatrixType::ColsAtCompileTime, 1> EigenvalueType;
    typedef Matrix<RealScalar, MatrixType::ColsAtCompileTime, 1> RealVectorType;
    typedef Matrix<RealScalar, Dynamic, 1> RealVectorTypeX;

    EigenSolver(const MatrixType& matrix)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_eivalues(matrix.cols())
    {
      _compute(matrix);
    }

    MatrixType eigenvectors(void) const { return m_eivec; }

    EigenvalueType eigenvalues(void) const { return m_eivalues; }

  private:

    void _compute(const MatrixType& matrix)
    {
      computeImpl(matrix, typename ei_meta_if<IsSelfadjoint, ei_meta_true, ei_meta_false>::ret());
    }
    void computeImpl(const MatrixType& matrix, ei_meta_true isSelfadjoint);
    void computeImpl(const MatrixType& matrix, ei_meta_false isNotSelfadjoint);

    void tridiagonalization(RealVectorType& eivalr, RealVectorType& eivali);
    void tql2(RealVectorType& eivalr, RealVectorType& eivali);

    void orthes(MatrixType& matH, RealVectorType& ort);
    void hqr2(MatrixType& matH);

  protected:
    MatrixType m_eivec;
    EigenvalueType m_eivalues;
};

template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::computeImpl(const MatrixType& matrix, ei_meta_true)
{
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();
  m_eivalues.resize(n,1);

  RealVectorType eivali(n);
  m_eivec = matrix;

  // Tridiagonalize.
  tridiagonalization(m_eivalues, eivali);

  // Diagonalize.
  tql2(m_eivalues, eivali);
}

template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::computeImpl(const MatrixType& matrix, ei_meta_false)
{
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();
  m_eivalues.resize(n,1);

  bool isSelfadjoint = true;
  for (int j = 0; (j < n) && isSelfadjoint; j++)
    for (int i = 0; (i < j) && isSelfadjoint; i++)
      isSelfadjoint = (matrix(i,j) == matrix(j,i));

  if (isSelfadjoint)
  {
    RealVectorType eivalr(n);
    RealVectorType eivali(n);
    m_eivec = matrix;

    // Tridiagonalize.
    tridiagonalization(eivalr, eivali);

    // Diagonalize.
    tql2(eivalr, eivali);

    m_eivalues = eivalr.template cast<Complex>();
  }
  else
  {
    MatrixType matH = matrix;
    RealVectorType ort(n);

    // Reduce to Hessenberg form.
    orthes(matH, ort);

    // Reduce Hessenberg to real Schur form.
    hqr2(matH);
  }
}


// Symmetric Householder reduction to tridiagonal form.
template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::tridiagonalization(RealVectorType& eivalr, RealVectorType& eivali)
{

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  int n = m_eivec.cols();
  eivalr = m_eivec.row(eivalr.size()-1);

  // Householder reduction to tridiagonal form.
  for (int i = n-1; i > 0; i--)
  {
    // Scale to avoid under/overflow.
    Scalar scale = 0.0;
    Scalar h = 0.0;
    scale = eivalr.start(i).cwiseAbs().sum();

    if (scale == 0.0)
    {
      eivali[i] = eivalr[i-1];
      eivalr.start(i) = m_eivec.row(i-1).start(i);
      m_eivec.corner(TopLeft, i, i) = m_eivec.corner(TopLeft, i, i).diagonal().asDiagonal();
    }
    else
    {
      // Generate Householder vector.
      eivalr.start(i) /= scale;
      h = eivalr.start(i).cwiseAbs2().sum();

      Scalar f = eivalr[i-1];
      Scalar g = ei_sqrt(h);
      if (f > 0)
        g = -g;
      eivali[i] = scale * g;
      h = h - f * g;
      eivalr[i-1] = f - g;
      eivali.start(i).setZero();

      // Apply similarity transformation to remaining columns.
      for (int j = 0; j < i; j++)
      {
        f = eivalr[j];
        m_eivec(j,i) = f;
        g = eivali[j] + m_eivec(j,j) * f;
        int bSize = i-j-1;
        if (bSize>0)
        {
          g += (m_eivec.col(j).block(j+1, bSize).transpose() * eivalr.block(j+1, bSize))(0,0);
          eivali.block(j+1, bSize) += m_eivec.col(j).block(j+1, bSize) * f;
        }
        eivali[j] = g;
      }

      f = (eivali.start(i).transpose() * eivalr.start(i))(0,0);
      eivali.start(i) = (eivali.start(i) - (f / (h + h)) * eivalr.start(i))/h;

      m_eivec.corner(TopLeft, i, i).template part<Lower>() -=
        ( (eivali.start(i) * eivalr.start(i).transpose()).lazy()
        + (eivalr.start(i) * eivali.start(i).transpose()).lazy());

      eivalr.start(i) = m_eivec.row(i-1).start(i);
      m_eivec.row(i).start(i).setZero();
    }
    eivalr[i] = h;
  }

  // Accumulate transformations.
  for (int i = 0; i < n-1; i++)
  {
    m_eivec(n-1,i) = m_eivec(i,i);
    m_eivec(i,i) = 1.0;
    Scalar h = eivalr[i+1];
    // FIXME this does not looks very stable ;)
    if (h != 0.0)
    {
      eivalr.start(i+1) = m_eivec.col(i+1).start(i+1) / h;
      m_eivec.corner(TopLeft, i+1, i+1) -= eivalr.start(i+1)
        * ( m_eivec.col(i+1).start(i+1).transpose() * m_eivec.corner(TopLeft, i+1, i+1) );
    }
    m_eivec.col(i+1).start(i+1).setZero();
  }
  eivalr = m_eivec.row(eivalr.size()-1);
  m_eivec.row(eivalr.size()-1).setZero();
  m_eivec(n-1,n-1) = 1.0;
  eivali[0] = 0.0;
}


// Symmetric tridiagonal QL algorithm.
template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::tql2(RealVectorType& eivalr, RealVectorType& eivali)
{
  //  This is derived from the Algol procedures tql2, by
  //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
  //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  int n = eivalr.size();

  for (int i = 1; i < n; i++) {
      eivali[i-1] = eivali[i];
  }
  eivali[n-1] = 0.0;

  Scalar f = 0.0;
  Scalar tst1 = 0.0;
  Scalar eps = std::pow(2.0,-52.0);
  for (int l = 0; l < n; l++)
  {
    // Find small subdiagonal element
    tst1 = std::max(tst1,ei_abs(eivalr[l]) + ei_abs(eivali[l]));
    int m = l;

    while ( (m < n) && (ei_abs(eivali[m]) > eps*tst1) )
      m++;

    // If m == l, eivalr[l] is an eigenvalue,
    // otherwise, iterate.
    if (m > l)
    {
      int iter = 0;
      do
      {
        iter = iter + 1;

        // Compute implicit shift
        Scalar g = eivalr[l];
        Scalar p = (eivalr[l+1] - g) / (2.0 * eivali[l]);
        Scalar r = hypot(p,1.0);
        if (p < 0)
          r = -r;

        eivalr[l] = eivali[l] / (p + r);
        eivalr[l+1] = eivali[l] * (p + r);
        Scalar dl1 = eivalr[l+1];
        Scalar h = g - eivalr[l];
        if (l+2<n)
          eivalr.end(n-l-2) -= RealVectorTypeX::constant(n-l-2, h);
        f = f + h;

        // Implicit QL transformation.
        p = eivalr[m];
        Scalar c = 1.0;
        Scalar c2 = c;
        Scalar c3 = c;
        Scalar el1 = eivali[l+1];
        Scalar s = 0.0;
        Scalar s2 = 0.0;
        for (int i = m-1; i >= l; i--)
        {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * eivali[i];
          h = c * p;
          r = hypot(p,eivali[i]);
          eivali[i+1] = s * r;
          s = eivali[i] / r;
          c = p / r;
          p = c * eivalr[i] - s * g;
          eivalr[i+1] = h + s * (c * g + s * eivalr[i]);

          // Accumulate transformation.
          for (int k = 0; k < n; k++)
          {
            h = m_eivec(k,i+1);
            m_eivec(k,i+1) = s * m_eivec(k,i) + c * h;
            m_eivec(k,i) = c * m_eivec(k,i) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * eivali[l] / dl1;
        eivali[l] = s * p;
        eivalr[l] = c * p;

        // Check for convergence.
      } while (ei_abs(eivali[l]) > eps*tst1);
    }
    eivalr[l] = eivalr[l] + f;
    eivali[l] = 0.0;
  }

  // Sort eigenvalues and corresponding vectors.
  // TODO use a better sort algorithm !!
  for (int i = 0; i < n-1; i++)
  {
    int k = i;
    Scalar minValue = eivalr[i];
    for (int j = i+1; j < n; j++)
    {
      if (eivalr[j] < minValue)
      {
        k = j;
        minValue = eivalr[j];
      }
    }
    if (k != i)
    {
      std::swap(eivalr[i], eivalr[k]);
      m_eivec.col(i).swap(m_eivec.col(k));
    }
  }
}


// Nonsymmetric reduction to Hessenberg form.
template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::orthes(MatrixType& matH, RealVectorType& ort)
{
  //  This is derived from the Algol procedures orthes and ortran,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutines in EISPACK.

  int n = m_eivec.cols();
  int low = 0;
  int high = n-1;

  for (int m = low+1; m <= high-1; m++)
  {
    // Scale column.
    Scalar scale = matH.block(m, m-1, high-m+1, 1).cwiseAbs().sum();
    if (scale != 0.0)
    {
      // Compute Householder transformation.
      Scalar h = 0.0;
      // FIXME could be rewritten, but this one looks better wrt cache
      for (int i = high; i >= m; i--)
      {
        ort[i] = matH(i,m-1)/scale;
        h += ort[i] * ort[i];
      }
      Scalar g = ei_sqrt(h);
      if (ort[m] > 0)
        g = -g;
      h = h - ort[m] * g;
      ort[m] = ort[m] - g;

      // Apply Householder similarity transformation
      // H = (I-u*u'/h)*H*(I-u*u')/h)
      int bSize = high-m+1;
      matH.block(m, m, bSize, n-m) -= ((ort.block(m, bSize)/h)
        * (ort.block(m, bSize).transpose() *  matH.block(m, m, bSize, n-m)).lazy()).lazy();

      matH.block(0, m, high+1, bSize) -= ((matH.block(0, m, high+1, bSize) * ort.block(m, bSize)).lazy()
        * (ort.block(m, bSize)/h).transpose()).lazy();

      ort[m] = scale*ort[m];
      matH(m,m-1) = scale*g;
    }
  }

  // Accumulate transformations (Algol's ortran).
  m_eivec.setIdentity();

  for (int m = high-1; m >= low+1; m--)
  {
    if (matH(m,m-1) != 0.0)
    {
      ort.block(m+1, high-m) = matH.col(m-1).block(m+1, high-m);

      int bSize = high-m+1;
      m_eivec.block(m, m, bSize, bSize) += ( (ort.block(m, bSize) /  (matH(m,m-1) * ort[m] ) )
        * (ort.block(m, bSize).transpose() * m_eivec.block(m, m, bSize, bSize)).lazy());
    }
  }
}


// Complex scalar division.
template<typename Scalar>
std::complex<Scalar> cdiv(Scalar xr, Scalar xi, Scalar yr, Scalar yi)
{
  Scalar r,d;
  if (ei_abs(yr) > ei_abs(yi))
  {
      r = yi/yr;
      d = yr + r*yi;
      return std::complex<Scalar>((xr + r*xi)/d, (xi - r*xr)/d);
  }
  else
  {
      r = yr/yi;
      d = yi + r*yr;
      return std::complex<Scalar>((r*xr + xi)/d, (r*xi - xr)/d);
  }
}


// Nonsymmetric reduction from Hessenberg to real Schur form.
template<typename MatrixType, bool IsSelfadjoint>
void EigenSolver<MatrixType,IsSelfadjoint>::hqr2(MatrixType& matH)
{
  //  This is derived from the Algol procedure hqr2,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutine in EISPACK.

  // Initialize
  int nn = m_eivec.cols();
  int n = nn-1;
  int low = 0;
  int high = nn-1;
  Scalar eps = pow(2.0,-52.0);
  Scalar exshift = 0.0;
  Scalar p=0,q=0,r=0,s=0,z=0,t,w,x,y;

  // Store roots isolated by balanc and compute matrix norm
  // FIXME to be efficient the following would requires a triangular reduxion code
  // Scalar norm = matH.upper().cwiseAbs().sum() + matH.corner(BottomLeft,n,n).diagonal().cwiseAbs().sum();
  Scalar norm = 0.0;
  for (int j = 0; j < nn; j++)
  {
    // FIXME what's the purpose of the following since the condition is always false
    if ((j < low) || (j > high))
    {
      m_eivalues[j].real() = matH(j,j);
      m_eivalues[j].imag() = 0.0;
    }
    norm += matH.col(j).start(std::min(j+1,nn)).cwiseAbs().sum();
  }

  // Outer loop over eigenvalue index
  int iter = 0;
  while (n >= low)
  {
    // Look for single small sub-diagonal element
    int l = n;
    while (l > low)
    {
      s = ei_abs(matH(l-1,l-1)) + ei_abs(matH(l,l));
      if (s == 0.0)
          s = norm;
      if (ei_abs(matH(l,l-1)) < eps * s)
        break;
      l--;
    }

    // Check for convergence
    // One root found
    if (l == n)
    {
      matH(n,n) = matH(n,n) + exshift;
      m_eivalues[n].real() = matH(n,n);
      m_eivalues[n].imag() = 0.0;
      n--;
      iter = 0;
    }
    else if (l == n-1) // Two roots found
    {
      w = matH(n,n-1) * matH(n-1,n);
      p = (matH(n-1,n-1) - matH(n,n)) / 2.0;
      q = p * p + w;
      z = ei_sqrt(ei_abs(q));
      matH(n,n) = matH(n,n) + exshift;
      matH(n-1,n-1) = matH(n-1,n-1) + exshift;
      x = matH(n,n);

      // Scalar pair
      if (q >= 0)
      {
        if (p >= 0)
          z = p + z;
        else
          z = p - z;

        m_eivalues[n-1].real() = x + z;
        m_eivalues[n].real() = m_eivalues[n-1].real();
        if (z != 0.0)
          m_eivalues[n].real() = x - w / z;

        m_eivalues[n-1].imag() = 0.0;
        m_eivalues[n].imag() = 0.0;
        x = matH(n,n-1);
        s = ei_abs(x) + ei_abs(z);
        p = x / s;
        q = z / s;
        r = ei_sqrt(p * p+q * q);
        p = p / r;
        q = q / r;

        // Row modification
        for (int j = n-1; j < nn; j++)
        {
          z = matH(n-1,j);
          matH(n-1,j) = q * z + p * matH(n,j);
          matH(n,j) = q * matH(n,j) - p * z;
        }

        // Column modification
        for (int i = 0; i <= n; i++)
        {
          z = matH(i,n-1);
          matH(i,n-1) = q * z + p * matH(i,n);
          matH(i,n) = q * matH(i,n) - p * z;
        }

        // Accumulate transformations
        for (int i = low; i <= high; i++)
        {
          z = m_eivec(i,n-1);
          m_eivec(i,n-1) = q * z + p * m_eivec(i,n);
          m_eivec(i,n) = q * m_eivec(i,n) - p * z;
        }
      }
      else // Complex pair
      {
        m_eivalues[n-1].real() = x + p;
        m_eivalues[n].real() = x + p;
        m_eivalues[n-1].imag() = z;
        m_eivalues[n].imag() = -z;
      }
      n = n - 2;
      iter = 0;
    }
    else // No convergence yet
    {
      // Form shift
      x = matH(n,n);
      y = 0.0;
      w = 0.0;
      if (l < n)
      {
          y = matH(n-1,n-1);
          w = matH(n,n-1) * matH(n-1,n);
      }

      // Wilkinson's original ad hoc shift
      if (iter == 10)
      {
        exshift += x;
        for (int i = low; i <= n; i++)
          matH(i,i) -= x;
        s = ei_abs(matH(n,n-1)) + ei_abs(matH(n-1,n-2));
        x = y = 0.75 * s;
        w = -0.4375 * s * s;
      }

      // MATLAB's new ad hoc shift
      if (iter == 30)
      {
        s = (y - x) / 2.0;
        s = s * s + w;
        if (s > 0)
        {
          s = ei_sqrt(s);
          if (y < x)
            s = -s;
          s = x - w / ((y - x) / 2.0 + s);
          for (int i = low; i <= n; i++)
            matH(i,i) -= s;
          exshift += s;
          x = y = w = 0.964;
        }
      }

      iter = iter + 1;   // (Could check iteration count here.)

      // Look for two consecutive small sub-diagonal elements
      int m = n-2;
      while (m >= l)
      {
        z = matH(m,m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / matH(m+1,m) + matH(m,m+1);
        q = matH(m+1,m+1) - z - r - s;
        r = matH(m+2,m+1);
        s = ei_abs(p) + ei_abs(q) + ei_abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) {
          break;
        }
        if (ei_abs(matH(m,m-1)) * (ei_abs(q) + ei_abs(r)) <
          eps * (ei_abs(p) * (ei_abs(matH(m-1,m-1)) + ei_abs(z) +
          ei_abs(matH(m+1,m+1)))))
        {
          break;
        }
        m--;
      }

      for (int i = m+2; i <= n; i++)
      {
        matH(i,i-2) = 0.0;
        if (i > m+2)
          matH(i,i-3) = 0.0;
      }

      // Double QR step involving rows l:n and columns m:n
      for (int k = m; k <= n-1; k++)
      {
        int notlast = (k != n-1);
        if (k != m) {
          p = matH(k,k-1);
          q = matH(k+1,k-1);
          r = (notlast ? matH(k+2,k-1) : 0.0);
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
            matH(k,k-1) = -s * x;
          else if (l != m)
            matH(k,k-1) = -matH(k,k-1);

          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          // Row modification
          for (int j = k; j < nn; j++)
          {
            p = matH(k,j) + q * matH(k+1,j);
            if (notlast)
            {
              p = p + r * matH(k+2,j);
              matH(k+2,j) = matH(k+2,j) - p * z;
            }
            matH(k,j) = matH(k,j) - p * x;
            matH(k+1,j) = matH(k+1,j) - p * y;
          }

          // Column modification
          for (int i = 0; i <= std::min(n,k+3); i++)
          {
            p = x * matH(i,k) + y * matH(i,k+1);
            if (notlast)
            {
              p = p + z * matH(i,k+2);
              matH(i,k+2) = matH(i,k+2) - p * r;
            }
            matH(i,k) = matH(i,k) - p;
            matH(i,k+1) = matH(i,k+1) - p * q;
          }

          // Accumulate transformations
          for (int i = low; i <= high; i++)
          {
            p = x * m_eivec(i,k) + y * m_eivec(i,k+1);
            if (notlast)
            {
              p = p + z * m_eivec(i,k+2);
              m_eivec(i,k+2) = m_eivec(i,k+2) - p * r;
            }
            m_eivec(i,k) = m_eivec(i,k) - p;
            m_eivec(i,k+1) = m_eivec(i,k+1) - p * q;
          }
        }  // (s != 0)
      }  // k loop
    }  // check convergence
  }  // while (n >= low)

  // Backsubstitute to find vectors of upper triangular form
  if (norm == 0.0)
  {
      return;
  }

  for (n = nn-1; n >= 0; n--)
  {
    p = m_eivalues[n].real();
    q = m_eivalues[n].imag();

    // Scalar vector
    if (q == 0)
    {
      int l = n;
      matH(n,n) = 1.0;
      for (int i = n-1; i >= 0; i--)
      {
        w = matH(i,i) - p;
        r = (matH.row(i).end(nn-l) * matH.col(n).end(nn-l))(0,0);

        if (m_eivalues[i].imag() < 0.0)
        {
          z = w;
          s = r;
        }
        else
        {
          l = i;
          if (m_eivalues[i].imag() == 0.0)
          {
            if (w != 0.0)
              matH(i,n) = -r / w;
            else
              matH(i,n) = -r / (eps * norm);
          }
          else // Solve real equations
          {
            x = matH(i,i+1);
            y = matH(i+1,i);
            q = (m_eivalues[i].real() - p) * (m_eivalues[i].real() - p) + m_eivalues[i].imag() * m_eivalues[i].imag();
            t = (x * s - z * r) / q;
            matH(i,n) = t;
            if (ei_abs(x) > ei_abs(z))
              matH(i+1,n) = (-r - w * t) / x;
            else
              matH(i+1,n) = (-s - y * t) / z;
          }

          // Overflow control
          t = ei_abs(matH(i,n));
          if ((eps * t) * t > 1)
            matH.col(n).end(nn-i) /= t;
        }
      }
    }
    else if (q < 0) // Complex vector
    {
      std::complex<Scalar> cc;
      int l = n-1;

      // Last vector component imaginary so matrix is triangular
      if (ei_abs(matH(n,n-1)) > ei_abs(matH(n-1,n)))
      {
        matH(n-1,n-1) = q / matH(n,n-1);
        matH(n-1,n) = -(matH(n,n) - p) / matH(n,n-1);
      }
      else
      {
        cc = cdiv<Scalar>(0.0,-matH(n-1,n),matH(n-1,n-1)-p,q);
        matH(n-1,n-1) = ei_real(cc);
        matH(n-1,n) = ei_imag(cc);
      }
      matH(n,n-1) = 0.0;
      matH(n,n) = 1.0;
      for (int i = n-2; i >= 0; i--)
      {
        Scalar ra,sa,vr,vi;
        ra = (matH.row(i).end(nn-l) * matH.col(n-1).end(nn-l)).lazy()(0,0);
        sa = (matH.row(i).end(nn-l) * matH.col(n).end(nn-l)).lazy()(0,0);
        w = matH(i,i) - p;

        if (m_eivalues[i].imag() < 0.0)
        {
          z = w;
          r = ra;
          s = sa;
        }
        else
        {
          l = i;
          if (m_eivalues[i].imag() == 0)
          {
            cc = cdiv(-ra,-sa,w,q);
            matH(i,n-1) = ei_real(cc);
            matH(i,n) = ei_imag(cc);
          }
          else
          {
            // Solve complex equations
            x = matH(i,i+1);
            y = matH(i+1,i);
            vr = (m_eivalues[i].real() - p) * (m_eivalues[i].real() - p) + m_eivalues[i].imag() * m_eivalues[i].imag() - q * q;
            vi = (m_eivalues[i].real() - p) * 2.0 * q;
            if ((vr == 0.0) && (vi == 0.0))
              vr = eps * norm * (ei_abs(w) + ei_abs(q) + ei_abs(x) + ei_abs(y) + ei_abs(z));

            cc= cdiv(x*r-z*ra+q*sa,x*s-z*sa-q*ra,vr,vi);
            matH(i,n-1) = ei_real(cc);
            matH(i,n) = ei_imag(cc);
            if (ei_abs(x) > (ei_abs(z) + ei_abs(q)))
            {
              matH(i+1,n-1) = (-ra - w * matH(i,n-1) + q * matH(i,n)) / x;
              matH(i+1,n) = (-sa - w * matH(i,n) - q * matH(i,n-1)) / x;
            }
            else
            {
              cc = cdiv(-r-y*matH(i,n-1),-s-y*matH(i,n),z,q);
              matH(i+1,n-1) = ei_real(cc);
              matH(i+1,n) = ei_imag(cc);
            }
          }

          // Overflow control
          t = std::max(ei_abs(matH(i,n-1)),ei_abs(matH(i,n)));
          if ((eps * t) * t > 1)
            matH.block(i, n-1, nn-i, 2) /= t;

        }
      }
    }
  }

  // Vectors of isolated roots
  for (int i = 0; i < nn; i++)
  {
    // FIXME again what's the purpose of this test ?
    // in this algo low==0 and high==nn-1 !!
    if (i < low || i > high)
    {
      m_eivec.row(i).end(nn-i) = matH.row(i).end(nn-i);
    }
  }

  // Back transformation to get eigenvectors of original matrix
  int bRows = high-low+1;
  for (int j = nn-1; j >= low; j--)
  {
    int bSize = std::min(j,high)-low+1;
    m_eivec.col(j).block(low, bRows) = (m_eivec.block(low, low, bRows, bSize) * matH.col(j).block(low, bSize));
  }
}

#endif // EIGEN_EIGENSOLVER_H
