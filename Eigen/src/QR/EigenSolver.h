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
  *
  * \note this code was adapted from JAMA (public domain)
  *
  * \sa MatrixBase::eigenvalues()
  */
template<typename _MatrixType> class EigenSolver
{
  public:

    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;

    EigenSolver(const MatrixType& matrix)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_eivalr(matrix.cols()), m_eivali(matrix.cols()),
        m_H(matrix.rows(), matrix.cols()),
        m_ort(matrix.cols())
    {
      _compute(matrix);
    }

    MatrixType eigenvectors(void) const { return m_eivec; }

    VectorType eigenvalues(void) const { return m_eivalr; }

  private:

    void _compute(const MatrixType& matrix);

    void tridiagonalization(void);
    void tql2(void);

    void orthes(void);
    void hqr2(void);

  protected:
    MatrixType m_eivec;
    VectorType m_eivalr, m_eivali;
    MatrixType m_H;
    VectorType m_ort;
    bool m_isSymmetric;
};

template<typename MatrixType>
void EigenSolver<MatrixType>::_compute(const MatrixType& matrix)
{
  assert(matrix.cols() == matrix.rows());

  m_isSymmetric = true;
  int n = matrix.cols();
  for (int j = 0; (j < n) && m_isSymmetric; j++) {
      for (int i = 0; (i < j) && m_isSymmetric; i++) {
        m_isSymmetric = (matrix(i,j) == matrix(j,i));
      }
  }

  m_eivalr.resize(n,1);
  m_eivali.resize(n,1);

  if (m_isSymmetric)
  {
    m_eivec = matrix;

    // Tridiagonalize.
    tridiagonalization();

    // Diagonalize.
    tql2();
  }
  else
  {
    m_H = matrix;
    m_ort.resize(n, 1);

    // Reduce to Hessenberg form.
    orthes();

    // Reduce Hessenberg to real Schur form.
    hqr2();
  }
  std::cout << m_eivali.transpose() << "\n";
}


// Symmetric Householder reduction to tridiagonal form.
template<typename MatrixType>
void EigenSolver<MatrixType>::tridiagonalization(void)
{

//  This is derived from the Algol procedures tred2 by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  int n = m_eivec.cols();
  m_eivalr = m_eivec.row(m_eivalr.size()-1);

  // Householder reduction to tridiagonal form.
  for (int i = n-1; i > 0; i--)
  {
    // Scale to avoid under/overflow.
    Scalar scale = 0.0;
    Scalar h = 0.0;
    scale = m_eivalr.start(i).cwiseAbs().sum();

    if (scale == 0.0)
    {
      m_eivali[i] = m_eivalr[i-1];
      m_eivalr.start(i) = m_eivec.row(i-1).start(i);
      m_eivec.corner(TopLeft, i, i) = m_eivec.corner(TopLeft, i, i).diagonal().asDiagonal();
    }
    else
    {
      // Generate Householder vector.
      m_eivalr.start(i) /= scale;
      h = m_eivalr.start(i).cwiseAbs2().sum();

      Scalar f = m_eivalr[i-1];
      Scalar g = ei_sqrt(h);
      if (f > 0)
        g = -g;
      m_eivali[i] = scale * g;
      h = h - f * g;
      m_eivalr[i-1] = f - g;
      m_eivali.start(i).setZero();

      // Apply similarity transformation to remaining columns.
      for (int j = 0; j < i; j++)
      {
        f = m_eivalr[j];
        m_eivec(j,i) = f;
        g = m_eivali[j] + m_eivec(j,j) * f;
        int bSize = i-j-1;
        if (bSize>0)
        {
          g += (m_eivec.col(j).block(j+1, bSize).transpose() * m_eivalr.block(j+1, bSize))(0,0);
          m_eivali.block(j+1, bSize) += m_eivec.col(j).block(j+1, bSize) * f;
        }
        m_eivali[j] = g;
      }

      f = (m_eivali.start(i).transpose() * m_eivalr.start(i))(0,0);
      m_eivali.start(i) = (m_eivali.start(i) - (f / (h + h)) * m_eivalr.start(i))/h;

      m_eivec.corner(TopLeft, i, i).lower() -=
        ( (m_eivali.start(i) * m_eivalr.start(i).transpose()).lazy()
        + (m_eivalr.start(i) * m_eivali.start(i).transpose()).lazy());

      m_eivalr.start(i) = m_eivec.row(i-1).start(i);
      m_eivec.row(i).start(i).setZero();
    }
    m_eivalr[i] = h;
  }

  // Accumulate transformations.
  for (int i = 0; i < n-1; i++)
  {
    m_eivec(n-1,i) = m_eivec(i,i);
    m_eivec(i,i) = 1.0;
    Scalar h = m_eivalr[i+1];
    // FIXME this does not looks very stable ;)
    if (h != 0.0)
    {
      m_eivalr.start(i+1) = m_eivec.col(i+1).start(i+1) / h;
      m_eivec.corner(TopLeft, i+1, i+1) -= m_eivalr.start(i+1)
        * ( m_eivec.col(i+1).start(i+1).transpose() * m_eivec.corner(TopLeft, i+1, i+1) );
    }
    m_eivec.col(i+1).start(i+1).setZero();
  }
  m_eivalr = m_eivec.row(m_eivalr.size()-1);
  m_eivec.row(m_eivalr.size()-1).setZero();
  m_eivec(n-1,n-1) = 1.0;
  m_eivali[0] = 0.0;
}


// Symmetric tridiagonal QL algorithm.
template<typename MatrixType>
void EigenSolver<MatrixType>::tql2(void)
{

//  This is derived from the Algol procedures tql2, by
//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
//  Fortran subroutine in EISPACK.

  int n = m_eivalr.size();

  for (int i = 1; i < n; i++) {
      m_eivali[i-1] = m_eivali[i];
  }
  m_eivali[n-1] = 0.0;

  Scalar f = 0.0;
  Scalar tst1 = 0.0;
  Scalar eps = std::pow(2.0,-52.0);
  for (int l = 0; l < n; l++)
  {
    // Find small subdiagonal element
    tst1 = std::max(tst1,ei_abs(m_eivalr[l]) + ei_abs(m_eivali[l]));
    int m = l;

    while ( (m < n) && (ei_abs(m_eivali[m]) > eps*tst1) )
      m++;

    // If m == l, m_eivalr[l] is an eigenvalue,
    // otherwise, iterate.
    if (m > l)
    {
      int iter = 0;
      do
      {
        iter = iter + 1;

        // Compute implicit shift
        Scalar g = m_eivalr[l];
        Scalar p = (m_eivalr[l+1] - g) / (2.0 * m_eivali[l]);
        Scalar r = hypot(p,1.0);
        if (p < 0)
          r = -r;

        m_eivalr[l] = m_eivali[l] / (p + r);
        m_eivalr[l+1] = m_eivali[l] * (p + r);
        Scalar dl1 = m_eivalr[l+1];
        Scalar h = g - m_eivalr[l];
        if (l+2<n)
          m_eivalr.end(n-l-2) -= VectorType::constant(n-l-2, h);
        f = f + h;

        // Implicit QL transformation.
        p = m_eivalr[m];
        Scalar c = 1.0;
        Scalar c2 = c;
        Scalar c3 = c;
        Scalar el1 = m_eivali[l+1];
        Scalar s = 0.0;
        Scalar s2 = 0.0;
        for (int i = m-1; i >= l; i--)
        {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * m_eivali[i];
          h = c * p;
          r = hypot(p,m_eivali[i]);
          m_eivali[i+1] = s * r;
          s = m_eivali[i] / r;
          c = p / r;
          p = c * m_eivalr[i] - s * g;
          m_eivalr[i+1] = h + s * (c * g + s * m_eivalr[i]);

          // Accumulate transformation.
          for (int k = 0; k < n; k++)
          {
            h = m_eivec(k,i+1);
            m_eivec(k,i+1) = s * m_eivec(k,i) + c * h;
            m_eivec(k,i) = c * m_eivec(k,i) - s * h;
          }
        }
        p = -s * s2 * c3 * el1 * m_eivali[l] / dl1;
        m_eivali[l] = s * p;
        m_eivalr[l] = c * p;

        // Check for convergence.
      } while (ei_abs(m_eivali[l]) > eps*tst1);
    }
    m_eivalr[l] = m_eivalr[l] + f;
    m_eivali[l] = 0.0;
  }

  // Sort eigenvalues and corresponding vectors.
  // TODO use a better sort algorithm !!
  for (int i = 0; i < n-1; i++)
  {
    int k = i;
    Scalar minValue = m_eivalr[i];
    for (int j = i+1; j < n; j++)
    {
      if (m_eivalr[j] < minValue)
      {
        k = j;
        minValue = m_eivalr[j];
      }
    }
    if (k != i)
    {
      std::swap(m_eivalr[i], m_eivalr[k]);
      m_eivec.col(i).swap(m_eivec.col(k));
    }
  }
}


// Nonsymmetric reduction to Hessenberg form.
template<typename MatrixType>
void EigenSolver<MatrixType>::orthes(void)
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
    Scalar scale = m_H.block(m, m-1, high-m+1, 1).cwiseAbs().sum();
    if (scale != 0.0)
    {
      // Compute Householder transformation.
      Scalar h = 0.0;
      // FIXME could be rewritten, but this one looks better wrt cache
      for (int i = high; i >= m; i--)
      {
        m_ort[i] = m_H(i,m-1)/scale;
        h += m_ort[i] * m_ort[i];
      }
      Scalar g = ei_sqrt(h);
      if (m_ort[m] > 0)
        g = -g;
      h = h - m_ort[m] * g;
      m_ort[m] = m_ort[m] - g;

      // Apply Householder similarity transformation
      // H = (I-u*u'/h)*H*(I-u*u')/h)
      int bSize = high-m+1;
      m_H.block(m, m, bSize, n-m) -= ((m_ort.block(m, bSize)/h)
        * (m_ort.block(m, bSize).transpose() *  m_H.block(m, m, bSize, n-m)).lazy()).lazy();

      m_H.block(0, m, high+1, bSize) -= ((m_H.block(0, m, high+1, bSize) * m_ort.block(m, bSize)).lazy()
        * (m_ort.block(m, bSize)/h).transpose()).lazy();

      m_ort[m] = scale*m_ort[m];
      m_H(m,m-1) = scale*g;
    }
  }

  // Accumulate transformations (Algol's ortran).
  m_eivec.setIdentity();

  for (int m = high-1; m >= low+1; m--)
  {
    if (m_H(m,m-1) != 0.0)
    {
      m_ort.block(m+1, high-m) = m_H.col(m-1).block(m+1, high-m);

      int bSize = high-m+1;
      m_eivec.block(m, m, bSize, bSize) += ( (m_ort.block(m, bSize) /  (m_H(m,m-1) * m_ort[m] ) )
        * (m_ort.block(m, bSize).transpose() * m_eivec.block(m, m, bSize, bSize)).lazy());
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
template<typename MatrixType>
void EigenSolver<MatrixType>::hqr2(void)
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
  // Scalar norm = m_H.upper().cwiseAbs().sum() + m_H.corner(BottomLeft,n,n).diagonal().cwiseAbs().sum();
  Scalar norm = 0.0;
  for (int j = 0; j < nn; j++)
  {
    // FIXME what's the purpose of the following since the condition is always false
    if ((j < low) || (j > high))
    {
      m_eivalr[j] = m_H(j,j);
      m_eivali[j] = 0.0;
    }
    norm += m_H.col(j).start(std::min(j+1,nn)).cwiseAbs().sum();
  }

  // Outer loop over eigenvalue index
  int iter = 0;
  while (n >= low)
  {
    // Look for single small sub-diagonal element
    int l = n;
    while (l > low)
    {
      s = ei_abs(m_H(l-1,l-1)) + ei_abs(m_H(l,l));
      if (s == 0.0)
          s = norm;
      if (ei_abs(m_H(l,l-1)) < eps * s)
        break;
      l--;
    }

    // Check for convergence
    // One root found
    if (l == n)
    {
      m_H(n,n) = m_H(n,n) + exshift;
      m_eivalr[n] = m_H(n,n);
      m_eivali[n] = 0.0;
      n--;
      iter = 0;
    }
    else if (l == n-1) // Two roots found
    {
      w = m_H(n,n-1) * m_H(n-1,n);
      p = (m_H(n-1,n-1) - m_H(n,n)) / 2.0;
      q = p * p + w;
      z = ei_sqrt(ei_abs(q));
      m_H(n,n) = m_H(n,n) + exshift;
      m_H(n-1,n-1) = m_H(n-1,n-1) + exshift;
      x = m_H(n,n);

      // Scalar pair
      if (q >= 0)
      {
        if (p >= 0)
          z = p + z;
        else
          z = p - z;

        m_eivalr[n-1] = x + z;
        m_eivalr[n] = m_eivalr[n-1];
        if (z != 0.0)
          m_eivalr[n] = x - w / z;

        m_eivali[n-1] = 0.0;
        m_eivali[n] = 0.0;
        x = m_H(n,n-1);
        s = ei_abs(x) + ei_abs(z);
        p = x / s;
        q = z / s;
        r = ei_sqrt(p * p+q * q);
        p = p / r;
        q = q / r;

        // Row modification
        for (int j = n-1; j < nn; j++)
        {
          z = m_H(n-1,j);
          m_H(n-1,j) = q * z + p * m_H(n,j);
          m_H(n,j) = q * m_H(n,j) - p * z;
        }

        // Column modification
        for (int i = 0; i <= n; i++)
        {
          z = m_H(i,n-1);
          m_H(i,n-1) = q * z + p * m_H(i,n);
          m_H(i,n) = q * m_H(i,n) - p * z;
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
        m_eivalr[n-1] = x + p;
        m_eivalr[n] = x + p;
        m_eivali[n-1] = z;
        m_eivali[n] = -z;
      }
      n = n - 2;
      iter = 0;
    }
    else // No convergence yet
    {
      // Form shift
      x = m_H(n,n);
      y = 0.0;
      w = 0.0;
      if (l < n)
      {
          y = m_H(n-1,n-1);
          w = m_H(n,n-1) * m_H(n-1,n);
      }

      // Wilkinson's original ad hoc shift
      if (iter == 10)
      {
        exshift += x;
        for (int i = low; i <= n; i++)
          m_H(i,i) -= x;
        s = ei_abs(m_H(n,n-1)) + ei_abs(m_H(n-1,n-2));
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
            m_H(i,i) -= s;
          exshift += s;
          x = y = w = 0.964;
        }
      }

      iter = iter + 1;   // (Could check iteration count here.)

      // Look for two consecutive small sub-diagonal elements
      int m = n-2;
      while (m >= l)
      {
        z = m_H(m,m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / m_H(m+1,m) + m_H(m,m+1);
        q = m_H(m+1,m+1) - z - r - s;
        r = m_H(m+2,m+1);
        s = ei_abs(p) + ei_abs(q) + ei_abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) {
          break;
        }
        if (ei_abs(m_H(m,m-1)) * (ei_abs(q) + ei_abs(r)) <
          eps * (ei_abs(p) * (ei_abs(m_H(m-1,m-1)) + ei_abs(z) +
          ei_abs(m_H(m+1,m+1)))))
        {
          break;
        }
        m--;
      }

      for (int i = m+2; i <= n; i++)
      {
        m_H(i,i-2) = 0.0;
        if (i > m+2)
          m_H(i,i-3) = 0.0;
      }

      // Double QR step involving rows l:n and columns m:n
      for (int k = m; k <= n-1; k++)
      {
        int notlast = (k != n-1);
        if (k != m) {
          p = m_H(k,k-1);
          q = m_H(k+1,k-1);
          r = (notlast ? m_H(k+2,k-1) : 0.0);
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
            m_H(k,k-1) = -s * x;
          else if (l != m)
            m_H(k,k-1) = -m_H(k,k-1);

          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          // Row modification
          for (int j = k; j < nn; j++)
          {
            p = m_H(k,j) + q * m_H(k+1,j);
            if (notlast)
            {
              p = p + r * m_H(k+2,j);
              m_H(k+2,j) = m_H(k+2,j) - p * z;
            }
            m_H(k,j) = m_H(k,j) - p * x;
            m_H(k+1,j) = m_H(k+1,j) - p * y;
          }

          // Column modification
          for (int i = 0; i <= std::min(n,k+3); i++)
          {
            p = x * m_H(i,k) + y * m_H(i,k+1);
            if (notlast)
            {
              p = p + z * m_H(i,k+2);
              m_H(i,k+2) = m_H(i,k+2) - p * r;
            }
            m_H(i,k) = m_H(i,k) - p;
            m_H(i,k+1) = m_H(i,k+1) - p * q;
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
    p = m_eivalr[n];
    q = m_eivali[n];

    // Scalar vector
    if (q == 0)
    {
      int l = n;
      m_H(n,n) = 1.0;
      for (int i = n-1; i >= 0; i--)
      {
        w = m_H(i,i) - p;
        r = (m_H.row(i).end(nn-l) * m_H.col(n).end(nn-l))(0,0);

        if (m_eivali[i] < 0.0)
        {
          z = w;
          s = r;
        }
        else
        {
          l = i;
          if (m_eivali[i] == 0.0)
          {
            if (w != 0.0)
              m_H(i,n) = -r / w;
            else
              m_H(i,n) = -r / (eps * norm);
          }
          else // Solve real equations
          {
            x = m_H(i,i+1);
            y = m_H(i+1,i);
            q = (m_eivalr[i] - p) * (m_eivalr[i] - p) + m_eivali[i] * m_eivali[i];
            t = (x * s - z * r) / q;
            m_H(i,n) = t;
            if (ei_abs(x) > ei_abs(z))
              m_H(i+1,n) = (-r - w * t) / x;
            else
              m_H(i+1,n) = (-s - y * t) / z;
          }

          // Overflow control
          t = ei_abs(m_H(i,n));
          if ((eps * t) * t > 1)
            m_H.col(n).end(nn-i) /= t;
        }
      }
    }
    else if (q < 0) // Complex vector
    {
      std::complex<Scalar> cc;
      int l = n-1;

      // Last vector component imaginary so matrix is triangular
      if (ei_abs(m_H(n,n-1)) > ei_abs(m_H(n-1,n)))
      {
        m_H(n-1,n-1) = q / m_H(n,n-1);
        m_H(n-1,n) = -(m_H(n,n) - p) / m_H(n,n-1);
      }
      else
      {
        cc = cdiv<Scalar>(0.0,-m_H(n-1,n),m_H(n-1,n-1)-p,q);
        m_H(n-1,n-1) = ei_real(cc);
        m_H(n-1,n) = ei_imag(cc);
      }
      m_H(n,n-1) = 0.0;
      m_H(n,n) = 1.0;
      for (int i = n-2; i >= 0; i--)
      {
        Scalar ra,sa,vr,vi;
        ra = (m_H.row(i).end(nn-l) * m_H.col(n-1).end(nn-l)).lazy()(0,0);
        sa = (m_H.row(i).end(nn-l) * m_H.col(n).end(nn-l)).lazy()(0,0);
        w = m_H(i,i) - p;

        if (m_eivali[i] < 0.0)
        {
          z = w;
          r = ra;
          s = sa;
        }
        else
        {
          l = i;
          if (m_eivali[i] == 0)
          {
            cc = cdiv(-ra,-sa,w,q);
            m_H(i,n-1) = ei_real(cc);
            m_H(i,n) = ei_imag(cc);
          }
          else
          {
            // Solve complex equations
            x = m_H(i,i+1);
            y = m_H(i+1,i);
            vr = (m_eivalr[i] - p) * (m_eivalr[i] - p) + m_eivali[i] * m_eivali[i] - q * q;
            vi = (m_eivalr[i] - p) * 2.0 * q;
            if ((vr == 0.0) && (vi == 0.0))
              vr = eps * norm * (ei_abs(w) + ei_abs(q) + ei_abs(x) + ei_abs(y) + ei_abs(z));

            cc= cdiv(x*r-z*ra+q*sa,x*s-z*sa-q*ra,vr,vi);
            m_H(i,n-1) = ei_real(cc);
            m_H(i,n) = ei_imag(cc);
            if (ei_abs(x) > (ei_abs(z) + ei_abs(q)))
            {
              m_H(i+1,n-1) = (-ra - w * m_H(i,n-1) + q * m_H(i,n)) / x;
              m_H(i+1,n) = (-sa - w * m_H(i,n) - q * m_H(i,n-1)) / x;
            }
            else
            {
              cc = cdiv(-r-y*m_H(i,n-1),-s-y*m_H(i,n),z,q);
              m_H(i+1,n-1) = ei_real(cc);
              m_H(i+1,n) = ei_imag(cc);
            }
          }

          // Overflow control
          t = std::max(ei_abs(m_H(i,n-1)),ei_abs(m_H(i,n)));
          if ((eps * t) * t > 1)
            m_H.block(i, n-1, nn-i, 2) /= t;

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
      m_eivec.row(i).end(nn-i) = m_H.row(i).end(nn-i);
    }
  }

  // Back transformation to get eigenvectors of original matrix
  int bRows = high-low+1;
  for (int j = nn-1; j >= low; j--)
  {
    int bSize = std::min(j,high)-low+1;
    m_eivec.col(j).block(low, bRows) = (m_eivec.block(low, low, bRows, bSize) * m_H.col(j).block(low, bSize));
  }
}

#endif // EIGEN_EIGENSOLVER_H
