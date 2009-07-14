// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SVD_H
#define EIGEN_SVD_H

/** \ingroup SVD_Module
  * \nonstableyet
  *
  * \class SVD
  *
  * \brief Standard SVD decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  *
  * This class performs a standard SVD decomposition of a real matrix A of size \c M x \c N.
  *
  * \sa MatrixBase::SVD()
  */
template<typename MatrixType> class SVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1,
      MinSize = EIGEN_ENUM_MIN(MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime)
    };

    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVector;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> RowVector;

    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, MatrixType::RowsAtCompileTime> MatrixUType;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, MatrixType::ColsAtCompileTime> MatrixVType;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> SingularValuesType;

  public:

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via SVD::compute(const MatrixType&).
    */
    SVD() : m_matU(), m_matV(), m_sigma(), m_isInitialized(false) {}

    SVD(const MatrixType& matrix)
      : m_matU(matrix.rows(), matrix.rows()),
        m_matV(matrix.cols(),matrix.cols()),
        m_sigma(matrix.cols()),
        m_isInitialized(false)
    {
      compute(matrix);
    }

    template<typename OtherDerived, typename ResultType>
    bool solve(const MatrixBase<OtherDerived> &b, ResultType* result) const;

    const MatrixUType& matrixU() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_matU;
    }

    const SingularValuesType& singularValues() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_sigma;
    }

    const MatrixVType& matrixV() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_matV;
    }

    void compute(const MatrixType& matrix);
    SVD& sort();

    template<typename UnitaryType, typename PositiveType>
    void computeUnitaryPositive(UnitaryType *unitary, PositiveType *positive) const;
    template<typename PositiveType, typename UnitaryType>
    void computePositiveUnitary(PositiveType *positive, UnitaryType *unitary) const;
    template<typename RotationType, typename ScalingType>
    void computeRotationScaling(RotationType *unitary, ScalingType *positive) const;
    template<typename ScalingType, typename RotationType>
    void computeScalingRotation(ScalingType *positive, RotationType *unitary) const;

  protected:
    // Computes (a^2 + b^2)^(1/2) without destructive underflow or overflow.
    inline static Scalar pythag(Scalar a, Scalar b)
    {
      Scalar abs_a = ei_abs(a);
      Scalar abs_b = ei_abs(b);
      if (abs_a > abs_b)
        return abs_a*ei_sqrt(Scalar(1.0)+ei_abs2(abs_b/abs_a));
      else
        return (abs_b == Scalar(0.0) ? Scalar(0.0) : abs_b*ei_sqrt(Scalar(1.0)+ei_abs2(abs_a/abs_b)));
    }

    inline static Scalar sign(Scalar a, Scalar b)
    {
      return (b >= Scalar(0.0) ? ei_abs(a) : -ei_abs(a));
    }
    
  protected:
    /** \internal */
    MatrixUType m_matU;
    /** \internal */
    MatrixVType m_matV;
    /** \internal */
    SingularValuesType m_sigma;
    bool m_isInitialized;
};

/** Computes / recomputes the SVD decomposition A = U S V^* of \a matrix
  *
  * \note this code has been adapted from Numerical Recipes, third edition.
  */
template<typename MatrixType>
void SVD<MatrixType>::compute(const MatrixType& matrix)
{
  const int m = matrix.rows();
  const int n = matrix.cols();

  m_matU.resize(m, m);
  m_matU.setZero();
  m_sigma.resize(n);
  m_matV.resize(n,n);

  int max_iters = 30;

  MatrixVType& V = m_matV;
  MatrixType A = matrix;
  SingularValuesType& W = m_sigma;

  bool flag;
  int i,its,j,jj,k,l,nm;
  Scalar anorm, c, f, g, h, s, scale, x, y, z;
  bool convergence = true;
  Scalar eps = precision<Scalar>();

  Matrix<Scalar,Dynamic,1> rv1(n);
  g = scale = anorm = 0;
  // Householder reduction to bidiagonal form.
  for (i=0; i<n; i++)
  {
    l = i+2;
    rv1[i] = scale*g;
    g = s = scale = 0.0;
    if (i < m)
    {
      scale = A.col(i).end(m-i).cwise().abs().sum();
      if (scale != Scalar(0))
      {
        for (k=i; k<m; k++)
        {
          A(k, i) /= scale;
          s += A(k, i)*A(k, i);
        }
        f = A(i, i);
        g = -sign( ei_sqrt(s), f );
        h = f*g - s;
        A(i, i)=f-g;
        for (j=l-1; j<n; j++)
        {
          s = A.col(i).end(m-i).dot(A.col(j).end(m-i));
          f = s/h;
          A.col(j).end(m-i) += f*A.col(i).end(m-i);
        }
        A.col(i).end(m-i) *= scale;
      }
    }
    W[i] = scale * g;
    g = s = scale = 0.0;
    if (i+1 <= m && i+1 != n)
    {
      scale = A.row(i).end(n-l+1).cwise().abs().sum();
      if (scale != Scalar(0))
      {
        for (k=l-1; k<n; k++)
        {
          A(i, k) /= scale;
          s += A(i, k)*A(i, k);
        }
        f = A(i,l-1);
        g = -sign(ei_sqrt(s),f);
        h = f*g - s;
        A(i,l-1) = f-g;
        rv1.end(n-l+1) = A.row(i).end(n-l+1)/h;
        for (j=l-1; j<m; j++)
        {
          s = A.row(j).end(n-l+1).dot(A.row(i).end(n-l+1));
          A.row(j).end(n-l+1) += s*rv1.end(n-l+1).transpose();
        }
        A.row(i).end(n-l+1) *= scale;
      }
    }
    anorm = std::max( anorm, (ei_abs(W[i])+ei_abs(rv1[i])) );
  }
  // Accumulation of right-hand transformations.
  for (i=n-1; i>=0; i--)
  {
    //Accumulation of right-hand transformations.
    if (i < n-1)
    {
      if (g != Scalar(0.0))
      {
        for (j=l; j<n; j++) //Double division to avoid possible underflow.
          V(j, i) = (A(i, j)/A(i, l))/g;
        for (j=l; j<n; j++)
        {
          s = A.row(i).end(n-l).dot(V.col(j).end(n-l));
          V.col(j).end(n-l) += s * V.col(i).end(n-l);
        }
      }
      V.row(i).end(n-l).setZero();
      V.col(i).end(n-l).setZero();
    }
    V(i, i) = 1.0;
    g = rv1[i];
    l = i;
  }
  // Accumulation of left-hand transformations.
  for (i=std::min(m,n)-1; i>=0; i--)
  {
    l = i+1;
    g = W[i];
    if (n-l>0)
      A.row(i).end(n-l).setZero();
    if (g != Scalar(0.0))
    {
      g = Scalar(1.0)/g;
      for (j=l; j<n; j++)
      {
        s = A.col(i).end(m-l).dot(A.col(j).end(m-l));
        f = (s/A(i,i))*g;
        A.col(j).end(m-i) += f * A.col(i).end(m-i);
      }
      A.col(i).end(m-i) *= g;
    }
    else
      A.col(i).end(m-i).setZero();
    ++A(i,i);
  }
  // Diagonalization of the bidiagonal form: Loop over
  // singular values, and over allowed iterations.
  for (k=n-1; k>=0; k--)
  {
    for (its=0; its<max_iters; its++)
    {
      flag = true;
      for (l=k; l>=0; l--)
      {
        // Test for splitting.
        nm = l-1;
        // Note that rv1[1] is always zero.
        //if ((double)(ei_abs(rv1[l])+anorm) == anorm)
        if (l==0 || ei_abs(rv1[l]) <= eps*anorm)
        {
          flag = false;
          break;
        }
        //if ((double)(ei_abs(W[nm])+anorm) == anorm)
        if (ei_abs(W[nm]) <= eps*anorm)
          break;
      }
      if (flag)
      {
        c = 0.0;  //Cancellation of rv1[l], if l > 0.
        s = 1.0;
        for (i=l ;i<k+1; i++)
        {
          f = s*rv1[i];
          rv1[i] = c*rv1[i];
          //if ((double)(ei_abs(f)+anorm) == anorm)
          if (ei_abs(f) <= eps*anorm)
            break;
          g = W[i];
          h = pythag(f,g);
          W[i] = h;
          h = Scalar(1.0)/h;
          c = g*h;
          s = -f*h;
          for (j=0; j<m; j++)
          {
            y = A(j,nm);
            z = A(j,i);
            A(j,nm) = y*c + z*s;
            A(j,i)  = z*c - y*s;
          }
        }
      }
      z = W[k];
      if (l == k)  //Convergence.
      {
        if (z < 0.0) { // Singular value is made nonnegative.
          W[k] = -z;
          V.col(k) = -V.col(k);
        }
        break;
      }
      if (its+1 == max_iters)
      {
        convergence = false;
      }
      x = W[l]; // Shift from bottom 2-by-2 minor.
      nm = k-1;
      y = W[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y-z)*(y+z) + (g-h)*(g+h))/(Scalar(2.0)*h*y);
      g = pythag(f,1.0);
      f = ((x-z)*(x+z) + h*((y/(f+sign(g,f)))-h))/x;
      c = s = 1.0;
      //Next QR transformation:
      for (j=l; j<=nm; j++)
      {
        i = j+1;
        g = rv1[i];
        y = W[i];
        h = s*g;
        g = c*g;
        z = pythag(f,h);
        rv1[j] = z;
        c = f/z;
        s = h/z;
        f = x*c + g*s;
        g = g*c - x*s;
        h = y*s;
        y *= c;
        for (jj=0; jj<n; jj++)
        {
          x = V(jj,j);
          z = V(jj,i);
          V(jj,j) = x*c + z*s;
          V(jj,i) = z*c - x*s;
        }
        z = pythag(f,h);
        W[j] = z;
        // Rotation can be arbitrary if z = 0.
        if (z!=Scalar(0))
        {
          z = Scalar(1.0)/z;
          c = f*z;
          s = h*z;
        }
        f = c*g + s*y;
        x = c*y - s*g;
        for (jj=0; jj<m; jj++)
        {
          y = A(jj,j);
          z = A(jj,i);
          A(jj,j) = y*c + z*s;
          A(jj,i) = z*c - y*s;
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      W[k]   = x;
    }
  }

  // sort the singular values:
  {
    for (int i=0; i<n; i++)
    {
      int k;
      W.end(n-i).minCoeff(&k);
      if (k != i)
      {
        std::swap(W[k],W[i]);
        A.col(i).swap(A.col(k));
        V.col(i).swap(V.col(k));
      }
    }
  }
  m_matU.setZero();
  if (m>=n)
    m_matU.block(0,0,m,n) = A;
  else
    m_matU = A.block(0,0,m,m);

  m_isInitialized = true;
}

template<typename MatrixType>
SVD<MatrixType>& SVD<MatrixType>::sort()
{
  ei_assert(m_isInitialized && "SVD is not initialized.");

  int mu = m_matU.rows();
  int mv = m_matV.rows();
  int n  = m_matU.cols();

  for (int i=0; i<n; ++i)
  {
    int  k = i;
    Scalar p = m_sigma.coeff(i);

    for (int j=i+1; j<n; ++j)
    {
      if (m_sigma.coeff(j) > p)
      {
        k = j;
        p = m_sigma.coeff(j);
      }
    }
    if (k != i)
    {
      m_sigma.coeffRef(k) = m_sigma.coeff(i);  // i.e.
      m_sigma.coeffRef(i) = p;                 // swaps the i-th and the k-th elements

      int j = mu;
      for(int s=0; j!=0; ++s, --j)
        std::swap(m_matU.coeffRef(s,i), m_matU.coeffRef(s,k));

      j = mv;
      for (int s=0; j!=0; ++s, --j)
        std::swap(m_matV.coeffRef(s,i), m_matV.coeffRef(s,k));
    }
  }
  return *this;
}

/** \returns the solution of \f$ A x = b \f$ using the current SVD decomposition of A.
  * The parts of the solution corresponding to zero singular values are ignored.
  *
  * \sa MatrixBase::svd(), LU::solve(), LLT::solve()
  */
template<typename MatrixType>
template<typename OtherDerived, typename ResultType>
bool SVD<MatrixType>::solve(const MatrixBase<OtherDerived> &b, ResultType* result) const
{
  ei_assert(m_isInitialized && "SVD is not initialized.");

  const int rows = m_matU.rows();
  ei_assert(b.rows() == rows);

  result->resize(m_matV.rows(), b.cols());

  Scalar maxVal = m_sigma.cwise().abs().maxCoeff();
  for (int j=0; j<b.cols(); ++j)
  {
    Matrix<Scalar,MatrixUType::RowsAtCompileTime,1> aux = m_matU.transpose() * b.col(j);

    for (int i = 0; i <m_matU.cols(); ++i)
    {
      Scalar si = m_sigma.coeff(i);
      if (ei_isMuchSmallerThan(ei_abs(si),maxVal))
        aux.coeffRef(i) = 0;
      else
        aux.coeffRef(i) /= si;
    }

    result->col(j) = m_matV * aux;
  }
  return true;
}

/** Computes the polar decomposition of the matrix, as a product unitary x positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * Only for square matrices.
  *
  * \sa computePositiveUnitary(), computeRotationScaling()
  */
template<typename MatrixType>
template<typename UnitaryType, typename PositiveType>
void SVD<MatrixType>::computeUnitaryPositive(UnitaryType *unitary,
                                             PositiveType *positive) const
{
  ei_assert(m_isInitialized && "SVD is not initialized.");
  ei_assert(m_matU.cols() == m_matV.cols() && "Polar decomposition is only for square matrices");
  if(unitary) *unitary = m_matU * m_matV.adjoint();
  if(positive) *positive = m_matV * m_sigma.asDiagonal() * m_matV.adjoint();
}

/** Computes the polar decomposition of the matrix, as a product positive x unitary.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * Only for square matrices.
  *
  * \sa computeUnitaryPositive(), computeRotationScaling()
  */
template<typename MatrixType>
template<typename UnitaryType, typename PositiveType>
void SVD<MatrixType>::computePositiveUnitary(UnitaryType *positive,
                                             PositiveType *unitary) const
{
  ei_assert(m_isInitialized && "SVD is not initialized.");
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  if(unitary) *unitary = m_matU * m_matV.adjoint();
  if(positive) *positive = m_matU * m_sigma.asDiagonal() * m_matU.adjoint();
}

/** decomposes the matrix as a product rotation x scaling, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * This method requires the Geometry module.
  *
  * \sa computeScalingRotation(), computeUnitaryPositive()
  */
template<typename MatrixType>
template<typename RotationType, typename ScalingType>
void SVD<MatrixType>::computeRotationScaling(RotationType *rotation, ScalingType *scaling) const
{
  ei_assert(m_isInitialized && "SVD is not initialized.");
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  Scalar x = (m_matU * m_matV.adjoint()).determinant(); // so x has absolute value 1
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> sv(m_sigma);
  sv.coeffRef(0) *= x;
  if(scaling) scaling->lazyAssign(m_matV * sv.asDiagonal() * m_matV.adjoint());
  if(rotation)
  {
    MatrixType m(m_matU);
    m.col(0) /= x;
    rotation->lazyAssign(m * m_matV.adjoint());
  }
}

/** decomposes the matrix as a product scaling x rotation, the scaling being
  * not necessarily positive.
  *
  * If either pointer is zero, the corresponding computation is skipped.
  *
  * This method requires the Geometry module.
  *
  * \sa computeRotationScaling(), computeUnitaryPositive()
  */
template<typename MatrixType>
template<typename ScalingType, typename RotationType>
void SVD<MatrixType>::computeScalingRotation(ScalingType *scaling, RotationType *rotation) const
{
  ei_assert(m_isInitialized && "SVD is not initialized.");
  ei_assert(m_matU.rows() == m_matV.rows() && "Polar decomposition is only for square matrices");
  Scalar x = (m_matU * m_matV.adjoint()).determinant(); // so x has absolute value 1
  Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> sv(m_sigma);
  sv.coeffRef(0) *= x;
  if(scaling) scaling->lazyAssign(m_matU * sv.asDiagonal() * m_matU.adjoint());
  if(rotation)
  {
    MatrixType m(m_matU);
    m.col(0) /= x;
    rotation->lazyAssign(m * m_matV.adjoint());
  }
}


/** \svd_module
  * \returns the SVD decomposition of \c *this
  */
template<typename Derived>
inline SVD<typename MatrixBase<Derived>::PlainMatrixType>
MatrixBase<Derived>::svd() const
{
  return SVD<PlainMatrixType>(derived());
}

#endif // EIGEN_SVD_H
