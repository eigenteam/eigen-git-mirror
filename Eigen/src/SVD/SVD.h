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

template<typename MatrixType, typename Rhs> struct ei_svd_solve_impl;

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
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;

    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      PacketSize = ei_packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1,
      MinSize = EIGEN_ENUM_MIN(RowsAtCompileTime, ColsAtCompileTime)
    };

    typedef Matrix<Scalar, RowsAtCompileTime, 1> ColVector;
    typedef Matrix<Scalar, ColsAtCompileTime, 1> RowVector;

    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime> MatrixUType;
    typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixVType;
    typedef Matrix<Scalar, ColsAtCompileTime, 1> SingularValuesType;

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

    /** \returns a solution of \f$ A x = b \f$ using the current SVD decomposition of A.
      *
      * \param b the right-hand-side of the equation to solve.
      *
      * \note_about_checking_solutions
      *
      * \note_about_arbitrary_choice_of_solution
      * \note_about_using_kernel_to_study_multiple_solutions
      *
      * \sa MatrixBase::svd(),
      */
    template<typename Rhs>
    inline const ei_svd_solve_impl<MatrixType, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return ei_svd_solve_impl<MatrixType, Rhs>(*this, b.derived());
    }

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

    SVD& compute(const MatrixType& matrix);

    template<typename UnitaryType, typename PositiveType>
    void computeUnitaryPositive(UnitaryType *unitary, PositiveType *positive) const;
    template<typename PositiveType, typename UnitaryType>
    void computePositiveUnitary(PositiveType *positive, UnitaryType *unitary) const;
    template<typename RotationType, typename ScalingType>
    void computeRotationScaling(RotationType *unitary, ScalingType *positive) const;
    template<typename ScalingType, typename RotationType>
    void computeScalingRotation(ScalingType *positive, RotationType *unitary) const;

    inline int rows() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_rows;
    }

    inline int cols() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_cols;
    }
    
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
    int m_rows, m_cols;
};

/** Computes / recomputes the SVD decomposition A = U S V^* of \a matrix
  *
  * \note this code has been adapted from Numerical Recipes, third edition.
  *
  * \returns a reference to *this
  */
template<typename MatrixType>
SVD<MatrixType>& SVD<MatrixType>::compute(const MatrixType& matrix)
{
  const int m = m_rows = matrix.rows();
  const int n = m_cols = matrix.cols();

  m_matU.resize(m, m);
  m_matU.setZero();
  m_sigma.resize(n);
  m_matV.resize(n,n);

  int max_iters = 30;

  MatrixVType& V = m_matV;
  MatrixType A = matrix;
  SingularValuesType& W = m_sigma;

  bool flag;
  int i,its,j,k,l,nm;
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
          s = A.col(j).end(m-i).dot(A.col(i).end(m-i));
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
          s = A.row(i).end(n-l+1).dot(A.row(j).end(n-l+1));
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
          s = V.col(j).end(n-l).dot(A.row(i).end(n-l));
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
      if (m-l)
      {
        for (j=l; j<n; j++)
        {
          s = A.col(j).end(m-l).dot(A.col(i).end(m-l));
          f = (s/A(i,i))*g;
          A.col(j).end(m-i) += f * A.col(i).end(m-i);
        }
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
          V.applyOnTheRight(i,nm,PlanarRotation<Scalar>(c,s));
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
        V.applyOnTheRight(i,j,PlanarRotation<Scalar>(c,s));

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
        A.applyOnTheRight(i,j,PlanarRotation<Scalar>(c,s));
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
      W.end(n-i).maxCoeff(&k);
      if (k != 0)
      {
        k += i;
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
  return *this;
}

template<typename MatrixType,typename Rhs>
struct ei_traits<ei_svd_solve_impl<MatrixType,Rhs> >
{
  typedef Matrix<typename Rhs::Scalar,
                 MatrixType::ColsAtCompileTime,
                 Rhs::ColsAtCompileTime,
                 Rhs::PlainMatrixType::Options,
                 MatrixType::MaxColsAtCompileTime,
                 Rhs::MaxColsAtCompileTime> ReturnMatrixType;
};

template<typename MatrixType, typename Rhs>
struct ei_svd_solve_impl : public ReturnByValue<ei_svd_solve_impl<MatrixType, Rhs> >
{
  typedef typename ei_cleantype<typename Rhs::Nested>::type RhsNested;
  typedef SVD<MatrixType> SVDType;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename MatrixType::Scalar Scalar;
  const SVDType& m_svd;
  const typename Rhs::Nested m_rhs;

  ei_svd_solve_impl(const SVDType& svd, const Rhs& rhs)
    : m_svd(svd), m_rhs(rhs)
  {}

  inline int rows() const { return m_svd.cols(); }
  inline int cols() const { return m_rhs.cols(); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    ei_assert(m_rhs.rows() == m_svd.rows());

    dst.resize(rows(), cols());

    for (int j=0; j<cols(); ++j)
    {
      Matrix<Scalar,SVDType::RowsAtCompileTime,1> aux = m_svd.matrixU().adjoint() * m_rhs.col(j);

      for (int i = 0; i <m_svd.rows(); ++i)
      {
        Scalar si = m_svd.singularValues().coeff(i);
        if(si == RealScalar(0))
          aux.coeffRef(i) = Scalar(0);
        else
          aux.coeffRef(i) /= si;
      }

      dst.col(j) = m_svd.matrixV() * aux;
    }
  }
};

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
