// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
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
  *
  *
  * \class SVD
  *
  * \brief Standard SVD decomposition of a matrix and associated features
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  *
  * This class performs a standard SVD decomposition of a real matrix A of size \c M x \c N.
  *
  * Requires M >= N, in other words, at least as many rows as columns.
  *
  * \sa MatrixBase::SVD()
  */
template<typename _MatrixType> class SVD
{
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef typename MatrixType::Index Index;

    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      PacketSize = ei_packet_traits<Scalar>::size,
      AlignmentMask = int(PacketSize)-1,
      MinSize = EIGEN_SIZE_MIN_PREFER_DYNAMIC(RowsAtCompileTime, ColsAtCompileTime),
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
      MatrixOptions = MatrixType::Options
    };

    typedef typename ei_plain_col_type<MatrixType>::type ColVector;
    typedef typename ei_plain_row_type<MatrixType>::type RowVector;

    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime, MatrixOptions, MaxRowsAtCompileTime, MaxRowsAtCompileTime> MatrixUType;
    typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime, MatrixOptions, MaxColsAtCompileTime, MaxColsAtCompileTime> MatrixVType;
    typedef ColVector SingularValuesType;

    /**
    * \brief Default Constructor.
    *
    * The default constructor is useful in cases in which the user intends to
    * perform decompositions via SVD::compute(const MatrixType&).
    */
    SVD() : m_matU(), m_matV(), m_sigma(), m_isInitialized(false) {}

    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa JacobiSVD()
      */
    SVD(Index rows, Index cols) : m_matU(rows, rows),
                              m_matV(cols,cols),
                              m_sigma(std::min(rows, cols)),
                              m_workMatrix(rows, cols),
                              m_rv1(cols),
                              m_isInitialized(false) {}

    SVD(const MatrixType& matrix) : m_matU(matrix.rows(), matrix.rows()),
                                    m_matV(matrix.cols(),matrix.cols()),
                                    m_sigma(std::min(matrix.rows(), matrix.cols())),
                                    m_workMatrix(matrix.rows(), matrix.cols()),
                                    m_rv1(matrix.cols()),
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
      *
      * \sa MatrixBase::svd(),
      */
    template<typename Rhs>
    inline const ei_solve_retval<SVD, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return ei_solve_retval<SVD, Rhs>(*this, b.derived());
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

    inline Index rows() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_rows;
    }

    inline Index cols() const
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
    MatrixType m_workMatrix;
    RowVector m_rv1;
    bool m_isInitialized;
    Index m_rows, m_cols;
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
  const Index m = m_rows = matrix.rows();
  const Index n = m_cols = matrix.cols();

  m_matU.resize(m, m);
  m_matU.setZero();
  m_sigma.resize(n);
  m_matV.resize(n,n);
  m_workMatrix = matrix;

  Index max_iters = 30;

  MatrixVType& V = m_matV;
  MatrixType& A = m_workMatrix;
  SingularValuesType& W = m_sigma;

  bool flag;
  Index i=0,its=0,j=0,k=0,l=0,nm=0;
  Scalar anorm, c, f, g, h, s, scale, x, y, z;
  bool convergence = true;
  Scalar eps = NumTraits<Scalar>::dummy_precision();

  m_rv1.resize(n);

  g = scale = anorm = 0;
  // Householder reduction to bidiagonal form.
  for (i=0; i<n; i++)
  {
    l = i+2;
    m_rv1[i] = scale*g;
    g = s = scale = 0.0;
    if (i < m)
    {
      scale = A.col(i).tail(m-i).cwiseAbs().sum();
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
          s = A.col(j).tail(m-i).dot(A.col(i).tail(m-i));
          f = s/h;
          A.col(j).tail(m-i) += f*A.col(i).tail(m-i);
        }
        A.col(i).tail(m-i) *= scale;
      }
    }
    W[i] = scale * g;
    g = s = scale = 0.0;
    if (i+1 <= m && i+1 != n)
    {
      scale = A.row(i).tail(n-l+1).cwiseAbs().sum();
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
        m_rv1.tail(n-l+1) = A.row(i).tail(n-l+1)/h;
        for (j=l-1; j<m; j++)
        {
          s = A.row(i).tail(n-l+1).dot(A.row(j).tail(n-l+1));
          A.row(j).tail(n-l+1) += s*m_rv1.tail(n-l+1).transpose();
        }
        A.row(i).tail(n-l+1) *= scale;
      }
    }
    anorm = std::max( anorm, (ei_abs(W[i])+ei_abs(m_rv1[i])) );
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
          s = V.col(j).tail(n-l).dot(A.row(i).tail(n-l));
          V.col(j).tail(n-l) += s * V.col(i).tail(n-l);
        }
      }
      V.row(i).tail(n-l).setZero();
      V.col(i).tail(n-l).setZero();
    }
    V(i, i) = 1.0;
    g = m_rv1[i];
    l = i;
  }
  // Accumulation of left-hand transformations.
  for (i=std::min(m,n)-1; i>=0; i--)
  {
    l = i+1;
    g = W[i];
    if (n-l>0)
      A.row(i).tail(n-l).setZero();
    if (g != Scalar(0.0))
    {
      g = Scalar(1.0)/g;
      if (m-l)
      {
        for (j=l; j<n; j++)
        {
          s = A.col(j).tail(m-l).dot(A.col(i).tail(m-l));
          f = (s/A(i,i))*g;
          A.col(j).tail(m-i) += f * A.col(i).tail(m-i);
        }
      }
      A.col(i).tail(m-i) *= g;
    }
    else
      A.col(i).tail(m-i).setZero();
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
        if (l==0 || ei_abs(m_rv1[l]) <= eps*anorm)
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
          f = s*m_rv1[i];
          m_rv1[i] = c*m_rv1[i];
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
      g = m_rv1[nm];
      h = m_rv1[k];
      f = ((y-z)*(y+z) + (g-h)*(g+h))/(Scalar(2.0)*h*y);
      g = pythag(f,1.0);
      f = ((x-z)*(x+z) + h*((y/(f+sign(g,f)))-h))/x;
      c = s = 1.0;
      //Next QR transformation:
      for (j=l; j<=nm; j++)
      {
        i = j+1;
        g = m_rv1[i];
        y = W[i];
        h = s*g;
        g = c*g;

        z = pythag(f,h);
        m_rv1[j] = z;
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
      m_rv1[l] = 0.0;
      m_rv1[k] = f;
      W[k]   = x;
    }
  }

  // sort the singular values:
  {
    for (Index i=0; i<n; i++)
    {
      Index k;
      W.tail(n-i).maxCoeff(&k);
      if (k != 0)
      {
        k += i;
        std::swap(W[k],W[i]);
        A.col(i).swap(A.col(k));
        V.col(i).swap(V.col(k));
      }
    }
  }
  m_matU.leftCols(n) = A;
  m_matU.rightCols(m-n).setZero();

  // Gram Schmidt orthogonalization to fill up U
  for (int col = A.cols(); col < A.rows(); ++col)
  {
    typename MatrixUType::ColXpr colVec = m_matU.col(col);
    colVec(col) = 1;
    for (int prevCol = 0; prevCol < col; ++prevCol)
    {
      typename MatrixUType::ColXpr prevColVec = m_matU.col(prevCol);
      colVec -= colVec.dot(prevColVec)*prevColVec;
    }
    m_matU.col(col) = colVec.normalized();
  }

  m_isInitialized = true;
  return *this;
}

template<typename _MatrixType, typename Rhs>
struct ei_solve_retval<SVD<_MatrixType>, Rhs>
  : ei_solve_retval_base<SVD<_MatrixType>, Rhs>
{
  EIGEN_MAKE_SOLVE_HELPERS(SVD<_MatrixType>,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    ei_assert(rhs().rows() == dec().rows());

    for (Index j=0; j<cols(); ++j)
    {
      Matrix<Scalar,MatrixType::RowsAtCompileTime,1> aux = dec().matrixU().adjoint() * rhs().col(j);

      for (Index i = 0; i < dec().singularValues().size(); ++i)
      {
        Scalar si = dec().singularValues().coeff(i);
        if(si == RealScalar(0))
          aux.coeffRef(i) = Scalar(0);
        else
          aux.coeffRef(i) /= si;
      }
      aux.tail(aux.size() - dec().singularValues().size()).setZero();

      const Index minsize = std::min(dec().rows(),dec().cols());
      dst.col(j).head(minsize) = aux.head(minsize);
      if(dec().cols()>dec().rows()) dst.col(j).tail(cols()-minsize).setZero();
      dst.col(j) = dec().matrixV() * dst.col(j);
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
inline SVD<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::svd() const
{
  return SVD<PlainObject>(derived());
}

#endif // EIGEN_SVD_H
