// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class EigenSolver
  *
  * \brief Computes eigenvalues and eigenvectors of general matrices
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the
  * eigendecomposition; this is expected to be an instantiation of the Matrix
  * class template. Currently, only real matrices are supported.
  *
  * The eigenvalues and eigenvectors of a matrix \f$ A \f$ are scalars
  * \f$ \lambda \f$ and vectors \f$ v \f$ such that \f$ Av = \lambda v \f$.  If
  * \f$ D \f$ is a diagonal matrix with the eigenvalues on the diagonal, and
  * \f$ V \f$ is a matrix with the eigenvectors as its columns, then \f$ A V =
  * V D \f$. The matrix \f$ V \f$ is almost always invertible, in which case we
  * have \f$ A = V D V^{-1} \f$. This is called the eigendecomposition.
  *
  * The eigenvalues and eigenvectors of a matrix may be complex, even when the
  * matrix is real. However, we can choose real matrices \f$ V \f$ and \f$ D
  * \f$ satisfying \f$ A V = V D \f$, just like the eigendecomposition, if the
  * matrix \f$ D \f$ is not required to be diagonal, but if it is allowed to
  * have blocks of the form
  * \f[ \begin{bmatrix} u & v \\ -v & u \end{bmatrix} \f]
  * (where \f$ u \f$ and \f$ v \f$ are real numbers) on the diagonal.  These
  * blocks correspond to complex eigenvalue pairs \f$ u \pm iv \f$. We call
  * this variant of the eigendecomposition the pseudo-eigendecomposition.
  *
  * Call the function compute() to compute the eigenvalues and eigenvectors of
  * a given matrix. Alternatively, you can use the
  * EigenSolver(const MatrixType&) constructor which computes the eigenvalues
  * and eigenvectors at construction time. Once the eigenvalue and eigenvectors
  * are computed, they can be retrieved with the eigenvalues() and
  * eigenvectors() functions. The pseudoEigenvalueMatrix() and
  * pseudoEigenvectors() methods allow the construction of the
  * pseudo-eigendecomposition.
  *
  * The documentation for EigenSolver(const MatrixType&) contains an example of
  * the typical use of this class.
  *
  * \note this code was adapted from JAMA (public domain)
  *
  * \sa MatrixBase::eigenvalues(), class ComplexEigenSolver, class SelfAdjointEigenSolver
  */
template<typename _MatrixType> class EigenSolver
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

    /** \brief Scalar type for matrices of type \p _MatrixType. */
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    /** \brief Complex scalar type for \p _MatrixType. 
      *
      * This is \c std::complex<Scalar> if #Scalar is real (e.g.,
      * \c float or \c double) and just \c Scalar if #Scalar is
      * complex.
      */
    typedef std::complex<RealScalar> Complex;

    /** \brief Type for vector of eigenvalues as returned by eigenvalues(). 
      *
      * This is a column vector with entries of type #Complex.
      * The length of the vector is the size of \p _MatrixType.
      */
    typedef Matrix<Complex, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> EigenvalueType;

    /** \brief Type for matrix of eigenvectors as returned by eigenvectors(). 
      *
      * This is a square matrix with entries of type #Complex. 
      * The size is the same as the size of \p _MatrixType.
      */
    typedef Matrix<Complex, RowsAtCompileTime, ColsAtCompileTime, Options, MaxRowsAtCompileTime, MaxColsAtCompileTime> EigenvectorType;

    /** \brief Default constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via EigenSolver::compute(const MatrixType&).
      *
      * \sa compute() for an example.
      */
    EigenSolver() : m_eivec(), m_eivalues(), m_isInitialized(false) {}

    /** \brief Constructor; computes eigendecomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose eigendecomposition is to be computed.
      *
      * This constructor calls compute() to compute the eigenvalues
      * and eigenvectors.
      *
      * Example: \include EigenSolver_EigenSolver_MatrixType.cpp
      * Output: \verbinclude EigenSolver_EigenSolver_MatrixType.out
      *
      * \sa compute()
      */
    EigenSolver(const MatrixType& matrix)
      : m_eivec(matrix.rows(), matrix.cols()),
        m_eivalues(matrix.cols()),
        m_isInitialized(false)
    {
      compute(matrix);
    }

    /** \brief Returns the eigenvectors of given matrix. 
      *
      * \returns  %Matrix whose columns are the (possibly complex) eigenvectors.
      *
      * \pre Either the constructor EigenSolver(const MatrixType&) or the
      * member function compute(const MatrixType&) has been called before.
      *
      * Column \f$ k \f$ of the returned matrix is an eigenvector corresponding
      * to eigenvalue number \f$ k \f$ as returned by eigenvalues().  The
      * eigenvectors are normalized to have (Euclidean) norm equal to one. The
      * matrix returned by this function is the matrix \f$ V \f$ in the
      * eigendecomposition \f$ A = V D V^{-1} \f$, if it exists.
      *
      * Example: \include EigenSolver_eigenvectors.cpp
      * Output: \verbinclude EigenSolver_eigenvectors.out
      *
      * \sa eigenvalues(), pseudoEigenvectors()
      */
    EigenvectorType eigenvectors() const;

    /** \brief Returns the pseudo-eigenvectors of given matrix. 
      *
      * \returns  Const reference to matrix whose columns are the pseudo-eigenvectors.
      *
      * \pre Either the constructor EigenSolver(const MatrixType&) or
      * the member function compute(const MatrixType&) has been called
      * before.
      *
      * The real matrix \f$ V \f$ returned by this function and the
      * block-diagonal matrix \f$ D \f$ returned by pseudoEigenvalueMatrix()
      * satisfy \f$ AV = VD \f$.
      *
      * Example: \include EigenSolver_pseudoEigenvectors.cpp
      * Output: \verbinclude EigenSolver_pseudoEigenvectors.out
      *
      * \sa pseudoEigenvalueMatrix(), eigenvectors()
      */
    const MatrixType& pseudoEigenvectors() const
    {
      ei_assert(m_isInitialized && "EigenSolver is not initialized.");
      return m_eivec;
    }

    /** \brief Returns the block-diagonal matrix in the pseudo-eigendecomposition.
      *
      * \returns  A block-diagonal matrix.
      *
      * \pre Either the constructor EigenSolver(const MatrixType&) or the
      * member function compute(const MatrixType&) has been called before.
      *
      * The matrix \f$ D \f$ returned by this function is real and
      * block-diagonal. The blocks on the diagonal are either 1-by-1 or 2-by-2
      * blocks of the form
      * \f$ \begin{bmatrix} u & v \\ -v & u \end{bmatrix} \f$.
      * The matrix \f$ D \f$ and the matrix \f$ V \f$ returned by
      * pseudoEigenvectors() satisfy \f$ AV = VD \f$.
      *
      * \sa pseudoEigenvectors() for an example, eigenvalues()
      */
    MatrixType pseudoEigenvalueMatrix() const;

    /** \brief Returns the eigenvalues of given matrix. 
      *
      * \returns Column vector containing the eigenvalues.
      *
      * \pre Either the constructor EigenSolver(const MatrixType&) or the
      * member function compute(const MatrixType&) has been called before.
      *
      * The eigenvalues are repeated according to their algebraic multiplicity,
      * so there are as many eigenvalues as rows in the matrix.
      *
      * Example: \include EigenSolver_eigenvalues.cpp
      * Output: \verbinclude EigenSolver_eigenvalues.out
      *
      * \sa eigenvectors(), pseudoEigenvalueMatrix(),
      *     MatrixBase::eigenvalues()
      */
    EigenvalueType eigenvalues() const
    {
      ei_assert(m_isInitialized && "EigenSolver is not initialized.");
      return m_eivalues;
    }

    /** \brief Computes eigendecomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose eigendecomposition is to be computed.
      * \returns    Reference to \c *this
      *
      * This function computes the eigenvalues and eigenvectors of \p matrix.
      * The eigenvalues() and eigenvectors() functions can be used to retrieve
      * the computed eigendecomposition.
      *
      * The matrix is first reduced to Schur form. The Schur decomposition is
      * then used to compute the eigenvalues and eigenvectors.
      *
      * The cost of the computation is dominated by the cost of the Schur
      * decomposition, which is \f$ O(n^3) \f$ where \f$ n \f$ is the size of
      * the matrix.
      *
      * This method reuses of the allocated data in the EigenSolver object.
      *
      * Example: \include EigenSolver_compute.cpp
      * Output: \verbinclude EigenSolver_compute.out
      */
    EigenSolver& compute(const MatrixType& matrix);

  private:

    typedef Matrix<RealScalar, ColsAtCompileTime, 1, Options, MaxColsAtCompileTime, 1> RealVectorType;
    void orthes(MatrixType& matH, RealVectorType& ort);
    void hqr2(MatrixType& matH);

  protected:
    MatrixType m_eivec;
    EigenvalueType m_eivalues;
    bool m_isInitialized;
};

template<typename MatrixType>
MatrixType EigenSolver<MatrixType>::pseudoEigenvalueMatrix() const
{
  ei_assert(m_isInitialized && "EigenSolver is not initialized.");
  int n = m_eivec.cols();
  MatrixType matD = MatrixType::Zero(n,n);
  for (int i=0; i<n; ++i)
  {
    if (ei_isMuchSmallerThan(ei_imag(m_eivalues.coeff(i)), ei_real(m_eivalues.coeff(i))))
      matD.coeffRef(i,i) = ei_real(m_eivalues.coeff(i));
    else
    {
      matD.template block<2,2>(i,i) <<  ei_real(m_eivalues.coeff(i)), ei_imag(m_eivalues.coeff(i)),
                                       -ei_imag(m_eivalues.coeff(i)), ei_real(m_eivalues.coeff(i));
      ++i;
    }
  }
  return matD;
}

template<typename MatrixType>
typename EigenSolver<MatrixType>::EigenvectorType EigenSolver<MatrixType>::eigenvectors() const
{
  ei_assert(m_isInitialized && "EigenSolver is not initialized.");
  int n = m_eivec.cols();
  EigenvectorType matV(n,n);
  for (int j=0; j<n; ++j)
  {
    if (ei_isMuchSmallerThan(ei_abs(ei_imag(m_eivalues.coeff(j))), ei_abs(ei_real(m_eivalues.coeff(j)))))
    {
      // we have a real eigen value
      matV.col(j) = m_eivec.col(j).template cast<Complex>();
    }
    else
    {
      // we have a pair of complex eigen values
      for (int i=0; i<n; ++i)
      {
        matV.coeffRef(i,j)   = Complex(m_eivec.coeff(i,j),  m_eivec.coeff(i,j+1));
        matV.coeffRef(i,j+1) = Complex(m_eivec.coeff(i,j), -m_eivec.coeff(i,j+1));
      }
      matV.col(j).normalize();
      matV.col(j+1).normalize();
      ++j;
    }
  }
  return matV;
}

template<typename MatrixType>
EigenSolver<MatrixType>& EigenSolver<MatrixType>::compute(const MatrixType& matrix)
{
  assert(matrix.cols() == matrix.rows());
  int n = matrix.cols();
  m_eivalues.resize(n,1);
  m_eivec.resize(n,n);

  MatrixType matH = matrix;
  RealVectorType ort(n);

  // Reduce to Hessenberg form.
  orthes(matH, ort);

  // Reduce Hessenberg to real Schur form.
  hqr2(matH);

  m_isInitialized = true;
  return *this;
}

// Nonsymmetric reduction to Hessenberg form.
template<typename MatrixType>
void EigenSolver<MatrixType>::orthes(MatrixType& matH, RealVectorType& ort)
{
  //  This is derived from the Algol procedures orthes and ortran,
  //  by Martin and Wilkinson, Handbook for Auto. Comp.,
  //  Vol.ii-Linear Algebra, and the corresponding
  //  Fortran subroutines in EISPACK.

  int n = m_eivec.cols();
  int low = 0;
  int high = n-1;

  for (int m = low+1; m <= high-1; ++m)
  {
    // Scale column.
    RealScalar scale = matH.block(m, m-1, high-m+1, 1).cwiseAbs().sum();
    if (scale != 0.0)
    {
      // Compute Householder transformation.
      RealScalar h = 0.0;
      // FIXME could be rewritten, but this one looks better wrt cache
      for (int i = high; i >= m; i--)
      {
        ort.coeffRef(i) = matH.coeff(i,m-1)/scale;
        h += ort.coeff(i) * ort.coeff(i);
      }
      RealScalar g = ei_sqrt(h);
      if (ort.coeff(m) > 0)
        g = -g;
      h = h - ort.coeff(m) * g;
      ort.coeffRef(m) = ort.coeff(m) - g;

      // Apply Householder similarity transformation
      // H = (I-u*u'/h)*H*(I-u*u')/h)
      int bSize = high-m+1;
      matH.block(m, m, bSize, n-m).noalias() -= ((ort.segment(m, bSize)/h)
        * (ort.segment(m, bSize).transpose() *  matH.block(m, m, bSize, n-m)));

      matH.block(0, m, high+1, bSize).noalias() -= ((matH.block(0, m, high+1, bSize) * ort.segment(m, bSize))
        * (ort.segment(m, bSize)/h).transpose());

      ort.coeffRef(m) = scale*ort.coeff(m);
      matH.coeffRef(m,m-1) = scale*g;
    }
  }

  // Accumulate transformations (Algol's ortran).
  m_eivec.setIdentity();

  for (int m = high-1; m >= low+1; m--)
  {
    if (matH.coeff(m,m-1) != 0.0)
    {
      ort.segment(m+1, high-m) = matH.col(m-1).segment(m+1, high-m);

      int bSize = high-m+1;
      m_eivec.block(m, m, bSize, bSize).noalias() += ( (ort.segment(m, bSize) /  (matH.coeff(m,m-1) * ort.coeff(m)))
        * (ort.segment(m, bSize).transpose() * m_eivec.block(m, m, bSize, bSize)) );
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
void EigenSolver<MatrixType>::hqr2(MatrixType& matH)
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
  Scalar eps = ei_pow(Scalar(2),ei_is_same_type<Scalar,float>::ret ? Scalar(-23) : Scalar(-52));
  Scalar exshift = 0.0;
  Scalar p=0,q=0,r=0,s=0,z=0,t,w,x,y;

  // Store roots isolated by balanc and compute matrix norm
  // FIXME to be efficient the following would requires a triangular reduxion code
  // Scalar norm = matH.upper().cwiseAbs().sum() + matH.corner(BottomLeft,n,n).diagonal().cwiseAbs().sum();
  Scalar norm = 0.0;
  for (int j = 0; j < nn; ++j)
  {
    // FIXME what's the purpose of the following since the condition is always false
    if ((j < low) || (j > high))
    {
      m_eivalues.coeffRef(j) = Complex(matH.coeff(j,j), 0.0);
    }
    norm += matH.row(j).segment(std::max(j-1,0), nn-std::max(j-1,0)).cwiseAbs().sum();
  }

  // Outer loop over eigenvalue index
  int iter = 0;
  while (n >= low)
  {
    // Look for single small sub-diagonal element
    int l = n;
    while (l > low)
    {
      s = ei_abs(matH.coeff(l-1,l-1)) + ei_abs(matH.coeff(l,l));
      if (s == 0.0)
          s = norm;
      if (ei_abs(matH.coeff(l,l-1)) < eps * s)
        break;
      l--;
    }

    // Check for convergence
    // One root found
    if (l == n)
    {
      matH.coeffRef(n,n) = matH.coeff(n,n) + exshift;
      m_eivalues.coeffRef(n) = Complex(matH.coeff(n,n), 0.0);
      n--;
      iter = 0;
    }
    else if (l == n-1) // Two roots found
    {
      w = matH.coeff(n,n-1) * matH.coeff(n-1,n);
      p = (matH.coeff(n-1,n-1) - matH.coeff(n,n)) * Scalar(0.5);
      q = p * p + w;
      z = ei_sqrt(ei_abs(q));
      matH.coeffRef(n,n) = matH.coeff(n,n) + exshift;
      matH.coeffRef(n-1,n-1) = matH.coeff(n-1,n-1) + exshift;
      x = matH.coeff(n,n);

      // Scalar pair
      if (q >= 0)
      {
        if (p >= 0)
          z = p + z;
        else
          z = p - z;

        m_eivalues.coeffRef(n-1) = Complex(x + z, 0.0);
        m_eivalues.coeffRef(n) = Complex(z!=0.0 ? x - w / z : m_eivalues.coeff(n-1).real(), 0.0);

        x = matH.coeff(n,n-1);
        s = ei_abs(x) + ei_abs(z);
        p = x / s;
        q = z / s;
        r = ei_sqrt(p * p+q * q);
        p = p / r;
        q = q / r;

        // Row modification
        for (int j = n-1; j < nn; ++j)
        {
          z = matH.coeff(n-1,j);
          matH.coeffRef(n-1,j) = q * z + p * matH.coeff(n,j);
          matH.coeffRef(n,j) = q * matH.coeff(n,j) - p * z;
        }

        // Column modification
        for (int i = 0; i <= n; ++i)
        {
          z = matH.coeff(i,n-1);
          matH.coeffRef(i,n-1) = q * z + p * matH.coeff(i,n);
          matH.coeffRef(i,n) = q * matH.coeff(i,n) - p * z;
        }

        // Accumulate transformations
        for (int i = low; i <= high; ++i)
        {
          z = m_eivec.coeff(i,n-1);
          m_eivec.coeffRef(i,n-1) = q * z + p * m_eivec.coeff(i,n);
          m_eivec.coeffRef(i,n) = q * m_eivec.coeff(i,n) - p * z;
        }
      }
      else // Complex pair
      {
        m_eivalues.coeffRef(n-1) = Complex(x + p, z);
        m_eivalues.coeffRef(n)   = Complex(x + p, -z);
      }
      n = n - 2;
      iter = 0;
    }
    else // No convergence yet
    {
      // Form shift
      x = matH.coeff(n,n);
      y = 0.0;
      w = 0.0;
      if (l < n)
      {
          y = matH.coeff(n-1,n-1);
          w = matH.coeff(n,n-1) * matH.coeff(n-1,n);
      }

      // Wilkinson's original ad hoc shift
      if (iter == 10)
      {
        exshift += x;
        for (int i = low; i <= n; ++i)
          matH.coeffRef(i,i) -= x;
        s = ei_abs(matH.coeff(n,n-1)) + ei_abs(matH.coeff(n-1,n-2));
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
            matH.coeffRef(i,i) -= s;
          exshift += s;
          x = y = w = Scalar(0.964);
        }
      }

      iter = iter + 1;   // (Could check iteration count here.)

      // Look for two consecutive small sub-diagonal elements
      int m = n-2;
      while (m >= l)
      {
        z = matH.coeff(m,m);
        r = x - z;
        s = y - z;
        p = (r * s - w) / matH.coeff(m+1,m) + matH.coeff(m,m+1);
        q = matH.coeff(m+1,m+1) - z - r - s;
        r = matH.coeff(m+2,m+1);
        s = ei_abs(p) + ei_abs(q) + ei_abs(r);
        p = p / s;
        q = q / s;
        r = r / s;
        if (m == l) {
          break;
        }
        if (ei_abs(matH.coeff(m,m-1)) * (ei_abs(q) + ei_abs(r)) <
          eps * (ei_abs(p) * (ei_abs(matH.coeff(m-1,m-1)) + ei_abs(z) +
          ei_abs(matH.coeff(m+1,m+1)))))
        {
          break;
        }
        m--;
      }

      for (int i = m+2; i <= n; ++i)
      {
        matH.coeffRef(i,i-2) = 0.0;
        if (i > m+2)
          matH.coeffRef(i,i-3) = 0.0;
      }

      // Double QR step involving rows l:n and columns m:n
      for (int k = m; k <= n-1; ++k)
      {
        int notlast = (k != n-1);
        if (k != m) {
          p = matH.coeff(k,k-1);
          q = matH.coeff(k+1,k-1);
          r = notlast ? matH.coeff(k+2,k-1) : Scalar(0);
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
            matH.coeffRef(k,k-1) = -s * x;
          else if (l != m)
            matH.coeffRef(k,k-1) = -matH.coeff(k,k-1);

          p = p + s;
          x = p / s;
          y = q / s;
          z = r / s;
          q = q / p;
          r = r / p;

          // Row modification
          for (int j = k; j < nn; ++j)
          {
            p = matH.coeff(k,j) + q * matH.coeff(k+1,j);
            if (notlast)
            {
              p = p + r * matH.coeff(k+2,j);
              matH.coeffRef(k+2,j) = matH.coeff(k+2,j) - p * z;
            }
            matH.coeffRef(k,j) = matH.coeff(k,j) - p * x;
            matH.coeffRef(k+1,j) = matH.coeff(k+1,j) - p * y;
          }

          // Column modification
          for (int i = 0; i <= std::min(n,k+3); ++i)
          {
            p = x * matH.coeff(i,k) + y * matH.coeff(i,k+1);
            if (notlast)
            {
              p = p + z * matH.coeff(i,k+2);
              matH.coeffRef(i,k+2) = matH.coeff(i,k+2) - p * r;
            }
            matH.coeffRef(i,k) = matH.coeff(i,k) - p;
            matH.coeffRef(i,k+1) = matH.coeff(i,k+1) - p * q;
          }

          // Accumulate transformations
          for (int i = low; i <= high; ++i)
          {
            p = x * m_eivec.coeff(i,k) + y * m_eivec.coeff(i,k+1);
            if (notlast)
            {
              p = p + z * m_eivec.coeff(i,k+2);
              m_eivec.coeffRef(i,k+2) = m_eivec.coeff(i,k+2) - p * r;
            }
            m_eivec.coeffRef(i,k) = m_eivec.coeff(i,k) - p;
            m_eivec.coeffRef(i,k+1) = m_eivec.coeff(i,k+1) - p * q;
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
    p = m_eivalues.coeff(n).real();
    q = m_eivalues.coeff(n).imag();

    // Scalar vector
    if (q == 0)
    {
      int l = n;
      matH.coeffRef(n,n) = 1.0;
      for (int i = n-1; i >= 0; i--)
      {
        w = matH.coeff(i,i) - p;
        r = matH.row(i).segment(l,n-l+1).dot(matH.col(n).segment(l, n-l+1));

        if (m_eivalues.coeff(i).imag() < 0.0)
        {
          z = w;
          s = r;
        }
        else
        {
          l = i;
          if (m_eivalues.coeff(i).imag() == 0.0)
          {
            if (w != 0.0)
              matH.coeffRef(i,n) = -r / w;
            else
              matH.coeffRef(i,n) = -r / (eps * norm);
          }
          else // Solve real equations
          {
            x = matH.coeff(i,i+1);
            y = matH.coeff(i+1,i);
            q = (m_eivalues.coeff(i).real() - p) * (m_eivalues.coeff(i).real() - p) + m_eivalues.coeff(i).imag() * m_eivalues.coeff(i).imag();
            t = (x * s - z * r) / q;
            matH.coeffRef(i,n) = t;
            if (ei_abs(x) > ei_abs(z))
              matH.coeffRef(i+1,n) = (-r - w * t) / x;
            else
              matH.coeffRef(i+1,n) = (-s - y * t) / z;
          }

          // Overflow control
          t = ei_abs(matH.coeff(i,n));
          if ((eps * t) * t > 1)
            matH.col(n).tail(nn-i) /= t;
        }
      }
    }
    else if (q < 0) // Complex vector
    {
      std::complex<Scalar> cc;
      int l = n-1;

      // Last vector component imaginary so matrix is triangular
      if (ei_abs(matH.coeff(n,n-1)) > ei_abs(matH.coeff(n-1,n)))
      {
        matH.coeffRef(n-1,n-1) = q / matH.coeff(n,n-1);
        matH.coeffRef(n-1,n) = -(matH.coeff(n,n) - p) / matH.coeff(n,n-1);
      }
      else
      {
        cc = cdiv<Scalar>(0.0,-matH.coeff(n-1,n),matH.coeff(n-1,n-1)-p,q);
        matH.coeffRef(n-1,n-1) = ei_real(cc);
        matH.coeffRef(n-1,n) = ei_imag(cc);
      }
      matH.coeffRef(n,n-1) = 0.0;
      matH.coeffRef(n,n) = 1.0;
      for (int i = n-2; i >= 0; i--)
      {
        Scalar ra,sa,vr,vi;
        ra = matH.row(i).segment(l, n-l+1).dot(matH.col(n-1).segment(l, n-l+1));
        sa = matH.row(i).segment(l, n-l+1).dot(matH.col(n).segment(l, n-l+1));
        w = matH.coeff(i,i) - p;

        if (m_eivalues.coeff(i).imag() < 0.0)
        {
          z = w;
          r = ra;
          s = sa;
        }
        else
        {
          l = i;
          if (m_eivalues.coeff(i).imag() == 0)
          {
            cc = cdiv(-ra,-sa,w,q);
            matH.coeffRef(i,n-1) = ei_real(cc);
            matH.coeffRef(i,n) = ei_imag(cc);
          }
          else
          {
            // Solve complex equations
            x = matH.coeff(i,i+1);
            y = matH.coeff(i+1,i);
            vr = (m_eivalues.coeff(i).real() - p) * (m_eivalues.coeff(i).real() - p) + m_eivalues.coeff(i).imag() * m_eivalues.coeff(i).imag() - q * q;
            vi = (m_eivalues.coeff(i).real() - p) * Scalar(2) * q;
            if ((vr == 0.0) && (vi == 0.0))
              vr = eps * norm * (ei_abs(w) + ei_abs(q) + ei_abs(x) + ei_abs(y) + ei_abs(z));

            cc= cdiv(x*r-z*ra+q*sa,x*s-z*sa-q*ra,vr,vi);
            matH.coeffRef(i,n-1) = ei_real(cc);
            matH.coeffRef(i,n) = ei_imag(cc);
            if (ei_abs(x) > (ei_abs(z) + ei_abs(q)))
            {
              matH.coeffRef(i+1,n-1) = (-ra - w * matH.coeff(i,n-1) + q * matH.coeff(i,n)) / x;
              matH.coeffRef(i+1,n) = (-sa - w * matH.coeff(i,n) - q * matH.coeff(i,n-1)) / x;
            }
            else
            {
              cc = cdiv(-r-y*matH.coeff(i,n-1),-s-y*matH.coeff(i,n),z,q);
              matH.coeffRef(i+1,n-1) = ei_real(cc);
              matH.coeffRef(i+1,n) = ei_imag(cc);
            }
          }

          // Overflow control
          t = std::max(ei_abs(matH.coeff(i,n-1)),ei_abs(matH.coeff(i,n)));
          if ((eps * t) * t > 1)
            matH.block(i, n-1, nn-i, 2) /= t;

        }
      }
    }
  }

  // Vectors of isolated roots
  for (int i = 0; i < nn; ++i)
  {
    // FIXME again what's the purpose of this test ?
    // in this algo low==0 and high==nn-1 !!
    if (i < low || i > high)
    {
      m_eivec.row(i).tail(nn-i) = matH.row(i).tail(nn-i);
    }
  }

  // Back transformation to get eigenvectors of original matrix
  int bRows = high-low+1;
  for (int j = nn-1; j >= low; j--)
  {
    int bSize = std::min(j,high)-low+1;
    m_eivec.col(j).segment(low, bRows) = (m_eivec.block(low, low, bRows, bSize) * matH.col(j).segment(low, bSize));
  }
}

#endif // EIGEN_EIGENSOLVER_H
