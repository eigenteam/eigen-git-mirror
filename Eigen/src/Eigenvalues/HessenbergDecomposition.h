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

#ifndef EIGEN_HESSENBERGDECOMPOSITION_H
#define EIGEN_HESSENBERGDECOMPOSITION_H

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class HessenbergDecomposition
  *
  * \brief Reduces a square matrix to Hessenberg form by an orthogonal similarity transformation
  *
  * \tparam _MatrixType the type of the matrix of which we are computing the Hessenberg decomposition
  *
  * This class performs an Hessenberg decomposition of a matrix \f$ A \f$. In
  * the real case, the Hessenberg decomposition consists of an orthogonal
  * matrix \f$ Q \f$ and a Hessenberg matrix \f$ H \f$ such that \f$ A = Q H
  * Q^T \f$. An orthogonal matrix is a matrix whose inverse equals its
  * transpose (\f$ Q^{-1} = Q^T \f$). A Hessenberg matrix has zeros below the
  * subdiagonal, so it is almost upper triangular. The Hessenberg decomposition
  * of a complex matrix is \f$ A = Q H Q^* \f$ with \f$ Q \f$ unitary (that is,
  * \f$ Q^{-1} = Q^* \f$).
  *
  * Call the function compute() to compute the Hessenberg decomposition of a
  * given matrix. Alternatively, you can use the 
  * HessenbergDecomposition(const MatrixType&) constructor which computes the
  * Hessenberg decomposition at construction time. Once the decomposition is
  * computed, you can use the matrixH() and matrixQ() functions to construct
  * the matrices H and Q in the decomposition. 
  *
  * The documentation for matrixH() contains an example of the typical use of
  * this class.
  *
  * \sa class ComplexSchur, class Tridiagonalization, \ref QR_Module "QR Module"
  */
template<typename _MatrixType> class HessenbergDecomposition
{
  public:

    typedef _MatrixType MatrixType;
    enum {
      Size = MatrixType::RowsAtCompileTime,
      SizeMinusOne = Size == Dynamic ? Dynamic : Size - 1,
      Options = MatrixType::Options,
      MaxSize = MatrixType::MaxRowsAtCompileTime,
      MaxSizeMinusOne = MaxSize == Dynamic ? Dynamic : MaxSize - 1
    };

    /** \brief Scalar type for matrices of type \p _MatrixType. */
    typedef typename MatrixType::Scalar Scalar;

    /** \brief Type for vector of Householder coefficients.
      *
      * This is column vector with entries of type #Scalar. The length of the
      * vector is one less than the size of \p _MatrixType, if it is a
      * fixed-side type.
      */
    typedef Matrix<Scalar, SizeMinusOne, 1, Options & ~RowMajor, MaxSizeMinusOne, 1> CoeffVectorType;

    /** \brief Default constructor; the decomposition will be computed later.
      *
      * \param [in] size  The size of the matrix whose Hessenberg decomposition will be computed.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via compute().  The \p size parameter is only
      * used as a hint. It is not an error to give a wrong \p size, but it may
      * impair performance.
      *
      * \sa compute() for an example.
      */
    HessenbergDecomposition(int size = Size==Dynamic ? 2 : Size)
      : m_matrix(size,size)
    {
      if(size>1)
        m_hCoeffs.resize(size-1);
    }

    /** \brief Constructor; computes Hessenberg decomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose Hessenberg decomposition is to be computed.
      *
      * This constructor calls compute() to compute the Hessenberg
      * decomposition.
      *
      * \sa matrixH() for an example.
      */
    HessenbergDecomposition(const MatrixType& matrix)
      : m_matrix(matrix)
    {
      if(matrix.rows()<2)
        return;
      m_hCoeffs.resize(matrix.rows()-1,1);
      _compute(m_matrix, m_hCoeffs);
    }

    /** \brief Computes Hessenberg decomposition of given matrix. 
      * 
      * \param[in]  matrix  Square matrix whose Hessenberg decomposition is to be computed.
      *
      * The Hessenberg decomposition is computed by bringing the columns of the
      * matrix successively in the required form using Householder reflections
      * (see, e.g., Algorithm 7.4.2 in Golub \& Van Loan, <i>%Matrix
      * Computations</i>). The cost is \f$ 10n^3/3 \f$ flops, where \f$ n \f$
      * denotes the size of the given matrix.
      *
      * This method reuses of the allocated data in the HessenbergDecomposition
      * object.
      *
      * Example: \include HessenbergDecomposition_compute.cpp
      * Output: \verbinclude HessenbergDecomposition_compute.out
      */
    void compute(const MatrixType& matrix)
    {
      m_matrix = matrix;
      if(matrix.rows()<2)
        return;
      m_hCoeffs.resize(matrix.rows()-1,1);
      _compute(m_matrix, m_hCoeffs);
    }

    /** \brief Returns the Householder coefficients.
      *
      * \returns a const reference to the vector of Householder coefficients
      *
      * \pre Either the constructor HessenbergDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * The Householder coefficients allow the reconstruction of the matrix 
      * \f$ Q \f$ in the Hessenberg decomposition from the packed data.
      *
      * \sa packedMatrix(), \ref Householder_Module "Householder module"
      */
    const CoeffVectorType& householderCoefficients() const { return m_hCoeffs; }

    /** \brief Returns the internal representation of the decomposition 
      *
      *	\returns a const reference to a matrix with the internal representation
      *	         of the decomposition.
      *
      * \pre Either the constructor HessenbergDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * The returned matrix contains the following information:
      *  - the upper part and lower sub-diagonal represent the Hessenberg matrix H
      *  - the rest of the lower part contains the Householder vectors that, combined with
      *    Householder coefficients returned by householderCoefficients(),
      *    allows to reconstruct the matrix Q as 
      *       \f$ Q = H_{N-1} \ldots H_1 H_0 \f$.
      *    Here, the matrices \f$ H_i \f$ are the Householder transformations 
      *       \f$ H_i = (I - h_i v_i v_i^T) \f$
      *    where \f$ h_i \f$ is the \f$ i \f$th Householder coefficient and 
      *    \f$ v_i \f$ is the Householder vector defined by
      *       \f$ v_i = [ 0, \ldots, 0, 1, M(i+2,i), \ldots, M(N-1,i) ]^T \f$
      *    with M the matrix returned by this function.
      *
      * See LAPACK for further details on this packed storage.
      *
      * Example: \include HessenbergDecomposition_packedMatrix.cpp
      * Output: \verbinclude HessenbergDecomposition_packedMatrix.out
      *
      * \sa householderCoefficients()
      */
    const MatrixType& packedMatrix(void) const { return m_matrix; }

    /** \brief Reconstructs the orthogonal matrix Q in the decomposition 
      *
      * \returns the matrix Q
      *
      * \pre Either the constructor HessenbergDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * This function reconstructs the matrix Q from the Householder
      * coefficients and the packed matrix stored internally. This
      * reconstruction requires \f$ 4n^3 / 3 \f$ flops.
      *
      * \sa matrixH() for an example
      */
    MatrixType matrixQ() const;

    /** \brief Constructs the Hessenberg matrix H in the decomposition
      *
      * \returns the matrix H
      *
      * \pre Either the constructor HessenbergDecomposition(const MatrixType&)
      * or the member function compute(const MatrixType&) has been called
      * before to compute the Hessenberg decomposition of a matrix.
      *
      * This function copies the matrix H from internal data. The upper part
      * (including the subdiagonal) of the packed matrix as returned by
      * packedMatrix() contains the matrix H. This function copies those
      * entries in a newly created matrix and sets the remaining entries to
      * zero. It may sometimes be sufficient to directly use the packed matrix
      * instead of creating a new one.
      *
      * Example: \include HessenbergDecomposition_matrixH.cpp
      * Output: \verbinclude HessenbergDecomposition_matrixH.out
      *
      * \sa matrixQ(), packedMatrix()
      */
    MatrixType matrixH() const;

  private:

    static void _compute(MatrixType& matA, CoeffVectorType& hCoeffs);
    typedef Matrix<Scalar, 1, Size, Options | RowMajor, 1, MaxSize> VectorType;
    typedef typename NumTraits<Scalar>::Real RealScalar;

  protected:
    MatrixType m_matrix;
    CoeffVectorType m_hCoeffs;
};

#ifndef EIGEN_HIDE_HEAVY_CODE

/** \internal
  * Performs a tridiagonal decomposition of \a matA in place.
  *
  * \param matA the input selfadjoint matrix
  * \param hCoeffs returned Householder coefficients
  *
  * The result is written in the lower triangular part of \a matA.
  *
  * Implemented from Golub's "%Matrix Computations", algorithm 8.3.1.
  *
  * \sa packedMatrix()
  */
template<typename MatrixType>
void HessenbergDecomposition<MatrixType>::_compute(MatrixType& matA, CoeffVectorType& hCoeffs)
{
  assert(matA.rows()==matA.cols());
  int n = matA.rows();
  VectorType temp(n);
  for (int i = 0; i<n-1; ++i)
  {
    // let's consider the vector v = i-th column starting at position i+1
    int remainingSize = n-i-1;
    RealScalar beta;
    Scalar h;
    matA.col(i).tail(remainingSize).makeHouseholderInPlace(h, beta);
    matA.col(i).coeffRef(i+1) = beta;
    hCoeffs.coeffRef(i) = h;

    // Apply similarity transformation to remaining columns,
    // i.e., compute A = H A H'

    // A = H A
    matA.corner(BottomRight, remainingSize, remainingSize)
        .applyHouseholderOnTheLeft(matA.col(i).tail(remainingSize-1), h, &temp.coeffRef(0));

    // A = A H'
    matA.corner(BottomRight, n, remainingSize)
        .applyHouseholderOnTheRight(matA.col(i).tail(remainingSize-1).conjugate(), ei_conj(h), &temp.coeffRef(0));
  }
}

template<typename MatrixType>
typename HessenbergDecomposition<MatrixType>::MatrixType
HessenbergDecomposition<MatrixType>::matrixQ() const
{
  int n = m_matrix.rows();
  MatrixType matQ = MatrixType::Identity(n,n);
  VectorType temp(n);
  for (int i = n-2; i>=0; i--)
  {
    matQ.corner(BottomRight,n-i-1,n-i-1)
        .applyHouseholderOnTheLeft(m_matrix.col(i).tail(n-i-2), ei_conj(m_hCoeffs.coeff(i)), &temp.coeffRef(0,0));
  }
  return matQ;
}

#endif // EIGEN_HIDE_HEAVY_CODE

template<typename MatrixType>
typename HessenbergDecomposition<MatrixType>::MatrixType
HessenbergDecomposition<MatrixType>::matrixH() const
{
  // FIXME should this function (and other similar) rather take a matrix as argument
  // and fill it (to avoid temporaries)
  int n = m_matrix.rows();
  MatrixType matH = m_matrix;
  if (n>2)
    matH.corner(BottomLeft,n-2, n-2).template triangularView<Lower>().setZero();
  return matH;
}

#endif // EIGEN_HESSENBERGDECOMPOSITION_H
