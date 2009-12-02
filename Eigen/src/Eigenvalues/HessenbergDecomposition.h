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

#ifndef EIGEN_HESSENBERGDECOMPOSITION_H
#define EIGEN_HESSENBERGDECOMPOSITION_H

/** \eigenvalues_module \ingroup Eigenvalues_Module
  * \nonstableyet
  *
  * \class HessenbergDecomposition
  *
  * \brief Reduces a squared matrix to an Hessemberg form
  *
  * \param MatrixType the type of the matrix of which we are computing the Hessenberg decomposition
  *
  * This class performs an Hessenberg decomposition of a matrix \f$ A \f$ such that:
  * \f$ A = Q H Q^* \f$ where \f$ Q \f$ is unitary and \f$ H \f$ a Hessenberg matrix.
  *
  * \sa class Tridiagonalization, class Qr
  */
template<typename _MatrixType> class HessenbergDecomposition
{
  public:

    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    enum {
      Size = MatrixType::RowsAtCompileTime,
      SizeMinusOne = MatrixType::RowsAtCompileTime==Dynamic
                   ? Dynamic
                   : MatrixType::RowsAtCompileTime-1
    };

    typedef Matrix<Scalar, SizeMinusOne, 1> CoeffVectorType;
    typedef Matrix<RealScalar, Size, 1> DiagonalType;
    typedef Matrix<RealScalar, SizeMinusOne, 1> SubDiagonalType;

    typedef typename Diagonal<MatrixType,0>::RealReturnType DiagonalReturnType;

    typedef typename Diagonal<
        Block<MatrixType,SizeMinusOne,SizeMinusOne>,0 >::RealReturnType SubDiagonalReturnType;

    /** This constructor initializes a HessenbergDecomposition object for
      * further use with HessenbergDecomposition::compute()
      */
    HessenbergDecomposition(int size = Size==Dynamic ? 2 : Size)
      : m_matrix(size,size), m_hCoeffs(size-1)
    {}

    HessenbergDecomposition(const MatrixType& matrix)
      : m_matrix(matrix),
        m_hCoeffs(matrix.cols()-1)
    {
      _compute(m_matrix, m_hCoeffs);
    }

    /** Computes or re-compute the Hessenberg decomposition for the matrix \a matrix.
      *
      * This method allows to re-use the allocated data.
      */
    void compute(const MatrixType& matrix)
    {
      m_matrix = matrix;
      m_hCoeffs.resize(matrix.rows()-1,1);
      _compute(m_matrix, m_hCoeffs);
    }

    /** \returns a const reference to the householder coefficients allowing to
      * reconstruct the matrix Q from the packed data.
      *
      * \sa packedMatrix()
      */
    const CoeffVectorType& householderCoefficients() const { return m_hCoeffs; }

    /** \returns a const reference to the internal representation of the decomposition.
      *
      * The returned matrix contains the following information:
      *  - the upper part and lower sub-diagonal represent the Hessenberg matrix H
      *  - the rest of the lower part contains the Householder vectors that, combined with
      *    Householder coefficients returned by householderCoefficients(),
      *    allows to reconstruct the matrix Q as follow:
      *       Q = H_{N-1} ... H_1 H_0
      *    where the matrices H are the Householder transformation:
      *       H_i = (I - h_i * v_i * v_i')
      *    where h_i == householderCoefficients()[i] and v_i is a Householder vector:
      *       v_i = [ 0, ..., 0, 1, M(i+2,i), ..., M(N-1,i) ]
      *
      * See LAPACK for further details on this packed storage.
      */
    const MatrixType& packedMatrix(void) const { return m_matrix; }

    MatrixType matrixQ() const;
    MatrixType matrixH() const;

  private:

    static void _compute(MatrixType& matA, CoeffVectorType& hCoeffs);

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
  * Implemented from Golub's "Matrix Computations", algorithm 8.3.1.
  *
  * \sa packedMatrix()
  */
template<typename MatrixType>
void HessenbergDecomposition<MatrixType>::_compute(MatrixType& matA, CoeffVectorType& hCoeffs)
{
  assert(matA.rows()==matA.cols());
  int n = matA.rows();
  Matrix<Scalar,1,Dynamic> temp(n);
  for (int i = 0; i<n-1; ++i)
  {
    // let's consider the vector v = i-th column starting at position i+1
    int remainingSize = n-i-1;
    RealScalar beta;
    Scalar h;
    matA.col(i).end(remainingSize).makeHouseholderInPlace(h, beta);
    matA.col(i).coeffRef(i+1) = beta;
    hCoeffs.coeffRef(i) = h;

    // Apply similarity transformation to remaining columns,
    // i.e., compute A = H A H'

    // A = H A
    matA.corner(BottomRight, remainingSize, remainingSize)
        .applyHouseholderOnTheLeft(matA.col(i).end(remainingSize-1), h, &temp.coeffRef(0));

    // A = A H'
    matA.corner(BottomRight, n, remainingSize)
        .applyHouseholderOnTheRight(matA.col(i).end(remainingSize-1).conjugate(), ei_conj(h), &temp.coeffRef(0));
  }
}

/** reconstructs and returns the matrix Q */
template<typename MatrixType>
typename HessenbergDecomposition<MatrixType>::MatrixType
HessenbergDecomposition<MatrixType>::matrixQ() const
{
  int n = m_matrix.rows();
  MatrixType matQ = MatrixType::Identity(n,n);
  Matrix<Scalar,1,MatrixType::ColsAtCompileTime> temp(n);
  for (int i = n-2; i>=0; i--)
  {
    matQ.corner(BottomRight,n-i-1,n-i-1)
        .applyHouseholderOnTheLeft(m_matrix.col(i).end(n-i-2), ei_conj(m_hCoeffs.coeff(i)), &temp.coeffRef(0,0));
  }
  return matQ;
}

#endif // EIGEN_HIDE_HEAVY_CODE

/** constructs and returns the matrix H.
  * Note that the matrix H is equivalent to the upper part of the packed matrix
  * (including the lower sub-diagonal). Therefore, it might be often sufficient
  * to directly use the packed matrix instead of creating a new one.
  */
template<typename MatrixType>
typename HessenbergDecomposition<MatrixType>::MatrixType
HessenbergDecomposition<MatrixType>::matrixH() const
{
  // FIXME should this function (and other similar) rather take a matrix as argument
  // and fill it (to avoid temporaries)
  int n = m_matrix.rows();
  MatrixType matH = m_matrix;
  if (n>2)
    matH.corner(BottomLeft,n-2, n-2).template triangularView<LowerTriangular>().setZero();
  return matH;
}

#endif // EIGEN_HESSENBERGDECOMPOSITION_H
