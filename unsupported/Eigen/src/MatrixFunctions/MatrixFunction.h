// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Jitse Niesen <jitse@maths.leeds.ac.uk>
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

#ifndef EIGEN_MATRIX_FUNCTION
#define EIGEN_MATRIX_FUNCTION

template <typename Scalar>
struct ei_stem_function
{
  typedef std::complex<typename NumTraits<Scalar>::Real> ComplexScalar;
  typedef ComplexScalar type(ComplexScalar, int);
};

/** \ingroup MatrixFunctions_Module
  *
  * \brief Compute a matrix function.
  *
  * \param[in]  M      argument of matrix function, should be a square matrix.
  * \param[in]  f      an entire function; \c f(x,n) should compute the n-th derivative of f at x.
  * \param[out] result pointer to the matrix in which to store the result, \f$ f(M) \f$.
  *
  * This function computes \f$ f(A) \f$ and stores the result in the
  * matrix pointed to by \p result.
  *
  * %Matrix functions are defined as follows.  Suppose that \f$ f \f$
  * is an entire function (that is, a function on the complex plane
  * that is everywhere complex differentiable).  Then its Taylor
  * series
  * \f[ f(0) + f'(0) x + \frac{f''(0)}{2} x^2 + \frac{f'''(0)}{3!} x^3 + \cdots \f]
  * converges to \f$ f(x) \f$. In this case, we can define the matrix
  * function by the same series:
  * \f[ f(M) = f(0) + f'(0) M + \frac{f''(0)}{2} M^2 + \frac{f'''(0)}{3!} M^3 + \cdots \f]
  *
  * This routine uses the algorithm described in:
  * Philip Davies and Nicholas J. Higham, 
  * "A Schur-Parlett algorithm for computing matrix functions", 
  * <em>SIAM J. %Matrix Anal. Applic.</em>, <b>25</b>:464&ndash;485, 2003.
  *
  * The actual work is done by the MatrixFunction class.
  *
  * Example: The following program checks that
  * \f[ \exp \left[ \begin{array}{ccc} 
  *       0 & \frac14\pi & 0 \\ 
  *       -\frac14\pi & 0 & 0 \\
  *       0 & 0 & 0 
  *     \end{array} \right] = \left[ \begin{array}{ccc}
  *       \frac12\sqrt2 & -\frac12\sqrt2 & 0 \\
  *       \frac12\sqrt2 & \frac12\sqrt2 & 0 \\
  *       0 & 0 & 1
  *     \end{array} \right]. \f]
  * This corresponds to a rotation of \f$ \frac14\pi \f$ radians around
  * the z-axis. This is the same example as used in the documentation
  * of ei_matrix_exponential().
  *
  * Note that the function \c expfn is defined for complex numbers \c x, 
  * even though the matrix \c A is over the reals.
  *
  * \include MatrixFunction.cpp
  * Output: \verbinclude MatrixFunction.out
  */
template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_function(const MatrixBase<Derived>& M, 
					    typename ei_stem_function<typename ei_traits<Derived>::Scalar>::type f,
					    typename MatrixBase<Derived>::PlainMatrixType* result);

#include "MatrixFunctionAtomic.h"


/** \ingroup MatrixFunctions_Module 
  * \brief Helper class for computing matrix functions. 
  */
template <typename MatrixType, int IsComplex = NumTraits<typename ei_traits<MatrixType>::Scalar>::IsComplex>
class MatrixFunction
{  
  private:

    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    typedef typename ei_stem_function<Scalar>::type StemFunction;

  public:

    /** \brief Constructor. Computes matrix function. 
      *
      * \param[in]  A      argument of matrix function, should be a square matrix.
      * \param[in]  f      an entire function; \c f(x,n) should compute the n-th derivative of f at x.
      * \param[out] result pointer to the matrix in which to store the result, \f$ f(A) \f$.
      *
      * This function computes \f$ f(A) \f$ and stores the result in
      * the matrix pointed to by \p result.
      *
      * See ei_matrix_function() for details.
      */
    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result);
};


/** \ingroup MatrixFunctions_Module 
  * \brief Partial specialization of MatrixFunction for real matrices \internal 
  */
template <typename MatrixType>
class MatrixFunction<MatrixType, 0>
{  
  private:

    typedef ei_traits<MatrixType> Traits;
    typedef typename Traits::Scalar Scalar;
    static const int Rows = Traits::RowsAtCompileTime;
    static const int Cols = Traits::ColsAtCompileTime;
    static const int Options = MatrixType::Options;
    static const int MaxRows = Traits::MaxRowsAtCompileTime;
    static const int MaxCols = Traits::MaxColsAtCompileTime;

    typedef std::complex<Scalar> ComplexScalar;
    typedef Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;
    typedef typename ei_stem_function<Scalar>::type StemFunction;

  public:

    /** \brief Constructor. Computes matrix function. 
      *
      * \param[in]  A      argument of matrix function, should be a square matrix.
      * \param[in]  f      an entire function; \c f(x,n) should compute the n-th derivative of f at x.
      * \param[out] result pointer to the matrix in which to store the result, \f$ f(A) \f$.
      *
      * This function converts the real matrix \c A to a complex matrix,
      * uses MatrixFunction<MatrixType,1> and then converts the result back to
      * a real matrix.
      */
    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result) 
    {
      ComplexMatrix CA = A.template cast<ComplexScalar>();
      ComplexMatrix Cresult;
      MatrixFunction<ComplexMatrix>(CA, f, &Cresult);
      *result = Cresult.real();
    }
};

      
/** \ingroup MatrixFunctions_Module 
  * \brief Partial specialization of MatrixFunction for complex matrices \internal 
  */
template <typename MatrixType>
class MatrixFunction<MatrixType, 1>
{
  private:

    typedef ei_traits<MatrixType> Traits;
    typedef typename Traits::Scalar Scalar;
    static const int RowsAtCompileTime = Traits::RowsAtCompileTime;
    static const int ColsAtCompileTime = Traits::ColsAtCompileTime;
    static const int Options = MatrixType::Options;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename ei_stem_function<Scalar>::type StemFunction;
    typedef Matrix<Scalar, Traits::RowsAtCompileTime, 1> VectorType;
    typedef Matrix<int, Traits::RowsAtCompileTime, 1> IntVectorType;
    typedef std::list<Scalar> Cluster;
    typedef std::list<Cluster> ListOfClusters;
    typedef Matrix<Scalar, Dynamic, Dynamic, Options, RowsAtCompileTime, ColsAtCompileTime> DynMatrixType;

  public:

    /** \brief Constructor. Computes matrix function. 
      *
      * \param[in]  A      argument of matrix function, should be a square matrix.
      * \param[in]  f      an entire function; \c f(x,n) should compute the n-th derivative of f at x.
      * \param[out] result pointer to the matrix in which to store the result, \f$ f(A) \f$.
      */
    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result);

  private:

    void computeSchurDecomposition(const MatrixType& A);
    void partitionEigenvalues();
    typename ListOfClusters::iterator findCluster(Scalar key);
    void computeClusterSize();
    void computeBlockStart();
    void constructPermutation();
    void permuteSchur();
    void swapEntriesInSchur(int index);
    void computeBlockAtomic();
    Block<MatrixType> block(const MatrixType& A, int i, int j);
    void computeOffDiagonal();
    DynMatrixType solveTriangularSylvester(const DynMatrixType& A, const DynMatrixType& B, const DynMatrixType& C);

    StemFunction *m_f; /**< \brief Stem function for matrix function under consideration */
    MatrixType m_T; /**< \brief Triangular part of Schur decomposition */
    MatrixType m_U; /**< \brief Unitary part of Schur decomposition */
    MatrixType m_fT; /**< \brief %Matrix function applied to #m_T */
    ListOfClusters m_clusters; /**< \brief Partition of eigenvalues into clusters of ei'vals "close" to each other */
    VectorXi m_eivalToCluster; /**< \brief m_eivalToCluster[i] = j means i-th ei'val is in j-th cluster */
    VectorXi m_clusterSize; /**< \brief Number of eigenvalues in each clusters  */
    VectorXi m_blockStart; /**< \brief Row index at which block corresponding to i-th cluster starts */
    IntVectorType m_permutation; /**< \brief Permutation which groups ei'vals in the same cluster together */

    /** \brief Maximum distance allowed between eigenvalues to be considered "close".
      *
      * This is morally a \c static \c const \c Scalar, but only
      * integers can be static constant class members in C++. The
      * separation constant is set to 0.01, a value taken from the
      * paper by Davies and Higham. */
    static const RealScalar separation() { return static_cast<RealScalar>(0.01); }
};

template <typename MatrixType>
MatrixFunction<MatrixType,1>::MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result) :
  m_f(f)
{
  computeSchurDecomposition(A);
  partitionEigenvalues();
  computeClusterSize();
  computeBlockStart();
  constructPermutation();
  permuteSchur();
  computeBlockAtomic();
  computeOffDiagonal();
  *result = m_U * m_fT * m_U.adjoint();
}

/** \brief Store the Schur decomposition of \p A in #m_T and #m_U */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::computeSchurDecomposition(const MatrixType& A)
{
  const ComplexSchur<MatrixType> schurOfA(A);  
  m_T = schurOfA.matrixT();
  m_U = schurOfA.matrixU();
}

/** \brief Partition eigenvalues in clusters of ei'vals close to each other
  * 
  * This function computes #m_clusters. This is a partition of the
  * eigenvalues of #m_T in clusters, such that
  * # Any eigenvalue in a certain cluster is at most separation() away
  *   from another eigenvalue in the same cluster.
  * # The distance between two eigenvalues in different clusters is
  *   more than separation().
  * The implementation follows Algorithm 4.1 in the paper of Davies
  * and Higham. 
  */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::partitionEigenvalues()
{
  const int rows = m_T.rows();
  VectorType diag = m_T.diagonal(); // contains eigenvalues of A

  for (int i=0; i<rows; ++i) {
    // Find set containing diag(i), adding a new set if necessary
    typename ListOfClusters::iterator qi = findCluster(diag(i));
    if (qi == m_clusters.end()) {
      Cluster l;
      l.push_back(diag(i));
      m_clusters.push_back(l);
      qi = m_clusters.end();
      --qi;
    }

    // Look for other element to add to the set
    for (int j=i+1; j<rows; ++j) {
      if (ei_abs(diag(j) - diag(i)) <= separation() && std::find(qi->begin(), qi->end(), diag(j)) == qi->end()) {
	typename ListOfClusters::iterator qj = findCluster(diag(j));
	if (qj == m_clusters.end()) {
	  qi->push_back(diag(j));
	} else {
	  qi->insert(qi->end(), qj->begin(), qj->end());
	  m_clusters.erase(qj);
	}
      }
    }
  }
}

/** \brief Find cluster in #m_clusters containing some value 
  * \param[in] key Value to find
  * \returns Iterator to cluster containing \c key, or
  * \c m_clusters.end() if no cluster in m_clusters contains \c key.
  */
template <typename MatrixType>
typename MatrixFunction<MatrixType,1>::ListOfClusters::iterator MatrixFunction<MatrixType,1>::findCluster(Scalar key)
{
  typename Cluster::iterator j;
  for (typename ListOfClusters::iterator i = m_clusters.begin(); i != m_clusters.end(); ++i) {
    j = std::find(i->begin(), i->end(), key);
    if (j != i->end())
      return i;
  }
  return m_clusters.end();
}

/** \brief Compute #m_clusterSize and #m_eivalToCluster using #m_clusters */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::computeClusterSize()
{
  const int rows = m_T.rows();
  VectorType diag = m_T.diagonal(); 
  const int numClusters = m_clusters.size();

  m_clusterSize.setZero(numClusters);
  m_eivalToCluster.resize(rows);
  int clusterIndex = 0;
  for (typename ListOfClusters::const_iterator cluster = m_clusters.begin(); cluster != m_clusters.end(); ++cluster) {
    for (int i = 0; i < diag.rows(); ++i) {
      if (std::find(cluster->begin(), cluster->end(), diag(i)) != cluster->end()) {
	++m_clusterSize[clusterIndex];
	m_eivalToCluster[i] = clusterIndex;
      }
    }
    ++clusterIndex;
  }
}

/** \brief Compute #m_blockStart using #m_clusterSize */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::computeBlockStart()
{
  m_blockStart.resize(m_clusterSize.rows());
  m_blockStart(0) = 0;
  for (int i = 1; i < m_clusterSize.rows(); i++) {
    m_blockStart(i) = m_blockStart(i-1) + m_clusterSize(i-1);
  }
}

/** \brief Compute #m_permutation using #m_eivalToCluster and #m_blockStart */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::constructPermutation()
{
  VectorXi indexNextEntry = m_blockStart;
  m_permutation.resize(m_T.rows());
  for (int i = 0; i < m_T.rows(); i++) {
    int cluster = m_eivalToCluster[i];
    m_permutation[i] = indexNextEntry[cluster];
    ++indexNextEntry[cluster];
  }
}  

/** \brief Permute Schur decomposition in #m_U and #m_T according to #m_permutation */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::permuteSchur()
{
  IntVectorType p = m_permutation;
  for (int i = 0; i < p.rows() - 1; i++) {
    int j;
    for (j = i; j < p.rows(); j++) {
      if (p(j) == i) break;
    }
    ei_assert(p(j) == i);
    for (int k = j-1; k >= i; k--) {
      swapEntriesInSchur(k);
      std::swap(p.coeffRef(k), p.coeffRef(k+1));
    }
  }
}

/** \brief Swap rows \a index and \a index+1 in Schur decomposition in #m_U and #m_T */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::swapEntriesInSchur(int index)
{
  PlanarRotation<Scalar> rotation;
  rotation.makeGivens(m_T(index, index+1), m_T(index+1, index+1) - m_T(index, index));
  m_T.applyOnTheLeft(index, index+1, rotation.adjoint());
  m_T.applyOnTheRight(index, index+1, rotation);
  m_U.applyOnTheRight(index, index+1, rotation);
}  

/** \brief Compute block diagonal part of #m_fT.
  *
  * This routine computes the matrix function #m_f applied to the block
  * diagonal part of #m_T, with the blocking given by #m_blockStart. The
  * result is stored in #m_fT. The off-diagonal parts of #m_fT are set
  * to zero.
  */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::computeBlockAtomic()
{ 
  m_fT.resize(m_T.rows(), m_T.cols());
  m_fT.setZero();
  MatrixFunctionAtomic<DynMatrixType> mfa(m_f);
  for (int i = 0; i < m_clusterSize.rows(); ++i) {
    block(m_fT, i, i) = mfa.compute(block(m_T, i, i));
  }
}

/** \brief Return block of matrix according to blocking given by #m_blockStart */
template <typename MatrixType>
Block<MatrixType> MatrixFunction<MatrixType,1>::block(const MatrixType& A, int i, int j)
{
  return A.block(m_blockStart(i), m_blockStart(j), m_clusterSize(i), m_clusterSize(j));
}

/** \brief Compute part of #m_fT above block diagonal.
  *
  * This routine assumes that the block diagonal part of #m_fT (which
  * equals #m_f applied to #m_T) has already been computed and computes
  * the part above the block diagonal. The part below the diagonal is
  * zero, because #m_T is upper triangular.
  */
template <typename MatrixType>
void MatrixFunction<MatrixType,1>::computeOffDiagonal()
{ 
  for (int diagIndex = 1; diagIndex < m_clusterSize.rows(); diagIndex++) {
    for (int blockIndex = 0; blockIndex < m_clusterSize.rows() - diagIndex; blockIndex++) {
      // compute (blockIndex, blockIndex+diagIndex) block
      DynMatrixType A = block(m_T, blockIndex, blockIndex);
      DynMatrixType B = -block(m_T, blockIndex+diagIndex, blockIndex+diagIndex);
      DynMatrixType C = block(m_fT, blockIndex, blockIndex) * block(m_T, blockIndex, blockIndex+diagIndex);
      C -= block(m_T, blockIndex, blockIndex+diagIndex) * block(m_fT, blockIndex+diagIndex, blockIndex+diagIndex);
      for (int k = blockIndex + 1; k < blockIndex + diagIndex; k++) {
	C += block(m_fT, blockIndex, k) * block(m_T, k, blockIndex+diagIndex);
	C -= block(m_T, blockIndex, k) * block(m_fT, k, blockIndex+diagIndex);
      }
      block(m_fT, blockIndex, blockIndex+diagIndex) = solveTriangularSylvester(A, B, C);
    }
  }
}

/** \brief Solve a triangular Sylvester equation AX + XB = C 
  *
  * \param[in]  A  the matrix A; should be square and upper triangular
  * \param[in]  B  the matrix B; should be square and upper triangular
  * \param[in]  C  the matrix C; should have correct size.
  *
  * \returns the solution X.
  *
  * If A is m-by-m and B is n-by-n, then both C and X are m-by-n. 
  * The (i,j)-th component of the Sylvester equation is
  * \f[ 
  *     \sum_{k=i}^m A_{ik} X_{kj} + \sum_{k=1}^j X_{ik} B_{kj} = C_{ij}. 
  * \f]
  * This can be re-arranged to yield:
  * \f[ 
  *     X_{ij} = \frac{1}{A_{ii} + B_{jj}} \Bigl( C_{ij}
  *     - \sum_{k=i+1}^m A_{ik} X_{kj} - \sum_{k=1}^{j-1} X_{ik} B_{kj} \Bigr).
  * \f]
  * It is assumed that A and B are such that the numerator is never
  * zero (otherwise the Sylvester equation does not have a unique
  * solution). In that case, these equations can be evaluated in the
  * order \f$ i=m,\ldots,1 \f$ and \f$ j=1,\ldots,n \f$.
  */
template <typename MatrixType>
typename MatrixFunction<MatrixType,1>::DynMatrixType MatrixFunction<MatrixType,1>::solveTriangularSylvester(
  const DynMatrixType& A, 
  const DynMatrixType& B, 
  const DynMatrixType& C)
{
  ei_assert(A.rows() == A.cols());
  ei_assert(A.isUpperTriangular());
  ei_assert(B.rows() == B.cols());
  ei_assert(B.isUpperTriangular());
  ei_assert(C.rows() == A.rows());
  ei_assert(C.cols() == B.rows());

  int m = A.rows();
  int n = B.rows();
  DynMatrixType X(m, n);

  for (int i = m - 1; i >= 0; --i) {
    for (int j = 0; j < n; ++j) {

      // Compute AX = \sum_{k=i+1}^m A_{ik} X_{kj}
      Scalar AX;
      if (i == m - 1) {
	AX = 0; 
      } else {
	Matrix<Scalar,1,1> AXmatrix = A.row(i).end(m-1-i) * X.col(j).end(m-1-i);
	AX = AXmatrix(0,0);
      }

      // Compute XB = \sum_{k=1}^{j-1} X_{ik} B_{kj}
      Scalar XB;
      if (j == 0) {
	XB = 0; 
      } else {
	Matrix<Scalar,1,1> XBmatrix = X.row(i).start(j) * B.col(j).start(j);
	XB = XBmatrix(0,0);
      }

      X(i,j) = (C(i,j) - AX - XB) / (A(i,i) + B(j,j));
    }
  }
  return X;
}


template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_function(const MatrixBase<Derived>& M, 
					    typename ei_stem_function<typename ei_traits<Derived>::Scalar>::type f,
					    typename MatrixBase<Derived>::PlainMatrixType* result)
{
  ei_assert(M.rows() == M.cols());
  typedef typename MatrixBase<Derived>::PlainMatrixType PlainMatrixType;
  MatrixFunction<PlainMatrixType>(M, f, result);
}

#endif // EIGEN_MATRIX_FUNCTION
