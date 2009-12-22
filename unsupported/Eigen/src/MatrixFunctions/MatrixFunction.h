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
  * Suppose that \f$ f \f$ is an entire function (that is, a function
  * on the complex plane that is everywhere complex differentiable).
  * Then its Taylor series
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


/** \ingroup MatrixFunctions_Module 
  * \class MatrixFunction
  * \brief Helper class for computing matrix functions. 
  */
template <typename MatrixType, 
  int IsComplex = NumTraits<typename ei_traits<MatrixType>::Scalar>::IsComplex,
  int IsDynamic = ( (ei_traits<MatrixType>::RowsAtCompileTime == Dynamic) 
		    && (ei_traits<MatrixType>::RowsAtCompileTime == Dynamic) ) >
class MatrixFunction;

/* Partial specialization of MatrixFunction for real matrices */

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols, int IsDynamic>
class MatrixFunction<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, 0, IsDynamic>
{  
  public:

    typedef std::complex<Scalar> ComplexScalar;
    typedef Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> MatrixType;
    typedef Matrix<ComplexScalar, Rows, Cols, Options, MaxRows, MaxCols> ComplexMatrix;
    typedef typename ei_stem_function<Scalar>::type StemFunction;

    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result) 
    {
      ComplexMatrix CA = A.template cast<ComplexScalar>();
      ComplexMatrix Cresult;
      MatrixFunction<ComplexMatrix>(CA, f, &Cresult);
      result->resize(A.cols(), A.rows());
      for (int j = 0; j < A.cols(); j++)
	for (int i = 0; i < A.rows(); i++)
	  (*result)(i,j) = std::real(Cresult(i,j));
    }
};
      
/* Partial specialization of MatrixFunction for complex static-size matrices */

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
class MatrixFunction<Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>, 1, 0>
{  
  public:

    typedef Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> MatrixType;
    typedef Matrix<Scalar, Dynamic, Dynamic, Options, MaxRows, MaxCols> DynamicMatrix;
    typedef typename ei_stem_function<Scalar>::type StemFunction;

    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result) 
    {
      DynamicMatrix DA = A;
      DynamicMatrix Dresult;
      MatrixFunction<DynamicMatrix>(DA, f, &Dresult);
      *result = Dresult;
    }
};
      
/* Partial specialization of MatrixFunction for complex dynamic-size matrices */
  
template <typename MatrixType>
class MatrixFunction<MatrixType, 1, 1>
{
  public:

    typedef ei_traits<MatrixType> Traits;
    typedef typename Traits::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef typename ei_stem_function<Scalar>::type StemFunction;
    typedef Matrix<Scalar, Traits::RowsAtCompileTime, 1> VectorType;
    typedef Matrix<int, Traits::RowsAtCompileTime, 1> IntVectorType;
    typedef std::list<Scalar> listOfScalars;
    typedef std::list<listOfScalars> listOfLists;

    /** \brief Compute matrix function. 
     *
     * \param A      argument of matrix function.
     * \param f      function to compute.
     * \param result pointer to the matrix in which to store the result.
     */
    MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result);

  private:

    // Prevent copying
    MatrixFunction(const MatrixFunction&);
    MatrixFunction& operator=(const MatrixFunction&);

    void separateBlocksInSchur(MatrixType& T, MatrixType& U, IntVectorType& blockSize);
    void permuteSchur(const IntVectorType& permutation, MatrixType& T, MatrixType& U);
    void swapEntriesInSchur(int index, MatrixType& T, MatrixType& U);
    void computeTriangular(const MatrixType& T, MatrixType& result, const IntVectorType& blockSize);
    void computeBlockAtomic(const MatrixType& T, MatrixType& result, const IntVectorType& blockSize);
    MatrixType solveSylvester(const MatrixType& A, const MatrixType& B, const MatrixType& C);
    MatrixType computeAtomic(const MatrixType& T);
    void divideInBlocks(const VectorType& v, listOfLists* result);
    void constructPermutation(const VectorType& diag, const listOfLists& blocks, 
			      IntVectorType& blockSize, IntVectorType& permutation);

    RealScalar computeMu(const MatrixType& M);
    bool taylorConverged(const MatrixType& T, int s, const MatrixType& F, 
			 const MatrixType& Fincr, const MatrixType& P, RealScalar mu);

    static const RealScalar separation() { return static_cast<RealScalar>(0.01); }
    StemFunction *m_f;
};

template <typename MatrixType>
MatrixFunction<MatrixType,1,1>::MatrixFunction(const MatrixType& A, StemFunction f, MatrixType* result) :
  m_f(f)
{
  if (A.rows() == 1) {
    result->resize(1,1);
    (*result)(0,0) = f(A(0,0), 0);
  } else {
    const ComplexSchur<MatrixType> schurOfA(A);  
    MatrixType T = schurOfA.matrixT();
    MatrixType U = schurOfA.matrixU();
    IntVectorType blockSize, permutation;
    separateBlocksInSchur(T, U, blockSize);
    MatrixType fT;
    computeTriangular(T, fT, blockSize);
    *result = U * fT * U.adjoint();
  }
}

template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::separateBlocksInSchur(MatrixType& T, MatrixType& U, IntVectorType& blockSize)
{
  const VectorType d = T.diagonal();
  listOfLists blocks;
  divideInBlocks(d, &blocks);

  IntVectorType permutation;
  constructPermutation(d, blocks, blockSize, permutation);
  permuteSchur(permutation, T, U);
}

template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::permuteSchur(const IntVectorType& permutation, MatrixType& T, MatrixType& U)
{
  IntVectorType p = permutation;
  for (int i = 0; i < p.rows() - 1; i++) {
    int j;
    for (j = i; j < p.rows(); j++) {
      if (p(j) == i) break;
    }
    ei_assert(p(j) == i);
    for (int k = j-1; k >= i; k--) {
      swapEntriesInSchur(k, T, U);
      std::swap(p.coeffRef(k), p.coeffRef(k+1));
    }
  }
}

// swap T(index, index) and T(index+1, index+1)
template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::swapEntriesInSchur(int index, MatrixType& T, MatrixType& U)
{
  PlanarRotation<Scalar> rotation;
  rotation.makeGivens(T(index, index+1), T(index+1, index+1) - T(index, index));
  T.applyOnTheLeft(index, index+1, rotation.adjoint());
  T.applyOnTheRight(index, index+1, rotation);
  U.applyOnTheRight(index, index+1, rotation);
}  

template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::computeTriangular(const MatrixType& T, MatrixType& result, 
						       const IntVectorType& blockSize)
{ 
  MatrixType expT;
  ei_matrix_exponential(T, &expT);
  computeBlockAtomic(T, result, blockSize);
  IntVectorType blockStart(blockSize.rows());
  blockStart(0) = 0;
  for (int i = 1; i < blockSize.rows(); i++) {
    blockStart(i) = blockStart(i-1) + blockSize(i-1);
  }
  for (int diagIndex = 1; diagIndex < blockSize.rows(); diagIndex++) {
    for (int blockIndex = 0; blockIndex < blockSize.rows() - diagIndex; blockIndex++) {
      // compute (blockIndex, blockIndex+diagIndex) block
      MatrixType A = T.block(blockStart(blockIndex), blockStart(blockIndex), blockSize(blockIndex), blockSize(blockIndex));
      MatrixType B = -T.block(blockStart(blockIndex+diagIndex), blockStart(blockIndex+diagIndex), blockSize(blockIndex+diagIndex), blockSize(blockIndex+diagIndex));
      MatrixType C = result.block(blockStart(blockIndex), blockStart(blockIndex), blockSize(blockIndex), blockSize(blockIndex)) * T.block(blockStart(blockIndex), blockStart(blockIndex+diagIndex), blockSize(blockIndex), blockSize(blockIndex+diagIndex));
      C -= T.block(blockStart(blockIndex), blockStart(blockIndex+diagIndex), blockSize(blockIndex), blockSize(blockIndex+diagIndex)) * result.block(blockStart(blockIndex+diagIndex), blockStart(blockIndex+diagIndex), blockSize(blockIndex+diagIndex), blockSize(blockIndex+diagIndex));
      for (int k = blockIndex + 1; k < blockIndex + diagIndex; k++) {
	C += result.block(blockStart(blockIndex), blockStart(k), blockSize(blockIndex), blockSize(k)) * T.block(blockStart(k), blockStart(blockIndex+diagIndex), blockSize(k), blockSize(blockIndex+diagIndex));
	C -= T.block(blockStart(blockIndex), blockStart(k), blockSize(blockIndex), blockSize(k)) * result.block(blockStart(k), blockStart(blockIndex+diagIndex), blockSize(k), blockSize(blockIndex+diagIndex));
      }
      result.block(blockStart(blockIndex), blockStart(blockIndex+diagIndex), blockSize(blockIndex), blockSize(blockIndex+diagIndex)) = solveSylvester(A, B, C);
    }
  }
}

// solve AX + XB = C  <=>  U* A' U X V V* + U* U X V B' V* = U* U C V V*  <=>  A' U X V + U X V B' = U C V 
// Schur: A* = U A'* U* (so A = U* A' U), B = V B' V*, define: X' = U X V, C' = U C V, to get: A' X' + X' B' = C'
// A is m-by-m, B is n-by-n, X is m-by-n, C is m-by-n, U is m-by-m, V is n-by-n
template <typename MatrixType>
MatrixType MatrixFunction<MatrixType,1,1>::solveSylvester(const MatrixType& A, const MatrixType& B, const MatrixType& C)
{
  MatrixType U = MatrixType::Zero(A.rows(), A.rows());
  for (int i = 0; i < A.rows(); i++) {
    U(i, A.rows() - 1 - i) = static_cast<Scalar>(1);
  }
  MatrixType Aprime = U * A * U;

  MatrixType Bprime = B;
  MatrixType V = MatrixType::Identity(B.rows(), B.rows());

  MatrixType Cprime = U * C * V;
  MatrixType Xprime(A.rows(), B.rows());
  for (int l = 0; l < B.rows(); l++) {
    for (int k = 0; k < A.rows(); k++) {
      Scalar tmp1, tmp2;
      if (k == 0) {
	tmp1 = 0; 
      } else {
	Matrix<Scalar,1,1> tmp1matrix = Aprime.row(k).start(k) * Xprime.col(l).start(k);
	tmp1 = tmp1matrix(0,0);
      }
      if (l == 0) {
	tmp2 = 0; 
      } else {
	Matrix<Scalar,1,1> tmp2matrix = Xprime.row(k).start(l) * Bprime.col(l).start(l);
	tmp2 = tmp2matrix(0,0);
      }
      Xprime(k,l) = (Cprime(k,l) - tmp1 - tmp2) / (Aprime(k,k) + Bprime(l,l));
    }
  }
  return U.adjoint() * Xprime * V.adjoint();
}


// does not touch irrelevant parts of T
template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::computeBlockAtomic(const MatrixType& T, MatrixType& result, 
							const IntVectorType& blockSize)
{ 
  int blockStart = 0;
  result.resize(T.rows(), T.cols());
  result.setZero();
  for (int i = 0; i < blockSize.rows(); i++) {
    result.block(blockStart, blockStart, blockSize(i), blockSize(i))
      = computeAtomic(T.block(blockStart, blockStart, blockSize(i), blockSize(i)));
    blockStart += blockSize(i);
  }
}

template <typename Scalar>
typename std::list<std::list<Scalar> >::iterator ei_find_in_list_of_lists(typename std::list<std::list<Scalar> >& ll, Scalar x)
{
  typename std::list<Scalar>::iterator j;
  for (typename std::list<std::list<Scalar> >::iterator i = ll.begin(); i != ll.end(); i++) {
    j = std::find(i->begin(), i->end(), x);
    if (j != i->end())
      return i;
  }
  return ll.end();
}

// Alg 4.1
template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::divideInBlocks(const VectorType& v, listOfLists* result)
{
  const int n = v.rows();
  for (int i=0; i<n; i++) {
    // Find set containing v(i), adding a new set if necessary
    typename listOfLists::iterator qi = ei_find_in_list_of_lists(*result, v(i));
    if (qi == result->end()) {
      listOfScalars l;
      l.push_back(v(i));
      result->push_back(l);
      qi = result->end();
      qi--;
    }
    // Look for other element to add to the set
    for (int j=i+1; j<n; j++) {
      if (ei_abs(v(j) - v(i)) <= separation() && std::find(qi->begin(), qi->end(), v(j)) == qi->end()) {
	typename listOfLists::iterator qj = ei_find_in_list_of_lists(*result, v(j));
	if (qj == result->end()) {
	  qi->push_back(v(j));
	} else {
	  qi->insert(qi->end(), qj->begin(), qj->end());
	  result->erase(qj);
	}
      }
    }
  }
}

// Construct permutation P, such that P(D) has eigenvalues clustered together
template <typename MatrixType>
void MatrixFunction<MatrixType,1,1>::constructPermutation(const VectorType& diag, const listOfLists& blocks, 
							  IntVectorType& blockSize, IntVectorType& permutation)
{
  const int n = diag.rows();
  const int numBlocks = blocks.size();

  // For every block in blocks, mark and count the entries in diag that
  // appear in that block
  blockSize.setZero(numBlocks);
  IntVectorType entryToBlock(n);
  int blockIndex = 0;
  for (typename listOfLists::const_iterator block = blocks.begin(); block != blocks.end(); block++) {
    for (int i = 0; i < diag.rows(); i++) {
      if (std::find(block->begin(), block->end(), diag(i)) != block->end()) {
	blockSize[blockIndex]++;
	entryToBlock[i] = blockIndex;
      }
    }
    blockIndex++;
  }

  // Compute index of first entry in every block as the sum of sizes
  // of all the preceding blocks
  IntVectorType indexNextEntry(numBlocks);
  indexNextEntry[0] = 0;
  for (blockIndex = 1; blockIndex < numBlocks; blockIndex++) {
    indexNextEntry[blockIndex] = indexNextEntry[blockIndex-1] + blockSize[blockIndex-1];
  }
      
  // Construct permutation 
  permutation.resize(n);
  for (int i = 0; i < n; i++) {
    int block = entryToBlock[i];
    permutation[i] = indexNextEntry[block];
    indexNextEntry[block]++;
  }
}  

template <typename MatrixType>
MatrixType MatrixFunction<MatrixType,1,1>::computeAtomic(const MatrixType& T)
{
  // TODO: Use that T is upper triangular
  const int n = T.rows();
  const Scalar sigma = T.trace() / Scalar(n);
  const MatrixType M = T - sigma * MatrixType::Identity(n, n);
  const RealScalar mu = computeMu(M);
  MatrixType F = m_f(sigma, 0) * MatrixType::Identity(n, n);
  MatrixType P = M;
  MatrixType Fincr;
  for (int s = 1; s < 1.1*n + 10; s++) { // upper limit is fairly arbitrary
    Fincr = m_f(sigma, s) * P;
    F += Fincr;
    P = (1/(s + 1.0)) * P * M;
    if (taylorConverged(T, s, F, Fincr, P, mu)) {
      return F;
    }
  }
  ei_assert("Taylor series does not converge" && 0);
  return F;
}

template <typename MatrixType>
typename MatrixFunction<MatrixType,1,1>::RealScalar MatrixFunction<MatrixType,1,1>::computeMu(const MatrixType& M)
{
  const int n = M.rows();
  const MatrixType N = MatrixType::Identity(n, n) - M;
  VectorType e = VectorType::Ones(n);
  N.template triangularView<UpperTriangular>().solveInPlace(e);
  return e.cwise().abs().maxCoeff();
}

template <typename MatrixType>
bool MatrixFunction<MatrixType,1,1>::taylorConverged(const MatrixType& T, int s, const MatrixType& F, 
						   const MatrixType& Fincr, const MatrixType& P, RealScalar mu)
{
  const int n = F.rows();
  const RealScalar F_norm = F.cwise().abs().rowwise().sum().maxCoeff();
  const RealScalar Fincr_norm = Fincr.cwise().abs().rowwise().sum().maxCoeff();
  if (Fincr_norm < epsilon<Scalar>() * F_norm) {
    RealScalar delta = 0;
    RealScalar rfactorial = 1;
    for (int r = 0; r < n; r++) {
      RealScalar mx = 0;
      for (int i = 0; i < n; i++) 
	mx = std::max(mx, std::abs(m_f(T(i, i), s+r)));
       if (r != 0)
	rfactorial *= r;
      delta = std::max(delta, mx / rfactorial);
    }
    const RealScalar P_norm = P.cwise().abs().rowwise().sum().maxCoeff();
    if (mu * delta * P_norm < epsilon<Scalar>() * F_norm) 
      return true;
  }
  return false;
}

template <typename Derived>
EIGEN_STRONG_INLINE void ei_matrix_function(const MatrixBase<Derived>& M, 
					    typename ei_stem_function<typename ei_traits<Derived>::Scalar>::type f,
					    typename MatrixBase<Derived>::PlainMatrixType* result)
{
  ei_assert(M.rows() == M.cols());
  MatrixFunction<typename MatrixBase<Derived>::PlainMatrixType>(M, f, result);
}

#endif // EIGEN_MATRIX_FUNCTION
