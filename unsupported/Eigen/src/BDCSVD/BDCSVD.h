// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
// 
// We used the "A Divide-And-Conquer Algorithm for the Bidiagonal SVD"
// research report written by Ming Gu and Stanley C.Eisenstat
// The code variable names correspond to the names they used in their 
// report
//
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
// Copyright (C) 2013 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BDCSVD_H
#define EIGEN_BDCSVD_H

namespace Eigen {

template<typename _MatrixType> class BDCSVD;

namespace internal {

template<typename _MatrixType> 
struct traits<BDCSVD<_MatrixType> >
{
  typedef _MatrixType MatrixType;
};  

} // end namespace internal
  
  
/** \ingroup SVD_Module
 *
 *
 * \class BDCSVD
 *
 * \brief class Bidiagonal Divide and Conquer SVD
 *
 * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
 * We plan to have a very similar interface to JacobiSVD on this class.
 * It should be used to speed up the calcul of SVD for big matrices. 
 */
template<typename _MatrixType> 
class BDCSVD : public SVDBase<BDCSVD<_MatrixType> >
{
  typedef SVDBase<BDCSVD> Base;
    
public:
  using Base::rows;
  using Base::cols;
  using Base::computeU;
  using Base::computeV;
  
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  typedef typename MatrixType::Index Index;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime, 
    ColsAtCompileTime = MatrixType::ColsAtCompileTime, 
    DiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_DYNAMIC(RowsAtCompileTime, ColsAtCompileTime), 
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime, 
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime, 
    MaxDiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(MaxRowsAtCompileTime, MaxColsAtCompileTime), 
    MatrixOptions = MatrixType::Options
  };

  typedef typename Base::MatrixUType MatrixUType;
  typedef typename Base::MatrixVType MatrixVType;
  typedef typename Base::SingularValuesType SingularValuesType;
  
  typedef Matrix<Scalar, Dynamic, Dynamic> MatrixX;
  typedef Matrix<RealScalar, Dynamic, Dynamic> MatrixXr;
  typedef Matrix<RealScalar, Dynamic, 1> VectorType;
  typedef Array<RealScalar, Dynamic, 1> ArrayXr;

  /** \brief Default Constructor.
   *
   * The default constructor is useful in cases in which the user intends to
   * perform decompositions via BDCSVD::compute(const MatrixType&).
   */
  BDCSVD() : m_algoswap(16), m_numIters(0)
  {}


  /** \brief Default Constructor with memory preallocation
   *
   * Like the default constructor but with preallocation of the internal data
   * according to the specified problem size.
   * \sa BDCSVD()
   */
  BDCSVD(Index rows, Index cols, unsigned int computationOptions = 0)
    : m_algoswap(16), m_numIters(0)
  {
    allocate(rows, cols, computationOptions);
  }

  /** \brief Constructor performing the decomposition of given matrix.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you want full or thin U or V unitaries to be computed.
   *                           By default, none is computed. This is a bit - field, the possible bits are #ComputeFullU, #ComputeThinU, 
   *                           #ComputeFullV, #ComputeThinV.
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number of columns (for example MatrixXf). They also are not
   * available with the (non - default) FullPivHouseholderQR preconditioner.
   */
  BDCSVD(const MatrixType& matrix, unsigned int computationOptions = 0)
    : m_algoswap(16), m_numIters(0)
  {
    compute(matrix, computationOptions);
  }

  ~BDCSVD() 
  {
  }
  
  /** \brief Method performing the decomposition of given matrix using custom options.
   *
   * \param matrix the matrix to decompose
   * \param computationOptions optional parameter allowing to specify if you want full or thin U or V unitaries to be computed.
   *                           By default, none is computed. This is a bit - field, the possible bits are #ComputeFullU, #ComputeThinU, 
   *                           #ComputeFullV, #ComputeThinV.
   *
   * Thin unitaries are only available if your matrix type has a Dynamic number of columns (for example MatrixXf). They also are not
   * available with the (non - default) FullPivHouseholderQR preconditioner.
   */
  BDCSVD& compute(const MatrixType& matrix, unsigned int computationOptions);

  /** \brief Method performing the decomposition of given matrix using current options.
   *
   * \param matrix the matrix to decompose
   *
   * This method uses the current \a computationOptions, as already passed to the constructor or to compute(const MatrixType&, unsigned int).
   */
  BDCSVD& compute(const MatrixType& matrix)
  {
    return compute(matrix, this->m_computationOptions);
  }

  void setSwitchSize(int s) 
  {
    eigen_assert(s>3 && "BDCSVD the size of the algo switch has to be greater than 3");
    m_algoswap = s;
  }
 
private:
  void allocate(Index rows, Index cols, unsigned int computationOptions);
  void divide(Index firstCol, Index lastCol, Index firstRowW, Index firstColW, Index shift);
  void computeSVDofM(Index firstCol, Index n, MatrixXr& U, VectorType& singVals, MatrixXr& V);
  void computeSingVals(const ArrayXr& col0, const ArrayXr& diag, VectorType& singVals,
                       ArrayXr& shifts, ArrayXr& mus);
  void perturbCol0(const ArrayXr& col0, const ArrayXr& diag, const VectorType& singVals,
                   const ArrayXr& shifts, const ArrayXr& mus, ArrayXr& zhat);
  void computeSingVecs(const ArrayXr& zhat, const ArrayXr& diag, const VectorType& singVals,
                       const ArrayXr& shifts, const ArrayXr& mus, MatrixXr& U, MatrixXr& V);
  void deflation43(Index firstCol, Index shift, Index i, Index size);
  void deflation44(Index firstColu , Index firstColm, Index firstRowW, Index firstColW, Index i, Index j, Index size);
  void deflation(Index firstCol, Index lastCol, Index k, Index firstRowW, Index firstColW, Index shift);
  template<typename HouseholderU, typename HouseholderV, typename NaiveU, typename NaiveV>
  void copyUV(const HouseholderU &householderU, const HouseholderV &householderV, const NaiveU &naiveU, const NaiveV &naivev);
  static void structured_update(Block<MatrixXr,Dynamic,Dynamic> A, const MatrixXr &B, Index n1);

protected:
  MatrixXr m_naiveU, m_naiveV;
  MatrixXr m_computed;
  Index m_nRec;
  int m_algoswap;
  bool m_isTranspose, m_compU, m_compV;
  
  using Base::m_singularValues;
  using Base::m_diagSize;
  using Base::m_computeFullU;
  using Base::m_computeFullV;
  using Base::m_computeThinU;
  using Base::m_computeThinV;
  using Base::m_matrixU;
  using Base::m_matrixV;
  using Base::m_isInitialized;
  using Base::m_nonzeroSingularValues;

public:  
  int m_numIters;
}; //end class BDCSVD


// Methode to allocate ans initialize matrix and attributs
template<typename MatrixType>
void BDCSVD<MatrixType>::allocate(Index rows, Index cols, unsigned int computationOptions)
{
  m_isTranspose = (cols > rows);
  if (Base::allocate(rows, cols, computationOptions))
    return;
  
  m_computed = MatrixXr::Zero(m_diagSize + 1, m_diagSize );
  m_compU = computeV();
  m_compV = computeU();
  if (m_isTranspose)
    std::swap(m_compU, m_compV);
  
  if (m_compU) m_naiveU = MatrixXr::Zero(m_diagSize + 1, m_diagSize + 1 );
  else         m_naiveU = MatrixXr::Zero(2, m_diagSize + 1 );
  
  if (m_compV) m_naiveV = MatrixXr::Zero(m_diagSize, m_diagSize);
}// end allocate

// Methode which compute the BDCSVD for the int
template<>
BDCSVD<Matrix<int, Dynamic, Dynamic> >& BDCSVD<Matrix<int, Dynamic, Dynamic> >::compute(const MatrixType& matrix, unsigned int computationOptions)
{
  allocate(matrix.rows(), matrix.cols(), computationOptions);
  m_nonzeroSingularValues = 0;
  m_computed = Matrix<int, Dynamic, Dynamic>::Zero(rows(), cols());
  
  m_singularValues.head(m_diagSize).setZero();
  
  if (m_computeFullU) m_matrixU.setZero(rows(), rows());
  if (m_computeFullV) m_matrixV.setZero(cols(), cols()); 
  m_isInitialized = true;
  return *this;
}


// Methode which compute the BDCSVD
template<typename MatrixType>
BDCSVD<MatrixType>& BDCSVD<MatrixType>::compute(const MatrixType& matrix, unsigned int computationOptions) 
{
  allocate(matrix.rows(), matrix.cols(), computationOptions);
  using std::abs;

  //**** step 1 Bidiagonalization  m_isTranspose = (matrix.cols()>matrix.rows()) ;
  MatrixType copy;
  if (m_isTranspose) copy = matrix.adjoint();
  else               copy = matrix;
  
  internal::UpperBidiagonalization<MatrixX> bid(copy);

  //**** step 2 Divide
  m_naiveU.setZero();
  m_naiveV.setZero();
  m_computed.topRows(m_diagSize) = bid.bidiagonal().toDenseMatrix().transpose();
  m_computed.template bottomRows<1>().setZero();
  divide(0, m_diagSize - 1, 0, 0, 0);

  //**** step 3 copy
  for (int i=0; i<m_diagSize; i++)
  {
    RealScalar a = abs(m_computed.coeff(i, i));
    m_singularValues.coeffRef(i) = a;
    if (a == 0)
    {
      m_nonzeroSingularValues = i;
      m_singularValues.tail(m_diagSize - i - 1).setZero();
      break;
    }
    else if (i == m_diagSize - 1)
    {
      m_nonzeroSingularValues = i + 1;
      break;
    }
  }
  if(m_isTranspose) copyUV(bid.householderV(), bid.householderU(), m_naiveV, m_naiveU);
  else              copyUV(bid.householderU(), bid.householderV(), m_naiveU, m_naiveV);
  m_isInitialized = true;
  return *this;
}// end compute


template<typename MatrixType>
template<typename HouseholderU, typename HouseholderV, typename NaiveU, typename NaiveV>
void BDCSVD<MatrixType>::copyUV(const HouseholderU &householderU, const HouseholderV &householderV, const NaiveU &naiveU, const NaiveV &naiveV)
{
  // Note exchange of U and V: m_matrixU is set from m_naiveV and vice versa
  if (computeU())
  {
    Index Ucols = m_computeThinU ? m_nonzeroSingularValues : householderU.cols();
    m_matrixU = MatrixX::Identity(householderU.cols(), Ucols);
    Index blockCols = m_computeThinU ? m_nonzeroSingularValues : m_diagSize;
    m_matrixU.topLeftCorner(m_diagSize, blockCols) = naiveV.template cast<Scalar>().topLeftCorner(m_diagSize, blockCols);
    householderU.applyThisOnTheLeft(m_matrixU);
  }
  if (computeV())
  {
    Index Vcols = m_computeThinV ? m_nonzeroSingularValues : householderV.cols();
    m_matrixV = MatrixX::Identity(householderV.cols(), Vcols);
    Index blockCols = m_computeThinV ? m_nonzeroSingularValues : m_diagSize;
    m_matrixV.topLeftCorner(m_diagSize, blockCols) = naiveU.template cast<Scalar>().topLeftCorner(m_diagSize, blockCols);
    householderV.applyThisOnTheLeft(m_matrixV);
  }
}

/** \internal
  * Performs A = A * B exploiting the special structure of the matrix A. Splitting A as:
  *  A = [A1]
  *      [A2]
  * such that A1.rows()==n1, then we assume that at least half of the columns of A1 and A2 are zeros.
  * We can thus pack them prior to the the matrix product. However, this is only worth the effort if the matrix is large
  * enough.
  */
template<typename MatrixType>
void BDCSVD<MatrixType>::structured_update(Block<MatrixXr,Dynamic,Dynamic> A, const MatrixXr &B, Index n1)
{
  Index n = A.rows();
  if(n>100)
  {
    // If the matrices are large enough, let's exploit the sparse strucure of A by
    // splitting it in half (wrt n1), and packing the non-zero columns.
    DenseIndex n2 = n - n1;
    MatrixXr A1(n1,n), A2(n2,n), B1(n,n), B2(n,n);
    Index k1=0, k2=0;
    for(Index j=0; j<n; ++j)
    {
      if( (A.col(j).head(n1).array()!=0).any() )
      {
        A1.col(k1) = A.col(j).head(n1);
        B1.row(k1) = B.row(j);
        ++k1;
      }
      if( (A.col(j).tail(n2).array()!=0).any() )
      {
        A2.col(k2) = A.col(j).tail(n2);
        B2.row(k2) = B.row(j);
        ++k2;
      }
    }
  
    A.topRows(n1).noalias()    = A1.leftCols(k1) * B1.topRows(k1);
    A.bottomRows(n2).noalias() = A2.leftCols(k2) * B2.topRows(k2);
  }
  else
    A *= B;  // FIXME this requires a temporary
}

// The divide algorithm is done "in place", we are always working on subsets of the same matrix. The divide methods takes as argument the 
// place of the submatrix we are currently working on.

//@param firstCol : The Index of the first column of the submatrix of m_computed and for m_naiveU;
//@param lastCol : The Index of the last column of the submatrix of m_computed and for m_naiveU; 
// lastCol + 1 - firstCol is the size of the submatrix.
//@param firstRowW : The Index of the first row of the matrix W that we are to change. (see the reference paper section 1 for more information on W)
//@param firstRowW : Same as firstRowW with the column.
//@param shift : Each time one takes the left submatrix, one must add 1 to the shift. Why? Because! We actually want the last column of the U submatrix 
// to become the first column (*coeff) and to shift all the other columns to the right. There are more details on the reference paper.
template<typename MatrixType>
void BDCSVD<MatrixType>::divide (Index firstCol, Index lastCol, Index firstRowW, Index firstColW, Index shift)
{
  // requires nbRows = nbCols + 1;
  using std::pow;
  using std::sqrt;
  using std::abs;
  const Index n = lastCol - firstCol + 1;
  const Index k = n/2;
  RealScalar alphaK;
  RealScalar betaK; 
  RealScalar r0; 
  RealScalar lambda, phi, c0, s0;
  VectorType l, f;
  // We use the other algorithm which is more efficient for small 
  // matrices.
  if (n < m_algoswap)
  {
    JacobiSVD<MatrixXr> b(m_computed.block(firstCol, firstCol, n + 1, n), ComputeFullU | (m_compV ? ComputeFullV : 0)) ;
    if (m_compU)
      m_naiveU.block(firstCol, firstCol, n + 1, n + 1).real() = b.matrixU();
    else 
    {
      m_naiveU.row(0).segment(firstCol, n + 1).real() = b.matrixU().row(0);
      m_naiveU.row(1).segment(firstCol, n + 1).real() = b.matrixU().row(n);
    }
    if (m_compV) m_naiveV.block(firstRowW, firstColW, n, n).real() = b.matrixV();
    m_computed.block(firstCol + shift, firstCol + shift, n + 1, n).setZero();
    m_computed.diagonal().segment(firstCol + shift, n) = b.singularValues().head(n);
    return;
  }
  // We use the divide and conquer algorithm
  alphaK =  m_computed(firstCol + k, firstCol + k);
  betaK = m_computed(firstCol + k + 1, firstCol + k);
  // The divide must be done in that order in order to have good results. Divide change the data inside the submatrices
  // and the divide of the right submatrice reads one column of the left submatrice. That's why we need to treat the 
  // right submatrix before the left one. 
  divide(k + 1 + firstCol, lastCol, k + 1 + firstRowW, k + 1 + firstColW, shift);
  divide(firstCol, k - 1 + firstCol, firstRowW, firstColW + 1, shift + 1);
  if (m_compU)
  {
    lambda = m_naiveU(firstCol + k, firstCol + k);
    phi = m_naiveU(firstCol + k + 1, lastCol + 1);
  } 
  else 
  {
    lambda = m_naiveU(1, firstCol + k);
    phi = m_naiveU(0, lastCol + 1);
  }
  r0 = sqrt((abs(alphaK * lambda) * abs(alphaK * lambda)) + abs(betaK * phi) * abs(betaK * phi));
  if (m_compU)
  {
    l = m_naiveU.row(firstCol + k).segment(firstCol, k);
    f = m_naiveU.row(firstCol + k + 1).segment(firstCol + k + 1, n - k - 1);
  } 
  else 
  {
    l = m_naiveU.row(1).segment(firstCol, k);
    f = m_naiveU.row(0).segment(firstCol + k + 1, n - k - 1);
  }
  if (m_compV) m_naiveV(firstRowW+k, firstColW) = 1;
  if (r0 == 0)
  {
    c0 = 1;
    s0 = 0;
  }
  else
  {
    c0 = alphaK * lambda / r0;
    s0 = betaK * phi / r0;
  }
  if (m_compU)
  {
    MatrixXr q1 (m_naiveU.col(firstCol + k).segment(firstCol, k + 1));     
    // we shiftW Q1 to the right
    for (Index i = firstCol + k - 1; i >= firstCol; i--) 
      m_naiveU.col(i + 1).segment(firstCol, k + 1) = m_naiveU.col(i).segment(firstCol, k + 1);
    // we shift q1 at the left with a factor c0
    m_naiveU.col(firstCol).segment( firstCol, k + 1) = (q1 * c0);
    // last column = q1 * - s0
    m_naiveU.col(lastCol + 1).segment(firstCol, k + 1) = (q1 * ( - s0));
    // first column = q2 * s0
    m_naiveU.col(firstCol).segment(firstCol + k + 1, n - k) = m_naiveU.col(lastCol + 1).segment(firstCol + k + 1, n - k) * s0; 
    // q2 *= c0
    m_naiveU.col(lastCol + 1).segment(firstCol + k + 1, n - k) *= c0;
  } 
  else 
  {
    RealScalar q1 = (m_naiveU(0, firstCol + k));
    // we shift Q1 to the right
    for (Index i = firstCol + k - 1; i >= firstCol; i--) 
      m_naiveU(0, i + 1) = m_naiveU(0, i);
    // we shift q1 at the left with a factor c0
    m_naiveU(0, firstCol) = (q1 * c0);
    // last column = q1 * - s0
    m_naiveU(0, lastCol + 1) = (q1 * ( - s0));
    // first column = q2 * s0
    m_naiveU(1, firstCol) = m_naiveU(1, lastCol + 1) *s0; 
    // q2 *= c0
    m_naiveU(1, lastCol + 1) *= c0;
    m_naiveU.row(1).segment(firstCol + 1, k).setZero();
    m_naiveU.row(0).segment(firstCol + k + 1, n - k - 1).setZero();
  }
  m_computed(firstCol + shift, firstCol + shift) = r0;
  m_computed.col(firstCol + shift).segment(firstCol + shift + 1, k) = alphaK * l.transpose().real();
  m_computed.col(firstCol + shift).segment(firstCol + shift + k + 1, n - k - 1) = betaK * f.transpose().real();

  // Second part: try to deflate singular values in combined matrix
  deflation(firstCol, lastCol, k, firstRowW, firstColW, shift);

  // Third part: compute SVD of combined matrix
  MatrixXr UofSVD, VofSVD;
  VectorType singVals;
  computeSVDofM(firstCol + shift, n, UofSVD, singVals, VofSVD);
  
  if (m_compU)  structured_update(m_naiveU.block(firstCol, firstCol, n + 1, n + 1), UofSVD, (n+2)/2);
  else          m_naiveU.middleCols(firstCol, n + 1) *= UofSVD; // FIXME this requires a temporary, and exploit that there are 2 rows at compile time
  
  if (m_compV)  structured_update(m_naiveV.block(firstRowW, firstColW, n, n), VofSVD, (n+1)/2);
  
  m_computed.block(firstCol + shift, firstCol + shift, n, n).setZero();
  m_computed.block(firstCol + shift, firstCol + shift, n, n).diagonal() = singVals;
}// end divide

// Compute SVD of m_computed.block(firstCol, firstCol, n + 1, n); this block only has non-zeros in
// the first column and on the diagonal and has undergone deflation, so diagonal is in increasing
// order except for possibly the (0,0) entry. The computed SVD is stored U, singVals and V, except
// that if m_compV is false, then V is not computed. Singular values are sorted in decreasing order.
//
// TODO Opportunities for optimization: better root finding algo, better stopping criterion, better
// handling of round-off errors, be consistent in ordering
template <typename MatrixType>
void BDCSVD<MatrixType>::computeSVDofM(Index firstCol, Index n, MatrixXr& U, VectorType& singVals, MatrixXr& V)
{
  // TODO Get rid of these copies (?)
  // FIXME at least preallocate them
  ArrayXr col0 = m_computed.col(firstCol).segment(firstCol, n);
  ArrayXr diag = m_computed.block(firstCol, firstCol, n, n).diagonal();
  diag(0) = 0;

  // compute singular values and vectors (in decreasing order)
  singVals.resize(n);
  U.resize(n+1, n+1);
  if (m_compV) V.resize(n, n);

  if (col0.hasNaN() || diag.hasNaN()) return;

  ArrayXr shifts(n), mus(n), zhat(n);
  
  computeSingVals(col0, diag, singVals, shifts, mus);
  perturbCol0(col0, diag, singVals, shifts, mus, zhat);
  computeSingVecs(zhat, diag, singVals, shifts, mus, U, V);

  // Reverse order so that singular values in increased order
  singVals.reverseInPlace();
  U.leftCols(n) = U.leftCols(n).rowwise().reverse().eval(); // FIXME this requires a temporary
  if (m_compV) V = V.rowwise().reverse().eval();            // FIXME this requires a temporary
}

template <typename MatrixType>
void BDCSVD<MatrixType>::computeSingVals(const ArrayXr& col0, const ArrayXr& diag, 
                                         VectorType& singVals, ArrayXr& shifts, ArrayXr& mus)
{
  using std::abs;
  using std::swap;
  using std::max;

  Index n = col0.size();
  for (Index k = 0; k < n; ++k)
  {
    if (col0(k) == 0)
    {
      // entry is deflated, so singular value is on diagonal
      singVals(k) = diag(k);
      mus(k) = 0;
      shifts(k) = diag(k);
      continue;
    } 

    // otherwise, use secular equation to find singular value
    RealScalar left = diag(k);
    RealScalar right = (k != n-1) ? diag(k+1) : (diag(n-1) + col0.matrix().norm());

    // first decide whether it's closer to the left end or the right end
    RealScalar mid = left + (right-left) / 2;
    RealScalar fMid = 1 + (col0.square() / ((diag + mid) * (diag - mid))).sum();

    RealScalar shift = (k == n-1 || fMid > 0) ? left : right;
    
    // measure everything relative to shift
    ArrayXr diagShifted = diag - shift;
    
    // initial guess
    RealScalar muPrev, muCur;
    if (shift == left)
    {
      muPrev = (right - left) * 0.1;
      if (k == n-1) muCur = right - left;
      else          muCur = (right - left) * 0.5; 
    }
    else
    {
      muPrev = -(right - left) * 0.1;
      muCur = -(right - left) * 0.5;
    }

    RealScalar fPrev = 1 + (col0.square() / ((diagShifted - muPrev) * (diag + shift + muPrev))).sum();
    RealScalar fCur = 1 + (col0.square() / ((diagShifted - muCur) * (diag + shift + muCur))).sum();
    if (abs(fPrev) < abs(fCur))
    {
      swap(fPrev, fCur);
      swap(muPrev, muCur);
    }

    // rational interpolation: fit a function of the form a / mu + b through the two previous
    // iterates and use its zero to compute the next iterate
    bool useBisection = false;
    while (abs(muCur - muPrev) > 8 * NumTraits<RealScalar>::epsilon() * (max)(abs(muCur), abs(muPrev)) && fCur != fPrev && !useBisection)
    {
      ++m_numIters;

      RealScalar a = (fCur - fPrev) / (1/muCur - 1/muPrev);
      RealScalar b = fCur - a / muCur;

      muPrev = muCur;
      fPrev = fCur;
      muCur = -a / b;
      fCur = 1 + (col0.square() / ((diagShifted - muCur) * (diag + shift + muCur))).sum();
      
      if (shift == left && (muCur < 0 || muCur > right - left)) useBisection = true;
      if (shift == right && (muCur < -(right - left) || muCur > 0)) useBisection = true;
    }

    // fall back on bisection method if rational interpolation did not work
    if (useBisection)
    {
      RealScalar leftShifted, rightShifted;
      if (shift == left)
      {
        leftShifted = 1e-30;
        if (k == 0) rightShifted = right - left;
        else        rightShifted = (right - left) * 0.6; // theoretically we can take 0.5, but let's be safe
      }
      else
      {
        leftShifted = -(right - left) * 0.6;
        rightShifted = -1e-30;
      }
      
      RealScalar fLeft = 1 + (col0.square() / ((diagShifted - leftShifted) * (diag + shift + leftShifted))).sum();
      RealScalar fRight = 1 + (col0.square() / ((diagShifted - rightShifted) * (diag + shift + rightShifted))).sum();
      assert(fLeft * fRight < 0);
      
      while (rightShifted - leftShifted > 2 * NumTraits<RealScalar>::epsilon() * (max)(abs(leftShifted), abs(rightShifted)))
      {
        RealScalar midShifted = (leftShifted + rightShifted) / 2;
        RealScalar fMid = 1 + (col0.square() / ((diagShifted - midShifted) * (diag + shift + midShifted))).sum();
        if (fLeft * fMid < 0)
        {
          rightShifted = midShifted;
          fRight = fMid;
        }
        else
        {
          leftShifted = midShifted;
          fLeft = fMid;
        }
      }

      muCur = (leftShifted + rightShifted) / 2;
    }
      
    singVals[k] = shift + muCur;
    shifts[k] = shift;
    mus[k] = muCur;

    // perturb singular value slightly if it equals diagonal entry to avoid division by zero later
    // (deflation is supposed to avoid this from happening)
    if (singVals[k] == left) singVals[k] *= 1 + NumTraits<RealScalar>::epsilon();
    if (singVals[k] == right) singVals[k] *= 1 - NumTraits<RealScalar>::epsilon();
  }
}


// zhat is perturbation of col0 for which singular vectors can be computed stably (see Section 3.1)
template <typename MatrixType>
void BDCSVD<MatrixType>::perturbCol0
   (const ArrayXr& col0, const ArrayXr& diag, const VectorType& singVals,
    const ArrayXr& shifts, const ArrayXr& mus, ArrayXr& zhat)
{
  using std::sqrt;
  Index n = col0.size();
  for (Index k = 0; k < n; ++k)
  {
    if (col0(k) == 0) 
      zhat(k) = 0;
    else
    {
      // see equation (3.6)
      RealScalar tmp = 
        sqrt(
             (singVals(n-1) + diag(k)) * (mus(n-1) + (shifts(n-1) - diag(k)))
             * (
                ((singVals.head(k).array() + diag(k)) * (mus.head(k) + (shifts.head(k) - diag(k))))
                / ((diag.head(k).array() + diag(k)) * (diag.head(k).array() - diag(k)))
               ).prod() 
             * (
                ((singVals.segment(k, n-k-1).array() + diag(k)) * (mus.segment(k, n-k-1) + (shifts.segment(k, n-k-1) - diag(k))))
                / ((diag.tail(n-k-1) + diag(k)) * (diag.tail(n-k-1) - diag(k)))
               ).prod()
             );
      if (col0(k) > 0) zhat(k) = tmp;
      else zhat(k) = -tmp;
    }
  }
}

// compute singular vectors
template <typename MatrixType>
void BDCSVD<MatrixType>::computeSingVecs
   (const ArrayXr& zhat, const ArrayXr& diag, const VectorType& singVals,
    const ArrayXr& shifts, const ArrayXr& mus, MatrixXr& U, MatrixXr& V)
{
  Index n = zhat.size();
  for (Index k = 0; k < n; ++k)
  {
    if (zhat(k) == 0)
    {
      U.col(k) = VectorType::Unit(n+1, k);
      if (m_compV) V.col(k) = VectorType::Unit(n, k);
    }
    else
    {
      U.col(k).head(n) = zhat / (((diag - shifts(k)) - mus(k)) * (diag + singVals[k]));
      U(n,k) = 0;
      U.col(k).normalize();
    
      if (m_compV)
      {
        V.col(k).tail(n-1) = (diag * zhat / (((diag - shifts(k)) - mus(k)) * (diag + singVals[k]))).tail(n-1);
        V(0,k) = -1;
        V.col(k).normalize();
      }
    }
  }
  U.col(n) = VectorType::Unit(n+1, n);
}


// page 12_13
// i >= 1, di almost null and zi non null.
// We use a rotation to zero out zi applied to the left of M
template <typename MatrixType>
void BDCSVD<MatrixType>::deflation43(Index firstCol, Index shift, Index i, Index size)
{
  using std::abs;
  using std::sqrt;
  using std::pow;
  RealScalar c = m_computed(firstCol + shift, firstCol + shift);
  RealScalar s = m_computed(i, firstCol + shift);
  RealScalar r = sqrt(pow(abs(c), 2) + pow(abs(s), 2));
  if (r == 0)
  {
    m_computed(i, i) = 0;
    return;
  }
  c/=r;
  s/=r;
  m_computed(firstCol + shift, firstCol + shift) = r;  
  m_computed(i, firstCol + shift) = 0;
  m_computed(i, i) = 0;
  if (m_compU)
  {
    m_naiveU.col(firstCol).segment(firstCol,size) = 
      c * m_naiveU.col(firstCol).segment(firstCol, size) - 
      s * m_naiveU.col(i).segment(firstCol, size) ;

    m_naiveU.col(i).segment(firstCol, size) = 
      (c + s*s/c) * m_naiveU.col(i).segment(firstCol, size) + 
      (s/c) * m_naiveU.col(firstCol).segment(firstCol,size);
  }
}// end deflation 43


// page 13
// i,j >= 1, i != j and |di - dj| < epsilon * norm2(M)
// We apply two rotations to have zj = 0;
template <typename MatrixType>
void BDCSVD<MatrixType>::deflation44(Index firstColu , Index firstColm, Index firstRowW, Index firstColW, Index i, Index j, Index size)
{
  using std::abs;
  using std::sqrt;
  using std::conj;
  using std::pow;
  RealScalar c = m_computed(firstColm, firstColm + j - 1);
  RealScalar s = m_computed(firstColm, firstColm + i - 1);
  RealScalar r = sqrt(pow(abs(c), 2) + pow(abs(s), 2));
  if (r==0)
  {
    m_computed(firstColm + i, firstColm + i) = m_computed(firstColm + j, firstColm + j);
    return;
  }
  c/=r;
  s/=r;
  m_computed(firstColm + i, firstColm) = r;  
  m_computed(firstColm + i, firstColm + i) = m_computed(firstColm + j, firstColm + j);
  m_computed(firstColm + j, firstColm) = 0;
  if (m_compU)
  {
    m_naiveU.col(firstColu + i).segment(firstColu, size) = 
      c * m_naiveU.col(firstColu + i).segment(firstColu, size) - 
      s * m_naiveU.col(firstColu + j).segment(firstColu, size) ;

    m_naiveU.col(firstColu + j).segment(firstColu, size) = 
      (c + s*s/c) *  m_naiveU.col(firstColu + j).segment(firstColu, size) + 
      (s/c) * m_naiveU.col(firstColu + i).segment(firstColu, size);
  } 
  if (m_compV)
  {
    m_naiveV.col(firstColW + i).segment(firstRowW, size - 1) = 
      c * m_naiveV.col(firstColW + i).segment(firstRowW, size - 1) + 
      s * m_naiveV.col(firstColW + j).segment(firstRowW, size - 1) ;

    m_naiveV.col(firstColW + j).segment(firstRowW, size - 1)  = 
      (c + s*s/c) * m_naiveV.col(firstColW + j).segment(firstRowW, size - 1) - 
      (s/c) * m_naiveV.col(firstColW + i).segment(firstRowW, size - 1);
  }
}// end deflation 44


// acts on block from (firstCol+shift, firstCol+shift) to (lastCol+shift, lastCol+shift) [inclusive]
template <typename MatrixType>
void BDCSVD<MatrixType>::deflation(Index firstCol, Index lastCol, Index k, Index firstRowW, Index firstColW, Index shift)
{
  //condition 4.1
  using std::sqrt;
  using std::abs;
  const Index length = lastCol + 1 - firstCol;
  RealScalar norm1 = m_computed.block(firstCol+shift, firstCol+shift, length, 1).squaredNorm();
  RealScalar norm2 = m_computed.block(firstCol+shift, firstCol+shift, length, length).diagonal().squaredNorm();
  RealScalar epsilon = 10 * NumTraits<RealScalar>::epsilon() * sqrt(norm1 + norm2);
  if (m_computed(firstCol + shift, firstCol + shift) < epsilon)
    m_computed(firstCol + shift, firstCol + shift) = epsilon;

  //condition 4.2
  for (Index i=firstCol + shift + 1;i<=lastCol + shift;i++)
    if (abs(m_computed(i, firstCol + shift)) < epsilon)
      m_computed(i, firstCol + shift) = 0;

  //condition 4.3
  for (Index i=firstCol + shift + 1;i<=lastCol + shift; i++)
    if (m_computed(i, i) < epsilon)
      deflation43(firstCol, shift, i, length);

  //condition 4.4
 
  Index i=firstCol + shift + 1, j=firstCol + shift + k + 1;
  //we stock the final place of each line
  Index *permutation = new Index[length]; // FIXME avoid repeated dynamic memory allocation

  for (Index p =1; p < length; p++)
  {
         if (i> firstCol + shift + k)             permutation[p] = j++;
    else if (j> lastCol + shift)                  permutation[p] = i++;
    else if (m_computed(i, i) < m_computed(j, j)) permutation[p] = j++;
    else                                          permutation[p] = i++;
  }
  //we do the permutation
  RealScalar aux;
  //we stock the current index of each col
  //and the column of each index
  Index *realInd = new Index[length];  // FIXME avoid repeated dynamic memory allocation
  Index *realCol = new Index[length];  // FIXME avoid repeated dynamic memory allocation
  for (int pos = 0; pos< length; pos++)
  {
    realCol[pos] = pos + firstCol + shift;
    realInd[pos] = pos;
  }
  const Index Zero = firstCol + shift;
  VectorType temp;
  for (int i = 1; i < length - 1; i++)
  {
    const Index I = i + Zero;
    const Index realI = realInd[i];
    const Index j  = permutation[length - i] - Zero;
    const Index J = realCol[j];
    
    //diag displace
    aux = m_computed(I, I); 
    m_computed(I, I) = m_computed(J, J);
    m_computed(J, J) = aux;
    
    //firstrow displace
    aux = m_computed(I, Zero); 
    m_computed(I, Zero) = m_computed(J, Zero);
    m_computed(J, Zero) = aux;

    // change columns
    if (m_compU)
    {
      temp = m_naiveU.col(I - shift).segment(firstCol, length + 1);
      m_naiveU.col(I - shift).segment(firstCol, length + 1) = m_naiveU.col(J - shift).segment(firstCol, length + 1);
      m_naiveU.col(J - shift).segment(firstCol, length + 1) = temp;
    } 
    else
    {
      temp = m_naiveU.col(I - shift).segment(0, 2);
      m_naiveU.col(I - shift).template head<2>() = m_naiveU.col(J - shift).segment(0, 2);
      m_naiveU.col(J - shift).template head<2>() = temp;      
    }
    if (m_compV)
    {
      const Index CWI = I + firstColW - Zero;
      const Index CWJ = J + firstColW - Zero;
      temp = m_naiveV.col(CWI).segment(firstRowW, length);
      m_naiveV.col(CWI).segment(firstRowW, length) = m_naiveV.col(CWJ).segment(firstRowW, length);
      m_naiveV.col(CWJ).segment(firstRowW, length) = temp;
    }

    //update real pos
    realCol[realI] = J;
    realCol[j] = I;
    realInd[J - Zero] = realI;
    realInd[I - Zero] = j;
  }
  for (Index i = firstCol + shift + 1; i<lastCol + shift;i++)
    if ((m_computed(i + 1, i + 1) - m_computed(i, i)) < epsilon)
      deflation44(firstCol, firstCol + shift, firstRowW, firstColW, i - Zero, i + 1 - Zero, length);
  
  delete[] permutation;
  delete[] realInd;
  delete[] realCol;
}//end deflation


  /** \svd_module
   *
   * \return the singular value decomposition of \c *this computed by 
   *  BDC Algorithm
   *
   * \sa class BDCSVD
   */
/*
template<typename Derived>
BDCSVD<typename MatrixBase<Derived>::PlainObject>
MatrixBase<Derived>::bdcSvd(unsigned int computationOptions) const
{
  return BDCSVD<PlainObject>(*this, computationOptions);
}
*/

} // end namespace Eigen

#endif
