// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_JACOBISVD_H
#define EIGEN_JACOBISVD_H

// forward declarations (needed by ICC)
// the empty bodies are required by MSVC
template<typename MatrixType, int QRPreconditioner,
         bool IsComplex = NumTraits<typename MatrixType::Scalar>::IsComplex>
struct ei_svd_precondition_2x2_block_to_be_real {};

template<typename MatrixType, int QRPreconditioner,
         bool PossiblyMoreRowsThanCols = (MatrixType::RowsAtCompileTime == Dynamic)
                                         || (MatrixType::RowsAtCompileTime > MatrixType::ColsAtCompileTime) >
struct ei_svd_precondition_if_more_rows_than_cols;

template<typename MatrixType, int QRPreconditioner,
         bool PossiblyMoreColsThanRows = (MatrixType::ColsAtCompileTime == Dynamic)
                                         || (MatrixType::ColsAtCompileTime > MatrixType::RowsAtCompileTime) >
struct ei_svd_precondition_if_more_cols_than_rows;


/*** QR preconditioners (R-SVD) ***/

enum { PreconditionIfMoreColsThanRows, PreconditionIfMoreRowsThanCols };

template<typename MatrixType, int QRPreconditioner, int Case>
struct ei_qr_preconditioner_should_do_anything
{
  enum { a = MatrixType::RowsAtCompileTime != Dynamic &&
              MatrixType::ColsAtCompileTime != Dynamic &&
              MatrixType::ColsAtCompileTime <= MatrixType::RowsAtCompileTime,
         b = MatrixType::RowsAtCompileTime != Dynamic &&
              MatrixType::ColsAtCompileTime != Dynamic &&
              MatrixType::RowsAtCompileTime <= MatrixType::ColsAtCompileTime,
         ret = !( (QRPreconditioner == NoQRPreconditioner) ||
                  (Case == PreconditionIfMoreColsThanRows && bool(a)) ||
                  (Case == PreconditionIfMoreRowsThanCols && bool(b)) )
  };
};

template<typename MatrixType, int QRPreconditioner, int Case,
         bool DoAnything = ei_qr_preconditioner_should_do_anything<MatrixType, QRPreconditioner, Case>::ret
> struct ei_qr_preconditioner_impl {};

template<typename MatrixType, int QRPreconditioner, int Case>
struct ei_qr_preconditioner_impl<MatrixType, QRPreconditioner, Case, false>
{
  static bool run(JacobiSVD<MatrixType, QRPreconditioner>&, const MatrixType&)
  {
    return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, FullPivHouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols, true>
{
  static bool run(JacobiSVD<MatrixType, FullPivHouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.rows() > matrix.cols())
    {
      FullPivHouseholderQR<MatrixType> qr(matrix);
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.cols(),matrix.cols()).template triangularView<Upper>();
      if(svd.m_computeU) svd.m_matrixU = qr.matrixQ();
      if(svd.m_computeV) svd.m_matrixV = qr.colsPermutation();
      return true;
    }
    return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, FullPivHouseholderQRPreconditioner, PreconditionIfMoreColsThanRows, true>
{
  static bool run(JacobiSVD<MatrixType, FullPivHouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.cols() > matrix.rows())
    {
      typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime,
                     MatrixType::Options, MatrixType::MaxColsAtCompileTime, MatrixType::MaxRowsAtCompileTime>
              TransposeTypeWithSameStorageOrder;
      FullPivHouseholderQR<TransposeTypeWithSameStorageOrder> qr(matrix.adjoint());
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.rows(),matrix.rows()).template triangularView<Upper>().adjoint();
      if(svd.m_computeV) svd.m_matrixV = qr.matrixQ();
      if(svd.m_computeU) svd.m_matrixU = qr.colsPermutation();
      return true;
    }
    else return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, ColPivHouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols, true>
{
  static bool run(JacobiSVD<MatrixType, ColPivHouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.rows() > matrix.cols())
    {
      ColPivHouseholderQR<MatrixType> qr(matrix);
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.cols(),matrix.cols()).template triangularView<Upper>();
      if(svd.m_computeU) svd.m_matrixU = qr.householderQ();
      if(svd.m_computeV) svd.m_matrixV = qr.colsPermutation();
      return true;
    }
    return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, ColPivHouseholderQRPreconditioner, PreconditionIfMoreColsThanRows, true>
{
  static bool run(JacobiSVD<MatrixType, ColPivHouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.cols() > matrix.rows())
    {
      typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime,
                     MatrixType::Options, MatrixType::MaxColsAtCompileTime, MatrixType::MaxRowsAtCompileTime>
              TransposeTypeWithSameStorageOrder;
      ColPivHouseholderQR<TransposeTypeWithSameStorageOrder> qr(matrix.adjoint());
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.rows(),matrix.rows()).template triangularView<Upper>().adjoint();
      if(svd.m_computeV) svd.m_matrixV = qr.householderQ();
      if(svd.m_computeU) svd.m_matrixU = qr.colsPermutation();
      return true;
    }
    else return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, HouseholderQRPreconditioner, PreconditionIfMoreRowsThanCols, true>
{
  static bool run(JacobiSVD<MatrixType, HouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.rows() > matrix.cols())
    {
      HouseholderQR<MatrixType> qr(matrix);
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.cols(),matrix.cols()).template triangularView<Upper>();
      if(svd.m_computeU) svd.m_matrixU = qr.householderQ();
      if(svd.m_computeV) svd.m_matrixV.setIdentity(matrix.cols(), matrix.cols());
      return true;
    }
    return false;
  }
};

template<typename MatrixType>
struct ei_qr_preconditioner_impl<MatrixType, HouseholderQRPreconditioner, PreconditionIfMoreColsThanRows, true>
{
  static bool run(JacobiSVD<MatrixType, HouseholderQRPreconditioner>& svd, const MatrixType& matrix)
  {
    if(matrix.cols() > matrix.rows())
    {
      typedef Matrix<typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime,
                     MatrixType::Options, MatrixType::MaxColsAtCompileTime, MatrixType::MaxRowsAtCompileTime>
              TransposeTypeWithSameStorageOrder;
      HouseholderQR<TransposeTypeWithSameStorageOrder> qr(matrix.adjoint());
      svd.m_workMatrix = qr.matrixQR().block(0,0,matrix.rows(),matrix.rows()).template triangularView<Upper>().adjoint();
      if(svd.m_computeV) svd.m_matrixV = qr.householderQ();
      if(svd.m_computeU) svd.m_matrixU.setIdentity(matrix.rows(), matrix.rows());
      return true;
    }
    else return false;
  }
};



/** \ingroup SVD_Module
  *
  *
  * \class JacobiSVD
  *
  * \brief Jacobi SVD decomposition of a square matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  * \param QRPreconditioner the type of QR decomposition that will be used internally for the R-SVD step
  *                        for non-square matrices. The default, FullPivHouseholderQR, is safest but slow.
  *                        Consider using ColPivHouseholderQR instead of greater speed while still being
  *                        quite safe, or even HouseholderQR to get closer to the speed and unsafety of
  *                        bidiagonalizing SVD implementations. Finally, if you don't need to handle non-square matrices,
  *                        you don't need any QR decomposition; you can then pass the dummy type NoQRDecomposition,
  *                        which will result in smaller executable size and shorter compilation times.
  *
  * \sa MatrixBase::jacobiSvd()
  */
template<typename MatrixType, int QRPreconditioner> class JacobiSVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    typedef typename MatrixType::Index Index;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      DiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_DYNAMIC(RowsAtCompileTime,ColsAtCompileTime),
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
      MaxDiagSizeAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(MaxRowsAtCompileTime,MaxColsAtCompileTime),
      MatrixOptions = MatrixType::Options
    };

    typedef Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime,
                   MatrixOptions, MaxRowsAtCompileTime, MaxRowsAtCompileTime>
            MatrixUType;
    typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime,
                   MatrixOptions, MaxColsAtCompileTime, MaxColsAtCompileTime>
            MatrixVType;
    typedef typename ei_plain_diag_type<MatrixType, RealScalar>::type SingularValuesType;
    typedef typename ei_plain_row_type<MatrixType>::type RowType;
    typedef typename ei_plain_col_type<MatrixType>::type ColType;
    typedef Matrix<Scalar, DiagSizeAtCompileTime, DiagSizeAtCompileTime,
                   MatrixOptions, MaxDiagSizeAtCompileTime, MaxDiagSizeAtCompileTime>
            WorkMatrixType;

  public:

    /** \brief Default Constructor.
      *
      * The default constructor is useful in cases in which the user intends to
      * perform decompositions via JacobiSVD::compute(const MatrixType&).
      */
    JacobiSVD() : m_isInitialized(false) {}


    /** \brief Default Constructor with memory preallocation
      *
      * Like the default constructor but with preallocation of the internal data
      * according to the specified problem \a size.
      * \sa JacobiSVD()
      */
    JacobiSVD(Index rows, Index cols) : m_matrixU(rows, rows),
                                    m_matrixV(cols, cols),
                                    m_singularValues(std::min(rows, cols)),
                                    m_workMatrix(rows, cols),
                                    m_isInitialized(false) {}

    JacobiSVD(const MatrixType& matrix, unsigned int computationOptions = 0)
        : m_matrixU(matrix.rows(), matrix.rows()),
          m_matrixV(matrix.cols(), matrix.cols()),
          m_singularValues(),
          m_workMatrix(),
          m_isInitialized(false)
    {
      const Index minSize = std::min(matrix.rows(), matrix.cols());
      m_singularValues.resize(minSize);
      m_workMatrix.resize(minSize, minSize);
      compute(matrix, computationOptions);
    }

    JacobiSVD& compute(const MatrixType& matrix, unsigned int computationOptions = 0);

    const MatrixUType& matrixU() const
    {
      ei_assert(m_isInitialized && "JacobiSVD is not initialized.");
      return m_matrixU;
    }

    const SingularValuesType& singularValues() const
    {
      ei_assert(m_isInitialized && "JacobiSVD is not initialized.");
      return m_singularValues;
    }

    const MatrixVType& matrixV() const
    {
      ei_assert(m_isInitialized && "JacobiSVD is not initialized.");
      return m_matrixV;
    }

  protected:
    MatrixUType m_matrixU;
    MatrixVType m_matrixV;
    SingularValuesType m_singularValues;
    WorkMatrixType m_workMatrix;
    bool m_isInitialized, m_computeU, m_computeV;

    template<typename _MatrixType, int _QRPreconditioner, bool _IsComplex>
    friend struct ei_svd_precondition_2x2_block_to_be_real;
    template<typename _MatrixType, int _QRPreconditioner, int _Case, bool _DoAnything>
    friend struct ei_qr_preconditioner_impl;
};

template<typename MatrixType, int QRPreconditioner>
struct ei_svd_precondition_2x2_block_to_be_real<MatrixType, QRPreconditioner, false>
{
  typedef JacobiSVD<MatrixType, QRPreconditioner> SVD;
  typedef typename SVD::Index Index;
  static void run(typename SVD::WorkMatrixType&, SVD&, Index, Index) {}
};

template<typename MatrixType, int QRPreconditioner>
struct ei_svd_precondition_2x2_block_to_be_real<MatrixType, QRPreconditioner, true>
{
  typedef JacobiSVD<MatrixType, QRPreconditioner> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef typename SVD::Index Index;
  static void run(typename SVD::WorkMatrixType& work_matrix, SVD& svd, Index p, Index q)
  {
    Scalar z;
    PlanarRotation<Scalar> rot;
    RealScalar n = ei_sqrt(ei_abs2(work_matrix.coeff(p,p)) + ei_abs2(work_matrix.coeff(q,p)));
    if(n==0)
    {
      z = ei_abs(work_matrix.coeff(p,q)) / work_matrix.coeff(p,q);
      work_matrix.row(p) *= z;
      if(svd.m_computeU) svd.m_matrixU.col(p) *= ei_conj(z);
      z = ei_abs(work_matrix.coeff(q,q)) / work_matrix.coeff(q,q);
      work_matrix.row(q) *= z;
      if(svd.m_computeU) svd.m_matrixU.col(q) *= ei_conj(z);
    }
    else
    {
      rot.c() = ei_conj(work_matrix.coeff(p,p)) / n;
      rot.s() = work_matrix.coeff(q,p) / n;
      work_matrix.applyOnTheLeft(p,q,rot);
      if(svd.m_computeU) svd.m_matrixU.applyOnTheRight(p,q,rot.adjoint());
      if(work_matrix.coeff(p,q) != Scalar(0))
      {
        Scalar z = ei_abs(work_matrix.coeff(p,q)) / work_matrix.coeff(p,q);
        work_matrix.col(q) *= z;
        if(svd.m_computeV) svd.m_matrixV.col(q) *= z;
      }
      if(work_matrix.coeff(q,q) != Scalar(0))
      {
        z = ei_abs(work_matrix.coeff(q,q)) / work_matrix.coeff(q,q);
        work_matrix.row(q) *= z;
        if(svd.m_computeU) svd.m_matrixU.col(q) *= ei_conj(z);
      }
    }
  }
};

template<typename MatrixType, typename RealScalar, typename Index>
void ei_real_2x2_jacobi_svd(const MatrixType& matrix, Index p, Index q,
                            PlanarRotation<RealScalar> *j_left,
                            PlanarRotation<RealScalar> *j_right)
{
  Matrix<RealScalar,2,2> m;
  m << ei_real(matrix.coeff(p,p)), ei_real(matrix.coeff(p,q)),
       ei_real(matrix.coeff(q,p)), ei_real(matrix.coeff(q,q));
  PlanarRotation<RealScalar> rot1;
  RealScalar t = m.coeff(0,0) + m.coeff(1,1);
  RealScalar d = m.coeff(1,0) - m.coeff(0,1);
  if(t == RealScalar(0))
  {
    rot1.c() = 0;
    rot1.s() = d > 0 ? 1 : -1;
  }
  else
  {
    RealScalar u = d / t;
    rot1.c() = RealScalar(1) / ei_sqrt(1 + ei_abs2(u));
    rot1.s() = rot1.c() * u;
  }
  m.applyOnTheLeft(0,1,rot1);
  j_right->makeJacobi(m,0,1);
  *j_left  = rot1 * j_right->transpose();
}

template<typename MatrixType, int QRPreconditioner>
JacobiSVD<MatrixType, QRPreconditioner>&
JacobiSVD<MatrixType, QRPreconditioner>::compute(const MatrixType& matrix, unsigned int computationOptions)
{
  m_computeU = computationOptions & ComputeU;
  m_computeV = computationOptions & ComputeV;
  Index rows = matrix.rows();
  Index cols = matrix.cols();
  Index diagSize = std::min(rows, cols);
  m_singularValues.resize(diagSize);
  const RealScalar precision = 2 * NumTraits<Scalar>::epsilon();

  if(!ei_qr_preconditioner_impl<MatrixType, QRPreconditioner, PreconditionIfMoreColsThanRows>::run(*this, matrix)
  && !ei_qr_preconditioner_impl<MatrixType, QRPreconditioner, PreconditionIfMoreRowsThanCols>::run(*this, matrix))
  {
    m_workMatrix = matrix.block(0,0,diagSize,diagSize);
    if(m_computeU) m_matrixU.setIdentity(rows,rows);
    if(m_computeV) m_matrixV.setIdentity(cols,cols);
  }

  bool finished = false;
  while(!finished)
  {
    finished = true;
    for(Index p = 1; p < diagSize; ++p)
    {
      for(Index q = 0; q < p; ++q)
      {
        if(std::max(ei_abs(m_workMatrix.coeff(p,q)),ei_abs(m_workMatrix.coeff(q,p)))
            > std::max(ei_abs(m_workMatrix.coeff(p,p)),ei_abs(m_workMatrix.coeff(q,q)))*precision)
        {
          finished = false;
          ei_svd_precondition_2x2_block_to_be_real<MatrixType, QRPreconditioner>::run(m_workMatrix, *this, p, q);

          PlanarRotation<RealScalar> j_left, j_right;
          ei_real_2x2_jacobi_svd(m_workMatrix, p, q, &j_left, &j_right);

          m_workMatrix.applyOnTheLeft(p,q,j_left);
          if(m_computeU) m_matrixU.applyOnTheRight(p,q,j_left.transpose());

          m_workMatrix.applyOnTheRight(p,q,j_right);
          if(m_computeV) m_matrixV.applyOnTheRight(p,q,j_right);
        }
      }
    }
  }

  for(Index i = 0; i < diagSize; ++i)
  {
    RealScalar a = ei_abs(m_workMatrix.coeff(i,i));
    m_singularValues.coeffRef(i) = a;
    if(m_computeU && (a!=RealScalar(0))) m_matrixU.col(i) *= m_workMatrix.coeff(i,i)/a;
  }

  for(Index i = 0; i < diagSize; i++)
  {
    Index pos;
    m_singularValues.tail(diagSize-i).maxCoeff(&pos);
    if(pos)
    {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if(m_computeU) m_matrixU.col(pos).swap(m_matrixU.col(i));
      if(m_computeV) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }

  m_isInitialized = true;
  return *this;
}
#endif // EIGEN_JACOBISVD_H
