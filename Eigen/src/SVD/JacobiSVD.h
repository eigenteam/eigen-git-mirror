// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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
// the empty bodies are required by VC
template<typename MatrixType, unsigned int Options, bool IsComplex = NumTraits<typename MatrixType::Scalar>::IsComplex>
struct ei_svd_precondition_2x2_block_to_be_real {};

template<typename MatrixType, unsigned int Options,
         bool PossiblyMoreRowsThanCols = (Options & AtLeastAsManyColsAsRows) == 0
                                         && (MatrixType::RowsAtCompileTime==Dynamic
                                             || (MatrixType::RowsAtCompileTime>MatrixType::ColsAtCompileTime))>
struct ei_svd_precondition_if_more_rows_than_cols;

template<typename MatrixType, unsigned int Options,
         bool PossiblyMoreColsThanRows = (Options & AtLeastAsManyRowsAsCols) == 0
                                         && (MatrixType::ColsAtCompileTime==Dynamic
                                             || (MatrixType::ColsAtCompileTime>MatrixType::RowsAtCompileTime))>
struct ei_svd_precondition_if_more_cols_than_rows;

/** \ingroup SVD_Module
  * \nonstableyet
  *
  * \class JacobiSVD
  *
  * \brief Jacobi SVD decomposition of a square matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  * \param Options a bit field of flags offering the following options: \c SkipU and \c SkipV allow to skip the computation of
  *                the unitaries \a U and \a V respectively; \c AtLeastAsManyRowsAsCols and \c AtLeastAsManyRowsAsCols allows
  *                to hint the compiler to only generate the corresponding code paths; \c Square is equivantent to the combination of
  *                the latter two bits and is useful when you know that the matrix is square. Note that when this information can
  *                be automatically deduced from the matrix type (e.g. a Matrix3f is always square), Eigen does it for you.
  *
  * \sa MatrixBase::jacobiSvd()
  */
template<typename MatrixType, unsigned int Options> class JacobiSVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    enum {
      ComputeU = (Options & SkipU) == 0,
      ComputeV = (Options & SkipV) == 0,
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      DiagSizeAtCompileTime = EIGEN_SIZE_MIN(RowsAtCompileTime,ColsAtCompileTime),
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
      MaxDiagSizeAtCompileTime = EIGEN_SIZE_MIN(MaxRowsAtCompileTime,MaxColsAtCompileTime),
      MatrixOptions = MatrixType::Options
    };

    typedef Matrix<Scalar, Dynamic, Dynamic, MatrixOptions> DummyMatrixType;
    typedef typename ei_meta_if<ComputeU,
                                Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime,
                                       MatrixOptions, MaxRowsAtCompileTime, MaxRowsAtCompileTime>,
                                DummyMatrixType>::ret MatrixUType;
    typedef typename ei_meta_if<ComputeV,
                                Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime,
                                       MatrixOptions, MaxColsAtCompileTime, MaxColsAtCompileTime>,
                                DummyMatrixType>::ret MatrixVType;
    typedef Matrix<RealScalar, DiagSizeAtCompileTime, 1,
                   MatrixOptions, MaxDiagSizeAtCompileTime, 1> SingularValuesType;
    typedef Matrix<Scalar, 1, RowsAtCompileTime, MatrixOptions, 1, MaxRowsAtCompileTime> RowType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1, MatrixOptions, MaxRowsAtCompileTime, 1> ColType;
    typedef Matrix<Scalar, DiagSizeAtCompileTime, DiagSizeAtCompileTime,
                   MatrixOptions, MaxDiagSizeAtCompileTime, MaxDiagSizeAtCompileTime>
              WorkMatrixType;

  public:

    JacobiSVD() : m_isInitialized(false) {}

    JacobiSVD(const MatrixType& matrix) : m_isInitialized(false)
    {
      compute(matrix);
    }

    JacobiSVD& compute(const MatrixType& matrix);

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
    bool m_isInitialized;

    template<typename _MatrixType, unsigned int _Options, bool _IsComplex>
    friend struct ei_svd_precondition_2x2_block_to_be_real;
    template<typename _MatrixType, unsigned int _Options, bool _PossiblyMoreRowsThanCols>
    friend struct ei_svd_precondition_if_more_rows_than_cols;
    template<typename _MatrixType, unsigned int _Options, bool _PossiblyMoreRowsThanCols>
    friend struct ei_svd_precondition_if_more_cols_than_rows;
};

template<typename MatrixType, unsigned int Options>
struct ei_svd_precondition_2x2_block_to_be_real<MatrixType, Options, false>
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  static void run(typename SVD::WorkMatrixType&, JacobiSVD<MatrixType, Options>&, int, int) {}
};

template<typename MatrixType, unsigned int Options>
struct ei_svd_precondition_2x2_block_to_be_real<MatrixType, Options, true>
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  enum { ComputeU = SVD::ComputeU, ComputeV = SVD::ComputeV };
  static void run(typename SVD::WorkMatrixType& work_matrix, JacobiSVD<MatrixType, Options>& svd, int p, int q)
  {
    Scalar z;
    PlanarRotation<Scalar> rot;
    RealScalar n = ei_sqrt(ei_abs2(work_matrix.coeff(p,p)) + ei_abs2(work_matrix.coeff(q,p)));
    if(n==0)
    {
      z = ei_abs(work_matrix.coeff(p,q)) / work_matrix.coeff(p,q);
      work_matrix.row(p) *= z;
      if(ComputeU) svd.m_matrixU.col(p) *= ei_conj(z);
      z = ei_abs(work_matrix.coeff(q,q)) / work_matrix.coeff(q,q);
      work_matrix.row(q) *= z;
      if(ComputeU) svd.m_matrixU.col(q) *= ei_conj(z);
    }
    else
    {
      rot.c() = ei_conj(work_matrix.coeff(p,p)) / n;
      rot.s() = work_matrix.coeff(q,p) / n;
      work_matrix.applyOnTheLeft(p,q,rot);
      if(ComputeU) svd.m_matrixU.applyOnTheRight(p,q,rot.adjoint());
      if(work_matrix.coeff(p,q) != Scalar(0))
      {
        Scalar z = ei_abs(work_matrix.coeff(p,q)) / work_matrix.coeff(p,q);
        work_matrix.col(q) *= z;
        if(ComputeV) svd.m_matrixV.col(q) *= z;
      }
      if(work_matrix.coeff(q,q) != Scalar(0))
      {
        z = ei_abs(work_matrix.coeff(q,q)) / work_matrix.coeff(q,q);
        work_matrix.row(q) *= z;
        if(ComputeU) svd.m_matrixU.col(q) *= ei_conj(z);
      }
    }
  }
};

template<typename MatrixType, typename RealScalar>
void ei_real_2x2_jacobi_svd(const MatrixType& matrix, int p, int q,
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

template<typename MatrixType, unsigned int Options, bool PossiblyMoreRowsThanCols>
struct ei_svd_precondition_if_more_rows_than_cols
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  static bool run(const MatrixType&, typename SVD::WorkMatrixType&, JacobiSVD<MatrixType, Options>&) { return false; }
};

template<typename MatrixType, unsigned int Options>
struct ei_svd_precondition_if_more_rows_than_cols<MatrixType, Options, true>
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  enum { ComputeU = SVD::ComputeU, ComputeV = SVD::ComputeV };
  static bool run(const MatrixType& matrix, typename SVD::WorkMatrixType& work_matrix, SVD& svd)
  {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int diagSize = cols;
    if(rows > cols)
    {
      FullPivHouseholderQR<MatrixType> qr(matrix);
      work_matrix = qr.matrixQR().block(0,0,diagSize,diagSize).template triangularView<Upper>();
      if(ComputeU) svd.m_matrixU = qr.matrixQ();
      if(ComputeV) svd.m_matrixV = qr.colsPermutation();

      return true;
    }
    else return false;
  }
};

template<typename MatrixType, unsigned int Options, bool PossiblyMoreColsThanRows>
struct ei_svd_precondition_if_more_cols_than_rows
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  static bool run(const MatrixType&, typename SVD::WorkMatrixType&, JacobiSVD<MatrixType, Options>&) { return false; }
};

template<typename MatrixType, unsigned int Options>
struct ei_svd_precondition_if_more_cols_than_rows<MatrixType, Options, true>
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  enum {
    ComputeU = SVD::ComputeU,
    ComputeV = SVD::ComputeV,
    RowsAtCompileTime = SVD::RowsAtCompileTime,
    ColsAtCompileTime = SVD::ColsAtCompileTime,
    MaxRowsAtCompileTime = SVD::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = SVD::MaxColsAtCompileTime,
    MatrixOptions = SVD::MatrixOptions
  };

  static bool run(const MatrixType& matrix, typename SVD::WorkMatrixType& work_matrix, SVD& svd)
  {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int diagSize = rows;
    if(cols > rows)
    {
      typedef Matrix<Scalar,ColsAtCompileTime,RowsAtCompileTime,
                      MatrixOptions,MaxColsAtCompileTime,MaxRowsAtCompileTime>
              TransposeTypeWithSameStorageOrder;
      FullPivHouseholderQR<TransposeTypeWithSameStorageOrder> qr(matrix.adjoint());
      work_matrix = qr.matrixQR().block(0,0,diagSize,diagSize).template triangularView<Upper>().adjoint();
      if(ComputeV) svd.m_matrixV = qr.matrixQ();
      if(ComputeU) svd.m_matrixU = qr.colsPermutation();
      return true;
    }
    else return false;
  }
};

template<typename MatrixType, unsigned int Options>
JacobiSVD<MatrixType, Options>& JacobiSVD<MatrixType, Options>::compute(const MatrixType& matrix)
{
  WorkMatrixType work_matrix;
  int rows = matrix.rows();
  int cols = matrix.cols();
  int diagSize = std::min(rows, cols);
  m_singularValues.resize(diagSize);
  const RealScalar precision = 2 * NumTraits<Scalar>::epsilon();

  if(!ei_svd_precondition_if_more_rows_than_cols<MatrixType, Options>::run(matrix, work_matrix, *this)
  && !ei_svd_precondition_if_more_cols_than_rows<MatrixType, Options>::run(matrix, work_matrix, *this))
  {
    work_matrix = matrix.block(0,0,diagSize,diagSize);
    if(ComputeU) m_matrixU.setIdentity(rows,rows);
    if(ComputeV) m_matrixV.setIdentity(cols,cols);
  }

  bool finished = false;
  while(!finished)
  {
    finished = true;
    for(int p = 1; p < diagSize; ++p)
    {
      for(int q = 0; q < p; ++q)
      {
        if(std::max(ei_abs(work_matrix.coeff(p,q)),ei_abs(work_matrix.coeff(q,p)))
            > std::max(ei_abs(work_matrix.coeff(p,p)),ei_abs(work_matrix.coeff(q,q)))*precision)
        {
          finished = false;
          ei_svd_precondition_2x2_block_to_be_real<MatrixType, Options>::run(work_matrix, *this, p, q);

          PlanarRotation<RealScalar> j_left, j_right;
          ei_real_2x2_jacobi_svd(work_matrix, p, q, &j_left, &j_right);

          work_matrix.applyOnTheLeft(p,q,j_left);
          if(ComputeU) m_matrixU.applyOnTheRight(p,q,j_left.transpose());

          work_matrix.applyOnTheRight(p,q,j_right);
          if(ComputeV) m_matrixV.applyOnTheRight(p,q,j_right);
        }
      }
    }
  }

  for(int i = 0; i < diagSize; ++i)
  {
    RealScalar a = ei_abs(work_matrix.coeff(i,i));
    m_singularValues.coeffRef(i) = a;
    if(ComputeU && (a!=RealScalar(0))) m_matrixU.col(i) *= work_matrix.coeff(i,i)/a;
  }

  for(int i = 0; i < diagSize; i++)
  {
    int pos;
    m_singularValues.tail(diagSize-i).maxCoeff(&pos);
    if(pos)
    {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if(ComputeU) m_matrixU.col(pos).swap(m_matrixU.col(i));
      if(ComputeV) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }

  m_isInitialized = true;
  return *this;
}
#endif // EIGEN_JACOBISVD_H
