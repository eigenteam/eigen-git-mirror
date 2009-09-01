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

/** \ingroup SVD_Module
  * \nonstableyet
  *
  * \class JacobiSVD
  *
  * \brief Jacobi SVD decomposition of a square matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  * \param ComputeU whether the U matrix should be computed
  * \param ComputeV whether the V matrix should be computed
  *
  * \sa MatrixBase::jacobiSvd()
  */
template<typename MatrixType, unsigned int Options> class JacobiSVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    enum {
      ComputeU = 1,
      ComputeV = 1,
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::ColsAtCompileTime,
      DiagSizeAtCompileTime = EIGEN_ENUM_MIN(RowsAtCompileTime,ColsAtCompileTime),
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
      MaxDiagSizeAtCompileTime = EIGEN_ENUM_MIN(MaxRowsAtCompileTime,MaxColsAtCompileTime),
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
                   Options, MaxDiagSizeAtCompileTime, 1> SingularValuesType;
    typedef Matrix<Scalar, 1, RowsAtCompileTime, MatrixOptions, 1, MaxRowsAtCompileTime> RowType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1, MatrixOptions, MaxRowsAtCompileTime, 1> ColType;

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

    const MatrixUType& matrixV() const
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
};

template<typename MatrixType, unsigned int Options, bool IsComplex = NumTraits<typename MatrixType::Scalar>::IsComplex>
struct ei_svd_precondition_2x2_block_to_be_real
{
  static void run(MatrixType&, JacobiSVD<MatrixType, Options>&, int, int) {}
};

template<typename MatrixType, unsigned int Options>
struct ei_svd_precondition_2x2_block_to_be_real<MatrixType, Options, true>
{
  typedef JacobiSVD<MatrixType, Options> SVD;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  
  enum { ComputeU = SVD::ComputeU, ComputeV = SVD::ComputeV };
  static void run(MatrixType& work_matrix, JacobiSVD<MatrixType, Options>& svd, int p, int q)
  {
    Scalar c, s, z;
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
      c = ei_conj(work_matrix.coeff(p,p)) / n;
      s = work_matrix.coeff(q,p) / n;
      work_matrix.applyJacobiOnTheLeft(p,q,c,s);
      if(ComputeU) svd.m_matrixU.applyJacobiOnTheRight(p,q,ei_conj(c),-s);
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
                            RealScalar *c_left, RealScalar *s_left,
                            RealScalar *c_right, RealScalar *s_right)
{
  Matrix<RealScalar,2,2> m;
  m << ei_real(matrix.coeff(p,p)), ei_real(matrix.coeff(p,q)),
        ei_real(matrix.coeff(q,p)), ei_real(matrix.coeff(q,q));
  RealScalar c1, s1;
  RealScalar t = m.coeff(0,0) + m.coeff(1,1);
  RealScalar d = m.coeff(1,0) - m.coeff(0,1);
  if(t == RealScalar(0))
  {
    c1 = 0;
    s1 = d > 0 ? 1 : -1;
  }
  else
  {
    RealScalar u = d / t;
    c1 = RealScalar(1) / ei_sqrt(1 + ei_abs2(u));
    s1 = c1 * u;
  }
  m.applyJacobiOnTheLeft(0,1,c1,s1);
  RealScalar c2, s2;
  m.makeJacobi(0,1,&c2,&s2);
  *c_left = c1*c2 + s1*s2;
  *s_left = s1*c2 - c1*s2;
  *c_right = c2;
  *s_right = s2;
}

template<typename MatrixType, unsigned int Options>
JacobiSVD<MatrixType, Options>& JacobiSVD<MatrixType, Options>::compute(const MatrixType& matrix)
{
  MatrixType work_matrix(matrix);
  int size = matrix.rows();
  if(ComputeU) m_matrixU = MatrixUType::Identity(size,size);
  if(ComputeV) m_matrixV = MatrixUType::Identity(size,size);
  m_singularValues.resize(size);
  const RealScalar precision = 2 * epsilon<Scalar>();

sweep_again:
  for(int p = 1; p < size; ++p)
  {
    for(int q = 0; q < p; ++q)
    {
      if(std::max(ei_abs(work_matrix.coeff(p,q)),ei_abs(work_matrix.coeff(q,p)))
          > std::max(ei_abs(work_matrix.coeff(p,p)),ei_abs(work_matrix.coeff(q,q)))*precision)
      {
        ei_svd_precondition_2x2_block_to_be_real<MatrixType, Options>::run(work_matrix, *this, p, q);

        RealScalar c_left, s_left, c_right, s_right;
        ei_real_2x2_jacobi_svd(work_matrix, p, q, &c_left, &s_left, &c_right, &s_right);
        
        work_matrix.applyJacobiOnTheLeft(p,q,c_left,s_left);
        if(ComputeU) m_matrixU.applyJacobiOnTheRight(p,q,c_left,-s_left);
        
        work_matrix.applyJacobiOnTheRight(p,q,c_right,s_right);
        if(ComputeV) m_matrixV.applyJacobiOnTheRight(p,q,c_right,s_right);
      }
    }
  }
  
  RealScalar biggestOnDiag = work_matrix.diagonal().cwise().abs().maxCoeff();
  RealScalar maxAllowedOffDiag = biggestOnDiag * precision;
  for(int p = 0; p < size; ++p)
  {
    for(int q = 0; q < p; ++q)
      if(ei_abs(work_matrix.coeff(p,q)) > maxAllowedOffDiag)
        goto sweep_again;
    for(int q = p+1; q < size; ++q)
      if(ei_abs(work_matrix.coeff(p,q)) > maxAllowedOffDiag)
        goto sweep_again;
  }
  
  for(int i = 0; i < size; ++i)
  {
    RealScalar a = ei_abs(work_matrix.coeff(i,i));
    m_singularValues.coeffRef(i) = a;
    if(ComputeU && (a!=RealScalar(0))) m_matrixU.col(i) *= work_matrix.coeff(i,i)/a;
  }

  for(int i = 0; i < size; i++)
  {
    int pos;
    m_singularValues.end(size-i).maxCoeff(&pos);
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
