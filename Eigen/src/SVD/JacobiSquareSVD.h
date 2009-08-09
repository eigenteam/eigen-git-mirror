// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_JACOBISQUARESVD_H
#define EIGEN_JACOBISQUARESVD_H

/** \ingroup SVD_Module
  * \nonstableyet
  *
  * \class JacobiSquareSVD
  *
  * \brief Jacobi SVD decomposition of a square matrix
  *
  * \param MatrixType the type of the matrix of which we are computing the SVD decomposition
  * \param ComputeU whether the U matrix should be computed
  * \param ComputeV whether the V matrix should be computed
  *
  * \sa MatrixBase::jacobiSvd()
  */
template<typename MatrixType, bool ComputeU, bool ComputeV> class JacobiSquareSVD
{
  private:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
    enum {
      RowsAtCompileTime = MatrixType::RowsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
      Options = MatrixType::Options
    };
    
    typedef Matrix<Scalar, Dynamic, Dynamic, Options> DummyMatrixType;
    typedef typename ei_meta_if<ComputeU,
                                Matrix<Scalar, RowsAtCompileTime, RowsAtCompileTime,
                                       Options, MaxRowsAtCompileTime, MaxRowsAtCompileTime>,
                                DummyMatrixType>::ret MatrixUType;
    typedef typename Diagonal<MatrixType,0>::PlainMatrixType SingularValuesType;
    typedef Matrix<Scalar, 1, RowsAtCompileTime, Options, 1, MaxRowsAtCompileTime> RowType;
    typedef Matrix<Scalar, RowsAtCompileTime, 1, Options, MaxRowsAtCompileTime, 1> ColType;

  public:

    JacobiSquareSVD() : m_isInitialized(false) {}

    JacobiSquareSVD(const MatrixType& matrix)
    {
      compute(matrix);
    }
    
    void compute(const MatrixType& matrix);
    
    const MatrixUType& matrixU() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_matrixU;
    }

    const SingularValuesType& singularValues() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_singularValues;
    }

    const MatrixUType& matrixV() const
    {
      ei_assert(m_isInitialized && "SVD is not initialized.");
      return m_matrixV;
    }

  protected:
    MatrixUType m_matrixU;
    MatrixUType m_matrixV;
    SingularValuesType m_singularValues;
    bool m_isInitialized;
};

template<typename MatrixType, bool ComputeU, bool ComputeV>
void JacobiSquareSVD<MatrixType, ComputeU, ComputeV>::compute(const MatrixType& matrix)
{
  MatrixType work_matrix(matrix);
  int size = matrix.rows();
  if(ComputeU) m_matrixU = MatrixUType::Identity(size,size);
  if(ComputeV) m_matrixV = MatrixUType::Identity(size,size);
  m_singularValues.resize(size);
  RealScalar max_coeff = work_matrix.cwise().abs().maxCoeff();
  for(int k = 1; k < 40; ++k) {
    bool finished = true;
    for(int p = 1; p < size; ++p)
    {
      for(int q = 0; q < p; ++q)
      {
        Scalar c, s;
        finished &= work_matrix.makeJacobiForAtA(p,q,max_coeff,&c,&s);
        work_matrix.applyJacobiOnTheRight(p,q,c,s);
        if(ComputeV) m_matrixV.applyJacobiOnTheRight(p,q,c,s);
      }
    }
    if(finished) break;
  }
  
  for(int i = 0; i < size; ++i)
  {
    m_singularValues.coeffRef(i) = work_matrix.col(i).norm();
  }

  int first_zero = size;
  RealScalar biggest = m_singularValues.maxCoeff();
  for(int i = 0; i < size; i++)
  {
    int pos;
    RealScalar biggest_remaining = m_singularValues.end(size-i).maxCoeff(&pos);
    if(first_zero == size && ei_isMuchSmallerThan(biggest_remaining, biggest)) first_zero = pos + i;
    if(pos)
    {
      pos += i;
      std::swap(m_singularValues.coeffRef(i), m_singularValues.coeffRef(pos));
      if(ComputeU) work_matrix.col(pos).swap(work_matrix.col(i));
      if(ComputeV) m_matrixV.col(pos).swap(m_matrixV.col(i));
    }
  }
  
  if(ComputeU)
  {
    for(int i = 0; i < first_zero; ++i)
    {
      m_matrixU.col(i) = work_matrix.col(i) / m_singularValues.coeff(i);
    }
    if(first_zero < size)
    {
      for(int i = first_zero; i < size; ++i)
      {
        for(int j = 0; j < size; ++j)
        {
          m_matrixU.col(i).setZero();
          m_matrixU.coeffRef(j,i) = Scalar(1);
          for(int k = 0; k < first_zero; ++k)
            m_matrixU.col(i) -= m_matrixU.col(i).dot(m_matrixU.col(k)) * m_matrixU.col(k);
          RealScalar n = m_matrixU.col(i).norm();
          if(!ei_isMuchSmallerThan(n, biggest))
          {
            m_matrixU.col(i) /= n;
            break;
          }
        }
      }
    }     
  }
}
#endif // EIGEN_JACOBISQUARESVD_H
