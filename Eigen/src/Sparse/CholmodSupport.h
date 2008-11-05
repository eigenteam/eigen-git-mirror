// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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

#ifndef EIGEN_CHOLMODSUPPORT_H
#define EIGEN_CHOLMODSUPPORT_H

template<typename Scalar, int Flags>
cholmod_sparse SparseMatrix<Scalar,Flags>::asCholmodMatrix()
{
  cholmod_sparse res;
  res.nzmax   = nonZeros();
  res.nrow    = rows();;
  res.ncol    = cols();
  res.p       = _outerIndexPtr();
  res.i       = _innerIndexPtr();
  res.x       = _valuePtr();
  res.xtype = CHOLMOD_REAL;
  res.itype = CHOLMOD_INT;
  res.sorted = 1;
  res.packed = 1;
  res.dtype = 0;
  res.stype = -1;

  if (ei_is_same_type<Scalar,float>::ret)
  {
    res.xtype = CHOLMOD_REAL;
    res.dtype = 1;
  }
  else if (ei_is_same_type<Scalar,double>::ret)
  {
    res.xtype = CHOLMOD_REAL;
    res.dtype = 0;
  }
  else if (ei_is_same_type<Scalar,std::complex<float> >::ret)
  {
    res.xtype = CHOLMOD_COMPLEX;
    res.dtype = 1;
  }
  else if (ei_is_same_type<Scalar,std::complex<double> >::ret)
  {
    res.xtype = CHOLMOD_COMPLEX;
    res.dtype = 0;
  }
  else
  {
    ei_assert(false && "Scalar type not supported by CHOLMOD");
  }

  if (Flags & SelfAdjoint)
  {
    if (Flags & Upper)
      res.stype = 1;
    else if (Flags & Lower)
      res.stype = -1;
    else
      res.stype = 0;
  }
  else
    res.stype = 0;

  return res;
}

template<typename Scalar, int Flags>
SparseMatrix<Scalar,Flags> SparseMatrix<Scalar,Flags>::Map(cholmod_sparse& cm)
{
  SparseMatrix res;
  res.m_innerSize = cm.nrow;
  res.m_outerSize = cm.ncol;
  res.m_outerIndex = reinterpret_cast<int*>(cm.p);
  SparseArray<Scalar> data = SparseArray<Scalar>::Map(
                                reinterpret_cast<int*>(cm.i),
                                reinterpret_cast<Scalar*>(cm.x),
                                res.m_outerIndex[cm.ncol]);
  res.m_data.swap(data);
  res.markAsRValue();
  return res;
}

template<typename MatrixType>
class SparseLLT<MatrixType,Cholmod> : public SparseLLT<MatrixType>
{
  protected:
    typedef SparseLLT<MatrixType> Base;
    using Base::Scalar;
    using Base::RealScalar;
    using Base::MatrixLIsDirty;
    using Base::SupernodalFactorIsDirty;
    using Base::m_flags;
    using Base::m_matrix;
    using Base::m_status;

  public:

    SparseLLT(int flags = 0)
      : Base(flags), m_cholmodFactor(0)
    {
      cholmod_start(&m_cholmod);
    }

    SparseLLT(const MatrixType& matrix, int flags = 0)
      : Base(flags), m_cholmodFactor(0)
    {
      cholmod_start(&m_cholmod);
      compute(matrix);
    }

    ~SparseLLT()
    {
      if (m_cholmodFactor)
        cholmod_free_factor(&m_cholmodFactor, &m_cholmod);
      cholmod_finish(&m_cholmod);
    }

    inline const typename Base::CholMatrixType& matrixL(void) const;

    template<typename Derived>
    void solveInPlace(MatrixBase<Derived> &b) const;

    void compute(const MatrixType& matrix);

  protected:
    mutable cholmod_common m_cholmod;
    cholmod_factor* m_cholmodFactor;
};

template<typename MatrixType>
void SparseLLT<MatrixType,Cholmod>::compute(const MatrixType& a)
{
  if (m_cholmodFactor)
  {
    cholmod_free_factor(&m_cholmodFactor, &m_cholmod);
    m_cholmodFactor = 0;
  }

  cholmod_sparse A = const_cast<MatrixType&>(a).asCholmodMatrix();
  // TODO
  if (m_flags&IncompleteFactorization)
  {
    m_cholmod.nmethods = 1;
    m_cholmod.method [0].ordering = CHOLMOD_NATURAL;
    m_cholmod.postorder = 0;
  }
  else
  {
    m_cholmod.nmethods = 1;
    m_cholmod.method[0].ordering = CHOLMOD_NATURAL;
    m_cholmod.postorder = 0;
  }
  m_cholmod.final_ll = 1;
  m_cholmodFactor = cholmod_analyze(&A, &m_cholmod);
  cholmod_factorize(&A, m_cholmodFactor, &m_cholmod);

  m_status = (m_status & ~SupernodalFactorIsDirty) | MatrixLIsDirty;
}

template<typename MatrixType>
inline const typename SparseLLT<MatrixType>::CholMatrixType&
SparseLLT<MatrixType,Cholmod>::matrixL() const
{
  if (m_status & MatrixLIsDirty)
  {
    ei_assert(!(m_status & SupernodalFactorIsDirty));

    cholmod_sparse* cmRes = cholmod_factor_to_sparse(m_cholmodFactor, &m_cholmod);
    const_cast<typename Base::CholMatrixType&>(m_matrix) = Base::CholMatrixType::Map(*cmRes);
    free(cmRes);

    m_status = (m_status & ~MatrixLIsDirty);
  }
  return m_matrix;
}

template<typename MatrixType>
template<typename Derived>
void SparseLLT<MatrixType,Cholmod>::solveInPlace(MatrixBase<Derived> &b) const
{
  if (m_status & MatrixLIsDirty)
    matrixL();

  const int size = m_matrix.rows();
  ei_assert(size==b.rows());

  Base::solveInPlace(b);
}

#endif // EIGEN_CHOLMODSUPPORT_H
