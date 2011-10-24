// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_UMFPACKSUPPORT_LEGACY_H
#define EIGEN_UMFPACKSUPPORT_LEGACY_H

/** \deprecated use class BiCGSTAB, class SuperLU, or class UmfPackLU */
template<typename _MatrixType>
class SparseLU<_MatrixType,UmfPack> : public SparseLU<_MatrixType>
{
  protected:
    typedef SparseLU<_MatrixType> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    typedef Matrix<int, 1, _MatrixType::ColsAtCompileTime> IntRowVectorType;
    typedef Matrix<int, _MatrixType::RowsAtCompileTime, 1> IntColVectorType;
    typedef SparseMatrix<Scalar,Lower|UnitDiag> LMatrixType;
    typedef SparseMatrix<Scalar,Upper> UMatrixType;
    using Base::m_flags;
    using Base::m_status;

  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Index Index;

    /** \deprecated the entire class is deprecated */
    EIGEN_DEPRECATED SparseLU(int flags = NaturalOrdering)
      : Base(flags), m_numeric(0)
    {
    }

    /** \deprecated the entire class is deprecated */
    EIGEN_DEPRECATED SparseLU(const MatrixType& matrix, int flags = NaturalOrdering)
      : Base(flags), m_numeric(0)
    {
      compute(matrix);
    }

    ~SparseLU()
    {
      if (m_numeric)
        umfpack_free_numeric(&m_numeric,Scalar());
    }

    inline const LMatrixType& matrixL() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_l;
    }

    inline const UMatrixType& matrixU() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_u;
    }

    inline const IntColVectorType& permutationP() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_p;
    }

    inline const IntRowVectorType& permutationQ() const
    {
      if (m_extractedDataAreDirty) extractData();
      return m_q;
    }

    Scalar determinant() const;

    template<typename BDerived, typename XDerived>
    bool solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived>* x) const;

    template<typename Rhs>
      inline const internal::solve_retval<SparseLU<MatrixType, UmfPack>, Rhs>
    solve(const MatrixBase<Rhs>& b) const
    {
      eigen_assert(true && "SparseLU is not initialized.");
      return internal::solve_retval<SparseLU<MatrixType, UmfPack>, Rhs>(*this, b.derived());
    }

    void compute(const MatrixType& matrix);

    inline Index cols() const { return m_matrixRef->cols(); }
    inline Index rows() const { return m_matrixRef->rows(); }

    inline const MatrixType& matrixLU() const
    {
      //eigen_assert(m_isInitialized && "LU is not initialized.");
      return *m_matrixRef;
    }

    const void* numeric() const
    {
      return m_numeric;
    }

  protected:

    void extractData() const;
  
  protected:
    // cached data:
    void* m_numeric;
    const MatrixType* m_matrixRef;
    mutable LMatrixType m_l;
    mutable UMatrixType m_u;
    mutable IntColVectorType m_p;
    mutable IntRowVectorType m_q;
    mutable bool m_extractedDataAreDirty;
};

namespace internal {

template<typename _MatrixType, typename Rhs>
  struct solve_retval<SparseLU<_MatrixType, UmfPack>, Rhs>
  : solve_retval_base<SparseLU<_MatrixType, UmfPack>, Rhs>
{
  typedef SparseLU<_MatrixType, UmfPack> SpLUDecType;
  EIGEN_MAKE_SOLVE_HELPERS(SpLUDecType,Rhs)

  template<typename Dest> void evalTo(Dest& dst) const
  {
    const int rhsCols = rhs().cols();

    eigen_assert((Rhs::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major rhs yet");
    eigen_assert((Dest::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major result yet");

    void* numeric = const_cast<void*>(dec().numeric());

    EIGEN_UNUSED int errorCode = 0;
    for (int j=0; j<rhsCols; ++j)
    {
      errorCode = umfpack_solve(UMFPACK_A,
                                dec().matrixLU()._outerIndexPtr(), dec().matrixLU()._innerIndexPtr(), dec().matrixLU()._valuePtr(),
                                &dst.col(j).coeffRef(0), &rhs().const_cast_derived().col(j).coeffRef(0), numeric, 0, 0);
      eigen_assert(!errorCode && "UmfPack could not solve the system.");
    }
  }
    
};

} // end namespace internal

template<typename MatrixType>
void SparseLU<MatrixType,UmfPack>::compute(const MatrixType& a)
{
  typedef typename MatrixType::Index Index;
  const Index rows = a.rows();
  const Index cols = a.cols();
  eigen_assert((MatrixType::Flags&RowMajorBit)==0 && "Row major matrices are not supported yet");

  m_matrixRef = &a;

  if (m_numeric)
    umfpack_free_numeric(&m_numeric,Scalar());

  void* symbolic;
  int errorCode = 0;
  errorCode = umfpack_symbolic(rows, cols, a._outerIndexPtr(), a._innerIndexPtr(), a._valuePtr(),
                                  &symbolic, 0, 0);
  if (errorCode==0)
    errorCode = umfpack_numeric(a._outerIndexPtr(), a._innerIndexPtr(), a._valuePtr(),
                                   symbolic, &m_numeric, 0, 0);

  umfpack_free_symbolic(&symbolic,Scalar());

  m_extractedDataAreDirty = true;

  Base::m_succeeded = (errorCode==0);
}

template<typename MatrixType>
void SparseLU<MatrixType,UmfPack>::extractData() const
{
  if (m_extractedDataAreDirty)
  {
    // get size of the data
    int lnz, unz, rows, cols, nz_udiag;
    umfpack_get_lunz(&lnz, &unz, &rows, &cols, &nz_udiag, m_numeric, Scalar());

    // allocate data
    m_l.resize(rows,(std::min)(rows,cols));
    m_l.resizeNonZeros(lnz);
    
    m_u.resize((std::min)(rows,cols),cols);
    m_u.resizeNonZeros(unz);

    m_p.resize(rows);
    m_q.resize(cols);

    // extract
    umfpack_get_numeric(m_l._outerIndexPtr(), m_l._innerIndexPtr(), m_l._valuePtr(),
                        m_u._outerIndexPtr(), m_u._innerIndexPtr(), m_u._valuePtr(),
                        m_p.data(), m_q.data(), 0, 0, 0, m_numeric);
    
    m_extractedDataAreDirty = false;
  }
}

template<typename MatrixType>
typename SparseLU<MatrixType,UmfPack>::Scalar SparseLU<MatrixType,UmfPack>::determinant() const
{
  Scalar det;
  umfpack_get_determinant(&det, 0, m_numeric, 0);
  return det;
}

template<typename MatrixType>
template<typename BDerived,typename XDerived>
bool SparseLU<MatrixType,UmfPack>::solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived> *x) const
{
  //const int size = m_matrix.rows();
  const int rhsCols = b.cols();
//   eigen_assert(size==b.rows());
  eigen_assert((BDerived::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major rhs yet");
  eigen_assert((XDerived::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major result yet");

  int errorCode;
  for (int j=0; j<rhsCols; ++j)
  {
    errorCode = umfpack_solve(UMFPACK_A,
        m_matrixRef->_outerIndexPtr(), m_matrixRef->_innerIndexPtr(), m_matrixRef->_valuePtr(),
        &x->col(j).coeffRef(0), &b.const_cast_derived().col(j).coeffRef(0), m_numeric, 0, 0);
    if (errorCode!=0)
      return false;
  }
//   errorCode = umfpack_di_solve(UMFPACK_A,
//       m_matrixRef._outerIndexPtr(), m_matrixRef._innerIndexPtr(), m_matrixRef._valuePtr(),
//       x->derived().data(), b.derived().data(), m_numeric, 0, 0);

  return true;
}

#endif // EIGEN_UMFPACKSUPPORT_H
