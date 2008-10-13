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

#ifndef EIGEN_TAUCSSUPPORT_H
#define EIGEN_TAUCSSUPPORT_H

template<typename Scalar, int Flags>
taucs_ccs_matrix SparseMatrix<Scalar,Flags>::asTaucsMatrix()
{
  taucs_ccs_matrix res;
  res.n         = cols();
  res.m         = rows();
  res.flags     = 0;
  res.colptr    = _outerIndexPtr();
  res.rowind    = _innerIndexPtr();
  res.values.v  = _valuePtr();
  if (ei_is_same_type<Scalar,int>::ret)
    res.flags |= TAUCS_INT;
  else if (ei_is_same_type<Scalar,float>::ret)
    res.flags |= TAUCS_SINGLE;
  else if (ei_is_same_type<Scalar,double>::ret)
    res.flags |= TAUCS_DOUBLE;
  else if (ei_is_same_type<Scalar,std::complex<float> >::ret)
    res.flags |= TAUCS_SCOMPLEX;
  else if (ei_is_same_type<Scalar,std::complex<double> >::ret)
    res.flags |= TAUCS_DCOMPLEX;
  else
  {
    ei_assert(false && "Scalar type not supported by TAUCS");
  }

  if (Flags & Upper)
    res.flags |= TAUCS_UPPER;
  if (Flags & Lower)
    res.flags |= TAUCS_LOWER;
  if (Flags & SelfAdjoint)
    res.flags |= (NumTraits<Scalar>::IsComplex ? TAUCS_HERMITIAN : TAUCS_SYMMETRIC);
  else if ((Flags & Upper) || (Flags & Lower))
    res.flags |= TAUCS_TRIANGULAR;

  return res;
}

template<typename Scalar, int Flags>
SparseMatrix<Scalar,Flags> SparseMatrix<Scalar,Flags>::Map(taucs_ccs_matrix& taucsMat)
{
  SparseMatrix res;
  res.m_innerSize = taucsMat.m;
  res.m_outerSize = taucsMat.n;
  res.m_outerIndex = taucsMat.colptr;
  SparseArray<Scalar> data = SparseArray<Scalar>::Map(
                                taucsMat.rowind,
                                reinterpret_cast<Scalar*>(taucsMat.values.v),
                                taucsMat.colptr[taucsMat.n]);
  res.m_data.swap(data);
  res.markAsRValue();
  return res;
}

template<typename MatrixType>
class SparseLLT<MatrixType,Taucs> : public SparseLLT<MatrixType>
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
      : Base(flags), m_taucsSupernodalFactor(0)
    {
    }

    SparseLLT(const MatrixType& matrix, int flags = 0)
      : Base(matrix, flags), m_taucsSupernodalFactor(0)
    {
      compute(matrix);
    }

    ~SparseLLT()
    {
      if (m_taucsSupernodalFactor)
        taucs_supernodal_factor_free(m_taucsSupernodalFactor);
    }

    inline const typename Base::CholMatrixType& matrixL(void) const;

    template<typename Derived>
    void solveInPlace(MatrixBase<Derived> &b) const;

    void compute(const MatrixType& matrix);

  protected:
    void* m_taucsSupernodalFactor;
};

template<typename MatrixType>
void SparseLLT<MatrixType,Taucs>::compute(const MatrixType& a)
{
  if (m_taucsSupernodalFactor)
  {
    taucs_supernodal_factor_free(m_taucsSupernodalFactor);
    m_taucsSupernodalFactor = 0;
  }

  if (m_flags & IncompleteFactorization)
  {
    taucs_ccs_matrix taucsMatA = const_cast<MatrixType&>(a).asTaucsMatrix();
    taucs_ccs_matrix* taucsRes = taucs_ccs_factor_llt(&taucsMatA, Base::m_precision, 0);
    m_matrix = Base::CholMatrixType::Map(*taucsRes);
    free(taucsRes);
    m_status = (m_status & ~(CompleteFactorization|MatrixLIsDirty))
             | IncompleteFactorization
             | SupernodalFactorIsDirty;
  }
  else
  {
    taucs_ccs_matrix taucsMatA = const_cast<MatrixType&>(a).asTaucsMatrix();
    if ( (m_flags & SupernodalLeftLooking)
      || ((!(m_flags & SupernodalMultifrontal)) && (m_flags & MemoryEfficient)) )
    {
      m_taucsSupernodalFactor = taucs_ccs_factor_llt_ll(&taucsMatA);
    }
    else
    {
      // use the faster Multifrontal routine
      m_taucsSupernodalFactor = taucs_ccs_factor_llt_ll(&taucsMatA);
    }
    m_status = (m_status & ~IncompleteFactorization) | CompleteFactorization | MatrixLIsDirty;
  }
}

template<typename MatrixType>
inline const typename SparseLLT<MatrixType>::CholMatrixType&
SparseLLT<MatrixType,Taucs>::matrixL() const
{
  if (m_status & MatrixLIsDirty)
  {
    ei_assert(!(m_status & SupernodalFactorIsDirty));

    taucs_ccs_matrix* taucsL = taucs_supernodal_factor_to_ccs(m_taucsSupernodalFactor);
    const_cast<typename Base::CholMatrixType&>(m_matrix) = Base::CholMatrixType::Map(*taucsL);
    free(taucsL);
    m_status = (m_status & ~MatrixLIsDirty);
  }
  return m_matrix;
}

template<typename MatrixType>
template<typename Derived>
void SparseLLT<MatrixType,Taucs>::solveInPlace(MatrixBase<Derived> &b) const
{
  const int size = m_matrix.rows();
  ei_assert(size==b.rows());

  if (m_status & MatrixLIsDirty)
  {
//     ei_assert(!(m_status & SupernodalFactorIsDirty));
//     taucs_supernodal_solve_llt(m_taucsSupernodalFactor,double* b);
    //matrixL();
  }
  else
  {
    Base::solveInPlace(b);
  }
}

#endif // EIGEN_TAUCSSUPPORT_H
