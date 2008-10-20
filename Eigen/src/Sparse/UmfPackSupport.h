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

#ifndef EIGEN_UMFPACKSUPPORT_H
#define EIGEN_UMFPACKSUPPORT_H

/* TODO extract L, extrac U, compute det, etc... */

inline int umfpack_symbolic(int n_row,int n_col,
                            const int Ap[], const int Ai[], const double Ax[], void **Symbolic,
                            const double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
  return umfpack_di_symbolic(n_row,n_col,Ap,Ai,Ax,Symbolic,Control,Info);
}

inline int umfpack_symbolic(int n_row,int n_col,
                            const int Ap[], const int Ai[], const std::complex<double> Ax[], void **Symbolic,
                            const double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
  return umfpack_zi_symbolic(n_row,n_col,Ap,Ai,&Ax[0].real(),0,Symbolic,Control,Info);
}

inline int umfpack_numeric( const int Ap[], const int Ai[], const double Ax[],
                            void *Symbolic, void **Numeric,
                            const double Control[UMFPACK_CONTROL],double Info [UMFPACK_INFO])
{
  return umfpack_di_numeric(Ap,Ai,Ax,Symbolic,Numeric,Control,Info);
}

inline int umfpack_numeric( const int Ap[], const int Ai[], const std::complex<double> Ax[],
                            void *Symbolic, void **Numeric,
                            const double Control[UMFPACK_CONTROL],double Info [UMFPACK_INFO])
{
  return umfpack_zi_numeric(Ap,Ai,&Ax[0].real(),0,Symbolic,Numeric,Control,Info);
}

inline int umfpack_solve( int sys, const int Ap[], const int Ai[], const double Ax[],
                          double X[], const double B[], void *Numeric,
                          const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
  return umfpack_di_solve(sys,Ap,Ai,Ax,X,B,Numeric,Control,Info);
}

inline int umfpack_solve( int sys, const int Ap[], const int Ai[], const std::complex<double> Ax[],
                          std::complex<double> X[], const std::complex<double> B[], void *Numeric,
                          const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
  return umfpack_zi_solve(sys,Ap,Ai,&Ax[0].real(),0,&X[0].real(),0,&B[0].real(),0,Numeric,Control,Info);
}


template<typename MatrixType>
class SparseLU<MatrixType,UmfPack> : public SparseLU<MatrixType>
{
  protected:
    typedef SparseLU<MatrixType> Base;
    typedef typename Base::Scalar Scalar;
    typedef typename Base::RealScalar RealScalar;
    typedef Matrix<Scalar,Dynamic,1> Vector;
    using Base::m_flags;
    using Base::m_status;

  public:

    SparseLU(int flags = NaturalOrdering)
      : Base(flags), m_numeric(0)
    {
    }

    SparseLU(const MatrixType& matrix, int flags = NaturalOrdering)
      : Base(flags), m_numeric(0)
    {
      compute(matrix);
    }

    ~SparseLU()
    {
      if (m_numeric)
        umfpack_di_free_numeric(&m_numeric);
    }

    template<typename BDerived, typename XDerived>
    bool solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived>* x) const;

    void compute(const MatrixType& matrix);

  protected:
    // cached data:
    void* m_numeric;
    const MatrixType* m_matrixRef;
};

template<typename MatrixType>
void SparseLU<MatrixType,UmfPack>::compute(const MatrixType& a)
{
  const int rows = a.rows();
  const int cols = a.cols();
  ei_assert((MatrixType::Flags&RowMajorBit)==0 && "Row major matrices are not supported yet");

  m_matrixRef = &a;

  if (m_numeric)
    umfpack_di_free_numeric(&m_numeric);

  void* symbolic;
  int errorCode = 0;
  errorCode = umfpack_symbolic(rows, cols, a._outerIndexPtr(), a._innerIndexPtr(), a._valuePtr(),
                                  &symbolic, 0, 0);
  if (errorCode==0)
    errorCode = umfpack_numeric(a._outerIndexPtr(), a._innerIndexPtr(), a._valuePtr(),
                                   symbolic, &m_numeric, 0, 0);

  umfpack_di_free_symbolic(&symbolic);

  Base::m_succeeded = (errorCode==0);
}

// template<typename MatrixType>
// inline const MatrixType&
// SparseLU<MatrixType,SuperLU>::matrixL() const
// {
//   ei_assert(false && "matrixL() is Not supported by the SuperLU backend");
//   return m_matrix;
// }
//
// template<typename MatrixType>
// inline const MatrixType&
// SparseLU<MatrixType,SuperLU>::matrixU() const
// {
//   ei_assert(false && "matrixU() is Not supported by the SuperLU backend");
//   return m_matrix;
// }

template<typename MatrixType>
template<typename BDerived,typename XDerived>
bool SparseLU<MatrixType,UmfPack>::solve(const MatrixBase<BDerived> &b, MatrixBase<XDerived> *x) const
{
  //const int size = m_matrix.rows();
  const int rhsCols = b.cols();
//   ei_assert(size==b.rows());
  ei_assert((BDerived::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major rhs yet");
  ei_assert((XDerived::Flags&RowMajorBit)==0 && "UmfPack backend does not support non col-major result yet");

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
