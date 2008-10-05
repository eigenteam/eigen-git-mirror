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
void SparseCholesky<MatrixType>::computeUsingCholmod(const MatrixType& a)
{
  cholmod_common c;
  cholmod_start(&c);
  cholmod_sparse A = const_cast<MatrixType&>(a).asCholmodMatrix();
  if (!(m_flags&CholPartial))
  {
    c.nmethods = 1;
    c.method [0].ordering = CHOLMOD_NATURAL;
    c.postorder = 0;
  }
  c.final_ll = 1;
  cholmod_factor *L = cholmod_analyze(&A, &c);
  cholmod_factorize(&A, L, &c);
  cholmod_sparse* cmRes = cholmod_factor_to_sparse(L, &c);
  m_matrix = CholMatrixType::Map(*cmRes);
  free(cmRes);
  cholmod_free_factor(&L, &c);
  cholmod_finish(&c);
}

#endif // EIGEN_CHOLMODSUPPORT_H
