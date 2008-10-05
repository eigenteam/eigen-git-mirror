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
void SparseCholesky<MatrixType>::computeUsingTaucs(const MatrixType& a)
{
  taucs_ccs_matrix taucsMatA = const_cast<MatrixType&>(a).asTaucsMatrix();
  taucs_ccs_matrix* taucsRes = taucs_ccs_factor_llt(&taucsMatA, 0, 0);
  m_matrix = CholMatrixType::Map(*taucsRes);
  free(taucsRes);
}

#endif // EIGEN_TAUCSSUPPORT_H
