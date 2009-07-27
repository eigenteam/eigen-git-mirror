// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SELFADJOINTRANK2UPTADE_H
#define EIGEN_SELFADJOINTRANK2UPTADE_H

/* Optimized selfadjoint matrix += alpha * uv' + vu'
 * It corresponds to the Level2 syr2 BLAS routine
 */

template<typename Scalar, typename UType, typename VType, int UpLo>
struct ei_selfadjoint_rank2_update_selector;

template<typename Scalar, typename UType, typename VType>
struct ei_selfadjoint_rank2_update_selector<Scalar,UType,VType,LowerTriangular>
{
  static void run(Scalar* mat, int stride, const UType& u, const VType& v, Scalar alpha)
  {
    const int size = u.size();
//     std::cerr << "lower \n" << u.transpose() << "\n" << v.transpose() << "\n\n";
    for (int i=0; i<size; ++i)
    {
//       std::cerr <<
      Map<Matrix<Scalar,Dynamic,1> >(mat+stride*i+i, size-i) +=
                        (alpha * ei_conj(u.coeff(i))) * v.end(size-i)
                      + (alpha * ei_conj(v.coeff(i))) * u.end(size-i);
    }
  }
};

template<typename Scalar, typename UType, typename VType>
struct ei_selfadjoint_rank2_update_selector<Scalar,UType,VType,UpperTriangular>
{
  static void run(Scalar* mat, int stride, const UType& u, const VType& v, Scalar alpha)
  {
    const int size = u.size();
    for (int i=0; i<size; ++i)
      Map<Matrix<Scalar,Dynamic,1> >(mat+stride*i, i+1) +=
                        (alpha * ei_conj(u.coeff(i))) * v.start(i+1)
                      + (alpha * ei_conj(v.coeff(i))) * u.start(i+1);
  }
};

template<bool Cond, typename T> struct ei_conj_expr_if
  : ei_meta_if<!Cond, const T&,
      CwiseUnaryOp<ei_scalar_conjugate_op<typename ei_traits<T>::Scalar>,T> > {};


template<typename MatrixType, unsigned int UpLo>
template<typename DerivedU, typename DerivedV>
SelfAdjointView<MatrixType,UpLo>& SelfAdjointView<MatrixType,UpLo>
::rankUpdate(const MatrixBase<DerivedU>& u, const MatrixBase<DerivedV>& v, Scalar alpha)
{
  typedef ei_blas_traits<DerivedU> UBlasTraits;
  typedef typename UBlasTraits::DirectLinearAccessType ActualUType;
  typedef typename ei_cleantype<ActualUType>::type _ActualUType;
  const ActualUType actualU = UBlasTraits::extract(u.derived());

  typedef ei_blas_traits<DerivedV> VBlasTraits;
  typedef typename VBlasTraits::DirectLinearAccessType ActualVType;
  typedef typename ei_cleantype<ActualVType>::type _ActualVType;
  const ActualVType actualV = VBlasTraits::extract(v.derived());

  Scalar actualAlpha = alpha * UBlasTraits::extractScalarFactor(u.derived())
                             * VBlasTraits::extractScalarFactor(v.derived());

  enum { IsRowMajor = (ei_traits<MatrixType>::Flags&RowMajorBit)?1:0 };
  ei_selfadjoint_rank2_update_selector<Scalar,
    typename ei_cleantype<typename ei_conj_expr_if<IsRowMajor ^ UBlasTraits::NeedToConjugate,_ActualUType>::ret>::type,
    typename ei_cleantype<typename ei_conj_expr_if<IsRowMajor ^ VBlasTraits::NeedToConjugate,_ActualVType>::ret>::type,
    (IsRowMajor ? (UpLo==UpperTriangular ? LowerTriangular : UpperTriangular) : UpLo)>
    ::run(const_cast<Scalar*>(_expression().data()),_expression().stride(),actualU,actualV,actualAlpha);

  return *this;
}

#endif // EIGEN_SELFADJOINTRANK2UPTADE_H
