// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SELFADJOINT_MATRIX_VECTOR_H
#define EIGEN_SELFADJOINT_MATRIX_VECTOR_H

/* Optimized selfadjoint matrix * vector product:
 * This algorithm processes 2 columns at onces that allows to both reduce
 * the number of load/stores of the result by a factor 2 and to reduce
 * the instruction dependency.
 */
template<typename Scalar, int StorageOrder, int UpLo>
static EIGEN_DONT_INLINE void ei_product_selfadjoint_vector(
  int size,
  const Scalar* lhs, int lhsStride,
  const Scalar* rhs, //int rhsIncr,
  Scalar* res)
{
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int PacketSize = sizeof(Packet)/sizeof(Scalar);

  enum {
    IsRowMajor = StorageOrder==RowMajorBit ? 1 : 0,
    IsLower = UpLo == LowerTriangularBit ? 1 : 0,
    FirstTriangular = IsRowMajor == IsLower
  };

  ei_conj_if<NumTraits<Scalar>::IsComplex && IsRowMajor> conj0;
  ei_conj_if<NumTraits<Scalar>::IsComplex && !IsRowMajor> conj1;

  for (int i=0;i<size;i++)
    res[i] = 0;

  int bound = std::max(0,size-8) & 0xfffffffE;
  if (FirstTriangular)
    bound = size - bound;

  for (int j=FirstTriangular ? bound : 0;
       j<(FirstTriangular ? size : bound);j+=2)
  {
    register const Scalar* EIGEN_RESTRICT A0 = lhs + j*lhsStride;
    register const Scalar* EIGEN_RESTRICT A1 = lhs + (j+1)*lhsStride;

    Scalar t0 = rhs[j];
    Packet ptmp0 = ei_pset1(t0);
    Scalar t1 = rhs[j+1];
    Packet ptmp1 = ei_pset1(t1);

    Scalar t2 = 0;
    Packet ptmp2 = ei_pset1(t2);
    Scalar t3 = 0;
    Packet ptmp3 = ei_pset1(t3);

    size_t starti = FirstTriangular ? 0 : j+2;
    size_t endi   = FirstTriangular ? j : size;
    size_t alignedEnd = starti;
    size_t alignedStart = (starti) + ei_alignmentOffset(&res[starti], endi-starti);
    alignedEnd = alignedStart + ((endi-alignedStart)/(PacketSize))*(PacketSize);

    res[j] += t0 * conj0(A0[j]);
    if(FirstTriangular)
    {
      res[j+1] += t1 * conj0(A1[j+1]);
      res[j]   += t1 * conj0(A1[j]);
      t3 += conj1(A1[j]) * rhs[j];
    }
    else
    {
      res[j+1] += t0 * conj0(A0[j+1]) + t1 * conj0(A1[j+1]);
      t2 += conj1(A0[j+1]) * rhs[j+1];
    }

    for (size_t i=starti; i<alignedStart; ++i)
    {
      res[i] += t0 * A0[i] + t1 * A1[i];
      t2 += ei_conj(A0[i]) * rhs[i];
      t3 += ei_conj(A1[i]) * rhs[i];
    }
    for (size_t i=alignedStart; i<alignedEnd; i+=PacketSize)
    {
      Packet A0i = ei_ploadu(&A0[i]);
      Packet A1i = ei_ploadu(&A1[i]);
      Packet Bi = ei_ploadu(&rhs[i]); // FIXME should be aligned in most cases
      Packet Xi = ei_pload(&res[i]);

      Xi = ei_padd(ei_padd(Xi, ei_pmul(ptmp0, conj0(A0i))), ei_pmul(ptmp1, conj0(A1i)));
      ptmp2 = ei_padd(ptmp2, ei_pmul(conj1(A0i), Bi));
      ptmp3 = ei_padd(ptmp3, ei_pmul(conj1(A1i), Bi));
      ei_pstore(&res[i],Xi);
    }
    for (size_t i=alignedEnd; i<endi; i++)
    {
      res[i] += t0 * conj0(A0[i]) + t1 * conj0(A1[i]);
      t2 += conj1(A0[i]) * rhs[i];
      t3 += conj1(A1[i]) * rhs[i];
    }

    res[j]   += t2 + ei_predux(ptmp2);
    res[j+1] += t3 + ei_predux(ptmp3);
  }
  for (int j=FirstTriangular ? 0 : bound;j<(FirstTriangular ? bound : size);j++)
  {
    register const Scalar* EIGEN_RESTRICT A0 = lhs + j*lhsStride;

    Scalar t1 = rhs[j];
    Scalar t2 = 0;
    res[j] += t1 * conj0(A0[j]);
    for (int i=FirstTriangular ? 0 : j+1; i<(FirstTriangular ? j : size); i++) {
      res[i] += t1 * conj0(A0[i]);
      t2 += conj1(A0[i]) * rhs[i];
    }
    res[j] += t2;
  }
}


#endif // EIGEN_SELFADJOINT_MATRIX_VECTOR_H
