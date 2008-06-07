// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_ASSIGN_H
#define EIGEN_ASSIGN_H

template<typename Derived1, typename Derived2, int UnrollCount>
struct ei_matrix_assignment_unroller
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_matrix_assignment_unroller<Derived1, Derived2, UnrollCount-1>::run(dst, src);
    dst.coeffRef(row, col) = src.coeff(row, col);
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, 1>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, 0>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

// Dynamic col-major
template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, -1>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = 0; i < dst.rows(); i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

// Dynamic row-major
template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, -2>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    // traverse in row-major order
    // in order to allow the compiler to unroll the inner loop
    for(int i = 0; i < dst.rows(); i++)
      for(int j = 0; j < dst.cols(); j++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

//----

template<typename Derived1, typename Derived2, int Index>
struct ei_matrix_assignment_packet_unroller
{
  enum {
    row = int(Derived1::Flags)&RowMajorBit ? Index / int(Derived1::ColsAtCompileTime) : Index % Derived1::RowsAtCompileTime,
    col = int(Derived1::Flags)&RowMajorBit ? Index % int(Derived1::ColsAtCompileTime) : Index / Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_matrix_assignment_packet_unroller<Derived1, Derived2,
      Index-ei_packet_traits<typename Derived1::Scalar>::size>::run(dst, src);
    dst.template writePacketCoeff<Aligned>(row, col, src.template packetCoeff<Aligned>(row, col));
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_packet_unroller<Derived1, Derived2, 0 >
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.template writePacketCoeff<Aligned>(0, 0, src.template packetCoeff<Aligned>(0, 0));
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_packet_unroller<Derived1, Derived2, Dynamic>
{
  inline static void run(Derived1 &, const Derived2 &)
  { ei_internal_assert(false && "ei_matrix_assignment_packet_unroller"); }
};

//----

template <typename Derived, typename OtherDerived,
bool Vectorize = (int(Derived::Flags) & int(OtherDerived::Flags) & VectorizableBit)
              && ((int(Derived::Flags)&RowMajorBit)==(int(OtherDerived::Flags)&RowMajorBit))
              && (   (int(Derived::Flags) & int(OtherDerived::Flags) & Like1DArrayBit)
                  || ((int(Derived::Flags) & RowMajorBit)
                    ?     int(Derived::ColsAtCompileTime)!=Dynamic
                      && (int(Derived::ColsAtCompileTime)%ei_packet_traits<typename Derived::Scalar>::size==0)
                    :     int(Derived::RowsAtCompileTime)!=Dynamic
                      && (int(Derived::RowsAtCompileTime)%ei_packet_traits<typename Derived::Scalar>::size==0)) ),
bool Unroll = Derived::SizeAtCompileTime * OtherDerived::CoeffReadCost <= EIGEN_UNROLLING_LIMIT>
struct ei_assignment_impl;

template<typename Derived>
template<typename OtherDerived>
inline Derived& MatrixBase<Derived>
  ::lazyAssign(const MatrixBase<OtherDerived>& other)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived,OtherDerived);
  ei_assert(rows() == other.rows() && cols() == other.cols());
  ei_assignment_impl<Derived, OtherDerived>::run(derived(),other.derived());
  return derived();
}

template<typename Derived, typename OtherDerived,
         bool EvalBeforeAssigning = (OtherDerived::Flags & EvalBeforeAssigningBit),
         bool NeedToTranspose = Derived::IsVectorAtCompileTime
                && OtherDerived::IsVectorAtCompileTime
                && (int)Derived::RowsAtCompileTime != (int)OtherDerived::RowsAtCompileTime
                && (int)Derived::ColsAtCompileTime != (int)OtherDerived::ColsAtCompileTime>
struct ei_assign_selector;

template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,true,true> {
  static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose().eval()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,true,false> {
  static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.eval()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,false,true> {
  static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,false,false> {
  static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.derived()); }
};

template<typename Derived>
template<typename OtherDerived>
inline Derived& MatrixBase<Derived>
  ::operator=(const MatrixBase<OtherDerived>& other)
{
  return ei_assign_selector<Derived,OtherDerived>::run(derived(), other.derived());
}

//----

// no vectorization
template <typename Derived, typename OtherDerived, bool Unroll>
struct ei_assignment_impl<Derived, OtherDerived, false, Unroll>
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    ei_matrix_assignment_unroller
      <Derived, OtherDerived,
      Unroll ? int(Derived::SizeAtCompileTime)
      : Derived::ColsAtCompileTime == Dynamic || Derived::RowsAtCompileTime != Dynamic ? -1 // col-major
      : -2 // row-major
      >::run(dst.derived(), src.derived());
  }
};

//----

template <typename Derived, typename OtherDerived>
struct ei_assignment_impl<Derived, OtherDerived, true, true> // vec + unrolling
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    ei_matrix_assignment_packet_unroller
      <Derived, OtherDerived,
       int(Derived::SizeAtCompileTime)-int(ei_packet_traits<typename Derived::Scalar>::size)
      >::run(dst.const_cast_derived(), src.derived());
  }
};

template <typename Derived, typename OtherDerived,
bool RowMajor = OtherDerived::Flags&RowMajorBit,
bool Complex1DArray = RowMajor
  ? (  (Derived::Flags & OtherDerived::Flags & Like1DArrayBit)
    && (   Derived::ColsAtCompileTime==Dynamic
       || Derived::ColsAtCompileTime%ei_packet_traits<typename Derived::Scalar>::size!=0) )
  : (  (Derived::Flags & OtherDerived::Flags & Like1DArrayBit)
    && (  Derived::RowsAtCompileTime==Dynamic
       || Derived::RowsAtCompileTime%ei_packet_traits<typename Derived::Scalar>::size!=0))>
struct ei_packet_assignment_seclector;

template <typename Derived, typename OtherDerived>
struct ei_assignment_impl<Derived, OtherDerived, true, false> // vec + no-unrolling
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    ei_packet_assignment_seclector<Derived,OtherDerived>::run(dst,src);
  }
};

template <typename Derived, typename OtherDerived>
struct ei_packet_assignment_seclector<Derived, OtherDerived, true, true> // row-major + complex 1D array like
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    const int size = dst.rows() * dst.cols();
    const int alignedSize = (size/ei_packet_traits<typename Derived::Scalar>::size)
                            * ei_packet_traits<typename Derived::Scalar>::size;
    int index = 0;
    for ( ; index<alignedSize ; index+=ei_packet_traits<typename Derived::Scalar>::size)
    {
      // FIXME the following is not really efficient
      int i = index/dst.cols();
      int j = index%dst.cols();
      dst.template writePacketCoeff<Aligned>(i, j, src.template packetCoeff<Aligned>(i, j));
    }
    for(int i = alignedSize/dst.cols(); i < dst.rows(); i++)
      for(int j = alignedSize%dst.cols(); j < dst.cols(); j++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

template <typename Derived, typename OtherDerived>
struct ei_packet_assignment_seclector<Derived, OtherDerived, true, false> // row-major + normal
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    for(int i = 0; i < dst.rows(); i++)
      for(int j = 0; j < dst.cols(); j+=ei_packet_traits<typename Derived::Scalar>::size)
        dst.template writePacketCoeff<Aligned>(i, j, src.template packetCoeff<Aligned>(i, j));
  }
};

template <typename Derived, typename OtherDerived>
struct ei_packet_assignment_seclector<Derived, OtherDerived, false, true> // col-major + complex 1D array like
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    const int size = dst.rows() * dst.cols();
    const int alignedSize = (size/ei_packet_traits<typename Derived::Scalar>::size)*ei_packet_traits<typename Derived::Scalar>::size;
    int index = 0;
    for ( ; index<alignedSize ; index+=ei_packet_traits<typename Derived::Scalar>::size)
    {
      // FIXME the following is not really efficient
      int i = index%dst.rows();
      int j = index/dst.rows();
      dst.template writePacketCoeff<Aligned>(i, j, src.template packetCoeff<Aligned>(i, j));
    }
    for(int j = alignedSize/dst.rows(); j < dst.cols(); j++)
      for(int i = alignedSize%dst.rows(); i < dst.rows(); i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

template <typename Derived, typename OtherDerived>
struct ei_packet_assignment_seclector<Derived, OtherDerived, false, false> // col-major + normal
{
  static void run(Derived & dst, const OtherDerived & src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = 0; i < dst.rows(); i+=ei_packet_traits<typename Derived::Scalar>::size)
        dst.template writePacketCoeff<Aligned>(i, j, src.template packetCoeff<Aligned>(i, j));
  }
};

#endif // EIGEN_ASSIGN_H
