// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007 Michael Olbrich <michael.olbrich@gmx.net>
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
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

/***************************************************************************
* Part 1 : the logic deciding a strategy for traversal and unrolling       *
***************************************************************************/

template <typename Derived, typename OtherDerived>
struct ei_assign_traits
{
public:
  enum {
    DstIsAligned = Derived::Flags & AlignedBit,
    DstHasDirectAccess = Derived::Flags & DirectAccessBit,
    SrcIsAligned = OtherDerived::Flags & AlignedBit,
    JointAlignment = DstIsAligned && SrcIsAligned ? Aligned : Unaligned
  };

private:
  enum {
    InnerSize = int(Derived::IsVectorAtCompileTime) ? int(Derived::SizeAtCompileTime)
              : int(Derived::Flags)&RowMajorBit ? int(Derived::ColsAtCompileTime)
              : int(Derived::RowsAtCompileTime),
    InnerMaxSize = int(Derived::IsVectorAtCompileTime) ? int(Derived::MaxSizeAtCompileTime)
              : int(Derived::Flags)&RowMajorBit ? int(Derived::MaxColsAtCompileTime)
              : int(Derived::MaxRowsAtCompileTime),
    MaxSizeAtCompileTime = Derived::SizeAtCompileTime,
    PacketSize = ei_packet_traits<typename Derived::Scalar>::size
  };

  enum {
    StorageOrdersAgree = (int(Derived::IsRowMajor) == int(OtherDerived::IsRowMajor)),
    MightVectorize = StorageOrdersAgree
                  && (int(Derived::Flags) & int(OtherDerived::Flags) & ActualPacketAccessBit),
    MayInnerVectorize  = MightVectorize && int(InnerSize)!=Dynamic && int(InnerSize)%int(PacketSize)==0
                       && int(DstIsAligned) && int(SrcIsAligned),
    MayLinearize = StorageOrdersAgree && (int(Derived::Flags) & int(OtherDerived::Flags) & LinearAccessBit),
    MayLinearVectorize = MightVectorize && MayLinearize && DstHasDirectAccess
                       && (DstIsAligned || MaxSizeAtCompileTime == Dynamic),
      /* If the destination isn't aligned, we have to do runtime checks and we don't unroll,
         so it's only good for large enough sizes. */
    MaySliceVectorize  = MightVectorize && DstHasDirectAccess && int(InnerMaxSize)>=3*PacketSize
      /* slice vectorization can be slow, so we only want it if the slices are big, which is
         indicated by InnerMaxSize rather than InnerSize, think of the case of a dynamic block
         in a fixed-size matrix */
  };

public:
  enum {
    Traversal = int(MayInnerVectorize)  ? int(InnerVectorizedTraversal)
              : int(MayLinearVectorize) ? int(LinearVectorizedTraversal)
              : int(MaySliceVectorize)  ? int(SliceVectorizedTraversal)
              : int(MayLinearize)       ? int(LinearTraversal)
                                        : int(DefaultTraversal),
    Vectorized = int(Traversal) == InnerVectorizedTraversal
              || int(Traversal) == LinearVectorizedTraversal
              || int(Traversal) == SliceVectorizedTraversal
  };

private:
  enum {
    UnrollingLimit      = EIGEN_UNROLLING_LIMIT * (Vectorized ? int(PacketSize) : 1),
    MayUnrollCompletely = int(Derived::SizeAtCompileTime) * int(OtherDerived::CoeffReadCost) <= int(UnrollingLimit),
    MayUnrollInner      = int(InnerSize) * int(OtherDerived::CoeffReadCost) <= int(UnrollingLimit)
  };

public:
  enum {
    Unrolling = (int(Traversal) == int(InnerVectorizedTraversal) || int(Traversal) == int(DefaultTraversal))
                ? (
                    int(MayUnrollCompletely) ? int(CompleteUnrolling)
                  : int(MayUnrollInner)      ? int(InnerUnrolling)
                                             : int(NoUnrolling)
                  )
              : int(Traversal) == int(LinearVectorizedTraversal)
                ? ( int(MayUnrollCompletely) && int(DstIsAligned) ? int(CompleteUnrolling) : int(NoUnrolling) )
              : int(Traversal) == int(LinearTraversal)
                ? ( int(MayUnrollCompletely) ? int(CompleteUnrolling) : int(NoUnrolling) )
              : int(NoUnrolling)
  };

#ifdef EIGEN_DEBUG_ASSIGN
  static void debug()
  {
    EIGEN_DEBUG_VAR(DstIsAligned)
    EIGEN_DEBUG_VAR(SrcIsAligned)
    EIGEN_DEBUG_VAR(JointAlignment)
    EIGEN_DEBUG_VAR(InnerSize)
    EIGEN_DEBUG_VAR(InnerMaxSize)
    EIGEN_DEBUG_VAR(PacketSize)
    EIGEN_DEBUG_VAR(StorageOrdersAgree)
    EIGEN_DEBUG_VAR(MightVectorize)
    EIGEN_DEBUG_VAR(MayLinearize)
    EIGEN_DEBUG_VAR(MayInnerVectorize)
    EIGEN_DEBUG_VAR(MayLinearVectorize)
    EIGEN_DEBUG_VAR(MaySliceVectorize)
    EIGEN_DEBUG_VAR(Traversal)
    EIGEN_DEBUG_VAR(UnrollingLimit)
    EIGEN_DEBUG_VAR(MayUnrollCompletely)
    EIGEN_DEBUG_VAR(MayUnrollInner)
    EIGEN_DEBUG_VAR(Unrolling)
  }
#endif
};

/***************************************************************************
* Part 2 : meta-unrollers
***************************************************************************/

/************************
*** Default traversal ***
************************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_DefaultTraversal_CompleteUnrolling
{
  enum {
    outer = Index / Derived1::InnerSizeAtCompileTime,
    inner = Index % Derived1::InnerSizeAtCompileTime
  };

  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.copyCoeffByOuterInner(outer, inner, src);
    ei_assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_DefaultTraversal_InnerUnrolling
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src, int outer)
  {
    dst.copyCoeffByOuterInner(outer, Index, src);
    ei_assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src, outer);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &, const Derived2 &, int) {}
};

/***********************
*** Linear traversal ***
***********************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_LinearTraversal_CompleteUnrolling
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.copyCoeff(Index, src);
    ei_assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &, const Derived2 &) {}
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_innervec_CompleteUnrolling
{
  enum {
    outer = Index / Derived1::InnerSizeAtCompileTime,
    inner = Index % Derived1::InnerSizeAtCompileTime,
    JointAlignment = ei_assign_traits<Derived1,Derived2>::JointAlignment
  };

  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.template copyPacketByOuterInner<Derived2, Aligned, JointAlignment>(outer, inner, src);
    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2,
      Index+ei_packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_innervec_InnerUnrolling
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src, int outer)
  {
    dst.template copyPacketByOuterInner<Derived2, Aligned, Aligned>(outer, Index, src);
    ei_assign_innervec_InnerUnrolling<Derived1, Derived2,
      Index+ei_packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src, outer);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_innervec_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &, const Derived2 &, int) {}
};

/***************************************************************************
* Part 3 : implementation of all cases
***************************************************************************/

template<typename Derived1, typename Derived2,
         int Traversal = ei_assign_traits<Derived1, Derived2>::Traversal,
         int Unrolling = ei_assign_traits<Derived1, Derived2>::Unrolling>
struct ei_assign_impl;

/************************
*** Default traversal ***
************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, DefaultTraversal, NoUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    const int innerSize = dst.innerSize();
    const int outerSize = dst.outerSize();
    for(int outer = 0; outer < outerSize; ++outer)
      for(int inner = 0; inner < innerSize; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, DefaultTraversal, CompleteUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, DefaultTraversal, InnerUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    const int outerSize = dst.outerSize();
    for(int outer = 0; outer < outerSize; ++outer)
      ei_assign_DefaultTraversal_InnerUnrolling<Derived1, Derived2, 0, Derived1::InnerSizeAtCompileTime>
        ::run(dst, src, outer);
  }
};

/***********************
*** Linear traversal ***
***********************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearTraversal, NoUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    const int size = dst.size();
    for(int i = 0; i < size; ++i)
      dst.copyCoeff(i, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearTraversal, CompleteUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_assign_LinearTraversal_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorizedTraversal, NoUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    const int innerSize = dst.innerSize();
    const int outerSize = dst.outerSize();
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    for(int outer = 0; outer < outerSize; ++outer)
      for(int inner = 0; inner < innerSize; inner+=packetSize)
        dst.template copyPacketByOuterInner<Derived2, Aligned, Aligned>(outer, inner, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorizedTraversal, CompleteUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorizedTraversal, InnerUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    const int outerSize = dst.outerSize();
    for(int outer = 0; outer < outerSize; ++outer)
      ei_assign_innervec_InnerUnrolling<Derived1, Derived2, 0, Derived1::InnerSizeAtCompileTime>
        ::run(dst, src, outer);
  }
};

/***************************
*** Linear vectorization ***
***************************/

template <bool IsAligned = false>
struct ei_unaligned_assign_impl
{
  template <typename Derived, typename OtherDerived>
  static EIGEN_STRONG_INLINE void run(const Derived&, OtherDerived&, int, int) {}
};

template <>
struct ei_unaligned_assign_impl<false>
{
  // MSVC must not inline this functions. If it does, it fails to optimize the
  // packet access path.
#ifdef _MSC_VER
  template <typename Derived, typename OtherDerived>
  static EIGEN_DONT_INLINE void run(const Derived& src, OtherDerived& dst, int start, int end)
#else
  template <typename Derived, typename OtherDerived>
  static EIGEN_STRONG_INLINE void run(const Derived& src, OtherDerived& dst, int start, int end)
#endif
  {
    for (int index = start; index < end; ++index)
      dst.copyCoeff(index, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearVectorizedTraversal, NoUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    const int size = dst.size();
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    const int alignedStart = ei_assign_traits<Derived1,Derived2>::DstIsAligned ? 0
                           : ei_first_aligned(&dst.coeffRef(0), size);
    const int alignedEnd = alignedStart + ((size-alignedStart)/packetSize)*packetSize;

    ei_unaligned_assign_impl<ei_assign_traits<Derived1,Derived2>::DstIsAligned!=0>::run(src,dst,0,alignedStart);

    for(int index = alignedStart; index < alignedEnd; index += packetSize)
    {
      dst.template copyPacket<Derived2, Aligned, ei_assign_traits<Derived1,Derived2>::JointAlignment>(index, src);
    }

    ei_unaligned_assign_impl<>::run(src,dst,alignedEnd,size);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearVectorizedTraversal, CompleteUnrolling>
{
  EIGEN_STRONG_INLINE static void run(Derived1 &dst, const Derived2 &src)
  {
    const int size = Derived1::SizeAtCompileTime;
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    const int alignedSize = (size/packetSize)*packetSize;

    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, alignedSize>::run(dst, src);
    ei_assign_DefaultTraversal_CompleteUnrolling<Derived1, Derived2, alignedSize, size>::run(dst, src);
  }
};

/**************************
*** Slice vectorization ***
***************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, SliceVectorizedTraversal, NoUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    const int packetAlignedMask = packetSize - 1;
    const int innerSize = dst.innerSize();
    const int outerSize = dst.outerSize();
    const int alignedStep = (packetSize - dst.outerStride() % packetSize) & packetAlignedMask;
    int alignedStart = ei_assign_traits<Derived1,Derived2>::DstIsAligned ? 0
                     : ei_first_aligned(&dst.coeffRef(0,0), innerSize);

    for(int outer = 0; outer < outerSize; ++outer)
    {
      const int alignedEnd = alignedStart + ((innerSize-alignedStart) & ~packetAlignedMask);
      // do the non-vectorizable part of the assignment
      for(int inner = 0; inner<alignedStart ; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);

      // do the vectorizable part of the assignment
      for(int inner = alignedStart; inner<alignedEnd; inner+=packetSize)
        dst.template copyPacketByOuterInner<Derived2, Aligned, Unaligned>(outer, inner, src);

      // do the non-vectorizable part of the assignment
      for(int inner = alignedEnd; inner<innerSize ; ++inner)
        dst.copyCoeffByOuterInner(outer, inner, src);

      alignedStart = std::min<int>((alignedStart+alignedStep)%packetSize, innerSize);
    }
  }
};

/***************************************************************************
* Part 4 : implementation of DenseBase methods
***************************************************************************/

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>
  ::lazyAssign(const DenseBase<OtherDerived>& other)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived,OtherDerived)
  EIGEN_STATIC_ASSERT((ei_is_same_type<typename Derived::Scalar, typename OtherDerived::Scalar>::ret),
    YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
#ifdef EIGEN_DEBUG_ASSIGN
  ei_assign_traits<Derived, OtherDerived>::debug();
#endif
  ei_assert(rows() == other.rows() && cols() == other.cols());
  ei_assign_impl<Derived, OtherDerived>::run(derived(),other.derived());
#ifndef EIGEN_NO_DEBUG
  checkTransposeAliasing(other.derived());
#endif
  return derived();
}

template<typename Derived, typename OtherDerived,
         bool EvalBeforeAssigning = (int(OtherDerived::Flags) & EvalBeforeAssigningBit) != 0,
         bool NeedToTranspose = Derived::IsVectorAtCompileTime
                && OtherDerived::IsVectorAtCompileTime
                && ((int(Derived::RowsAtCompileTime) == 1 && int(OtherDerived::ColsAtCompileTime) == 1)
                      |  // FIXME | instead of || to please GCC 4.4.0 stupid warning "suggest parentheses around &&".
                         // revert to || as soon as not needed anymore.
                    (int(Derived::ColsAtCompileTime) == 1 && int(OtherDerived::RowsAtCompileTime) == 1))
                && int(Derived::SizeAtCompileTime) != 1>
struct ei_assign_selector;

template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,false,false> {
  EIGEN_STRONG_INLINE static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.derived()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,true,false> {
  EIGEN_STRONG_INLINE static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.eval()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,false,true> {
  EIGEN_STRONG_INLINE static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose()); }
};
template<typename Derived, typename OtherDerived>
struct ei_assign_selector<Derived,OtherDerived,true,true> {
  EIGEN_STRONG_INLINE static Derived& run(Derived& dst, const OtherDerived& other) { return dst.lazyAssign(other.transpose().eval()); }
};

template<typename Derived>
template<typename OtherDerived>
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase<OtherDerived>& other)
{
  return ei_assign_selector<Derived,OtherDerived>::run(derived(), other.derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE Derived& DenseBase<Derived>::operator=(const DenseBase& other)
{
  return ei_assign_selector<Derived,Derived>::run(derived(), other.derived());
}

template<typename Derived>
EIGEN_STRONG_INLINE Derived& MatrixBase<Derived>::operator=(const MatrixBase& other)
{
  return ei_assign_selector<Derived,Derived>::run(derived(), other.derived());
}

#endif // EIGEN_ASSIGN_H
