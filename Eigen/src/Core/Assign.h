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

/***************************************************************************
* Part 1 : the logic deciding a strategy for vectorization and unrolling
***************************************************************************/

enum {
  NoVectorization,
  InnerVectorization,
  LinearVectorization,
  SliceVectorization
};

enum {
  CompleteUnrolling,
  InnerUnrolling,
  NoUnrolling
};

template <typename Derived, typename OtherDerived>
struct ei_assign_traits
{
private:
  enum {
    InnerSize = int(Derived::Flags)&RowMajorBit
              ? Derived::ColsAtCompileTime
              : Derived::RowsAtCompileTime,
    PacketSize = ei_packet_traits<typename Derived::Scalar>::size
  };

  enum {
    MightVectorize = (int(Derived::Flags) & int(OtherDerived::Flags) & PacketAccessBit)
             && ((int(Derived::Flags)&RowMajorBit)==(int(OtherDerived::Flags)&RowMajorBit)),
    MayInnerVectorize = MightVectorize && InnerSize!=Dynamic && int(InnerSize)%int(PacketSize)==0,
    MayLinearVectorize = MightVectorize && (int(Derived::Flags) & int(OtherDerived::Flags) & LinearAccessBit),
    MaySliceVectorize = MightVectorize && InnerSize==Dynamic
  };

public:
  enum {
    Vectorization = MayInnerVectorize  ? InnerVectorization
                  : MayLinearVectorize ? LinearVectorization
                  : MaySliceVectorize ? SliceVectorization
                                       : NoVectorization
  };

private:
  enum {
    UnrollingLimit      = EIGEN_UNROLLING_LIMIT * (int(Vectorization) == int(NoVectorization) ? 1 : int(PacketSize)),
    MayUnrollCompletely = int(Derived::SizeAtCompileTime) * int(OtherDerived::CoeffReadCost) <= int(UnrollingLimit),
    MayUnrollInner      = int(InnerSize * OtherDerived::CoeffReadCost) <= int(UnrollingLimit)
  };

public:
  enum {
    Unrolling = (int(Vectorization) == int(InnerVectorization) || int(Vectorization) == int(NoVectorization))
              ? (
                   MayUnrollCompletely ? CompleteUnrolling
                 : MayUnrollInner      ? InnerUnrolling
                                       : NoUnrolling
                )
              : int(Vectorization) == int(LinearVectorization)
              ? ( MayUnrollCompletely ? CompleteUnrolling : NoUnrolling )
              : NoUnrolling
  };
};

/***************************************************************************
* Part 2 : meta-unrollers
***************************************************************************/

/***********************
*** No vectorization ***
***********************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_novec_CompleteUnrolling
{
  enum {
    row = int(Derived1::Flags)&RowMajorBit
        ? Index / int(Derived1::ColsAtCompileTime)
        : Index % Derived1::RowsAtCompileTime,
    col = int(Derived1::Flags)&RowMajorBit
        ? Index % int(Derived1::ColsAtCompileTime)
        : Index / Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(row, col) = src.coeff(row, col);
    ei_assign_novec_CompleteUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_novec_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_novec_InnerUnrolling
{
  inline static void run(Derived1 &dst, const Derived2 &src, int row_or_col)
  {
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int row = rowMajor ? row_or_col : Index;
    const int col = rowMajor ? Index : row_or_col;
    dst.coeffRef(row, col) = src.coeff(row, col);
    ei_assign_novec_InnerUnrolling<Derived1, Derived2, Index+1, Stop>::run(dst, src, row_or_col);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_novec_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  inline static void run(Derived1 &, const Derived2 &, int) {}
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_innervec_CompleteUnrolling
{
  enum {
    row = int(Derived1::Flags)&RowMajorBit
        ? Index / int(Derived1::ColsAtCompileTime)
        : Index % Derived1::RowsAtCompileTime,
    col = int(Derived1::Flags)&RowMajorBit
        ? Index % int(Derived1::ColsAtCompileTime)
        : Index / Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.template writePacket<Aligned>(row, col, src.template packet<Aligned>(row, col));
    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2,
      Index+ei_packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, Stop, Stop>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, int Index, int Stop>
struct ei_assign_innervec_InnerUnrolling
{
  inline static void run(Derived1 &dst, const Derived2 &src, int row_or_col)
  {
    const int row = int(Derived1::Flags)&RowMajorBit ? row_or_col : Index;
    const int col = int(Derived1::Flags)&RowMajorBit ? Index : row_or_col;
    dst.template writePacket<Aligned>(row, col, src.template packet<Aligned>(row, col));
    ei_assign_innervec_InnerUnrolling<Derived1, Derived2,
      Index+ei_packet_traits<typename Derived1::Scalar>::size, Stop>::run(dst, src, row_or_col);
  }
};

template<typename Derived1, typename Derived2, int Stop>
struct ei_assign_innervec_InnerUnrolling<Derived1, Derived2, Stop, Stop>
{
  inline static void run(Derived1 &, const Derived2 &, int) {}
};

/***************************************************************************
* Part 3 : implementation of all cases
***************************************************************************/

template<typename Derived1, typename Derived2,
         int Vectorization = ei_assign_traits<Derived1, Derived2>::Vectorization,
         int Unrolling = ei_assign_traits<Derived1, Derived2>::Unrolling>
struct ei_assign_impl;

/***********************
*** No vectorization ***
***********************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, NoVectorization, NoUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int innerSize = rowMajor ? dst.cols() : dst.rows();
    const int outerSize = rowMajor ? dst.rows() : dst.cols();
    for(int j = 0; j < outerSize; j++)
      for(int i = 0; i < innerSize; i++)
      {
        const int row = rowMajor ? j : i;
        const int col = rowMajor ? i : j;
        dst.coeffRef(row, col) = src.coeff(row, col);
      }
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, NoVectorization, CompleteUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_assign_novec_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, NoVectorization, InnerUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int innerSize = rowMajor ? Derived1::ColsAtCompileTime : Derived1::RowsAtCompileTime;
    const int outerSize = rowMajor ? dst.rows() : dst.cols();
    for(int j = 0; j < outerSize; j++)
      ei_assign_novec_InnerUnrolling<Derived1, Derived2, 0, innerSize>
        ::run(dst, src, j);
  }
};

/**************************
*** Inner vectorization ***
**************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorization, NoUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int innerSize = rowMajor ? dst.cols() : dst.rows();
    const int outerSize = rowMajor ? dst.rows() : dst.cols();
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    for(int j = 0; j < outerSize; j++)
    {
      for(int i = 0; i < innerSize; i+=packetSize)
      {
        const int row = rowMajor ? j : i;
        const int col = rowMajor ? i : j;
        dst.template writePacket<Aligned>(row, col, src.template packet<Aligned>(row, col));
      }
    }
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorization, CompleteUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, Derived1::SizeAtCompileTime>
      ::run(dst, src);
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, InnerVectorization, InnerUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int innerSize = rowMajor ? Derived1::ColsAtCompileTime : Derived1::RowsAtCompileTime;
    const int outerSize = rowMajor ? dst.rows() : dst.cols();
    for(int j = 0; j < outerSize; j++)
      ei_assign_innervec_InnerUnrolling<Derived1, Derived2, 0, innerSize>
        ::run(dst, src, j);
  }
};

/***************************
*** Linear vectorization ***
***************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearVectorization, NoUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    const int size = dst.size();
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    const int alignedSize = (size/packetSize)*packetSize;
    const bool rowMajor = Derived1::Flags&RowMajorBit;
    const int innerSize = rowMajor ? dst.cols() : dst.rows();
    const int outerSize = rowMajor ? dst.rows() : dst.cols();
    int index = 0;

    // do the vectorizable part of the assignment
    for ( ; index<alignedSize ; index+=packetSize)
    {
      // FIXME the following is not really efficient
      const int row = rowMajor ? index/innerSize : index%innerSize;
      const int col = rowMajor ? index%innerSize : index/innerSize;
      dst.template writePacket<Aligned>(row, col, src.template packet<Aligned>(row, col));
    }

    // now we must do the rest without vectorization.
    if(alignedSize == size) return;
    const int k = alignedSize/innerSize;

    // do the remainder of the current row or col
    for(int i = alignedSize%innerSize; i < innerSize; i++)
    {
      const int row = rowMajor ? k : i;
      const int col = rowMajor ? i : k;
      dst.coeffRef(row, col) = src.coeff(row, col);
    }

    // do the remaining rows or cols
    for(int j = k+1; j < outerSize; j++)
      for(int i = 0; i < innerSize; i++)
      {
        const int row = rowMajor ? i : j;
        const int col = rowMajor ? j : i;
        dst.coeffRef(row, col) = src.coeff(row, col);
      }
  }
};

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, LinearVectorization, CompleteUnrolling>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    const int size = Derived1::SizeAtCompileTime;
    const int packetSize = ei_packet_traits<typename Derived1::Scalar>::size;
    const int alignedSize = (size/packetSize)*packetSize;
    const bool rowMajor = int(Derived1::Flags)&RowMajorBit;
    const int innerSize = rowMajor ? int(Derived1::ColsAtCompileTime) : int(Derived1::RowsAtCompileTime);
    const int outerSize = rowMajor ? int(Derived1::RowsAtCompileTime) : int(Derived1::ColsAtCompileTime);

    // do the vectorizable part of the assignment
    ei_assign_innervec_CompleteUnrolling<Derived1, Derived2, 0, alignedSize>::run(dst, src);

    // now we must do the rest without vectorization.
    const int k = alignedSize/innerSize;
    const int i = alignedSize%innerSize;

    // do the remainder of the current row or col
    ei_assign_novec_InnerUnrolling<Derived1, Derived2, i, k<outerSize ? innerSize : 0>::run(dst, src, k);

    // do the remaining rows or cols
    for(int j = k+1; j < outerSize; j++)
      ei_assign_novec_InnerUnrolling<Derived1, Derived2, 0, innerSize>::run(dst, src, j);
  }
};

/**************************
*** Slice vectorization ***
***************************/

template<typename Derived1, typename Derived2>
struct ei_assign_impl<Derived1, Derived2, SliceVectorization, NoUnrolling>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    //FIXME unimplemented, so for now we fall back to non-vectorized path
    ei_assign_impl<Derived1, Derived2, NoVectorization, NoUnrolling>::run(dst, src);
  }
};

/***************************************************************************
* Part 4 : implementation of MatrixBase methods
***************************************************************************/

template<typename Derived>
template<typename OtherDerived>
inline Derived& MatrixBase<Derived>
  ::lazyAssign(const MatrixBase<OtherDerived>& other)
{
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Derived,OtherDerived);
  ei_assert(rows() == other.rows() && cols() == other.cols());
  ei_assign_impl<Derived, OtherDerived>::run(derived(),other.derived());
  return derived();
}

template<typename Derived, typename OtherDerived,
         bool EvalBeforeAssigning = int(OtherDerived::Flags) & EvalBeforeAssigningBit,
         bool NeedToTranspose = Derived::IsVectorAtCompileTime
                && OtherDerived::IsVectorAtCompileTime
                && int(Derived::RowsAtCompileTime) == int(OtherDerived::ColsAtCompileTime)
                && int(Derived::ColsAtCompileTime) == int(OtherDerived::RowsAtCompileTime)
                && int(Derived::SizeAtCompileTime) != 1>
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

#endif // EIGEN_ASSIGN_H
