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

  static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_matrix_assignment_unroller<Derived1, Derived2, UnrollCount-1>::run(dst, src);
    dst.coeffRef(row, col) = src.coeff(row, col);
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, 1>
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, 0>
{
  static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_unroller<Derived1, Derived2, Dynamic>
{
  static void run(Derived1 &, const Derived2 &) {}
};

//----

template<typename Derived1, typename Derived2, int Index>
struct ei_matrix_assignment_packet_unroller
{
  enum {
    row = Derived1::Flags&RowMajorBit ? Index / Derived1::ColsAtCompileTime : Index % Derived1::RowsAtCompileTime,
    col = Derived1::Flags&RowMajorBit ? Index % Derived1::ColsAtCompileTime : Index / Derived1::RowsAtCompileTime
  };

  static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_matrix_assignment_packet_unroller<Derived1, Derived2,
      Index-ei_packet_traits<typename Derived1::Scalar>::size>::run(dst, src);
    dst.writePacketCoeff(row, col, src.packetCoeff(row, col));
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_packet_unroller<Derived1, Derived2, 0 >
{
  static void run(Derived1 &dst, const Derived2 &src)
  {
    dst.writePacketCoeff(0, 0, src.packetCoeff(0, 0));
  }
};

template<typename Derived1, typename Derived2>
struct ei_matrix_assignment_packet_unroller<Derived1, Derived2, Dynamic>
{
  static void run(Derived1 &, const Derived2 &) { ei_internal_assert(false && "ei_matrix_assignment_packet_unroller"); }
};

template <typename Derived, typename OtherDerived,
bool Vectorize = (Derived::Flags & OtherDerived::Flags & VectorizableBit)
              && ((Derived::Flags&RowMajorBit)==(OtherDerived::Flags&RowMajorBit))>
struct ei_assignment_impl;

template<typename Derived>
template<typename OtherDerived>
Derived& MatrixBase<Derived>
  ::lazyAssign(const MatrixBase<OtherDerived>& other)
{
  ei_assignment_impl<Derived,OtherDerived>::execute(derived(),other.derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
Derived& MatrixBase<Derived>
  ::operator=(const MatrixBase<OtherDerived>& other)
{
  const bool need_to_transpose = Derived::IsVectorAtCompileTime
                              && OtherDerived::IsVectorAtCompileTime
                              && (int)Derived::RowsAtCompileTime != (int)OtherDerived::RowsAtCompileTime;
  if(OtherDerived::Flags & EvalBeforeAssigningBit)
  {
    if(need_to_transpose)
      return lazyAssign(other.transpose().eval());
    else
      return lazyAssign(other.eval());
  }
  else
  {
    if(need_to_transpose)
      return lazyAssign(other.transpose());
    else
      return lazyAssign(other.derived());
  }
}

template<typename T1, typename T2> bool ei_should_parallelize_assignment(const T1& t, const T2&)
{
  return (T1::Flags & T2::Flags & LargeBit) && t.size() >= EIGEN_PARALLELIZATION_TRESHOLD;
}

template <typename Derived, typename OtherDerived>
struct ei_assignment_impl<Derived, OtherDerived, false>
{
  static void execute(Derived & dst, const OtherDerived & src)
  {
    const bool unroll = Derived::SizeAtCompileTime * OtherDerived::CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
    ei_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    if(unroll)
    {
      ei_matrix_assignment_unroller
        <Derived, OtherDerived,
        unroll ? Derived::SizeAtCompileTime : Dynamic
        >::run(dst.derived(), src.derived());
    }
    else
    {
      if(Derived::ColsAtCompileTime == Dynamic || Derived::RowsAtCompileTime != Dynamic)
      {
        #define EIGEN_THE_PARALLELIZABLE_LOOP \
            for(int j = 0; j < dst.cols(); j++) \
              for(int i = 0; i < dst.rows(); i++) \
                dst.coeffRef(i, j) = src.coeff(i, j);
        EIGEN_RUN_PARALLELIZABLE_LOOP(ei_should_parallelize_assignment(dst, src))
        #undef EIGEN_THE_PARALLELIZABLE_LOOP
      }
      else
      {
        // traverse in row-major order
        // in order to allow the compiler to unroll the inner loop
        #define EIGEN_THE_PARALLELIZABLE_LOOP \
          for(int i = 0; i < dst.rows(); i++) \
            for(int j = 0; j < dst.cols(); j++) \
              dst.coeffRef(i, j) = src.coeff(i, j);
        EIGEN_RUN_PARALLELIZABLE_LOOP(ei_should_parallelize_assignment(dst, src))
        #undef EIGEN_THE_PARALLELIZABLE_LOOP
      }
    }
  }
};

template <typename Derived, typename OtherDerived>
struct ei_assignment_impl<Derived, OtherDerived, true>
{
  static void execute(Derived & dst, const OtherDerived & src)
  {
    const bool unroll = Derived::SizeAtCompileTime * OtherDerived::CoeffReadCost <= EIGEN_UNROLLING_LIMIT;
    ei_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    if(unroll)
    {
      ei_matrix_assignment_packet_unroller
        <Derived, OtherDerived,
          unroll && int(Derived::SizeAtCompileTime)>=ei_packet_traits<typename Derived::Scalar>::size
            ? Derived::SizeAtCompileTime-ei_packet_traits<typename Derived::Scalar>::size
            : Dynamic>::run(dst.const_cast_derived(), src.derived());
    }
    else
    {
      if(OtherDerived::Flags&RowMajorBit)
      {
        #define EIGEN_THE_PARALLELIZABLE_LOOP \
        for(int i = 0; i < dst.rows(); i++) \
          for(int j = 0; j < dst.cols(); j+=ei_packet_traits<typename Derived::Scalar>::size) \
            dst.writePacketCoeff(i, j, src.packetCoeff(i, j));
        EIGEN_RUN_PARALLELIZABLE_LOOP(ei_should_parallelize_assignment(dst, src))
        #undef EIGEN_THE_PARALLELIZABLE_LOOP
      }
      else
      {
        #define EIGEN_THE_PARALLELIZABLE_LOOP \
        for(int j = 0; j < dst.cols(); j++) \
          for(int i = 0; i < dst.rows(); i+=ei_packet_traits<typename Derived::Scalar>::size) \
            dst.writePacketCoeff(i, j, src.packetCoeff(i, j));
        EIGEN_RUN_PARALLELIZABLE_LOOP(ei_should_parallelize_assignment(dst, src))
        #undef EIGEN_THE_PARALLELIZABLE_LOOP
      }
    }
  }
};

#endif // EIGEN_ASSIGN_H
