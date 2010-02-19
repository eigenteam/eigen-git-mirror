// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_STRIDE_H
#define EIGEN_STRIDE_H

template<int _InnerStrideAtCompileTime, int _OuterStrideAtCompileTime>
class Stride
{
  public:

    enum {
      InnerStrideAtCompileTime = _InnerStrideAtCompileTime,
      OuterStrideAtCompileTime = _OuterStrideAtCompileTime
    };

    Stride()
      : m_inner(InnerStrideAtCompileTime), m_outer(OuterStrideAtCompileTime)
    {
      ei_assert(InnerStrideAtCompileTime != Dynamic && OuterStrideAtCompileTime != Dynamic);
    }

    Stride(int innerStride, int outerStride)
      : m_inner(innerStride), m_outer(outerStride)
    {
      ei_assert(innerStride>=0 && outerStride>=0);
    }

    Stride(const Stride& other)
      : m_inner(other.inner()), m_outer(other.outer())
    {}

    inline int inner() const { return m_inner.value(); }
    inline int outer() const { return m_outer.value(); }

    template<int OtherInnerStrideAtCompileTime, int OtherOuterStrideAtCompileTime>
    Stride<EIGEN_ENUM_MAX(InnerStrideAtCompileTime, OtherInnerStrideAtCompileTime),
           EIGEN_ENUM_MAX(OuterStrideAtCompileTime, OtherOuterStrideAtCompileTime)>
    operator|(const Stride<OtherInnerStrideAtCompileTime, OtherOuterStrideAtCompileTime>& other)
    {
      EIGEN_STATIC_ASSERT(!((InnerStrideAtCompileTime && OtherInnerStrideAtCompileTime)
                         || (OuterStrideAtCompileTime && OtherOuterStrideAtCompileTime)),
                          YOU_ALREADY_SPECIFIED_THIS_STRIDE)
      int result_inner = InnerStrideAtCompileTime ? inner() : other.inner();
      int result_outer = OuterStrideAtCompileTime ? outer() : other.outer();
      return Stride<EIGEN_ENUM_MAX(InnerStrideAtCompileTime, OtherInnerStrideAtCompileTime),
                    EIGEN_ENUM_MAX(OuterStrideAtCompileTime, OtherOuterStrideAtCompileTime)>
                    (result_inner, result_outer);
    }
  protected:
    ei_int_if_dynamic<InnerStrideAtCompileTime> m_inner;
    ei_int_if_dynamic<OuterStrideAtCompileTime> m_outer;
};

template<int Value = Dynamic>
class InnerStride : public Stride<Value, 0>
{
    typedef Stride<Value,0> Base;
  public:
    InnerStride() : Base() {}
    InnerStride(int v) : Base(v,0) {}
};

template<int Value = Dynamic>
class OuterStride : public Stride<0, Value>
{
    typedef Stride<0,Value> Base;
  public:
    OuterStride() : Base() {}
    OuterStride(int v) : Base(0,v) {}
};

template<typename T, bool HasDirectAccess = int(ei_traits<T>::Flags)&DirectAccessBit>
struct ei_outer_stride_or_outer_size_impl
{
  static inline int value(const T& x) { return x.outerStride(); }
};

template<typename T>
struct ei_outer_stride_or_outer_size_impl<T, false>
{
  static inline int value(const T& x) { return x.outerSize(); }
};

template<typename T>
inline int ei_outer_stride_or_outer_size(const T& x)
{
  return ei_outer_stride_or_outer_size_impl<T>::value(x);
}

template<typename T, bool HasDirectAccess = int(ei_traits<typename ei_cleantype<T>::type>::Flags)&DirectAccessBit>
struct ei_inner_stride_at_compile_time
{
  enum { ret = ei_traits<typename ei_cleantype<T>::type>::InnerStrideAtCompileTime };
};

template<typename T>
struct ei_inner_stride_at_compile_time<T, false>
{
  enum { ret = 1 };
};

template<typename T, bool HasDirectAccess = int(ei_traits<typename ei_cleantype<T>::type>::Flags)&DirectAccessBit>
struct ei_outer_stride_at_compile_time
{
  enum { ret = ei_traits<typename ei_cleantype<T>::type>::OuterStrideAtCompileTime };
};

template<typename T>
struct ei_outer_stride_at_compile_time<T, false>
{
  enum { ret = 1 };
};

#endif // EIGEN_STRIDE_H
