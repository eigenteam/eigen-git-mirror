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

#endif // EIGEN_STRIDE_H
