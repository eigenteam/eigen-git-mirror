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

#ifndef EIGEN_DENSEDIRECTACCESSBASE_H
#define EIGEN_DENSEDIRECTACCESSBASE_H

template<typename Derived> struct ei_has_direct_access
{
  enum { ret = (ei_traits<Derived>::Flags & DirectAccessBit) ? 1 : 0 };
};

template<typename Derived, bool _HasDirectAccess = ei_has_direct_access<Derived>::ret>
struct ei_inner_stride_at_compile_time
{
  enum { ret = ei_traits<Derived>::InnerStrideAtCompileTime };
};

template<typename Derived>
struct ei_inner_stride_at_compile_time<Derived, false>
{
  enum { ret = 0 };
};

template<typename Derived, bool _HasDirectAccess = ei_has_direct_access<Derived>::ret>
struct ei_outer_stride_at_compile_time
{
  enum { ret = ei_traits<Derived>::OuterStrideAtCompileTime };
};

template<typename Derived>
struct ei_outer_stride_at_compile_time<Derived, false>
{
  enum { ret = 0 };
};

template<typename Derived, typename XprKind = typename ei_traits<Derived>::XprKind>
struct ei_dense_xpr_base
{
  /* ei_dense_xpr_base should only ever be used on dense expressions, thus falling either into the MatrixXpr or into the ArrayXpr cases */
};

template<typename Derived>
struct ei_dense_xpr_base<Derived, MatrixXpr>
{
  typedef MatrixBase<Derived> type;
};

template<typename Derived>
struct ei_dense_xpr_base<Derived, ArrayXpr>
{
  typedef ArrayBase<Derived> type;
};

template<typename Derived> class DenseDirectAccessBase
  : public ei_dense_xpr_base<Derived>::type
{
  public:

    typedef typename ei_dense_xpr_base<Derived>::type Base;

    using Base::IsVectorAtCompileTime;
    using Base::IsRowMajor;
    using Base::derived;
    using Base::operator=;

    typedef typename Base::CoeffReturnType CoeffReturnType;

    enum {
      InnerStrideAtCompileTime = ei_traits<Derived>::InnerStrideAtCompileTime,
      OuterStrideAtCompileTime = ei_traits<Derived>::OuterStrideAtCompileTime
    };

    /** \returns the pointer increment between two consecutive elements within a slice in the inner direction.
      *
      * \sa outerStride(), rowStride(), colStride()
      */
    inline int innerStride() const
    {
      return derived().innerStride();
    }

    /** \returns the pointer increment between two consecutive inner slices (for example, between two consecutive columns
      *          in a column-major matrix).
      *
      * \sa innerStride(), rowStride(), colStride()
      */
    inline int outerStride() const
    {
      return derived().outerStride();
    }

    inline int stride() const
    {
      return IsVectorAtCompileTime ? innerStride() : outerStride();
    }

    /** \returns the pointer increment between two consecutive rows.
      *
      * \sa innerStride(), outerStride(), colStride()
      */
    inline int rowStride() const
    {
      return IsRowMajor ? outerStride() : innerStride();
    }

    /** \returns the pointer increment between two consecutive columns.
      *
      * \sa innerStride(), outerStride(), rowStride()
      */
    inline int colStride() const
    {
      return IsRowMajor ? innerStride() : outerStride();
    }

};

#endif // EIGEN_DENSEDIRECTACCESSBASE_H
