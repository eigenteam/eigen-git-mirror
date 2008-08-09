// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_MAPBASE_H
#define EIGEN_MAPBASE_H

/** \internal
  *
  * \class MapBase
  *
  * \brief Base class for Map and Block expression with direct access
  *
  * \sa class Map, class Block
  */
template<typename Derived> class MapBase
  : public MatrixBase<Derived>
{
  public:

    typedef MatrixBase<Derived> Base;
    enum {
      IsRowMajor = int(ei_traits<Derived>::Flags) & RowMajorBit ? 1 : 0,
      PacketAccess = ei_traits<Derived>::PacketAccess,
      RowsAtCompileTime = ei_traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = ei_traits<Derived>::ColsAtCompileTime,
      SizeAtCompileTime = Base::SizeAtCompileTime
    };
    
    typedef typename ei_traits<Derived>::AlignedDerivedType AlignedDerivedType;
    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename Base::PacketScalar PacketScalar;
    using Base::derived;

    inline int rows() const { return m_rows.value(); }
    inline int cols() const { return m_cols.value(); }

    inline int stride() const { return derived().stride(); }
    AlignedDerivedType allowAligned() { return derived().allowAligned(); }

    inline const Scalar& coeff(int row, int col) const
    {
      if(IsRowMajor)
        return m_data[col + row * stride()];
      else // column-major
        return m_data[row + col * stride()];
    }

    inline Scalar& coeffRef(int row, int col)
    {
      if(IsRowMajor)
        return const_cast<Scalar*>(m_data)[col + row * stride()];
      else // column-major
        return const_cast<Scalar*>(m_data)[row + col * stride()];
    }
    
    inline const Scalar coeff(int index) const
    {
      ei_assert(Derived::IsVectorAtCompileTime || (ei_traits<Derived>::Flags & LinearAccessBit));
      if ( ((RowsAtCompileTime == 1) == IsRowMajor) )
        return m_data[index];
      else
        return m_data[index*stride()];
    }

    inline Scalar& coeffRef(int index)
    {
      return *const_cast<Scalar*>(m_data + index);
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return ei_ploadt<Scalar, int(PacketAccess) == Aligned ? Aligned : LoadMode>
               (m_data + (IsRowMajor ? col + row * stride()
                                     : row + col * stride()));
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return ei_ploadt<Scalar, int(PacketAccess) == Aligned ? Aligned : LoadMode>(m_data + index);
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, int(PacketAccess) == Aligned ? Aligned : StoreMode>
               (const_cast<Scalar*>(m_data) + (IsRowMajor ? col + row * stride()
                                                          : row + col * stride()), x);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, int(PacketAccess) == Aligned ? Aligned : StoreMode>
        (const_cast<Scalar*>(m_data) + index, x);
    }

    inline MapBase(const Scalar* data) : m_data(data), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
    }

    inline MapBase(const Scalar* data, int size)
            : m_data(data),
              m_rows(RowsAtCompileTime == Dynamic ? size : RowsAtCompileTime),
              m_cols(ColsAtCompileTime == Dynamic ? size : ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
      ei_assert(size > 0);
      ei_assert(SizeAtCompileTime == Dynamic || SizeAtCompileTime == size);
    }

    inline MapBase(const Scalar* data, int rows, int cols)
            : m_data(data), m_rows(rows), m_cols(cols)
    {
      ei_assert(rows > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
             && cols > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MapBase)

//     EIGEN_INHERIT_ASSIGNMENT_OPERATOR(MapBase, =)

    template<typename OtherDerived>
    Derived& operator+=(const MatrixBase<OtherDerived>& other)
    { return derived() = allowAligned() + other; }
    
    template<typename OtherDerived>
    Derived& operator-=(const MatrixBase<OtherDerived>& other)
    { return derived() = allowAligned() - other; }

    Derived& operator*=(const Scalar& other)
    { return derived() = allowAligned() * other; }
    
    Derived& operator/=(const Scalar& other)
    { return derived() = allowAligned() / other; }

  protected:
    const Scalar* __restrict__ m_data;
    const ei_int_if_dynamic<RowsAtCompileTime> m_rows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_cols;
};

#endif // EIGEN_MAPBASE_H
