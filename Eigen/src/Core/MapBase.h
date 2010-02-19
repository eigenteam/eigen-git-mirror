// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2007-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
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

/** \class MapBase
  *
  * \brief Base class for Map and Block expression with direct access
  *
  * \sa class Map, class Block
  */
template<typename Derived, typename Base> class MapBase
  : public Base
{
  public:

    enum {
      IsRowMajor = (int(ei_traits<Derived>::Flags) & RowMajorBit) ? 1 : 0,
      RowsAtCompileTime = ei_traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = ei_traits<Derived>::ColsAtCompileTime,
      SizeAtCompileTime = Base::SizeAtCompileTime,
      InnerStrideAtCompileTime = ei_traits<Derived>::InnerStrideAtCompileTime
    };

    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename Base::PacketScalar PacketScalar;
    using Base::derived;

    inline int rows() const { return m_rows.value(); }
    inline int cols() const { return m_cols.value(); }

    /** \returns the pointer increment between two consecutive elements.
      *
      * \note For vectors, the storage order is ignored. For matrices (non-vectors), we're looking
      *       at the increment between two consecutive elements within a slice in the inner direction.
      *
      * \sa outerStride(), data(), rowStride(), colStride()
      */
    inline int innerStride() const { return derived().innerStride(); }

    /** \returns the pointer increment between two consecutive inner slices (for example, between two consecutive columns
      *          in a column-major matrix).
      *
      * \note For vectors, the storage order is ignored, there is only one inner slice, and so this method returns 1.
      *       For matrices (non-vectors), the notion of inner slice depends on the storage order.
      *
      * \sa innerStride(), data(), rowStride(), colStride()
      */
    inline int outerStride() const { return derived().outerStride(); }

    /** \returns the pointer increment between two consecutive rows.
      *
      * \sa data(), innerStride(), outerStride(), colStride()
      */
    inline int rowStride() const
    {
      return (RowsAtCompileTime==1 || IsRowMajor) ? outerStride() : innerStride();
    }

    /** \returns the pointer increment between two consecutive columns.
      *
      * \sa data(), innerStride(), outerStride(), rowStride()
      */
    inline int colStride() const
    {
      return (RowsAtCompileTime==1 || IsRowMajor) ? innerStride() : outerStride();
    }

    /** Returns a pointer to the first coefficient of the matrix or vector.
      *
      * \note When addressing this data, make sure to honor the strides returned by innerStride() and outerStride().
      *
      * \sa innerStride(), outerStride()
      */
    inline const Scalar* data() const { return m_data; }

    inline const Scalar& coeff(int row, int col) const
    {
      return m_data[col * colStride() + row * rowStride()];
    }

    inline Scalar& coeffRef(int row, int col)
    {
      return const_cast<Scalar*>(m_data)[col * colStride() + row * rowStride()];
    }

    inline const Scalar& coeff(int index) const
    {
      ei_assert(Derived::IsVectorAtCompileTime || (ei_traits<Derived>::Flags & LinearAccessBit));
      return m_data[index * innerStride()];
    }

    inline Scalar& coeffRef(int index)
    {
      ei_assert(Derived::IsVectorAtCompileTime || (ei_traits<Derived>::Flags & LinearAccessBit));
      return const_cast<Scalar*>(m_data)[index * innerStride()];
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return ei_ploadt<Scalar, LoadMode>
               (m_data + (col * colStride() + row * rowStride()));
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return ei_ploadt<Scalar, LoadMode>(m_data + index * innerStride());
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>
               (const_cast<Scalar*>(m_data) + (col * colStride() + row * rowStride()), x);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode>
        (const_cast<Scalar*>(m_data) + index * innerStride(), x);
    }

    inline MapBase(const Scalar* data) : m_data(data), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived)
      checkSanity();
    }

    inline MapBase(const Scalar* data, int size)
            : m_data(data),
              m_rows(RowsAtCompileTime == Dynamic ? size : RowsAtCompileTime),
              m_cols(ColsAtCompileTime == Dynamic ? size : ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
      ei_assert(size >= 0);
      ei_assert(data == 0 || SizeAtCompileTime == Dynamic || SizeAtCompileTime == size);
      checkSanity();
    }

    inline MapBase(const Scalar* data, int rows, int cols)
            : m_data(data), m_rows(rows), m_cols(cols)
    {
      ei_assert( (data == 0)
              || (   rows >= 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
                  && cols >= 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols)));
      checkSanity();
    }

    Derived& operator=(const MapBase& other)
    {
      return Base::operator=(other);
    }

    using Base::operator=;

  protected:

    void checkSanity() const
    {
      ei_assert( ((!(ei_traits<Derived>::Flags&AlignedBit))
                  || ((size_t(m_data)&0xf)==0)) && "data is not aligned");
      ei_assert( ((!(ei_traits<Derived>::Flags&PacketAccessBit))
                  || (innerStride()==1)) && "packet access incompatible with inner stride greater than 1");
    }

    const Scalar* EIGEN_RESTRICT m_data;
    const ei_int_if_dynamic<RowsAtCompileTime> m_rows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_cols;
};

#endif // EIGEN_MAPBASE_H
