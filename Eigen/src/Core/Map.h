// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_MAP_H
#define EIGEN_MAP_H

/** \class Map
  *
  * \brief A matrix or vector expression mapping an existing array of data.
  *
  * \param Alignment can be either Aligned or Unaligned. Tells whether the array is suitably aligned for
  *                  vectorization on the present CPU architecture. Defaults to Unaligned.
  *
  * This class represents a matrix or vector expression mapping an existing array of data.
  * It can be used to let Eigen interface without any overhead with non-Eigen data structures,
  * such as plain C arrays or structures from other libraries.
  *
  * This class is the return type of Matrix::map() but can also be used directly.
  *
  * \sa Matrix::map()
  */
template<typename MatrixType, int Alignment>
struct ei_traits<Map<MatrixType, Alignment> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags
          & ( (HereditaryBits | LinearAccessBit | DirectAccessBit)
              | (Alignment == Aligned ? PacketAccessBit : 0) ),
    CoeffReadCost = NumTraits<Scalar>::ReadCost
  };
};

template<typename MatrixType, int Alignment> class Map
  : public MatrixBase<Map<MatrixType, Alignment> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Map)

    inline int rows() const { return m_rows.value(); }
    inline int cols() const { return m_cols.value(); }

    inline int stride() const { return this->innerSize(); }

    inline const Scalar& coeff(int row, int col) const
    {
      if(Flags & RowMajorBit)
        return m_data[col + row * m_cols.value()];
      else // column-major
        return m_data[row + col * m_rows.value()];
    }

    inline Scalar& coeffRef(int row, int col)
    {
      if(Flags & RowMajorBit)
        return const_cast<Scalar*>(m_data)[col + row * m_cols.value()];
      else // column-major
        return const_cast<Scalar*>(m_data)[row + col * m_rows.value()];
    }

    inline const Scalar& coeff(int index) const
    {
      return m_data[index];
    }

    inline Scalar& coeffRef(int index)
    {
      return *const_cast<Scalar*>(m_data + index);
    }

    template<int LoadMode>
    inline PacketScalar packet(int row, int col) const
    {
      return ei_ploadt<Scalar, LoadMode == Aligned ? Alignment : Unaligned>
               (m_data + (Flags & RowMajorBit
                         ? col + row * m_cols.value()
                         : row + col * m_rows.value()));
    }

    template<int LoadMode>
    inline PacketScalar packet(int index) const
    {
      return ei_ploadt<Scalar, LoadMode == Aligned ? Alignment : Unaligned>(m_data + index);
    }

    template<int StoreMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode == Aligned ? Alignment : Unaligned>
               (const_cast<Scalar*>(m_data) + (Flags & RowMajorBit
                         ? col + row * m_cols.value()
                         : row + col * m_rows.value()), x);
    }

    template<int StoreMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      ei_pstoret<Scalar, PacketScalar, StoreMode == Aligned ? Alignment : Unaligned>
        (const_cast<Scalar*>(m_data) + index, x);
    }

    inline Map(const Scalar* data) : m_data(data), m_rows(RowsAtCompileTime), m_cols(ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_FIXED_SIZE(MatrixType)
    }

    inline Map(const Scalar* data, int size)
            : m_data(data),
              m_rows(RowsAtCompileTime == Dynamic ? size : RowsAtCompileTime),
              m_cols(ColsAtCompileTime == Dynamic ? size : ColsAtCompileTime)
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(MatrixType)
      ei_assert(size > 0);
      ei_assert(SizeAtCompileTime == Dynamic || SizeAtCompileTime == size);
    }

    inline Map(const Scalar* data, int rows, int cols)
            : m_data(data), m_rows(rows), m_cols(cols)
    {
      ei_assert(rows > 0 && (RowsAtCompileTime == Dynamic || RowsAtCompileTime == rows)
               && cols > 0 && (ColsAtCompileTime == Dynamic || ColsAtCompileTime == cols));
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Map)

  protected:
    const Scalar* m_data;
    const ei_int_if_dynamic<RowsAtCompileTime> m_rows;
    const ei_int_if_dynamic<ColsAtCompileTime> m_cols;
};

/** Constructor copying an existing array of data.
  * Only for fixed-size matrices and vectors.
  * \param data The array of data to copy
  *
  * For dynamic-size matrices and vectors, see the variants taking additional int parameters
  * for the dimensions.
  *
  * \sa Matrix(const Scalar *, int), Matrix(const Scalar *, int, int),
  * Matrix::map(const Scalar *)
  */
template<typename _Scalar, int _Rows, int _Cols, int _MaxRows, int _MaxCols, unsigned int _Flags>
inline Matrix<_Scalar, _Rows, _Cols, _MaxRows, _MaxCols, _Flags>
  ::Matrix(const Scalar *data)
{
  *this = Map<Matrix>(data);
}

#endif // EIGEN_MAP_H
