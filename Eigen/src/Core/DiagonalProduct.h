// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_DIAGONALPRODUCT_H
#define EIGEN_DIAGONALPRODUCT_H

template<typename Lhs, typename Rhs>
struct ei_traits<Product<Lhs, Rhs, DiagonalProduct> >
{
  typedef typename Lhs::Scalar Scalar;
  typedef typename ei_nested<Lhs>::type LhsNested;
  typedef typename ei_nested<Rhs>::type RhsNested;
  typedef typename ei_unref<LhsNested>::type _LhsNested;
  typedef typename ei_unref<RhsNested>::type _RhsNested;
  enum {
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,
    RowsAtCompileTime = Lhs::RowsAtCompileTime,
    ColsAtCompileTime = Rhs::ColsAtCompileTime,
    MaxRowsAtCompileTime = Lhs::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Rhs::MaxColsAtCompileTime,
    _RhsVectorizable =  (RhsFlags & RowMajorBit) && (RhsFlags & VectorizableBit)
                     && (ColsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LhsVectorizable =  (!(LhsFlags & RowMajorBit)) && (LhsFlags & VectorizableBit)
                     && (RowsAtCompileTime % ei_packet_traits<Scalar>::size == 0),
    _LostBits = ~(((RhsFlags & RowMajorBit) && (!_LhsVectorizable) ? 0 : RowMajorBit)
                | ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic) ? 0 : LargeBit)),
    Flags = ((unsigned int)(LhsFlags | RhsFlags) & HereditaryBits & _LostBits)
          | (_LhsVectorizable || _RhsVectorizable ? VectorizableBit : 0),
    CoeffReadCost = NumTraits<Scalar>::MulCost + _LhsNested::CoeffReadCost + _RhsNested::CoeffReadCost
  };
};

template<typename Lhs, typename Rhs> class Product<Lhs, Rhs, DiagonalProduct> : ei_no_assignment_operator,
  public MatrixBase<Product<Lhs, Rhs, DiagonalProduct> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)
    typedef typename ei_traits<Product>::LhsNested LhsNested;
    typedef typename ei_traits<Product>::RhsNested RhsNested;
    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

    enum {
      PacketSize = ei_packet_traits<Scalar>::size
    };

    inline Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

  private:

    inline int _rows() const { return m_lhs.rows(); }
    inline int _cols() const { return m_rhs.cols(); }

    const Scalar _coeff(int row, int col) const
    {
      int unique = ((Rhs::Flags&Diagonal)==Diagonal) ? col : row;
      return m_lhs.coeff(row, unique) * m_rhs.coeff(unique, col);
    }

    template<int LoadMode>
    const PacketScalar _packetCoeff(int row, int col) const
    {
      if ((Rhs::Flags&Diagonal)==Diagonal)
      {
        ei_assert((_LhsNested::Flags&RowMajorBit)==0);
        return ei_pmul(m_lhs.template packetCoeff<LoadMode>(row, col), ei_pset1(m_rhs.coeff(col, col)));
      }
      else
      {
        ei_assert(_RhsNested::Flags&RowMajorBit);
        return ei_pmul(ei_pset1(m_lhs.coeff(row, row)), m_rhs.template packetCoeff<LoadMode>(row, col));
      }
    }

  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
};

#endif // EIGEN_DIAGONALPRODUCT_H
