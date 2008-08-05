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

#ifndef EIGEN_SWAP_H
#define EIGEN_SWAP_H

/** \class SwapWrapper
  *
  * \brief Expression which must be nested by value
  *
  * \param ExpressionType the type of the object of which we are requiring nesting-by-value
  *
  * This class is the return type of MatrixBase::nestByValue()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::nestByValue()
  */
template<typename ExpressionType>
struct ei_traits<SwapWrapper<ExpressionType> >
{
  typedef typename ExpressionType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ExpressionType::RowsAtCompileTime,
    ColsAtCompileTime = ExpressionType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ExpressionType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ExpressionType::MaxColsAtCompileTime,
    Flags = ExpressionType::Flags,
    CoeffReadCost = ExpressionType::CoeffReadCost
  };
};

template<typename ExpressionType> class SwapWrapper
  : public MatrixBase<SwapWrapper<ExpressionType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(SwapWrapper)
    typedef typename ei_packet_traits<Scalar>::type Packet;

    inline SwapWrapper(ExpressionType& matrix) : m_expression(matrix) {}

    inline int rows() const { return m_expression.rows(); }
    inline int cols() const { return m_expression.cols(); }
    inline int stride() const { return m_expression.stride(); }

    template<typename OtherDerived>
    void copyCoeff(int row, int col, const MatrixBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      Scalar tmp = m_expression.coeff(row, col);
      m_expression.coeffRef(row, col) = _other.coeff(row, col);
      _other.coeffRef(row, col) = tmp;
    }

    template<typename OtherDerived>
    void copyCoeff(int index, const MatrixBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_expression.size());
      Scalar tmp = m_expression.coeff(index);
      m_expression.coeffRef(index) = _other.coeff(index);
      _other.coeffRef(index) = tmp;
    }

    template<typename OtherDerived, int LoadStoreMode>
    void copyPacket(int row, int col, const MatrixBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                        && col >= 0 && col < cols());
      Packet tmp = m_expression.template packet<LoadStoreMode>(row, col);
      m_expression.template writePacket<LoadStoreMode>(row, col,
        _other.template packet<LoadStoreMode>(row, col)
      );
      _other.template writePacket<LoadStoreMode>(row, col, tmp);
    }

    template<typename OtherDerived, int LoadStoreMode>
    void copyPacket(int index, const MatrixBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_expression.size());
      Packet tmp = m_expression.template packet<LoadStoreMode>(index);
      m_expression.template writePacket<LoadStoreMode>(index,
        _other.template packet<LoadStoreMode>(index)
      );
      _other.template writePacket<LoadStoreMode>(index, tmp);
    }

  protected:
    ExpressionType m_expression;
};

/** swaps *this with the expression \a other.
  *
  * \note \a other is only marked const because I couln't find another way
  * to get g++ (4.2 and 4.3) to accept that template parameter resolution.
  * The problem seems to be that when swapping expressions as in
  * m.row(i).swap(m.row(j)); the Row object returned by row(j) is a temporary
  * and g++ doesn't dare to pass it by non-constant reference.
  * It gets const_cast'd of course. TODO: get rid of const here.
  */
template<typename Derived>
template<typename OtherDerived>
void MatrixBase<Derived>::swap(const MatrixBase<OtherDerived>& other)
{
  SwapWrapper<Derived>(derived()).lazyAssign(other);
}

#endif // EIGEN_SWAP_H
