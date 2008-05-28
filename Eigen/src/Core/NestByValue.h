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

#ifndef EIGEN_NESTBYVALUE_H
#define EIGEN_NESTBYVALUE_H

/** \class NestByValue
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
struct ei_traits<NestByValue<ExpressionType> >
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

template<typename ExpressionType> class NestByValue
  : public MatrixBase<NestByValue<ExpressionType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(NestByValue)

    inline NestByValue(const ExpressionType& matrix) : m_expression(matrix) {}

  private:

    inline int _rows() const { return m_expression.rows(); }
    inline int _cols() const { return m_expression.cols(); }
    inline int _stride() const { return m_expression.stride(); }

    inline const Scalar _coeff(int row, int col) const
    {
      return m_expression.coeff(row, col);
    }

    inline Scalar& _coeffRef(int row, int col)
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    template<int LoadMode>
    inline const PacketScalar _packetCoeff(int row, int col) const
    {
      return m_expression.template packetCoeff<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void _writePacketCoeff(int row, int col, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacketCoeff<LoadMode>(row, col, x);
    }

  protected:
    const ExpressionType m_expression;
};

/** \returns an expression of the temporary version of *this.
  */
template<typename Derived>
inline const NestByValue<Derived>
MatrixBase<Derived>::nestByValue() const
{
  return NestByValue<Derived>(derived());
}

#endif // EIGEN_NESTBYVALUE_H
