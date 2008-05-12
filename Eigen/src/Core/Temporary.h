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

#ifndef EIGEN_TEMPORARY_H
#define EIGEN_TEMPORARY_H

/** \class Temporary
  *
  * \brief Expression with the temporary flag set
  *
  * \param ExpressionType the type of the object of which we are taking the temporary version
  *
  * This class represents the temporary version of an expression.
  * It is the return type of MatrixBase::temporary()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::temporary()
  */
template<typename ExpressionType>
struct ei_traits<Temporary<ExpressionType> >
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

template<typename ExpressionType> class Temporary
  : public MatrixBase<Temporary<ExpressionType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Temporary)

    inline Temporary(const ExpressionType& matrix) : m_expression(matrix) {}

  private:

    inline int _rows() const { return m_expression.rows(); }
    inline int _cols() const { return m_expression.cols(); }

    inline const Scalar _coeff(int row, int col) const
    {
      return m_expression.coeff(row, col);
    }

    template<int LoadMode>
    inline const PacketScalar _packetCoeff(int row, int col) const
    {
      return m_expression.template packetCoeff<LoadMode>(row, col);
    }

  protected:
    const ExpressionType m_expression;
};

/** \returns an expression of the temporary version of *this.
  */
template<typename Derived>
inline const Temporary<Derived>
MatrixBase<Derived>::temporary() const
{
  return Temporary<Derived>(derived());
}

#endif // EIGEN_TEMPORARY_H
