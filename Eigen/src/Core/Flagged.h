// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_FLAGGED_H
#define EIGEN_FLAGGED_H

/** \class Flagged
  *
  * \brief Expression with modified flags
  *
  * \param ExpressionType the type of the object of which we are modifying the flags
  * \param Added the flags added to the expression
  * \param Removed the flags removed from the expression (has priority over Added).
  *
  * This class represents an expression whose flags have been modified
  * It is the return type of MatrixBase::flagged()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::flagged()
  */
template<typename ExpressionType, unsigned int Added, unsigned int Removed>
struct ei_traits<Flagged<ExpressionType, Added, Removed> >
{
  typedef typename ExpressionType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ExpressionType::RowsAtCompileTime,
    ColsAtCompileTime = ExpressionType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ExpressionType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ExpressionType::MaxColsAtCompileTime,
    Flags = (ExpressionType::Flags | Added) & ~Removed,
    CoeffReadCost = ExpressionType::CoeffReadCost
  };
};

template<typename ExpressionType, unsigned int Added, unsigned int Removed> class Flagged
  : public MatrixBase<Flagged<ExpressionType, Added, Removed> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Flagged)

    inline Flagged(const ExpressionType& matrix) : m_expression(matrix) {}

    /** \internal */
    inline ExpressionType _expression() const { return m_expression; }

  private:

    inline int _rows() const { return m_expression.rows(); }
    inline int _cols() const { return m_expression.cols(); }

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
template<unsigned int Added, unsigned int Removed>
inline const Flagged<Derived, Added, Removed>
MatrixBase<Derived>::flagged() const
{
  return Flagged<Derived, Added, Removed>(derived());
}

#endif // EIGEN_FLAGGED_H
