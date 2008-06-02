// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
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

#ifndef EIGEN_CROSS_H
#define EIGEN_CROSS_H

/** \class Cross
  *
  * \brief Expression of the cross product of two vectors
  *
  * \param Lhs the type of the left-hand side
  * \param Rhs the type of the right-hand side
  *
  * This class represents an expression of the cross product of two 3D vectors.
  * It is the return type of MatrixBase::cross(), and most
  * of the time this is the only way it is used.
  */
template<typename Lhs, typename Rhs>
struct ei_traits<Cross<Lhs, Rhs> >
{
  typedef typename Lhs::Scalar Scalar;
  typedef typename ei_nested<Lhs,2>::type LhsNested;
  typedef typename ei_nested<Rhs,2>::type RhsNested;
  typedef typename ei_unref<LhsNested>::type _LhsNested;
  typedef typename ei_unref<RhsNested>::type _RhsNested;
  enum {
    RowsAtCompileTime = 3,
    ColsAtCompileTime = 1,
    MaxRowsAtCompileTime = 3,
    MaxColsAtCompileTime = 1,
    Flags = ((_RhsNested::Flags | _LhsNested::Flags) & HereditaryBits)
          | EvalBeforeAssigningBit,
    CoeffReadCost = NumTraits<Scalar>::AddCost + 2 * NumTraits<Scalar>::MulCost
  };
};

template<typename Lhs, typename Rhs> class Cross : ei_no_assignment_operator,
    public MatrixBase<Cross<Lhs, Rhs> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Cross)
    typedef typename ei_traits<Cross>::LhsNested LhsNested;
    typedef typename ei_traits<Cross>::RhsNested RhsNested;

    Cross(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      assert(lhs.isVector());
      assert(rhs.isVector());
      assert(lhs.size() == 3 && rhs.size() == 3);
    }

  private:

    int _rows() const { return 3; }
    int _cols() const { return 1; }

    Scalar _coeff(int i, int) const
    {
      return m_lhs[(i+1)%3]*m_rhs[(i+2)%3] - m_lhs[(i+2)%3]*m_rhs[(i+1)%3];
    }

  protected:
    const LhsNested m_lhs;
    const RhsNested m_rhs;
};

/** \returns an expression of the cross product of \c *this and \a other
  *
  * \sa class Cross
  */
template<typename Derived>
template<typename OtherDerived>
const Cross<Derived,OtherDerived>
MatrixBase<Derived>::cross(const MatrixBase<OtherDerived>& other) const
{
    return Cross<Derived,OtherDerived>(derived(),other.derived());
}

#endif // EIGEN_CROSS_H
