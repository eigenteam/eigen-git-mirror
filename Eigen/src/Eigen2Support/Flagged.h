// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

/** \deprecated it is only used by lazy() which is deprecated
  *
  * \class Flagged
  *
  * \brief Expression with modified flags
  *
  * \param ExpressionType the type of the object of which we are modifying the flags
  * \param Added the flags added to the expression
  * \param Removed the flags removed from the expression (has priority over Added).
  *
  * This class represents an expression whose flags have been modified.
  * It is the return type of MatrixBase::flagged()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::flagged()
  */
template<typename ExpressionType, unsigned int Added, unsigned int Removed>
struct ei_traits<Flagged<ExpressionType, Added, Removed> > : ei_traits<ExpressionType>
{
  enum { Flags = (ExpressionType::Flags | Added) & ~Removed };
};

template<typename ExpressionType, unsigned int Added, unsigned int Removed> class Flagged
  : public MatrixBase<Flagged<ExpressionType, Added, Removed> >
{
  public:

    typedef MatrixBase<Flagged> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(Flagged)
    typedef typename ei_meta_if<ei_must_nest_by_value<ExpressionType>::ret,
        ExpressionType, const ExpressionType&>::ret ExpressionTypeNested;
    typedef typename ExpressionType::InnerIterator InnerIterator;

    inline Flagged(const ExpressionType& matrix) : m_matrix(matrix) {}

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }
    inline int stride() const { return m_matrix.stride(); }

    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row, col);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    inline const Scalar coeff(int index) const
    {
      return m_matrix.coeff(index);
    }

    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int row, int col) const
    {
      return m_matrix.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int index) const
    {
      return m_matrix.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

    const ExpressionType& _expression() const { return m_matrix; }

    template<typename OtherDerived>
    typename ExpressionType::PlainMatrixType solveTriangular(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    void solveTriangularInPlace(const MatrixBase<OtherDerived>& other) const;

  protected:
    ExpressionTypeNested m_matrix;
};

/** \deprecated it is only used by lazy() which is deprecated
  *
  * \returns an expression of *this with added flags
  *
  * Example: \include MatrixBase_marked.cpp
  * Output: \verbinclude MatrixBase_marked.out
  *
  * \sa class Flagged, extract(), part()
  */
template<typename Derived>
template<unsigned int Added>
inline const Flagged<Derived, Added, 0>
MatrixBase<Derived>::marked() const
{
  return derived();
}

/** \deprecated use MatrixBase::noalias()
  *
  * \returns an expression of *this with the EvalBeforeAssigningBit flag removed.
  *
  * Example: \include MatrixBase_lazy.cpp
  * Output: \verbinclude MatrixBase_lazy.out
  *
  * \sa class Flagged, marked()
  */
template<typename Derived>
inline const Flagged<Derived, 0, EvalBeforeAssigningBit>
MatrixBase<Derived>::lazy() const
{
  return derived();
}


/** \internal
  * Overloaded to perform an efficient C += (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator+=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeAssigningBit>& other)
{
  other._expression().derived().addTo(derived()); return derived();
}

/** \internal
  * Overloaded to perform an efficient C -= (A*B).lazy() */
template<typename Derived>
template<typename ProductDerived, typename Lhs, typename Rhs>
Derived& MatrixBase<Derived>::operator-=(const Flagged<ProductBase<ProductDerived, Lhs,Rhs>, 0,
                                                       EvalBeforeAssigningBit>& other)
{
  other._expression().derived().subTo(derived()); return derived();
}

#endif // EIGEN_FLAGGED_H
