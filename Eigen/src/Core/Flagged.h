// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_FLAGGED_H
#define EIGEN_FLAGGED_H

namespace Eigen { 

/** \class Flagged
  * \ingroup Core_Module
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

namespace internal {
template<typename ExpressionType, unsigned int Added, unsigned int Removed>
struct traits<Flagged<ExpressionType, Added, Removed> > : traits<ExpressionType>
{
  enum { Flags = (ExpressionType::Flags | Added) & ~Removed };
};
}

template<typename ExpressionType, unsigned int Added, unsigned int Removed> class Flagged
  : public MatrixBase<Flagged<ExpressionType, Added, Removed> >
{
  public:

    typedef MatrixBase<Flagged> Base;
    
    EIGEN_DENSE_PUBLIC_INTERFACE(Flagged)
    typedef typename internal::conditional<internal::must_nest_by_value<ExpressionType>::ret,
        ExpressionType, const ExpressionType&>::type ExpressionTypeNested;
    typedef typename ExpressionType::InnerIterator InnerIterator;

    explicit inline Flagged(const ExpressionType& matrix) : m_matrix(matrix) {}

    EIGEN_DEVICE_FUNC inline Index rows() const { return m_matrix.rows(); }
    EIGEN_DEVICE_FUNC inline Index cols() const { return m_matrix.cols(); }
    EIGEN_DEVICE_FUNC inline Index outerStride() const { return m_matrix.outerStride(); }
    EIGEN_DEVICE_FUNC inline Index innerStride() const { return m_matrix.innerStride(); }

    EIGEN_DEVICE_FUNC inline CoeffReturnType coeff(Index row, Index col) const
    {
      return m_matrix.coeff(row, col);
    }

    EIGEN_DEVICE_FUNC inline CoeffReturnType coeff(Index index) const
    {
      return m_matrix.coeff(index);
    }
    
    EIGEN_DEVICE_FUNC inline const Scalar& coeffRef(Index row, Index col) const
    {
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    EIGEN_DEVICE_FUNC inline const Scalar& coeffRef(Index index) const
    {
      return m_matrix.const_cast_derived().coeffRef(index);
    }

    EIGEN_DEVICE_FUNC inline Scalar& coeffRef(Index row, Index col)
    {
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    EIGEN_DEVICE_FUNC inline Scalar& coeffRef(Index index)
    {
      return m_matrix.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index row, Index col) const
    {
      return m_matrix.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(Index row, Index col, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(Index index) const
    {
      return m_matrix.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(Index index, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

    EIGEN_DEVICE_FUNC const ExpressionType& _expression() const { return m_matrix; }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC typename ExpressionType::PlainObject solveTriangular(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC void solveTriangularInPlace(const MatrixBase<OtherDerived>& other) const;

  protected:
    ExpressionTypeNested m_matrix;
};

/** \returns an expression of *this with added and removed flags
  *
  * This is mostly for internal use.
  *
  * \sa class Flagged
  */
template<typename Derived>
template<unsigned int Added,unsigned int Removed>
inline const Flagged<Derived, Added, Removed>
DenseBase<Derived>::flagged() const
{
  return Flagged<Derived, Added, Removed>(derived());
}

} // end namespace Eigen

#endif // EIGEN_FLAGGED_H
