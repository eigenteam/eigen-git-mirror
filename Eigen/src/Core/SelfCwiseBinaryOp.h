// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SELFCWISEBINARYOP_H
#define EIGEN_SELFCWISEBINARYOP_H

/** \class SelfCwiseBinaryOp
  *
  * \internal
  *
  * \brief Internal helper class for optimizing operators like +=, -=
  *
  * This is a pseudo expression class re-implementing the copyCoeff/copyPacket
  * method to directly performs a +=/-= operations in an optimal way. In particular,
  * this allows to make sure that the input/output data are loaded only once using
  * aligned packet loads.
  *
  * \sa class SwapWrapper for a similar trick.
  */
template<typename BinaryOp, typename MatrixType>
struct ei_traits<SelfCwiseBinaryOp<BinaryOp,MatrixType> > : ei_traits<MatrixType>
{
  
};

template<typename BinaryOp, typename MatrixType> class SelfCwiseBinaryOp
  : public ei_dense_xpr_base< SelfCwiseBinaryOp<BinaryOp, MatrixType> >::type
{
  public:

    typedef typename ei_dense_xpr_base<SelfCwiseBinaryOp>::type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SelfCwiseBinaryOp)

    typedef typename ei_packet_traits<Scalar>::type Packet;

    using Base::operator=;

    inline SelfCwiseBinaryOp(MatrixType& xpr, const BinaryOp& func = BinaryOp()) : m_matrix(xpr), m_functor(func) {}

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }
    inline Index outerStride() const { return m_matrix.outerStride(); }
    inline Index innerStride() const { return m_matrix.innerStride(); }
    inline const Scalar* data() const { return m_matrix.data(); }

    // note that this function is needed by assign to correctly align loads/stores
    // TODO make Assign use .data()
    inline Scalar& coeffRef(Index row, Index col)
    {
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    // note that this function is needed by assign to correctly align loads/stores
    // TODO make Assign use .data()
    inline Scalar& coeffRef(Index index)
    {
      return m_matrix.const_cast_derived().coeffRef(index);
    }

    template<typename OtherDerived>
    void copyCoeff(Index row, Index col, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      Scalar& tmp = m_matrix.coeffRef(row,col);
      tmp = m_functor(tmp, _other.coeff(row,col));
    }

    template<typename OtherDerived>
    void copyCoeff(Index index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_matrix.size());
      Scalar& tmp = m_matrix.coeffRef(index);
      tmp = m_functor(tmp, _other.coeff(index));
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(Index row, Index col, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                        && col >= 0 && col < cols());
      m_matrix.template writePacket<StoreMode>(row, col,
        m_functor.packetOp(m_matrix.template packet<StoreMode>(row, col),_other.template packet<LoadMode>(row, col)) );
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(Index index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_matrix.size());
      m_matrix.template writePacket<StoreMode>(index,
        m_functor.packetOp(m_matrix.template packet<StoreMode>(index),_other.template packet<LoadMode>(index)) );
    }

    // reimplement lazyAssign to handle complex *= real
    // see CwiseBinaryOp ctor for details
    template<typename RhsDerived>
    EIGEN_STRONG_INLINE SelfCwiseBinaryOp& lazyAssign(const DenseBase<RhsDerived>& rhs)
    {
      EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(MatrixType,RhsDerived)

      EIGEN_STATIC_ASSERT((ei_functor_allows_mixing_real_and_complex<BinaryOp>::ret
                           ? int(ei_is_same_type<typename MatrixType::RealScalar, typename RhsDerived::RealScalar>::ret)
                           : int(ei_is_same_type<typename MatrixType::Scalar, typename RhsDerived::Scalar>::ret)),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
      
    #ifdef EIGEN_DEBUG_ASSIGN
      ei_assign_traits<SelfCwiseBinaryOp, RhsDerived>::debug();
    #endif
      ei_assert(rows() == rhs.rows() && cols() == rhs.cols());
      ei_assign_impl<SelfCwiseBinaryOp, RhsDerived>::run(*this,rhs.derived());
    #ifndef EIGEN_NO_DEBUG
      checkTransposeAliasing(rhs.derived());
    #endif
      return *this;
    }

  protected:
    MatrixType& m_matrix;
    const BinaryOp& m_functor;

  private:
    SelfCwiseBinaryOp& operator=(const SelfCwiseBinaryOp&);
};

template<typename Derived>
inline Derived& DenseBase<Derived>::operator*=(const Scalar& other)
{
  SelfCwiseBinaryOp<ei_scalar_product_op<Scalar>, Derived> tmp(derived());
  typedef typename Derived::PlainObject PlainObject;
  tmp = PlainObject::Constant(rows(),cols(),other);
  return derived();
}

template<typename Derived>
inline Derived& DenseBase<Derived>::operator/=(const Scalar& other)
{
  SelfCwiseBinaryOp<typename ei_meta_if<NumTraits<Scalar>::IsInteger,
                                        ei_scalar_quotient_op<Scalar>,
                                        ei_scalar_product_op<Scalar> >::ret, Derived> tmp(derived());
  typedef typename Derived::PlainObject PlainObject;
  tmp = PlainObject::Constant(rows(),cols(), NumTraits<Scalar>::IsInteger ? other : Scalar(1)/other);
  return derived();
}

#endif // EIGEN_SELFCWISEBINARYOP_H
