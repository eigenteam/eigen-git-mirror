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
struct ei_traits<SelfCwiseBinaryOp<BinaryOp,MatrixType> > : ei_traits<MatrixType> {};

template<typename BinaryOp, typename MatrixType> class SelfCwiseBinaryOp
  : public MatrixType::template MakeBase< SelfCwiseBinaryOp<BinaryOp, MatrixType> >::Type
{
  public:

    typedef typename MatrixType::template MakeBase< SelfCwiseBinaryOp<BinaryOp, MatrixType> >::Type Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(SelfCwiseBinaryOp)

    typedef typename ei_packet_traits<Scalar>::type Packet;

    using Base::operator=;

    inline SelfCwiseBinaryOp(MatrixType& xpr, const BinaryOp& func = BinaryOp()) : m_matrix(xpr), m_functor(func) {}

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }
    inline int stride() const { return m_matrix.stride(); }
    inline const Scalar* data() const { return m_matrix.data(); }

    // note that this function is needed by assign to correctly align loads/stores
    // TODO make Assign use .data()
    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    // note that this function is needed by assign to correctly align loads/stores
    // TODO make Assign use .data()
    inline Scalar& coeffRef(int index)
    {
      return m_matrix.const_cast_derived().coeffRef(index);
    }

    template<typename OtherDerived>
    void copyCoeff(int row, int col, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                         && col >= 0 && col < cols());
      Scalar& tmp = m_matrix.coeffRef(row,col);
      tmp = m_functor(tmp, _other.coeff(row,col));
    }

    template<typename OtherDerived>
    void copyCoeff(int index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_matrix.size());
      Scalar& tmp = m_matrix.coeffRef(index);
      tmp = m_functor(tmp, _other.coeff(index));
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int row, int col, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(row >= 0 && row < rows()
                        && col >= 0 && col < cols());
      m_matrix.template writePacket<StoreMode>(row, col,
        m_functor.packetOp(m_matrix.template packet<StoreMode>(row, col),_other.template packet<LoadMode>(row, col)) );
    }

    template<typename OtherDerived, int StoreMode, int LoadMode>
    void copyPacket(int index, const DenseBase<OtherDerived>& other)
    {
      OtherDerived& _other = other.const_cast_derived();
      ei_internal_assert(index >= 0 && index < m_matrix.size());
      m_matrix.template writePacket<StoreMode>(index,
        m_functor.packetOp(m_matrix.template packet<StoreMode>(index),_other.template packet<LoadMode>(index)) );
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
  typedef typename Derived::PlainMatrixType PlainMatrixType;
  tmp = PlainMatrixType::Constant(rows(),cols(),other);
  return derived();
}

template<typename Derived>
inline Derived& DenseBase<Derived>::operator/=(const Scalar& other)
{
  SelfCwiseBinaryOp<typename ei_meta_if<NumTraits<Scalar>::HasFloatingPoint,ei_scalar_product_op<Scalar>,ei_scalar_quotient_op<Scalar> >::ret, Derived> tmp(derived());
  typedef typename Derived::PlainMatrixType PlainMatrixType;
  tmp = PlainMatrixType::Constant(rows(),cols(), NumTraits<Scalar>::HasFloatingPoint ? Scalar(1)/other : other);
  return derived();
}

#endif // EIGEN_SELFCWISEBINARYOP_H
