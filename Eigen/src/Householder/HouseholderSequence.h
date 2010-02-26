// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2010 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_HOUSEHOLDER_SEQUENCE_H
#define EIGEN_HOUSEHOLDER_SEQUENCE_H

/** \ingroup Householder_Module
  * \householder_module
  * \class HouseholderSequence
  * \brief Represents a sequence of householder reflections with decreasing size
  *
  * This class represents a product sequence of householder reflections \f$ H = \Pi_0^{n-1} H_i \f$
  * where \f$ H_i \f$ is the i-th householder transformation \f$ I - h_i v_i v_i^* \f$,
  * \f$ v_i \f$ is the i-th householder vector \f$ [ 1, m_vectors(i+1,i), m_vectors(i+2,i), ...] \f$
  * and \f$ h_i \f$ is the i-th householder coefficient \c m_coeffs[i].
  *
  * Typical usages are listed below, where H is a HouseholderSequence:
  * \code
  * A.applyOnTheRight(H);             // A = A * H
  * A.applyOnTheLeft(H);              // A = H * A
  * A.applyOnTheRight(H.adjoint());   // A = A * H^*
  * A.applyOnTheLeft(H.adjoint());    // A = H^* * A
  * MatrixXd Q = H;                   // conversion to a dense matrix
  * \endcode
  * In addition to the adjoint, you can also apply the inverse (=adjoint), the transpose, and the conjugate.
  *
  * \sa MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */

template<typename VectorsType, typename CoeffsType, int Side>
struct ei_traits<HouseholderSequence<VectorsType,CoeffsType,Side> >
{
  typedef typename VectorsType::Scalar Scalar;
  enum {
    RowsAtCompileTime = Side==OnTheLeft ? ei_traits<VectorsType>::RowsAtCompileTime
                                        : ei_traits<VectorsType>::ColsAtCompileTime,
    ColsAtCompileTime = RowsAtCompileTime,
    MaxRowsAtCompileTime = Side==OnTheLeft ? ei_traits<VectorsType>::MaxRowsAtCompileTime
                                           : ei_traits<VectorsType>::MaxColsAtCompileTime,
    MaxColsAtCompileTime = MaxRowsAtCompileTime,
    Flags = 0
  };
};

template<typename VectorsType, typename CoeffsType, int Side>
struct ei_hseq_side_dependent_impl
{
  typedef Block<VectorsType, Dynamic, 1> EssentialVectorType;
  typedef HouseholderSequence<VectorsType, CoeffsType, OnTheLeft> HouseholderSequenceType;
  static inline const EssentialVectorType essentialVector(const HouseholderSequenceType& h, int k)
  {
    const int start = k+1+h.m_shift;
    return Block<VectorsType,Dynamic,1>(h.m_vectors, start, k, h.rows()-start, 1);
  }
};

template<typename VectorsType, typename CoeffsType>
struct ei_hseq_side_dependent_impl<VectorsType, CoeffsType, OnTheRight>
{
  typedef Transpose<Block<VectorsType, 1, Dynamic> > EssentialVectorType;
  typedef HouseholderSequence<VectorsType, CoeffsType, OnTheRight> HouseholderSequenceType;
  static inline const EssentialVectorType essentialVector(const HouseholderSequenceType& h, int k)
  {
    const int start = k+1+h.m_shift;
    return Block<VectorsType,1,Dynamic>(h.m_vectors, k, start, 1, h.rows()-start).transpose();
  }
};

template<typename OtherScalarType, typename MatrixType> struct ei_matrix_type_times_scalar_type
{
  typedef typename ei_scalar_product_traits<OtherScalarType, typename MatrixType::Scalar>::ReturnType
    ResultScalar;
  typedef Matrix<ResultScalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime,
                 0, MatrixType::MaxRowsAtCompileTime, MatrixType::MaxColsAtCompileTime> Type;
};

template<typename VectorsType, typename CoeffsType, int Side> class HouseholderSequence
  : public EigenBase<HouseholderSequence<VectorsType,CoeffsType,Side> >
{
    enum {
      RowsAtCompileTime = ei_traits<HouseholderSequence>::RowsAtCompileTime,
      ColsAtCompileTime = ei_traits<HouseholderSequence>::ColsAtCompileTime,
      MaxRowsAtCompileTime = ei_traits<HouseholderSequence>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = ei_traits<HouseholderSequence>::MaxColsAtCompileTime
    };
    typedef typename ei_traits<HouseholderSequence>::Scalar Scalar;

    typedef typename ei_hseq_side_dependent_impl<VectorsType,CoeffsType,Side>::EssentialVectorType
            EssentialVectorType;

  public:

    typedef HouseholderSequence<
      VectorsType,
      typename ei_meta_if<NumTraits<Scalar>::IsComplex,
        typename ei_cleantype<typename CoeffsType::ConjugateReturnType>::type,
        CoeffsType>::ret,
      Side
    > ConjugateReturnType;

    HouseholderSequence(const VectorsType& v, const CoeffsType& h, bool trans = false)
      : m_vectors(v), m_coeffs(h), m_trans(trans), m_actualVectors(v.diagonalSize()),
        m_shift(0)
    {
    }

    HouseholderSequence(const VectorsType& v, const CoeffsType& h, bool trans, int actualVectors, int shift)
      : m_vectors(v), m_coeffs(h), m_trans(trans), m_actualVectors(actualVectors), m_shift(shift)
    {
    }

    int rows() const { return Side==OnTheLeft ? m_vectors.rows() : m_vectors.cols(); }
    int cols() const { return rows(); }

    const EssentialVectorType essentialVector(int k) const
    {
      ei_assert(k >= 0 && k < m_actualVectors);
      return ei_hseq_side_dependent_impl<VectorsType,CoeffsType,Side>::essentialVector(*this, k);
    }

    HouseholderSequence transpose() const
    { return HouseholderSequence(m_vectors, m_coeffs, !m_trans, m_actualVectors, m_shift); }

    ConjugateReturnType conjugate() const
    { return ConjugateReturnType(m_vectors, m_coeffs.conjugate(), m_trans, m_actualVectors, m_shift); }

    ConjugateReturnType adjoint() const
    { return ConjugateReturnType(m_vectors, m_coeffs.conjugate(), !m_trans, m_actualVectors, m_shift); }

    ConjugateReturnType inverse() const { return adjoint(); }

    /** \internal */
    template<typename DestType> void evalTo(DestType& dst) const
    {
      int vecs = m_actualVectors;
      dst.setIdentity(rows(), rows());
      Matrix<Scalar,1,DestType::RowsAtCompileTime> temp(rows());
      for(int k = vecs-1; k >= 0; --k)
      {
        int cornerSize = rows() - k - m_shift;
        if(m_trans)
          dst.corner(BottomRight, cornerSize, cornerSize)
          .applyHouseholderOnTheRight(essentialVector(k), m_coeffs.coeff(k), &temp.coeffRef(0));
        else
          dst.corner(BottomRight, cornerSize, cornerSize)
            .applyHouseholderOnTheLeft(essentialVector(k), m_coeffs.coeff(k), &temp.coeffRef(0));
      }
    }

    /** \internal */
    template<typename Dest> inline void applyThisOnTheRight(Dest& dst) const
    {
      Matrix<Scalar,1,Dest::RowsAtCompileTime> temp(dst.rows());
      for(int k = 0; k < m_actualVectors; ++k)
      {
        int actual_k = m_trans ? m_actualVectors-k-1 : k;
        dst.corner(BottomRight, dst.rows(), rows()-m_shift-actual_k)
           .applyHouseholderOnTheRight(essentialVector(actual_k), m_coeffs.coeff(actual_k), &temp.coeffRef(0));
      }
    }

    /** \internal */
    template<typename Dest> inline void applyThisOnTheLeft(Dest& dst) const
    {
      Matrix<Scalar,1,Dest::ColsAtCompileTime> temp(dst.cols());
      for(int k = 0; k < m_actualVectors; ++k)
      {
        int actual_k = m_trans ? k : m_actualVectors-k-1;
        dst.corner(BottomRight, rows()-m_shift-actual_k, dst.cols())
           .applyHouseholderOnTheLeft(essentialVector(actual_k), m_coeffs.coeff(actual_k), &temp.coeffRef(0));
      }
    }

    template<typename OtherDerived>
    typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::Type operator*(const MatrixBase<OtherDerived>& other) const
    {
      typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::Type
        res(other.template cast<typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::ResultScalar>());
      applyThisOnTheLeft(res);
      return res;
    }

    template<typename OtherDerived> friend
    typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::Type operator*(const MatrixBase<OtherDerived>& other, const HouseholderSequence& h)
    {
      typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::Type
        res(other.template cast<typename ei_matrix_type_times_scalar_type<Scalar, OtherDerived>::ResultScalar>());
      h.applyThisOnTheRight(res);
      return res;
    }

    template<typename _VectorsType, typename _CoeffsType, int _Side> friend struct ei_hseq_side_dependent_impl;

  protected:
    typename VectorsType::Nested m_vectors;
    typename CoeffsType::Nested m_coeffs;
    bool m_trans;
    int m_actualVectors;
    int m_shift;
};

template<typename VectorsType, typename CoeffsType>
HouseholderSequence<VectorsType,CoeffsType> householderSequence(const VectorsType& v, const CoeffsType& h, bool trans=false)
{
  return HouseholderSequence<VectorsType,CoeffsType,OnTheLeft>(v, h, trans);
}

template<typename VectorsType, typename CoeffsType>
HouseholderSequence<VectorsType,CoeffsType> householderSequence(const VectorsType& v, const CoeffsType& h, bool trans, int actualVectors, int shift)
{
  return HouseholderSequence<VectorsType,CoeffsType,OnTheLeft>(v, h, trans, actualVectors, shift);
}

template<typename VectorsType, typename CoeffsType>
HouseholderSequence<VectorsType,CoeffsType> rightHouseholderSequence(const VectorsType& v, const CoeffsType& h, bool trans=false)
{
  return HouseholderSequence<VectorsType,CoeffsType,OnTheRight>(v, h, trans);
}

template<typename VectorsType, typename CoeffsType>
HouseholderSequence<VectorsType,CoeffsType> rightHouseholderSequence(const VectorsType& v, const CoeffsType& h, bool trans, int actualVectors, int shift)
{
  return HouseholderSequence<VectorsType,CoeffsType,OnTheRight>(v, h, trans, actualVectors, shift);
}

#endif // EIGEN_HOUSEHOLDER_SEQUENCE_H
