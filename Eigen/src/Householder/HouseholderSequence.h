// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

template<typename VectorsType, typename CoeffsType>
struct ei_traits<HouseholderSequence<VectorsType,CoeffsType> >
{
  typedef typename VectorsType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ei_traits<VectorsType>::RowsAtCompileTime,
    ColsAtCompileTime = ei_traits<VectorsType>::RowsAtCompileTime,
    MaxRowsAtCompileTime = ei_traits<VectorsType>::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ei_traits<VectorsType>::MaxRowsAtCompileTime,
    Flags = 0
  };
};

template<typename VectorsType, typename CoeffsType> class HouseholderSequence
  : public AnyMatrixBase<HouseholderSequence<VectorsType,CoeffsType> >
{
    typedef typename VectorsType::Scalar Scalar;
  public:

    typedef HouseholderSequence<VectorsType,
      typename ei_meta_if<NumTraits<Scalar>::IsComplex,
        NestByValue<typename ei_cleantype<typename CoeffsType::ConjugateReturnType>::type >,
        CoeffsType>::ret> ConjugateReturnType;

    HouseholderSequence(const VectorsType& v, const CoeffsType& h, bool trans = false)
      : m_vectors(v), m_coeffs(h), m_trans(trans)
    {}

    int rows() const { return m_vectors.rows(); }
    int cols() const { return m_vectors.rows(); }

    HouseholderSequence transpose() const
    { return HouseholderSequence(m_vectors, m_coeffs, !m_trans); }

    ConjugateReturnType conjugate() const
    { return ConjugateReturnType(m_vectors, m_coeffs.conjugate(), m_trans); }

    ConjugateReturnType adjoint() const
    { return ConjugateReturnType(m_vectors, m_coeffs.conjugate(), !m_trans); }

    ConjugateReturnType inverse() const { return adjoint(); }

    /** \internal */
    template<typename DestType> void evalTo(DestType& dst) const
    {
      int vecs = std::min(m_vectors.cols(),m_vectors.rows());
      int length = m_vectors.rows();
      dst.setIdentity();
      Matrix<Scalar,1,DestType::RowsAtCompileTime> temp(dst.rows());
      for(int k = vecs-1; k >= 0; --k)
      {
        if(m_trans)
          dst.corner(BottomRight, length-k, length-k)
          .applyHouseholderOnTheRight(m_vectors.col(k).end(length-k-1), m_coeffs.coeff(k), &temp.coeffRef(0));
        else
          dst.corner(BottomRight, length-k, length-k)
            .applyHouseholderOnTheLeft(m_vectors.col(k).end(length-k-1), m_coeffs.coeff(k), &temp.coeffRef(k));
      }
    }

    /** \internal */
    template<typename Dest> inline void applyThisOnTheRight(Dest& dst) const
    {
      int vecs = std::min(m_vectors.cols(),m_vectors.rows()); // number of householder vectors
      int length = m_vectors.rows();                          // size of the largest householder vector
      Matrix<Scalar,1,Dest::ColsAtCompileTime> temp(dst.rows());
      for(int k = 0; k < vecs; ++k)
      {
        int actual_k = m_trans ? vecs-k-1 : k;
        dst.corner(BottomRight, dst.rows(), length-k)
           .applyHouseholderOnTheRight(m_vectors.col(k).end(length-k-1), m_coeffs.coeff(k), &temp.coeffRef(0));
      }
    }

    /** \internal */
    template<typename Dest> inline void applyThisOnTheLeft(Dest& dst) const
    {
      int vecs = std::min(m_vectors.cols(),m_vectors.rows()); // number of householder vectors
      int length = m_vectors.rows();                          // size of the largest householder vector
      Matrix<Scalar,1,Dest::ColsAtCompileTime> temp(dst.cols());
      for(int k = 0; k < vecs; ++k)
      {
        int actual_k = m_trans ? k : vecs-k-1;
        dst.corner(BottomRight, length-actual_k, dst.cols())
           .applyHouseholderOnTheLeft(m_vectors.col(actual_k).end(length-actual_k-1), m_coeffs.coeff(actual_k), &temp.coeffRef(0));
      }
    }

    template<typename OtherDerived>
    typename OtherDerived::PlainMatrixType operator*(const MatrixBase<OtherDerived>& other) const
    {
      typename OtherDerived::PlainMatrixType res(other);
      applyThisOnTheLeft(res);
      return res;
    }

    template<typename OtherDerived> friend
    typename OtherDerived::PlainMatrixType operator*(const MatrixBase<OtherDerived>& other, const HouseholderSequence& h)
    {
      typename OtherDerived::PlainMatrixType res(other);
      h.applyThisOnTheRight(res);
      return res;
    }

  protected:
    typename VectorsType::Nested m_vectors;
    typename CoeffsType::Nested m_coeffs;
    bool m_trans;

private:
  HouseholderSequence& operator=(const HouseholderSequence&);
};

template<typename VectorsType, typename CoeffsType>
HouseholderSequence<VectorsType,CoeffsType> householderSequence(const VectorsType& v, const CoeffsType& h, bool trans=false)
{
  return HouseholderSequence<VectorsType,CoeffsType>(v, h, trans);
}

#endif // EIGEN_HOUSEHOLDER_SEQUENCE_H
