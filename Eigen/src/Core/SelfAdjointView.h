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

#ifndef EIGEN_SELFADJOINTMATRIX_H
#define EIGEN_SELFADJOINTMATRIX_H

/** \class SelfAdjointView
  * \nonstableyet
  *
  * \brief Expression of a selfadjoint matrix from a triangular part of a dense matrix
  *
  * \param MatrixType the type of the dense matrix storing the coefficients
  * \param TriangularPart can be either \c LowerTriangular or \c UpperTriangular
  *
  * This class is an expression of a sefladjoint matrix from a triangular part of a matrix
  * with given dense storage of the coefficients. It is the return type of MatrixBase::selfadjointView()
  * and most of the time this is the only way that it is used.
  *
  * \sa class TriangularBase, MatrixBase::selfAdjointView()
  */
template<typename MatrixType, unsigned int TriangularPart>
struct ei_traits<SelfAdjointView<MatrixType, TriangularPart> > : ei_traits<MatrixType>
{
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  typedef MatrixType ExpressionType;
  enum {
    Mode = TriangularPart | SelfAdjointBit,
    Flags =  _MatrixTypeNested::Flags & (HereditaryBits)
           & (~(PacketAccessBit | DirectAccessBit | LinearAccessBit)), // FIXME these flags should be preserved
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename Lhs,typename Rhs>
struct ei_selfadjoint_vector_product_returntype;

// FIXME could also be called SelfAdjointWrapper to be consistent with DiagonalWrapper ??
template<typename MatrixType, unsigned int UpLo> class SelfAdjointView
  : public TriangularBase<SelfAdjointView<MatrixType, UpLo> >
{
  public:

    typedef TriangularBase<SelfAdjointView> Base;
    typedef typename ei_traits<SelfAdjointView>::Scalar Scalar;
    enum {
      Mode = ei_traits<SelfAdjointView>::Mode
    };
    typedef typename MatrixType::PlainMatrixType PlainMatrixType;

    inline SelfAdjointView(const MatrixType& matrix) : m_matrix(matrix)
    { ei_assert(ei_are_flags_consistent<Mode>::ret); }

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }
    inline int stride() const { return m_matrix.stride(); }

    /** \sa MatrixBase::coeff()
      * \warning the coordinates must fit into the referenced triangular part
      */
    inline Scalar coeff(int row, int col) const
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.coeff(row, col);
    }

    /** \sa MatrixBase::coeffRef()
      * \warning the coordinates must fit into the referenced triangular part
      */
    inline Scalar& coeffRef(int row, int col)
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    /** \internal */
    const MatrixType& _expression() const { return m_matrix; }

    /** Efficient self-adjoint matrix times vector product */
    // TODO this product is far to be ready
    template<typename OtherDerived>
    ei_selfadjoint_vector_product_returntype<SelfAdjointView,OtherDerived>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return ei_selfadjoint_vector_product_returntype<SelfAdjointView,OtherDerived>(*this, rhs.derived());
    }

    /** Perform a symmetric rank 2 update of the selfadjoint matrix \c *this:
      * \f$ this = this + \alpha ( u v^* + v u^*) \f$
      * 
      * The vectors \a u and \c v \b must be column vectors, however they can be
      * a adjoint expression without any overhead. Only the meaningful triangular
      * part of the matrix is updated, the rest is left unchanged.
      */
    template<typename DerivedU, typename DerivedV>
    void rank2update(const MatrixBase<DerivedU>& u, const MatrixBase<DerivedV>& v, Scalar alpha = Scalar(1));

/////////// Cholesky module ///////////

    const LLT<PlainMatrixType, UpLo> llt() const;
    const LDLT<PlainMatrixType> ldlt() const;

  protected:

    const typename MatrixType::Nested m_matrix;
};

/***************************************************************************
* Wrapper to ei_product_selfadjoint_vector
***************************************************************************/

template<typename Lhs,typename Rhs>
struct ei_selfadjoint_vector_product_returntype
  : public ReturnByValue<ei_selfadjoint_vector_product_returntype<Lhs,Rhs>,
                         Matrix<typename ei_traits<Rhs>::Scalar,
                                Rhs::RowsAtCompileTime,Rhs::ColsAtCompileTime> >
{
  typedef typename ei_cleantype<typename Rhs::Nested>::type RhsNested;
  ei_selfadjoint_vector_product_returntype(const Lhs& lhs, const Rhs& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  {}

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dst.resize(m_rhs.rows(), m_rhs.cols());
    ei_product_selfadjoint_vector<typename Lhs::Scalar,ei_traits<Lhs>::Flags&RowMajorBit,
      Lhs::Mode&(UpperTriangularBit|LowerTriangularBit)>
      (
        m_lhs.rows(), // size
        m_lhs._expression().data(), // lhs
        m_lhs.stride(), // lhsStride,
        m_rhs.data(), // rhs
        // int rhsIncr,
        dst.data() // res
      );
  }

  const Lhs m_lhs;
  const typename Rhs::Nested m_rhs;
};

/***************************************************************************
* Implementation of MatrixBase methods
***************************************************************************/

template<typename Derived>
template<unsigned int Mode>
const SelfAdjointView<Derived, Mode> MatrixBase<Derived>::selfadjointView() const
{
  return derived();
}

template<typename Derived>
template<unsigned int Mode>
SelfAdjointView<Derived, Mode> MatrixBase<Derived>::selfadjointView()
{
  return derived();
}

#endif // EIGEN_SELFADJOINTMATRIX_H
