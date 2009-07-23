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

template <typename Lhs, int LhsMode, bool LhsIsVector,
          typename Rhs, int RhsMode, bool RhsIsVector>
struct ei_selfadjoint_product_returntype;

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

    /** Efficient self-adjoint matrix times vector/matrix product */
    template<typename OtherDerived>
    ei_selfadjoint_product_returntype<MatrixType,Mode,false,OtherDerived,0,OtherDerived::IsVectorAtCompileTime>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return ei_selfadjoint_product_returntype
              <MatrixType,Mode,false,OtherDerived,0,OtherDerived::IsVectorAtCompileTime>
              (m_matrix, rhs.derived());
    }

    /** Efficient vector/matrix times self-adjoint matrix product */
    template<typename OtherDerived> friend
    ei_selfadjoint_product_returntype<OtherDerived,0,OtherDerived::IsVectorAtCompileTime,MatrixType,Mode,false>
    operator*(const MatrixBase<OtherDerived>& lhs, const SelfAdjointView& rhs)
    {
      return ei_selfadjoint_product_returntype
              <OtherDerived,0,OtherDerived::IsVectorAtCompileTime,MatrixType,Mode,false>
              (lhs.derived(),rhs.m_matrix);
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


// template<typename OtherDerived, typename MatrixType, unsigned int UpLo>
// ei_selfadjoint_matrix_product_returntype<OtherDerived,SelfAdjointView<MatrixType,UpLo> >
// operator*(const MatrixBase<OtherDerived>& lhs, const SelfAdjointView<MatrixType,UpLo>& rhs)
// {
//   return ei_matrix_selfadjoint_product_returntype<OtherDerived,SelfAdjointView<MatrixType,UpLo> >(lhs.derived(),rhs);
// }

template<typename Derived1, typename Derived2, int UnrollCount, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, SelfAdjoint, UnrollCount, ClearOpposite>
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_triangular_assignment_selector<Derived1, Derived2, SelfAdjoint, UnrollCount-1, ClearOpposite>::run(dst, src);

    if(row == col)
      dst.coeffRef(row, col) = ei_real(src.coeff(row, col));
    else if(row < col)
      dst.coeffRef(col, row) = ei_conj(dst.coeffRef(row, col) = src.coeff(row, col));
  }
};

// selfadjoint to dense matrix
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, SelfAdjoint, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = 0; i < j; ++i)
        dst.coeffRef(j, i) = ei_conj(dst.coeffRef(i, j) = src.coeff(i, j));
      dst.coeffRef(j, j) = ei_real(src.coeff(j, j));
    }
  }
};

/***************************************************************************
* Wrapper to ei_product_selfadjoint_vector
***************************************************************************/

template<typename Lhs, int LhsMode, typename Rhs>
struct ei_selfadjoint_product_returntype<Lhs,LhsMode,false,Rhs,0,true>
  : public ReturnByValue<ei_selfadjoint_product_returntype<Lhs,LhsMode,false,Rhs,0,true>,
                         Matrix<typename ei_traits<Rhs>::Scalar,
                                Rhs::RowsAtCompileTime,Rhs::ColsAtCompileTime> >
{
  typedef typename Lhs::Scalar Scalar;

  typedef typename Lhs::Nested LhsNested;
  typedef typename ei_cleantype<LhsNested>::type _LhsNested;
  typedef ei_blas_traits<_LhsNested> LhsBlasTraits;
  typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
  typedef typename ei_cleantype<ActualLhsType>::type _ActualLhsType;

  typedef typename Rhs::Nested RhsNested;
  typedef typename ei_cleantype<RhsNested>::type _RhsNested;
  typedef ei_blas_traits<_RhsNested> RhsBlasTraits;
  typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
  typedef typename ei_cleantype<ActualRhsType>::type _ActualRhsType;

  enum {
    LhsUpLo = LhsMode&(UpperTriangularBit|LowerTriangularBit)
  };

  ei_selfadjoint_product_returntype(const Lhs& lhs, const Rhs& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  {}

  template<typename Dest> inline void _addTo(Dest& dst) const
  { evalTo(dst,1); }
  template<typename Dest> inline void _subTo(Dest& dst) const
  { evalTo(dst,-1); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dst.resize(m_lhs.rows(), m_rhs.cols());
    dst.setZero();
    evalTo(dst,1);
  }

  template<typename Dest> void evalTo(Dest& dst, Scalar alpha) const
  {
    const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
    const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    ei_assert((&dst.coeff(1))-(&dst.coeff(0))==1 && "not implemented yet");
    ei_product_selfadjoint_vector<Scalar, ei_traits<_ActualLhsType>::Flags&RowMajorBit, int(LhsUpLo), bool(LhsBlasTraits::NeedToConjugate), bool(RhsBlasTraits::NeedToConjugate)>
      (
        lhs.rows(),                                     // size
        &lhs.coeff(0,0), lhs.stride(),                  // lhs info
        &rhs.coeff(0), (&rhs.coeff(1))-(&rhs.coeff(0)), // rhs info
        &dst.coeffRef(0),                               // result info
        actualAlpha                                     // scale factor
      );
  }

  const LhsNested m_lhs;
  const RhsNested m_rhs;
};

/***************************************************************************
* Wrapper to ei_product_selfadjoint_matrix
***************************************************************************/

template<typename Lhs, int LhsMode, typename Rhs, int RhsMode>
struct ei_selfadjoint_product_returntype<Lhs,LhsMode,false,Rhs,RhsMode,false>
  : public ReturnByValue<ei_selfadjoint_product_returntype<Lhs,LhsMode,false,Rhs,RhsMode,false>,
                         Matrix<typename ei_traits<Rhs>::Scalar,
                                Lhs::RowsAtCompileTime,Rhs::ColsAtCompileTime> >
{
  ei_selfadjoint_product_returntype(const Lhs& lhs, const Rhs& rhs)
    : m_lhs(lhs), m_rhs(rhs)
  {}

  typedef typename Lhs::Scalar Scalar;

  typedef typename Lhs::Nested LhsNested;
  typedef typename ei_cleantype<LhsNested>::type _LhsNested;
  typedef ei_blas_traits<_LhsNested> LhsBlasTraits;
  typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
  typedef typename ei_cleantype<ActualLhsType>::type _ActualLhsType;

  typedef typename Rhs::Nested RhsNested;
  typedef typename ei_cleantype<RhsNested>::type _RhsNested;
  typedef ei_blas_traits<_RhsNested> RhsBlasTraits;
  typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
  typedef typename ei_cleantype<ActualRhsType>::type _ActualRhsType;

  enum {
    LhsUpLo = LhsMode&(UpperTriangularBit|LowerTriangularBit),
    LhsIsSelfAdjoint = (LhsMode&SelfAdjointBit)==SelfAdjointBit,
    RhsUpLo = RhsMode&(UpperTriangularBit|LowerTriangularBit),
    RhsIsSelfAdjoint = (RhsMode&SelfAdjointBit)==SelfAdjointBit
  };

  template<typename Dest> inline void _addTo(Dest& dst) const
  { evalTo(dst,1); }
  template<typename Dest> inline void _subTo(Dest& dst) const
  { evalTo(dst,-1); }

  template<typename Dest> void evalTo(Dest& dst) const
  {
    dst.resize(m_lhs.rows(), m_rhs.cols());
    dst.setZero();
    evalTo(dst,1);
  }

  template<typename Dest> void evalTo(Dest& dst, Scalar alpha) const
  {
    const ActualLhsType lhs = LhsBlasTraits::extract(m_lhs);
    const ActualRhsType rhs = RhsBlasTraits::extract(m_rhs);

    Scalar actualAlpha = alpha * LhsBlasTraits::extractScalarFactor(m_lhs)
                               * RhsBlasTraits::extractScalarFactor(m_rhs);

    ei_product_selfadjoint_matrix<Scalar,
      EIGEN_LOGICAL_XOR(LhsUpLo==UpperTriangular,
                        ei_traits<Lhs>::Flags &RowMajorBit) ? RowMajor : ColMajor, LhsIsSelfAdjoint,
      NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(LhsUpLo==UpperTriangular,bool(LhsBlasTraits::NeedToConjugate)),
      EIGEN_LOGICAL_XOR(RhsUpLo==UpperTriangular,
                        ei_traits<Rhs>::Flags &RowMajorBit) ? RowMajor : ColMajor, RhsIsSelfAdjoint,
      NumTraits<Scalar>::IsComplex && EIGEN_LOGICAL_XOR(RhsUpLo==UpperTriangular,bool(RhsBlasTraits::NeedToConjugate)),
      ei_traits<Dest>::Flags&RowMajorBit  ? RowMajor : ColMajor>
      ::run(
        lhs.rows(), rhs.cols(),           // sizes
        &lhs.coeff(0,0),    lhs.stride(), // lhs info
        &rhs.coeff(0,0),    rhs.stride(), // rhs info
        &dst.coeffRef(0,0), dst.stride(), // result info
        actualAlpha                       // alpha
      );
  }

  const LhsNested m_lhs;
  const RhsNested m_rhs;
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
