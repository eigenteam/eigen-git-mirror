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

#ifndef EIGEN_ARRAYBASE_H
#define EIGEN_ARRAYBASE_H

template<typename ExpressionType> class MatrixWrapper;

/** \ingroup Array_Module
  *
  * \class ArrayBase
  *
  * \brief Base class for all 1D and 2D array, and related expressions
  *
  * An array is similar to a dense vector or matrix. While matrices are mathematical
  * objects with well defined linear algebra operators, an array is just a collection
  * of scalar values arranged in a one or two dimensionnal fashion. The main consequence,
  * is that all operations applied to an array are performed coefficient wise. Furthermore,
  * arays support scalar math functions of the c++ standard library, and convenient
  * constructors allowing to easily write generic code working for both scalar values
  * and arrays.
  *
  * This class is the base that is inherited by all array expression types.
  *
  * \param Derived is the derived type, e.g. an array type, or an expression, etc.
  *
  * \sa class ArrayBase
  */
template<typename Derived> class ArrayBase
  : public DenseBase<Derived>
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** The base class for a given storage type. */
    typedef ArrayBase StorageBaseType;
    /** Construct the base class type for the derived class OtherDerived */
    template <typename OtherDerived> struct MakeBase { typedef ArrayBase<OtherDerived> Type; };

    using ei_special_scalar_op_base<Derived,typename ei_traits<Derived>::Scalar,
                typename NumTraits<typename ei_traits<Derived>::Scalar>::Real>::operator*;

    class InnerIterator;

    typedef typename ei_traits<Derived>::Scalar Scalar;
    typedef typename ei_packet_traits<Scalar>::type PacketScalar;

    typedef DenseBase<Derived> Base;
    using Base::RowsAtCompileTime;
    using Base::ColsAtCompileTime;
    using Base::SizeAtCompileTime;
    using Base::MaxRowsAtCompileTime;
    using Base::MaxColsAtCompileTime;
    using Base::MaxSizeAtCompileTime;
    using Base::IsVectorAtCompileTime;
    using Base::Flags;
    using Base::CoeffReadCost;
    using Base::_HasDirectAccess;
    
    using Base::rows;
    using Base::cols;
    using Base::size;
    using Base::coeff;
    using Base::coeffRef;
    using Base::operator=;
    
    typedef typename Base::RealScalar RealScalar;
    typedef typename Base::CoeffReturnType CoeffReturnType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** \internal the plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix! It is however guaranteed that the return type of eval() is either
      * PlainMatrixType or const PlainMatrixType&.
      */
    typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType;
    /** \internal the column-major plain matrix type corresponding to this expression. Note that is not necessarily
      * exactly the return type of eval(): in the case of plain matrices, the return type of eval() is a const
      * reference to a matrix, not a matrix!
      * The only difference from PlainMatrixType is that PlainMatrixType_ColMajor is guaranteed to be column-major.
      */
    typedef typename ei_plain_matrix_type<Derived>::type PlainMatrixType_ColMajor;


    /** \internal Represents a matrix with all coefficients equal to one another*/
    typedef CwiseNullaryOp<ei_scalar_constant_op<Scalar>,Derived> ConstantReturnType;
#endif // not EIGEN_PARSED_BY_DOXYGEN

#ifndef EIGEN_PARSED_BY_DOXYGEN
    using AnyMatrixBase<Derived>::derived;
    inline Derived& const_cast_derived() const
    { return *static_cast<Derived*>(const_cast<ArrayBase*>(this)); }
#endif // not EIGEN_PARSED_BY_DOXYGEN

#define EIGEN_CURRENT_STORAGE_BASE_CLASS Eigen::ArrayBase
#  include "../plugins/CommonCwiseUnaryOps.h"
#  include "../plugins/MatrixCwiseUnaryOps.h"
#  include "../plugins/ArrayCwiseUnaryOps.h"
#  include "../plugins/CommonCwiseBinaryOps.h"
#  include "../plugins/ArrayCwiseBinaryOps.h"
#undef EIGEN_CURRENT_STORAGE_BASE_CLASS


    /** Copies \a other into *this. \returns a reference to *this. */
//     template<typename OtherDerived>
//     Derived& operator=(const ArrayBase<OtherDerived>& other);

    /** Special case of the template operator=, in order to prevent the compiler
      * from generating a default operator= (issue hit with g++ 4.1)
      */
    Derived& operator=(const ArrayBase& other)
    {
      return ei_assign_selector<Derived,Derived>::run(derived(), other.derived());
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** Copies \a other into *this without evaluating other. \returns a reference to *this. */
//     template<typename OtherDerived>
//     Derived& lazyAssign(const ArrayBase<OtherDerived>& other);
#endif // not EIGEN_PARSED_BY_DOXYGEN

    Derived& operator+=(const Scalar& scalar)
    { return *this = derived() + scalar; }

    template<typename OtherDerived>
    Derived& operator+=(const ArrayBase<OtherDerived>& other);
    template<typename OtherDerived>
    Derived& operator-=(const ArrayBase<OtherDerived>& other);

    template<typename OtherDerived>
    Derived& operator*=(const ArrayBase<OtherDerived>& other);

    template<typename OtherDerived>
    inline bool operator==(const ArrayBase<OtherDerived>& other) const
    { return cwiseEqual(other).all(); }

    template<typename OtherDerived>
    inline bool operator!=(const ArrayBase<OtherDerived>& other) const
    { return cwiseNotEqual(other).all(); }


    /** \returns the matrix or vector obtained by evaluating this expression.
      *
      * Notice that in the case of a plain matrix or vector (not an expression) this function just returns
      * a const reference, in order to avoid a useless copy.
      */
//     EIGEN_STRONG_INLINE const typename ei_eval<Derived>::type eval() const
//     { return typename ei_eval<Derived>::type(derived()); }

//     template<typename OtherDerived>
//     void swap(ArrayBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other);


//     const VectorwiseOp<Derived,Horizontal> rowwise() const;
//     VectorwiseOp<Derived,Horizontal> rowwise();
//     const VectorwiseOp<Derived,Vertical> colwise() const;
//     VectorwiseOp<Derived,Vertical> colwise();

    #ifdef EIGEN_ARRAYBASE_PLUGIN
    #include EIGEN_ARRAYBASE_PLUGIN
    #endif

  public:
    MatrixWrapper<Derived> asMatrix() { return derived(); }
    const MatrixWrapper<Derived> asMatrix() const { return derived(); }

    template<typename Dest>
    inline void evalTo(Dest& dst) const { dst = asMatrix(); }

  protected:
    /** Default constructor. Do nothing. */
    ArrayBase()
    {
      /* Just checks for self-consistency of the flags.
       * Only do it when debugging Eigen, as this borders on paranoiac and could slow compilation down
       */
#ifdef EIGEN_INTERNAL_DEBUGGING
      EIGEN_STATIC_ASSERT(ei_are_flags_consistent<Flags>::ret,
                          INVALID_MATRIXBASE_TEMPLATE_PARAMETERS)
#endif
    }

  private:
    explicit ArrayBase(int);
    ArrayBase(int,int);
    template<typename OtherDerived> explicit ArrayBase(const ArrayBase<OtherDerived>&);
};

#endif // EIGEN_ARRAYBASE_H
