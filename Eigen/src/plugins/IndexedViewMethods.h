// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_PARSED_BY_DOXYGEN

// This file is automatically included twice to generate const and non-const versions

#ifndef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#define EIGEN_INDEXED_VIEW_METHOD_CONST const
#define EIGEN_INDEXED_VIEW_METHOD_TYPE  ConstIndexedViewType
#else
#define EIGEN_INDEXED_VIEW_METHOD_CONST
#define EIGEN_INDEXED_VIEW_METHOD_TYPE IndexedViewType
#endif

template<typename RowIndices, typename ColIndices>
struct EIGEN_INDEXED_VIEW_METHOD_TYPE {
  typedef IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,
                      typename internal::MakeIndexing<RowIndices>::type,
                      typename internal::MakeIndexing<ColIndices>::type> type;
};

// This is the generic version

template<typename RowIndices, typename ColIndices>
typename internal::enable_if<
  !  (internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::IsBlockAlike
  || (internal::is_integral<RowIndices>::value && internal::is_integral<ColIndices>::value)),
  typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type >::type
operator()(const RowIndices& rowIndices, const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  return typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type
            (derived(), internal::make_indexing(rowIndices,derived().rows()), internal::make_indexing(colIndices,derived().cols()));
}

// The folowing overload returns a Block<> object

template<typename RowIndices, typename ColIndices>
typename internal::enable_if<
      internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::IsBlockAlike
  && !(internal::is_integral<RowIndices>::value && internal::is_integral<ColIndices>::value),
  typename internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::BlockType>::type
operator()(const RowIndices& rowIndices, const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  typedef typename internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::BlockType BlockType;
  typename internal::MakeIndexing<RowIndices>::type actualRowIndices = internal::make_indexing(rowIndices,derived().rows());
  typename internal::MakeIndexing<ColIndices>::type actualColIndices = internal::make_indexing(colIndices,derived().cols());
  return BlockType(derived(),
                   internal::first(actualRowIndices),
                   internal::first(actualColIndices),
                   internal::size(actualRowIndices),
                   internal::size(actualColIndices));
}

// The folowing three overloads are needed to handle raw Index[N] arrays.

template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],typename internal::MakeIndexing<ColIndices>::type>
operator()(const RowIndicesT (&rowIndices)[RowIndicesN], const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],typename internal::MakeIndexing<ColIndices>::type>
                    (derived(), rowIndices, internal::make_indexing(colIndices,derived().cols()));
}

template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<RowIndices>::type, const ColIndicesT (&)[ColIndicesN]>
operator()(const RowIndices& rowIndices, const ColIndicesT (&colIndices)[ColIndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<RowIndices>::type,const ColIndicesT (&)[ColIndicesN]>
                    (derived(), internal::make_indexing(rowIndices,derived().rows()), colIndices);
}

template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndicesT, std::size_t ColIndicesN>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN], const ColIndicesT (&)[ColIndicesN]>
operator()(const RowIndicesT (&rowIndices)[RowIndicesN], const ColIndicesT (&colIndices)[ColIndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],const ColIndicesT (&)[ColIndicesN]>
                    (derived(), rowIndices, colIndices);
}

// Overloads for 1D vectors/arrays

template<typename Indices>
typename internal::enable_if<
  IsRowMajor && (!(internal::get_compile_time_incr<typename internal::MakeIndexing<Indices>::type>::value==1 || internal::is_integral<Indices>::value)),
  IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Index>::type,typename internal::MakeIndexing<Indices>::type> >::type
operator()(const Indices& indices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Index>::type,typename internal::MakeIndexing<Indices>::type>
            (derived(), internal::make_indexing(0,derived().rows()), internal::make_indexing(indices,derived().cols()));
}

template<typename Indices>
typename internal::enable_if<
  (!IsRowMajor) && (!(internal::get_compile_time_incr<typename internal::MakeIndexing<Indices>::type>::value==1 || internal::is_integral<Indices>::value)),
  IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Indices>::type,typename internal::MakeIndexing<Index>::type> >::type
operator()(const Indices& indices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Indices>::type,typename internal::MakeIndexing<Index>::type>
            (derived(), internal::make_indexing(indices,derived().rows()), internal::make_indexing(Index(0),derived().cols()));
}

template<typename Indices>
typename internal::enable_if<
  (internal::get_compile_time_incr<typename internal::MakeIndexing<Indices>::type>::value==1) && (!internal::is_integral<Indices>::value),
  VectorBlock<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,internal::get_compile_time_size<Indices,SizeAtCompileTime>::value> >::type
operator()(const Indices& indices) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  typename internal::MakeIndexing<Indices>::type actualIndices = internal::make_indexing(indices,derived().size());
  return VectorBlock<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,internal::get_compile_time_size<Indices,SizeAtCompileTime>::value>
            (derived(), internal::first(actualIndices), internal::size(actualIndices));
}

template<typename IndicesT, std::size_t IndicesN>
typename internal::enable_if<IsRowMajor,
  IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Index>::type,const IndicesT (&)[IndicesN]> >::type
operator()(const IndicesT (&indices)[IndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<Index>::type,const IndicesT (&)[IndicesN]>
            (derived(), internal::make_indexing(0,derived().rows()), indices);
}

template<typename IndicesT, std::size_t IndicesN>
typename internal::enable_if<!IsRowMajor,
  IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const IndicesT (&)[IndicesN],typename internal::MakeIndexing<Index>::type> >::type
operator()(const IndicesT (&indices)[IndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived)
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const IndicesT (&)[IndicesN],typename internal::MakeIndexing<Index>::type>
            (derived(), indices, internal::make_indexing(0,derived().rows()));
}

#undef EIGEN_INDEXED_VIEW_METHOD_CONST
#undef EIGEN_INDEXED_VIEW_METHOD_TYPE

#ifndef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#define EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#include "IndexedViewMethods.h"
#undef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#endif

#else // EIGEN_PARSED_BY_DOXYGEN

/**
  * \returns a generic submatrix view defined by the rows and columns indexed \a rowIndices and \a colIndices respectively.
  *
  * Each parameter must either be:
  *  - An integer indexing a single row or column
  *  - Eigen::all indexing the full set of respective rows or columns in increasing order
  *  - An ArithemeticSequence as returned by the seq and seqN functions
  *  - Any %Eigen's vector/array of integers or expressions
  *  - Plain C arrays: \c int[N]
  *  - And more generally any type exposing the following two member functions:
  * \code
  * <integral type> operator[](<integral type>) const;
  * <integral type> size() const;
  * \endcode
  * where \c <integral \c type>  stands for any integer type compatible with Eigen::Index (i.e. \c std::ptrdiff_t).
  *
  * The last statement implies compatibility with \c std::vector, \c std::valarray, \c std::array, many of the Range-v3's ranges, etc.
  *
  * If the submatrix can be represented using a starting position \c (i,j) and positive sizes \c (rows,columns), then this
  * method will returns a Block object after extraction of the relevant information from the passed arguments. This is the case
  * when all arguments are either:
  *  - An integer
  *  - Eigen::all
  *  - An ArithemeticSequence with compile-time increment strictly equal to 1, as returned by seq(a,b), and seqN(a,N).
  *
  * Otherwise a more general IndexedView<Derived,RowIndices',ColIndices'> object will be returned, after conversion of the inputs
  * to more suitable types \c RowIndices' and \c ColIndices'.
  *
  * For 1D vectors and arrays, you better use the operator()(const Indices&) overload, which behave the same way but taking a single parameter.
  *
  * \sa operator()(const Indices&), class Block, class IndexedView, DenseBase::block(Index,Index,Index,Index)
  */
template<typename RowIndices, typename ColIndices>
IndexedView_or_Block
operator()(const RowIndices& rowIndices, const ColIndices& colIndices);

/** This is an overload of operator()(const RowIndices&, const ColIndices&) for 1D vectors or arrays
  *
  * \only_for_vectors
  */
template<typename Indices>
IndexedView_or_VectorBlock
operator()(const Indices& indices);

#endif // EIGEN_PARSED_BY_DOXYGEN

