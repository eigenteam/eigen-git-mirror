// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// This file is automatically included twice to generate const and non-const versions

#ifndef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#define EIGEN_INDEXED_VIEW_METHOD_CONST const
#define EIGEN_INDEXED_VIEW_METHOD_TYPE ConstIndexedViewType
#else
#define EIGEN_INDEXED_VIEW_METHOD_CONST
#define EIGEN_INDEXED_VIEW_METHOD_TYPE IndexedViewType
#endif

template<typename RowIndices, typename ColIndices>
struct EIGEN_INDEXED_VIEW_METHOD_TYPE {
  typedef IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<RowIndices>::type,typename internal::MakeIndexing<ColIndices>::type> type;
};

template<typename RowIndices, typename ColIndices>
typename internal::enable_if<
  !  (internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::IsBlockAlike
  || (internal::is_integral<RowIndices>::value && internal::is_integral<ColIndices>::value)),
  typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type >::type
operator()(const RowIndices& rowIndices, const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST {
  return typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type(
            derived(), internal::make_indexing(rowIndices,derived().rows()), internal::make_indexing(colIndices,derived().cols()));
}

template<typename RowIndices, typename ColIndices>
typename internal::enable_if<
      internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::IsBlockAlike
  && !(internal::is_integral<RowIndices>::value && internal::is_integral<ColIndices>::value),
  typename internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::BlockType>::type
operator()(const RowIndices& rowIndices, const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST {
  typedef typename internal::traits<typename EIGEN_INDEXED_VIEW_METHOD_TYPE<RowIndices,ColIndices>::type>::BlockType BlockType;
  typename internal::MakeIndexing<RowIndices>::type actualRowIndices = internal::make_indexing(rowIndices,derived().rows());
  typename internal::MakeIndexing<ColIndices>::type actualColIndices = internal::make_indexing(colIndices,derived().cols());
  return BlockType(derived(),
                    internal::first(actualRowIndices),
                    internal::first(actualColIndices),
                    internal::size(actualRowIndices),
                    internal::size(actualColIndices));
}

template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndices>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],typename internal::MakeIndexing<ColIndices>::type>
operator()(const RowIndicesT (&rowIndices)[RowIndicesN], const ColIndices& colIndices) EIGEN_INDEXED_VIEW_METHOD_CONST {
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],typename internal::MakeIndexing<ColIndices>::type>(
            derived(), rowIndices, internal::make_indexing(colIndices,derived().cols()));
}

template<typename RowIndices, typename ColIndicesT, std::size_t ColIndicesN>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<RowIndices>::type, const ColIndicesT (&)[ColIndicesN]>
operator()(const RowIndices& rowIndices, const ColIndicesT (&colIndices)[ColIndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST {
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,typename internal::MakeIndexing<RowIndices>::type,const ColIndicesT (&)[ColIndicesN]>(
            derived(), internal::make_indexing(rowIndices,derived().rows()), colIndices);
}

template<typename RowIndicesT, std::size_t RowIndicesN, typename ColIndicesT, std::size_t ColIndicesN>
IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN], const ColIndicesT (&)[ColIndicesN]>
operator()(const RowIndicesT (&rowIndices)[RowIndicesN], const ColIndicesT (&colIndices)[ColIndicesN]) EIGEN_INDEXED_VIEW_METHOD_CONST {
  return IndexedView<EIGEN_INDEXED_VIEW_METHOD_CONST Derived,const RowIndicesT (&)[RowIndicesN],const ColIndicesT (&)[ColIndicesN]>(
            derived(), rowIndices, colIndices);
}

#undef EIGEN_INDEXED_VIEW_METHOD_CONST
#undef EIGEN_INDEXED_VIEW_METHOD_TYPE

#ifndef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#define EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#include "IndexedViewMethods.h"
#undef EIGEN_INDEXED_VIEW_METHOD_2ND_PASS
#endif

