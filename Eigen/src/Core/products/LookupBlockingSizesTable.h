// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Jacob <benoitjacob@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_LOOKUP_BLOCKING_SIZES_TABLE_H
#define EIGEN_LOOKUP_BLOCKING_SIZES_TABLE_H

namespace Eigen {

namespace internal {

template <typename LhsScalar,
          typename RhsScalar,
          bool HasLookupTable = BlockingSizesLookupTable<LhsScalar, RhsScalar>::NumSizes != 0 >
struct LookupBlockingSizesFromTableImpl
{
  static bool run(Index&, Index&, Index&, Index)
  {
    return false;
  }
};

inline size_t floor_log2_helper(unsigned short& x, size_t offset)
{
  unsigned short y = x >> offset;
  if (y) {
    x = y;
    return offset;
  } else {
    return 0;
  }
}

inline size_t floor_log2(unsigned short x)
{
  return floor_log2_helper(x, 8)
       + floor_log2_helper(x, 4)
       + floor_log2_helper(x, 2)
       + floor_log2_helper(x, 1);
}

inline size_t ceil_log2(unsigned short x)
{
  return x > 1 ? floor_log2(x - 1) + 1 : 0;
}

template <typename LhsScalar,
          typename RhsScalar>
struct LookupBlockingSizesFromTableImpl<LhsScalar, RhsScalar, true>
{
  static bool run(Index& k, Index& m, Index& n, Index)
  {
    using std::min;
    using std::max;
    typedef BlockingSizesLookupTable<LhsScalar, RhsScalar> Table;
    const unsigned short minsize = Table::BaseSize;
    const unsigned short maxsize = minsize << (Table::NumSizes - 1);
    const unsigned short k_clamped = max<unsigned short>(minsize, min<Index>(k, maxsize));
    const unsigned short m_clamped = max<unsigned short>(minsize, min<Index>(m, maxsize));
    const unsigned short n_clamped = max<unsigned short>(minsize, min<Index>(n, maxsize));
    const size_t k_index = ceil_log2(k_clamped / minsize);
    const size_t m_index = ceil_log2(m_clamped / minsize);
    const size_t n_index = ceil_log2(n_clamped / minsize);
    const size_t index = n_index + Table::NumSizes * (m_index + Table::NumSizes * k_index);
    const unsigned short table_entry = Table::Data()[index];
    k = min<Index>(k, 1 << ((table_entry & 0xf00) >> 8));
    m = min<Index>(m, 1 << ((table_entry & 0x0f0) >> 4));
    n = min<Index>(n, 1 << ((table_entry & 0x00f) >> 0));
    return true;
  }
};

template <typename LhsScalar,
          typename RhsScalar>
bool lookupBlockingSizesFromTable(Index& k, Index& m, Index& n, Index num_threads)
{
  if (num_threads > 1) {
    // We don't currently have lookup tables recorded for multithread performance,
    // and we have confirmed experimentally that our single-thread-recorded LUTs are
    // poor for multithread performance, and our LUTs don't currently contain
    // any annotation about multithread status (FIXME - we need that).
    // So for now, we just early-return here.
    return false;
  }
  return LookupBlockingSizesFromTableImpl<LhsScalar, RhsScalar>::run(k, m, n, num_threads);
}

}

}

#endif // EIGEN_LOOKUP_BLOCKING_SIZES_TABLE_H
