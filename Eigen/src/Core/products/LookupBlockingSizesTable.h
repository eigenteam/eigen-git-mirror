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

inline uint8_t floor_log2_helper(uint16_t& x, size_t offset)
{
  uint16_t y = x >> offset;
  if (y) {
    x = y;
    return offset;
  } else {
    return 0;
  }
}

inline uint8_t floor_log2(uint16_t x)
{
  return floor_log2_helper(x, 8)
       + floor_log2_helper(x, 4)
       + floor_log2_helper(x, 2)
       + floor_log2_helper(x, 1);
}

inline uint8_t ceil_log2(uint16_t x)
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
    const uint16_t minsize = Table::BaseSize;
    const uint16_t maxsize = minsize << (Table::NumSizes + 1);
    const uint16_t k_clamped = max<uint16_t>(minsize, min<Index>(k, maxsize));
    const uint16_t m_clamped = max<uint16_t>(minsize, min<Index>(m, maxsize));
    const uint16_t n_clamped = max<uint16_t>(minsize, min<Index>(n, maxsize));
    const size_t k_index = ceil_log2(k_clamped / minsize);
    const size_t m_index = ceil_log2(m_clamped / minsize);
    const size_t n_index = ceil_log2(n_clamped / minsize);
    const size_t index = n_index + Table::NumSizes * (m_index + Table::NumSizes * k_index);
    const uint16_t table_entry = Table::Data()[index];
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
  return LookupBlockingSizesFromTableImpl<LhsScalar, RhsScalar>::run(k, m, n, num_threads);
}

}

}

#endif // EIGEN_LOOKUP_BLOCKING_SIZES_TABLE_H
