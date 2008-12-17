// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_RANDOMSETTER_H
#define EIGEN_RANDOMSETTER_H

template<typename Scalar> struct StdMapTraits
{
  typedef int KeyType;
  typedef std::map<KeyType,Scalar> Type;
  enum {
    IsSorted = 1
  };

  static void setInvalidKey(Type&, const KeyType&) {}
};

#ifdef _HASH_MAP
template<typename Scalar> struct GnuHashMapTraits
{
  typedef int KeyType;
  typedef __gnu_cxx::hash_map<KeyType,Scalar> Type;
  enum {
    IsSorted = 0
  };

  static void setInvalidKey(Type&, const KeyType&) {}
};
#endif

#ifdef _DENSE_HASH_MAP_H_
template<typename Scalar> struct GoogleDenseHashMapTraits
{
  typedef int KeyType;
  typedef google::dense_hash_map<KeyType,Scalar> Type;
  enum {
    IsSorted = 0
  };

  static void setInvalidKey(Type& map, const KeyType& k)
  { map.set_empty_key(k); }
};
#endif

#ifdef _SPARSE_HASH_MAP_H_
template<typename Scalar> struct GoogleSparseHashMapTraits
{
  typedef int KeyType;
  typedef google::sparse_hash_map<KeyType,Scalar> Type;
  enum {
    IsSorted = 0
  };

  static void setInvalidKey(Type&, const KeyType&) {}
};
#endif

/** \class RandomSetter
  *
  * Typical usage:
  * \code
  * SparseMatrix<double> m(rows,cols);
  * {
  *   RandomSetter<SparseMatrix<double> > w(m);
  *   // don't use m but w instead with read/write random access to the coefficients:
  *   for(;;)
  *     w(rand(),rand()) = rand;
  * }
  * // when w is deleted, the data are copied back to m
  * // and m is ready to use.
  * \endcode
  *
  * \note for performance and memory consumption reasons it is highly recommended to use
  * Google's hash library. To do so you have two options:
  *  - include <google/dense_hash_map> yourself \b before Eigen/Sparse header
  *  - define EIGEN_GOOGLEHASH_SUPPORT
  * In the later case the inclusion of <google/dense_hash_map> is made for you.
  */
template<typename SparseMatrixType,
         template <typename T> class MapTraits =
#if defined _DENSE_HASH_MAP_H_
          GoogleDenseHashMapTraits
#elif defined _HASH_MAP
          GnuHashMapTraits
#else
          StdMapTraits
#endif
         ,int OuterPacketBits = 6>
class RandomSetter
{
    typedef typename ei_traits<SparseMatrixType>::Scalar Scalar;
    struct ScalarWrapper
    {
      ScalarWrapper() : value(0) {}
      Scalar value;
    };
    typedef typename MapTraits<ScalarWrapper>::KeyType KeyType;
    typedef typename MapTraits<ScalarWrapper>::Type HashMapType;
    static const int OuterPacketMask = (1 << OuterPacketBits) - 1;
    enum {
      SwapStorage = 1 - MapTraits<ScalarWrapper>::IsSorted,
      TargetRowMajor = (SparseMatrixType::Flags & RowMajorBit) ? 1 : 0,
      SetterRowMajor = SwapStorage ? 1-TargetRowMajor : TargetRowMajor
    };

  public:

    inline RandomSetter(SparseMatrixType& target)
      : mp_target(&target)
    {
      const int outerSize = SwapStorage ? target.innerSize() : target.outerSize();
      const int innerSize = SwapStorage ? target.outerSize() : target.innerSize();
      m_outerPackets = outerSize >> OuterPacketBits;
      if (outerSize&OuterPacketMask)
        m_outerPackets += 1;
      m_hashmaps = new HashMapType[m_outerPackets];
      // compute number of bits needed to store inner indices
      int aux = innerSize - 1;
      m_keyBitsOffset = 0;
      while (aux)
      {
        ++m_keyBitsOffset;
        aux = aux >> 1;
      }
      KeyType ik = (1<<(OuterPacketBits+m_keyBitsOffset));
      for (int k=0; k<m_outerPackets; ++k)
        MapTraits<ScalarWrapper>::setInvalidKey(m_hashmaps[k],ik);

      // insert current coeffs
      for (int j=0; j<mp_target->outerSize(); ++j)
        for (typename SparseMatrixType::InnerIterator it(*mp_target,j); it; ++it)
          (*this)(TargetRowMajor?j:it.index(), TargetRowMajor?it.index():j) = it.value();
    }

    ~RandomSetter()
    {
      KeyType keyBitsMask = (1<<m_keyBitsOffset)-1;
      if (!SwapStorage) // also means the map is sorted
      {
        mp_target->startFill(nonZeros());
        for (int k=0; k<m_outerPackets; ++k)
        {
          const int outerOffset = (1<<OuterPacketBits) * k;
          typename HashMapType::iterator end = m_hashmaps[k].end();
          for (typename HashMapType::iterator it = m_hashmaps[k].begin(); it!=end; ++it)
          {
            const int outer = (it->first >> m_keyBitsOffset) + outerOffset;
            const int inner = it->first & keyBitsMask;
            mp_target->fill(TargetRowMajor ? outer : inner, TargetRowMajor ? inner : outer) = it->second.value;
          }
        }
        mp_target->endFill();
      }
      else
      {
        VectorXi positions(mp_target->outerSize());
        positions.setZero();
        // pass 1
        for (int k=0; k<m_outerPackets; ++k)
        {
          typename HashMapType::iterator end = m_hashmaps[k].end();
          for (typename HashMapType::iterator it = m_hashmaps[k].begin(); it!=end; ++it)
          {
            const int outer = it->first & keyBitsMask;
            ++positions[outer];
          }
        }
        // prefix sum
        int count = 0;
        for (int j=0; j<mp_target->outerSize(); ++j)
        {
          int tmp = positions[j];
          mp_target->_outerIndexPtr()[j] = count;
          positions[j] = count;
          count += tmp;
        }
        mp_target->_outerIndexPtr()[mp_target->outerSize()] = count;
        mp_target->resizeNonZeros(count);
        // pass 2
        for (int k=0; k<m_outerPackets; ++k)
        {
          const int outerOffset = (1<<OuterPacketBits) * k;
          typename HashMapType::iterator end = m_hashmaps[k].end();
          for (typename HashMapType::iterator it = m_hashmaps[k].begin(); it!=end; ++it)
          {
            const int inner = (it->first >> m_keyBitsOffset) + outerOffset;
            const int outer = it->first & keyBitsMask;
            // sorted insertion
            // Note that we have to deal with at most 2^OuterPacketBits unsorted coefficients,
            // moreover those 2^OuterPacketBits coeffs are likely to be sparse, an so only a
            // small fraction of them have to be sorted, whence the following simple procedure:
            int posStart = mp_target->_outerIndexPtr()[outer];
            int i = (positions[outer]++) - 1;
            while ( (i >= posStart) && (mp_target->_innerIndexPtr()[i] > inner) )
            {
              mp_target->_valuePtr()[i+1] = mp_target->_valuePtr()[i];
              mp_target->_innerIndexPtr()[i+1] = mp_target->_innerIndexPtr()[i];
              --i;
            }
            mp_target->_innerIndexPtr()[i+1] = inner;
            mp_target->_valuePtr()[i+1] = it->second.value;
          }
        }
      }
      delete[] m_hashmaps;
    }

    Scalar& operator() (int row, int col)
    {
      const int outer = SetterRowMajor ? row : col;
      const int inner = SetterRowMajor ? col : row;
      const int outerMajor = outer >> OuterPacketBits; // index of the packet/map
      const int outerMinor = outer & OuterPacketMask;  // index of the inner vector in the packet
      const KeyType key = (KeyType(outerMinor)<<m_keyBitsOffset) | inner;
      return m_hashmaps[outerMajor][key].value;
    }

    // might be slow
    int nonZeros() const
    {
      int nz = 0;
      for (int k=0; k<m_outerPackets; ++k)
        nz += m_hashmaps[k].size();
      return nz;
    }


  protected:

    HashMapType* m_hashmaps;
    SparseMatrixType* mp_target;
    int m_outerPackets;
    unsigned char m_keyBitsOffset;
};

#endif // EIGEN_RANDOMSETTER_H
