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

  static void setInvalidKey(Type&, const KeyType&) {}
};

#ifdef _HASH_MAP
template<typename Scalar> struct GnuHashMapTraits
{
  typedef int KeyType;
  typedef __gnu_cxx::hash_map<KeyType,Scalar> Type;

  static void setInvalidKey(Type&, const KeyType&) {}
};
#endif

#ifdef _DENSE_HASH_MAP_H_
template<typename Scalar> struct GoogleDenseHashMapTraits
{
  typedef int KeyType;
  typedef google::dense_hash_map<KeyType,Scalar> Type;

  static void setInvalidKey(Type& map, const KeyType& k)
  { map.set_empty_key(k); }
};
#endif

#ifdef _SPARSE_HASH_MAP_H_
template<typename Scalar> struct GoogleSparseHashMapTraits
{
  typedef int KeyType;
  typedef google::sparse_hash_map<KeyType,Scalar> Type;

  static void setInvalidKey(Type&, const KeyType&) {}
};
#endif

/** \class RandomSetter
  *
  */
template<typename SparseMatrixType,
         template <typename T> class HashMapTraits = StdMapTraits,
         int OuterPacketBits = 6>
class RandomSetter
{
    typedef typename ei_traits<SparseMatrixType>::Scalar Scalar;
    struct ScalarWrapper
    {
      ScalarWrapper() : value(0) {}
      Scalar value;
    };
    typedef typename HashMapTraits<ScalarWrapper>::KeyType KeyType;
    typedef typename HashMapTraits<ScalarWrapper>::Type HashMapType;
    static const int OuterPacketMask = (1 << OuterPacketBits) - 1;
    enum {
      RowMajor = SparseMatrixType::Flags & RowMajorBit
    };

  public:

    inline RandomSetter(SparseMatrixType& target)
      : mp_target(&target)
    {
      m_outerPackets = target.outerSize() >> OuterPacketBits;
      if (target.outerSize()&OuterPacketMask)
        m_outerPackets += 1;
      m_hashmaps = new HashMapType[m_outerPackets];
      KeyType ik = (1<<OuterPacketBits)*mp_target->innerSize()+1;
      for (int k=0; k<m_outerPackets; ++k)
        HashMapTraits<ScalarWrapper>::setInvalidKey(m_hashmaps[k],ik);
    }

    ~RandomSetter()
    {
      delete[] m_hashmaps;
    }

    Scalar& operator() (int row, int col)
    {
      const int outer = RowMajor ? row : col;
      const int inner = RowMajor ? col : row;
      const int outerMajor = outer >> OuterPacketBits;
      const int outerMinor = outer & OuterPacketMask;
      const KeyType key = inner + outerMinor * mp_target->innerSize();

      return m_hashmaps[outerMajor][key].value;
    }

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
};

#endif // EIGEN_RANDOMSETTER_H
