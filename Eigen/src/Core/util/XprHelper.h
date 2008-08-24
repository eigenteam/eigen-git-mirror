// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_XPRHELPER_H
#define EIGEN_XPRHELPER_H

// just a workaround because GCC seems to not really like empty structs
#ifdef __GNUG__
  struct ei_empty_struct{char _ei_dummy_;};
  #define EIGEN_EMPTY_STRUCT : Eigen::ei_empty_struct
#else
  #define EIGEN_EMPTY_STRUCT
#endif

//classes inheriting ei_no_assignment_operator don't generate a default operator=.
class ei_no_assignment_operator
{
  private:
    ei_no_assignment_operator& operator=(const ei_no_assignment_operator&);
};

template<int Value> class ei_int_if_dynamic EIGEN_EMPTY_STRUCT
{
  public:
    ei_int_if_dynamic() {}
    explicit ei_int_if_dynamic(int) {}
    static int value() { return Value; }
    void setValue(int) {}
};

template<> class ei_int_if_dynamic<Dynamic>
{
    int m_value;
    ei_int_if_dynamic() {}
  public:
    explicit ei_int_if_dynamic(int value) : m_value(value) {}
    int value() const { return m_value; }
    void setValue(int value) { m_value = value; }
};

template<typename T> struct ei_functor_traits
{
  enum
  {
    Cost = 10,
    PacketAccess = false
  };
};

template<typename T> struct ei_packet_traits
{
  typedef T type;
  enum {size=1};
};

template<typename T> struct ei_unpacket_traits
{
  typedef T type;
  enum {size=1};
};


template<typename Scalar, int Rows, int Cols, int StorageOrder, int MaxRows, int MaxCols>
class ei_compute_matrix_flags
{
    enum {
      row_major_bit = (Rows != 1 && Cols != 1)  // if this is not a vector,
                                                // then the storage order really matters,
                                                // so let us strictly honor the user's choice.
                    ? StorageOrder
                    : Cols > 1 ? RowMajorBit : 0,
      inner_max_size = row_major_bit ? MaxCols : MaxRows,
      is_big = inner_max_size == Dynamic,
      is_packet_size_multiple = (Cols * Rows)%ei_packet_traits<Scalar>::size==0,
      packet_access_bit = ei_packet_traits<Scalar>::size > 1
                          && (is_big || is_packet_size_multiple) ? PacketAccessBit : 0,
      aligned_bit = packet_access_bit && (is_big || is_packet_size_multiple) ? AlignedBit : 0
    };

  public:
    enum { ret = LinearAccessBit | DirectAccessBit | packet_access_bit | row_major_bit | aligned_bit };
};

template<int _Rows, int _Cols> struct ei_size_at_compile_time
{
  enum { ret = (_Rows==Dynamic || _Cols==Dynamic) ? Dynamic : _Rows * _Cols };
};

template<typename T, int Sparseness = ei_traits<T>::Flags&SparseBit> class ei_eval;

template<typename T> struct ei_eval<T,Dense>
{
  typedef Matrix<typename ei_traits<T>::Scalar,
                ei_traits<T>::RowsAtCompileTime,
                ei_traits<T>::ColsAtCompileTime,
                ei_traits<T>::Flags&RowMajorBit ? RowMajor : ColMajor,
                ei_traits<T>::MaxRowsAtCompileTime,
                ei_traits<T>::MaxColsAtCompileTime
          > type;
};

template<typename T> struct ei_must_nest_by_value { enum { ret = false }; };
template<typename T> struct ei_must_nest_by_value<NestByValue<T> > { enum { ret = true }; };

template<typename T, int n=1, typename EvalType = typename ei_eval<T>::type> struct ei_nested
{
  enum {
    CostEval   = (n+1) * int(NumTraits<typename ei_traits<T>::Scalar>::ReadCost),
    CostNoEval = (n-1) * int(ei_traits<T>::CoeffReadCost)
  };
  typedef typename ei_meta_if<
    ei_must_nest_by_value<T>::ret,
    T,
    typename ei_meta_if<
      (int(ei_traits<T>::Flags) & EvalBeforeNestingBit)
      || ( int(CostEval) <= int(CostNoEval) ),
      EvalType,
      const T&
    >::ret
  >::ret type;
};

template<unsigned int Flags> struct ei_are_flags_consistent
{
  enum { ret = !( (Flags&UnitDiagBit && Flags&ZeroDiagBit) )
  };
};

/** \internal Gives the type of a sub-matrix or sub-vector of a matrix of type \a ExpressionType and size \a Size
  * TODO: could be a good idea to define a big ReturnType struct ??
  */
template<typename ExpressionType, int RowsOrSize=Dynamic, int Cols=Dynamic> struct BlockReturnType {
  typedef Block<ExpressionType, (ei_traits<ExpressionType>::RowsAtCompileTime == 1 ? 1 : RowsOrSize),
                                (ei_traits<ExpressionType>::ColsAtCompileTime == 1 ? 1 : RowsOrSize)> SubVectorType;
  typedef Block<ExpressionType, RowsOrSize, Cols> Type;
};

#endif // EIGEN_XPRHELPER_H
