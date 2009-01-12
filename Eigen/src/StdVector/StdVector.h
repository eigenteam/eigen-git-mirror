// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Alex Stapleton <alex.stapleton@gmail.com>
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

#ifndef EIGEN_STDVECTOR_H
#define EIGEN_STDVECTOR_H

template <typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols, typename _Alloc>
class vector<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols>, _Alloc>
  : public vector<Eigen::ei_unaligned_type<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> >,
                  Eigen::aligned_allocator<Eigen::ei_unaligned_type<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> > > >
{
public:
  typedef Eigen::ei_unaligned_type<Eigen::Matrix<_Scalar,_Rows,_Cols,_Options,_MaxRows,_MaxCols> > value_type;
  typedef Eigen::aligned_allocator<value_type> allocator_type;
  typedef vector<value_type, allocator_type > unaligned_base;
  typedef typename unaligned_base::size_type size_type;
  typedef typename unaligned_base::iterator iterator;

  explicit vector(const allocator_type& __a = allocator_type()) : unaligned_base(__a) {}
  vector(const vector& c) : unaligned_base(c) {}
  vector(size_type num, const value_type& val = value_type()) : unaligned_base(num, val) {}
  vector(iterator start, iterator end) : unaligned_base(start, end) {}
  vector& operator=(const vector& __x) {
    unaligned_base::operator=(__x);
    return *this;
  }
};

#endif // EIGEN_STDVECTOR_H
