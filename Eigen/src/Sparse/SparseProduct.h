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

#ifndef EIGEN_SPARSEPRODUCT_H
#define EIGEN_SPARSEPRODUCT_H

#define DENSE_TMP 1
#define MAP_TMP 2
#define LIST_TMP 3

#define TMP_TMP 3

template<typename Scalar>
struct ListEl
{
  int next;
  int index;
  Scalar value;
};

template<typename Lhs, typename Rhs>
static void ei_sparse_product(const Lhs& lhs, const Rhs& rhs, SparseMatrix<typename ei_traits<Lhs>::Scalar>& res)
{
  int rows = lhs.rows();
  int cols = rhs.rows();
  int size = lhs.cols();

  float ratio = std::max(float(lhs.nonZeros())/float(lhs.rows()*lhs.cols()), float(rhs.nonZeros())/float(rhs.rows()*rhs.cols()));
  std::cout << ratio << "\n";

  ei_assert(size == rhs.rows());
  typedef typename ei_traits<Lhs>::Scalar Scalar;
  #if (TMP_TMP == MAP_TMP)
  std::map<int,Scalar> tmp;
  #elif (TMP_TMP == LIST_TMP)
  std::vector<ListEl<Scalar> > tmp(2*rows);
  #else
  std::vector<Scalar> tmp(rows);
  #endif
  res.resize(rows, cols);
  res.startFill(2*std::max(rows, cols));
  for (int j=0; j<cols; ++j)
  {
    #if (TMP_TMP == MAP_TMP)
    tmp.clear();
    #elif (TMP_TMP == LIST_TMP)
    int tmp_size = 0;
    int tmp_start = -1;
    #else
    for (int k=0; k<rows; ++k)
      tmp[k] = 0;
    #endif
    for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
    {
      #if (TMP_TMP == MAP_TMP)
      typename std::map<int,Scalar>::iterator hint = tmp.begin();
      typename std::map<int,Scalar>::iterator r;
      #elif (TMP_TMP == LIST_TMP)
      int tmp_el = tmp_start;
      #endif
      for (typename Lhs::InnerIterator lhsIt(lhs, rhsIt.index()); lhsIt; ++lhsIt)
      {
        #if (TMP_TMP == MAP_TMP)
        r = hint;
        Scalar v = lhsIt.value() * rhsIt.value();
        int id = lhsIt.index();
        while (r!=tmp.end() && r->first < id)
          ++r;
        if (r!=tmp.end() && r->first==id)
        {
          r->second += v;
          hint = r;
        }
        else
          hint = tmp.insert(r, std::pair<int,Scalar>(id, v));
        ++hint;
        #elif (TMP_TMP == LIST_TMP)
        Scalar v = lhsIt.value() * rhsIt.value();
        int id = lhsIt.index();
        if (tmp_size==0)
        {
          tmp_start = 0;
          tmp_el = 0;
          tmp_size++;
          tmp[0].value = v;
          tmp[0].index = id;
          tmp[0].next = -1;
        }
        else if (id<tmp[tmp_start].index)
        {
          tmp[tmp_size].value = v;
          tmp[tmp_size].index = id;
          tmp[tmp_size].next = tmp_start;
          tmp_start = tmp_size;
          tmp_size++;
        }
        else
        {
          int nextel = tmp[tmp_el].next;
          while (nextel >= 0 && tmp[nextel].index<=id)
          {
            tmp_el = nextel;
            nextel = tmp[nextel].next;
          }

          if (tmp[tmp_el].index==id)
          {
            tmp[tmp_el].value += v;
          }
          else
          {
            tmp[tmp_size].value = v;
            tmp[tmp_size].index = id;
            tmp[tmp_size].next = tmp[tmp_el].next;
            tmp[tmp_el].next = tmp_size;
            tmp_size++;
          }
        }
        #else
        tmp[lhsIt.index()] += lhsIt.value() * rhsIt.value();
        #endif
        //res.coeffRef(lhsIt.index(), j) += lhsIt.value() * rhsIt.value();
      }
    }
    #if (TMP_TMP == MAP_TMP)
    for (typename std::map<int,Scalar>::const_iterator k=tmp.begin(); k!=tmp.end(); ++k)
      if (k->second!=0)
        res.fill(k->first, j) = k->second;
    #elif (TMP_TMP == LIST_TMP)
    int k = tmp_start;
    while (k>=0)
    {
      if (tmp[k].value!=0)
        res.fill(tmp[k].index, j) = tmp[k].value;
      k = tmp[k].next;
    }
    #else
    for (int k=0; k<rows; ++k)
      if (tmp[k]!=0)
        res.fill(k, j) = tmp[k];
    #endif
  }
  res.endFill();

  std::cout << "  => " << float(res.nonZeros())/float(res.rows()*res.cols()) << "\n";
}

#endif // EIGEN_SPARSEPRODUCT_H
