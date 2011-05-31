// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Gael Guennebaud <gael.guennebaud@inria.fr>
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

#ifndef EIGEN_SPARSE_MARKET_IO_H
#define EIGEN_SPARSE_MARKET_IO_H


template<typename SparseMatrixType>
bool loadMarket(SparseMatrixType& mat, const std::string& filename)
{
  typedef typename SparseMatrixType::Scalar Scalar;
  std::ifstream input(filename.c_str(),std::ios::in);
  if(!input)
    return false;
  
  const int maxBuffersize = 2048;
  char buffer[maxBuffersize];
  
  bool readsizes = false;
  
  int M(-1), N(-1), NNZ(-1);
  int count = 0;
  
  while(input.getline(buffer, maxBuffersize))
  {
    // skip comments
    if(buffer[0]=='%')
      continue;
    
    std::stringstream line(buffer);
    
    if(!readsizes)
    {
      line >> M >> N >> NNZ;
      readsizes = true;
      std::cout << "sizes: " << M << "," << N << "," << NNZ << "\n";
      mat.resize(M,N);
      mat.reserve(NNZ);
    }
    else
    {
      int i(-1), j(-1);
      Scalar v;
      line >> i >> j >> v;
      i--;
      j--;
      if(i>=0 && j>=0 && i<M && j<N)
      {
        ++ count;
        //std::cout << "M[" << i << "," << j << "] = " << v << "\n";
        mat.insert(i,j) = v;
      }
      else
        std::cerr << "Invalid read: " << i << "," << j << "\n";
    }
  }
  
  if(count!=NNZ)
    std::cerr << count << "!=" << NNZ << "\n";
  
  input.close();
  return true;
}

template<typename SparseMatrixType>
bool saveMarket(const SparseMatrixType& mat, const std::string& filename)
{
  std::ofstream out(filename.c_str(),std::ios::out);
  if(!out)
    return false;
  
  out.flags(std::ios_base::scientific);
  out.precision(64);
  out << mat.rows() << " " << mat.cols() << " " << mat.nonZeros() << "\n";
  int count = 0;
  for(int j=0; j<mat.outerSize(); ++j)
    for(typename SparseMatrixType::InnerIterator it(mat,j); it; ++it)
    {
      ++ count;
      out << it.row()+1 << " " << it.col()+1 << " " << it.value() << "\n";
    }
  out.close();
  return true;
}

#endif // EIGEN_SPARSE_MARKET_IO_H
