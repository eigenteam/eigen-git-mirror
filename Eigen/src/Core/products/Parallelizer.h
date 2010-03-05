// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_PARALLELIZER_H
#define EIGEN_PARALLELIZER_H

struct GemmParallelInfo
{
  GemmParallelInfo() : sync(-1), users(0) {}

  int volatile sync;
  int volatile users;

  int rhs_start;
  int rhs_length;
  float* blockB;
};

template<bool Condition,typename Functor>
void ei_parallelize_gemm(const Functor& func, int rows, int cols)
{
#ifndef EIGEN_HAS_OPENMP
  func(0,rows, 0,cols);
#else

  int threads = omp_get_max_threads();
  if((!Condition)||(threads==1))
    return func(0,rows, 0,cols);

  int blockCols = (cols / threads) & ~0x3;
  int blockRows = (rows / threads) & ~0x7;

  float* sharedBlockB = new float[2048*2048*4];

  GemmParallelInfo* info = new GemmParallelInfo[threads];

  #pragma omp parallel for schedule(static,1)
  for(int i=0; i<threads; ++i)
  {
    int r0 = i*blockRows;
    int actualBlockRows = (i+1==threads) ? rows-r0 : blockRows;

    int c0 = i*blockCols;
    int actualBlockCols = (i+1==threads) ? cols-c0 : blockCols;

    info[i].rhs_start = c0;
    info[i].rhs_length = actualBlockCols;
    info[i].blockB = sharedBlockB;

    func(r0, actualBlockRows, 0,cols, info);
  }

  delete[] sharedBlockB;
  delete[] info;
#endif
}

#endif // EIGEN_PARALLELIZER_H
