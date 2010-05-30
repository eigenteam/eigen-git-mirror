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

template<typename BlockBScalar, typename Index> struct GemmParallelInfo
{
  GemmParallelInfo() : sync(-1), users(0), rhs_start(0), rhs_length(0), blockB(0) {}

  int volatile sync;
  int volatile users;

  Index rhs_start;
  Index rhs_length;
  BlockBScalar* blockB;
};

template<bool Condition, typename Functor, typename Index>
void ei_parallelize_gemm(const Functor& func, Index rows, Index cols)
{
#ifndef EIGEN_HAS_OPENMP
  func(0,rows, 0,cols);
#else

  // Dynamically check whether we should enable or disable OpenMP.
  // The conditions are:
  // - the max number of threads we can create is greater than 1
  // - we are not already in a parallel code
  // - the sizes are large enough

  // 1- are we already in a parallel session?
  // FIXME omp_get_num_threads()>1 only works for openmp, what if the user does not use openmp?
  if((!Condition) || (omp_get_num_threads()>1))
    return func(0,rows, 0,cols);

  // 2- compute the maximal number of threads from the size of the product:
  // FIXME this has to be fine tuned
  Index max_threads = std::max(1,rows / 32);

  // 3 - compute the number of threads we are going to use
  Index threads = std::min<Index>(omp_get_max_threads(), max_threads);

  if(threads==1)
    return func(0,rows, 0,cols);

  Index blockCols = (cols / threads) & ~Index(0x3);
  Index blockRows = (rows / threads) & ~Index(0x7);

  typedef typename Functor::BlockBScalar BlockBScalar;
  BlockBScalar* sharedBlockB = new BlockBScalar[func.sharedBlockBSize()];

  GemmParallelInfo<BlockBScalar>* info = new GemmParallelInfo<BlockBScalar>[threads];

  #pragma omp parallel for schedule(static,1) num_threads(threads)
  for(Index i=0; i<threads; ++i)
  {
    Index r0 = i*blockRows;
    Index actualBlockRows = (i+1==threads) ? rows-r0 : blockRows;

    Index c0 = i*blockCols;
    Index actualBlockCols = (i+1==threads) ? cols-c0 : blockCols;

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
