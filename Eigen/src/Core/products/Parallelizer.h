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

template<bool Parallelize,typename Functor>
void ei_run_parallel_1d(const Functor& func, int size)
{
#ifndef EIGEN_HAS_OPENMP
  func(0,size);
#else
  if(!Parallelize)
    return func(0,size);

  int threads = omp_get_num_procs();
  int blockSize = size / threads;
  #pragma omp parallel for schedule(static,1)
  for(int i=0; i<threads; ++i)
  {
    int blockStart = i*blockSize;
    int actualBlockSize = std::min(blockSize, size - blockStart);

    func(blockStart, actualBlockSize);
  }
#endif
}

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
