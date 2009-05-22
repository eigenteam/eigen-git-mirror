// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_GENERAL_MATRIX_MATRIX_H
#define EIGEN_GENERAL_MATRIX_MATRIX_H

template <int L2MemorySize,typename Scalar>
struct ei_L2_block_traits {
  enum {width = 8 * ei_meta_sqrt<L2MemorySize/(64*sizeof(Scalar))>::ret };
};

#ifndef EIGEN_EXTERN_INSTANTIATIONS

template<typename Scalar>
static void ei_cache_friendly_product(
  int _rows, int _cols, int depth,
  bool _lhsRowMajor, const Scalar* _lhs, int _lhsStride,
  bool _rhsRowMajor, const Scalar* _rhs, int _rhsStride,
  bool resRowMajor, Scalar* res, int resStride)
{
  const Scalar* EIGEN_RESTRICT lhs;
  const Scalar* EIGEN_RESTRICT rhs;
  int lhsStride, rhsStride, rows, cols;
  bool lhsRowMajor;

  if (resRowMajor)
  {
    lhs = _rhs;
    rhs = _lhs;
    lhsStride = _rhsStride;
    rhsStride = _lhsStride;
    cols = _rows;
    rows = _cols;
    lhsRowMajor = !_rhsRowMajor;
    ei_assert(_lhsRowMajor);
  }
  else
  {
    lhs = _lhs;
    rhs = _rhs;
    lhsStride = _lhsStride;
    rhsStride = _rhsStride;
    rows = _rows;
    cols = _cols;
    lhsRowMajor = _lhsRowMajor;
    ei_assert(!_rhsRowMajor);
  }

  typedef typename ei_packet_traits<Scalar>::type PacketType;



#ifndef EIGEN_USE_ALT_PRODUCT

  enum {
    PacketSize = sizeof(PacketType)/sizeof(Scalar),
    #if (defined __i386__)
    HalfRegisterCount = 4,
    #else
    HalfRegisterCount = 8,
    #endif

    // register block size along the N direction
    nr = HalfRegisterCount/2,

    // register block size along the M direction
    mr = 2 * PacketSize,

    // max cache block size along the K direction
    Max_kc = ei_L2_block_traits<EIGEN_TUNE_FOR_CPU_CACHE_SIZE,Scalar>::width,

    // max cache block size along the M direction
    Max_mc = 2*Max_kc
  };

  int kc = std::min<int>(Max_kc,depth);  // cache block size along the K direction
  int mc = std::min<int>(Max_mc,rows);   // cache block size along the M direction

  Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
  Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*PacketSize);

  // number of columns which can be processed by packet of nr columns
  int packet_cols = (cols/nr)*nr;

  // GEMM_VAR1
  for(int k2=0; k2<depth; k2+=kc)
  {
    const int actual_kc = std::min(k2+kc,depth)-k2;

    // we have selected one row panel of rhs and one column panel of lhs
    // pack rhs's panel into a sequential chunk of memory
    // and expand each coeff to a constant packet for further reuse
    {
      int count = 0;
      for(int j2=0; j2<packet_cols; j2+=nr)
      {
        const Scalar* b0 = &rhs[(j2+0)*rhsStride + k2];
        const Scalar* b1 = &rhs[(j2+1)*rhsStride + k2];
        const Scalar* b2 = &rhs[(j2+2)*rhsStride + k2];
        const Scalar* b3 = &rhs[(j2+3)*rhsStride + k2];
        for(int k=0; k<actual_kc; k++)
        {
          ei_pstore(&blockB[count+0*PacketSize], ei_pset1(b0[k]));
          ei_pstore(&blockB[count+1*PacketSize], ei_pset1(b1[k]));
          if (nr==4)
          {
            ei_pstore(&blockB[count+2*PacketSize], ei_pset1(b2[k]));
            ei_pstore(&blockB[count+3*PacketSize], ei_pset1(b3[k]));
          }
          count += nr*PacketSize;
        }
      }
    }

    // => GEPP_VAR1
    for(int i2=0; i2<rows; i2+=mc)
    {
      const int actual_mc = std::min(i2+mc,rows)-i2;

      // We have selected a mc x kc block of lhs
      // Let's pack it in a clever order for further purely sequential access
      int count = 0;
      const int peeled_mc = (actual_mc/mr)*mr;
      if (lhsRowMajor)
      {
        for(int i=0; i<peeled_mc; i+=mr)
          for(int k=0; k<actual_kc; k++)
            for(int w=0; w<mr; w++)
              blockA[count++] = lhs[(k2+k) + (i2+i+w)*lhsStride];
        for(int i=peeled_mc; i<actual_mc; i++)
        {
          const Scalar* llhs = &lhs[(k2) + (i2+i)*lhsStride];
          for(int k=0; k<actual_kc; k++)
            blockA[count++] = llhs[k];
        }
      }
      else
      {
        for(int i=0; i<peeled_mc; i+=mr)
          for(int k=0; k<actual_kc; k++)
            for(int w=0; w<mr; w++)
              blockA[count++] = lhs[(k2+k)*lhsStride + i2+i+w];
        for(int i=peeled_mc; i<actual_mc; i++)
          for(int k=0; k<actual_kc; k++)
            blockA[count++] = lhs[(k2+k)*lhsStride + i2+i];
      }

      // GEBP
      // loops on each cache friendly block of the result/rhs
      for(int j2=0; j2<packet_cols; j2+=nr)
      {
        // loops on each register blocking of lhs/res
        const int peeled_mc = (actual_mc/mr)*mr;
        for(int i=0; i<peeled_mc; i+=mr)
        {
          const Scalar* blA = &blockA[i*actual_kc];
          #ifdef EIGEN_VECTORIZE_SSE
          _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
          #endif

          // TODO move the res loads to the stores

          // gets res block as register
          PacketType C0, C1, C2, C3, C4, C5, C6, C7;
                    C0 = ei_ploadu(&res[(j2+0)*resStride + i2 + i]);
                    C1 = ei_ploadu(&res[(j2+1)*resStride + i2 + i]);
          if(nr==4) C2 = ei_ploadu(&res[(j2+2)*resStride + i2 + i]);
          if(nr==4) C3 = ei_ploadu(&res[(j2+3)*resStride + i2 + i]);
                    C4 = ei_ploadu(&res[(j2+0)*resStride + i2 + i + PacketSize]);
                    C5 = ei_ploadu(&res[(j2+1)*resStride + i2 + i + PacketSize]);
          if(nr==4) C6 = ei_ploadu(&res[(j2+2)*resStride + i2 + i + PacketSize]);
          if(nr==4) C7 = ei_ploadu(&res[(j2+3)*resStride + i2 + i + PacketSize]);

          // performs "inner" product
          // TODO let's check wether the flowing peeled loop could not be
          //      optimized via optimal prefetching from one loop to the other
          const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
          const int peeled_kc = (actual_kc/4)*4;
          for(int k=0; k<peeled_kc; k+=4)
          {
            PacketType B0, B1, B2, B3, A0, A1;

                      A0 = ei_pload(&blA[0*PacketSize]);
                      A1 = ei_pload(&blA[1*PacketSize]);
                      B0 = ei_pload(&blB[0*PacketSize]);
                      B1 = ei_pload(&blB[1*PacketSize]);
                      C0 = ei_pmadd(B0, A0, C0);
            if(nr==4) B2 = ei_pload(&blB[2*PacketSize]);
                      C4 = ei_pmadd(B0, A1, C4);
            if(nr==4) B3 = ei_pload(&blB[3*PacketSize]);
                      B0 = ei_pload(&blB[(nr==4 ? 4 : 2)*PacketSize]);
                      C1 = ei_pmadd(B1, A0, C1);
                      C5 = ei_pmadd(B1, A1, C5);
                      B1 = ei_pload(&blB[(nr==4 ? 5 : 3)*PacketSize]);
            if(nr==4) C2 = ei_pmadd(B2, A0, C2);
            if(nr==4) C6 = ei_pmadd(B2, A1, C6);
            if(nr==4) B2 = ei_pload(&blB[6*PacketSize]);
            if(nr==4) C3 = ei_pmadd(B3, A0, C3);
                      A0 = ei_pload(&blA[2*PacketSize]);
            if(nr==4) C7 = ei_pmadd(B3, A1, C7);
                      A1 = ei_pload(&blA[3*PacketSize]);
            if(nr==4) B3 = ei_pload(&blB[7*PacketSize]);
                      C0 = ei_pmadd(B0, A0, C0);
                      C4 = ei_pmadd(B0, A1, C4);
                      B0 = ei_pload(&blB[(nr==4 ? 8 : 4)*PacketSize]);
                      C1 = ei_pmadd(B1, A0, C1);
                      C5 = ei_pmadd(B1, A1, C5);
                      B1 = ei_pload(&blB[(nr==4 ? 9 : 5)*PacketSize]);
            if(nr==4) C2 = ei_pmadd(B2, A0, C2);
            if(nr==4) C6 = ei_pmadd(B2, A1, C6);
            if(nr==4) B2 = ei_pload(&blB[10*PacketSize]);
            if(nr==4) C3 = ei_pmadd(B3, A0, C3);
                      A0 = ei_pload(&blA[4*PacketSize]);
            if(nr==4) C7 = ei_pmadd(B3, A1, C7);
                      A1 = ei_pload(&blA[5*PacketSize]);
            if(nr==4) B3 = ei_pload(&blB[11*PacketSize]);

                      C0 = ei_pmadd(B0, A0, C0);
                      C4 = ei_pmadd(B0, A1, C4);
                      B0 = ei_pload(&blB[(nr==4 ? 12 : 6)*PacketSize]);
                      C1 = ei_pmadd(B1, A0, C1);
                      C5 = ei_pmadd(B1, A1, C5);
                      B1 = ei_pload(&blB[(nr==4 ? 13 : 7)*PacketSize]);
            if(nr==4) C2 = ei_pmadd(B2, A0, C2);
            if(nr==4) C6 = ei_pmadd(B2, A1, C6);
            if(nr==4) B2 = ei_pload(&blB[14*PacketSize]);
            if(nr==4) C3 = ei_pmadd(B3, A0, C3);
                      A0 = ei_pload(&blA[6*PacketSize]);
            if(nr==4) C7 = ei_pmadd(B3, A1, C7);
                      A1 = ei_pload(&blA[7*PacketSize]);
            if(nr==4) B3 = ei_pload(&blB[15*PacketSize]);
                      C0 = ei_pmadd(B0, A0, C0);
                      C4 = ei_pmadd(B0, A1, C4);
                      C1 = ei_pmadd(B1, A0, C1);
                      C5 = ei_pmadd(B1, A1, C5);
            if(nr==4) C2 = ei_pmadd(B2, A0, C2);
            if(nr==4) C6 = ei_pmadd(B2, A1, C6);
            if(nr==4) C3 = ei_pmadd(B3, A0, C3);
            if(nr==4) C7 = ei_pmadd(B3, A1, C7);

            blB += 4*nr*PacketSize;
            blA += 4*mr;
          }
          // process remaining peeled loop
          for(int k=peeled_kc; k<actual_kc; k++)
          {
            PacketType B0, B1, B2, B3, A0, A1;

                      A0 = ei_pload(&blA[0*PacketSize]);
                      A1 = ei_pload(&blA[1*PacketSize]);
                      B0 = ei_pload(&blB[0*PacketSize]);
                      B1 = ei_pload(&blB[1*PacketSize]);
                      C0 = ei_pmadd(B0, A0, C0);
            if(nr==4) B2 = ei_pload(&blB[2*PacketSize]);
                      C4 = ei_pmadd(B0, A1, C4);
            if(nr==4) B3 = ei_pload(&blB[3*PacketSize]);
                      C1 = ei_pmadd(B1, A0, C1);
                      C5 = ei_pmadd(B1, A1, C5);
            if(nr==4) C2 = ei_pmadd(B2, A0, C2);
            if(nr==4) C6 = ei_pmadd(B2, A1, C6);
            if(nr==4) C3 = ei_pmadd(B3, A0, C3);
            if(nr==4) C7 = ei_pmadd(B3, A1, C7);

            blB += nr*PacketSize;
            blA += mr;
          }

                    ei_pstoreu(&res[(j2+0)*resStride + i2 + i], C0);
                    ei_pstoreu(&res[(j2+1)*resStride + i2 + i], C1);
          if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i2 + i], C2);
          if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i2 + i], C3);
                    ei_pstoreu(&res[(j2+0)*resStride + i2 + i + PacketSize], C4);
                    ei_pstoreu(&res[(j2+1)*resStride + i2 + i + PacketSize], C5);
          if(nr==4) ei_pstoreu(&res[(j2+2)*resStride + i2 + i + PacketSize], C6);
          if(nr==4) ei_pstoreu(&res[(j2+3)*resStride + i2 + i + PacketSize], C7);
        }
        for(int i=peeled_mc; i<actual_mc; i++)
        {
          const Scalar* blA = &blockA[i*actual_kc];
          #ifdef EIGEN_VECTORIZE_SSE
          _mm_prefetch((const char*)(&blA[0]), _MM_HINT_T0);
          #endif

          // gets a 1 x nr res block as registers
          Scalar C0(0), C1(0), C2(0), C3(0);
          const Scalar* blB = &blockB[j2*actual_kc*PacketSize];
          for(int k=0; k<actual_kc; k++)
          {
            Scalar B0, B1, B2, B3, A0;

                      A0 =  blA[k];
                      B0 =  blB[0*PacketSize];
                      B1 =  blB[1*PacketSize];
                      C0 += B0 * A0;
            if(nr==4) B2 =  blB[2*PacketSize];
            if(nr==4) B3 =  blB[3*PacketSize];
                      C1 += B1 * A0;
            if(nr==4) C2 += B2 * A0;
            if(nr==4) C3 += B3 * A0;

            blB += nr*PacketSize;
          }
          res[(j2+0)*resStride + i2 + i] += C0;
          res[(j2+1)*resStride + i2 + i] += C1;
          if(nr==4) res[(j2+2)*resStride + i2 + i] += C2;
          if(nr==4) res[(j2+3)*resStride + i2 + i] += C3;
        }
      }
      // remaining rhs/res columns (<nr)
      for(int j2=packet_cols; j2<cols; j2++)
      {
        for(int i=0; i<actual_mc; i++)
        {
          Scalar c0 = res[(j2)*resStride + i2+i];
          if (lhsRowMajor)
            for(int k=0; k<actual_kc; k++)
              c0 += lhs[(k2+k)+(i2+i)*lhsStride] * rhs[j2*rhsStride + k2 + k];
          else
            for(int k=0; k<actual_kc; k++)
              c0 += lhs[(k2+k)*lhsStride + i2+i] * rhs[j2*rhsStride + k2 + k];
          res[(j2)*resStride + i2+i] = c0;
        }
      }
    }
  }

  ei_aligned_stack_delete(Scalar, blockA, kc*mc);
  ei_aligned_stack_delete(Scalar, blockB, kc*cols*PacketSize);

#else // alternate product from cylmor

  enum {
    PacketSize = sizeof(PacketType)/sizeof(Scalar),
    #if (defined __i386__)
    // i386 architecture provides only 8 xmm registers,
    // so let's reduce the max number of rows processed at once.
    MaxBlockRows = 4,
    MaxBlockRows_ClampingMask = 0xFFFFFC,
    #else
    MaxBlockRows = 8,
    MaxBlockRows_ClampingMask = 0xFFFFF8,
    #endif
    // maximal size of the blocks fitted in L2 cache
    MaxL2BlockSize = ei_L2_block_traits<EIGEN_TUNE_FOR_CPU_CACHE_SIZE,Scalar>::width
  };

  const bool resIsAligned = (PacketSize==1) || (((resStride%PacketSize) == 0) && (size_t(res)%16==0));

  const int remainingSize = depth % PacketSize;
  const int size = depth - remainingSize; // third dimension of the product clamped to packet boundaries

  const int l2BlockRows = MaxL2BlockSize > rows ? rows : 512;
  const int l2BlockCols = MaxL2BlockSize > cols ? cols : 128;
  const int l2BlockSize = MaxL2BlockSize > size ? size : 256;
  const int l2BlockSizeAligned = (1 + std::max(l2BlockSize,l2BlockCols)/PacketSize)*PacketSize;
  const bool needRhsCopy = (PacketSize>1) && ((rhsStride%PacketSize!=0) || (size_t(rhs)%16!=0));

  Scalar* EIGEN_RESTRICT block = new Scalar[l2BlockRows*size];
//   for(int i=0; i<l2BlockRows*l2BlockSize; ++i)
//     block[i] = 0;
  // loops on each L2 cache friendly blocks of lhs
  for(int l2k=0; l2k<depth; l2k+=l2BlockSize)
  {
    for(int l2i=0; l2i<rows; l2i+=l2BlockRows)
    {
      // We have selected a block of lhs
      // Packs this block into 'block'
      int count = 0;
      for(int k=0; k<l2BlockSize; k+=MaxBlockRows)
      {
        for(int i=0; i<l2BlockRows; i+=2*PacketSize)
          for (int w=0; w<MaxBlockRows; ++w)
            for (int y=0; y<2*PacketSize; ++y)
              block[count++] = lhs[(k+l2k+w)*lhsStride + l2i+i+ y];
      }

      // loops on each L2 cache firendly block of the result/rhs
      for(int l2j=0; l2j<cols; l2j+=l2BlockCols)
      {
        for(int k=0; k<l2BlockSize; k+=MaxBlockRows)
        {
          for(int j=0; j<l2BlockCols; ++j)
          {
            PacketType A0, A1, A2, A3, A4, A5, A6, A7;

            // Load the packets from rhs and reorder them

            // Here we need some vector reordering
            // Right now its hardcoded to packets of 4 elements
            const Scalar* lrhs = &rhs[(j+l2j)*rhsStride+(k+l2k)];
            A0 = ei_pset1(lrhs[0]);
            A1 = ei_pset1(lrhs[1]);
            A2 = ei_pset1(lrhs[2]);
            A3 = ei_pset1(lrhs[3]);
            if (MaxBlockRows==8)
            {
              A4 = ei_pset1(lrhs[4]);
              A5 = ei_pset1(lrhs[5]);
              A6 = ei_pset1(lrhs[6]);
              A7 = ei_pset1(lrhs[7]);
            }

            Scalar * lb = &block[l2BlockRows * k];
            for(int i=0; i<l2BlockRows; i+=2*PacketSize)
            {
              PacketType R0, R1, L0, L1, T0, T1;

              // We perform "cross products" of vectors to avoid
              // reductions (horizontal ops) afterwards
              T0 = ei_pload(&res[(j+l2j)*resStride+l2i+i]);
              T1 = ei_pload(&res[(j+l2j)*resStride+l2i+i+PacketSize]);

              R0 = ei_pload(&lb[0*PacketSize]);
              L0 = ei_pload(&lb[1*PacketSize]);
              R1 = ei_pload(&lb[2*PacketSize]);
              L1 = ei_pload(&lb[3*PacketSize]);
              T0 = ei_pmadd(R0, A0, T0);
              T1 = ei_pmadd(L0, A0, T1);
              R0 = ei_pload(&lb[4*PacketSize]);
              L0 = ei_pload(&lb[5*PacketSize]);
              T0 = ei_pmadd(R1, A1, T0);
              T1 = ei_pmadd(L1, A1, T1);
              R1 = ei_pload(&lb[6*PacketSize]);
              L1 = ei_pload(&lb[7*PacketSize]);
              T0 = ei_pmadd(R0, A2, T0);
              T1 = ei_pmadd(L0, A2, T1);
              if(MaxBlockRows==8)
              {
                R0 = ei_pload(&lb[8*PacketSize]);
                L0 = ei_pload(&lb[9*PacketSize]);
              }
              T0 = ei_pmadd(R1, A3, T0);
              T1 = ei_pmadd(L1, A3, T1);
              if(MaxBlockRows==8)
              {
                R1 = ei_pload(&lb[10*PacketSize]);
                L1 = ei_pload(&lb[11*PacketSize]);
                T0 = ei_pmadd(R0, A4, T0);
                T1 = ei_pmadd(L0, A4, T1);
                R0 = ei_pload(&lb[12*PacketSize]);
                L0 = ei_pload(&lb[13*PacketSize]);
                T0 = ei_pmadd(R1, A5, T0);
                T1 = ei_pmadd(L1, A5, T1);
                R1 = ei_pload(&lb[14*PacketSize]);
                L1 = ei_pload(&lb[15*PacketSize]);
                T0 = ei_pmadd(R0, A6, T0);
                T1 = ei_pmadd(L0, A6, T1);
                T0 = ei_pmadd(R1, A7, T0);
                T1 = ei_pmadd(L1, A7, T1);
              }
              lb += MaxBlockRows*2*PacketSize;

              ei_pstore(&res[(j+l2j)*resStride+l2i+i], T0);
              ei_pstore(&res[(j+l2j)*resStride+l2i+i+PacketSize], T1);
            }
          }
        }
      }
    }
  }
  delete[] block;
#endif


}

#endif // EIGEN_EXTERN_INSTANTIATIONS

#endif // EIGEN_GENERAL_MATRIX_MATRIX_H
