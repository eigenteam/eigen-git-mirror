// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_H

// if the rhs is row major, we have to evaluate it in a temporary colmajor matrix
template <typename Scalar, int Side, int Mode, bool Conjugate, int TriStorageOrder>
struct ei_triangular_solve_matrix<Scalar,Side,Mode,Conjugate,TriStorageOrder,RowMajor>
{
  static EIGEN_DONT_INLINE void run(
    int size, int cols,
    const Scalar*  tri, int triStride,
    Scalar* _other, int otherStride)
  {
    
    ei_triangular_solve_matrix<
      Scalar, Side==OnTheLeft?OnTheRight:OnTheLeft,
      (Mode&UnitDiagBit) | (Mode&UpperTriangular) ? LowerTriangular : UpperTriangular,
      !Conjugate, TriStorageOrder, ColMajor>
      ::run(size, cols, tri, triStride, _other, otherStride);

//     Map<Matrix<Scalar,Dynamic,Dynamic> > other(_other, otherStride, cols);
//     Matrix<Scalar,Dynamic,Dynamic> aux = other.block(0,0,size,cols);
//     ei_triangular_solve_matrix<Scalar,Side,Mode,Conjugate,TriStorageOrder,ColMajor>
//       ::run(size, cols, tri, triStride, aux.data(), aux.stride());
//     other.block(0,0,size,cols) = aux;
  }
};

/* Optimized triangular solver with multiple right hand side (_TRSM)
 */
template <typename Scalar, int Mode, bool Conjugate, int TriStorageOrder>
struct ei_triangular_solve_matrix<Scalar,OnTheLeft,Mode,Conjugate,TriStorageOrder,ColMajor>
{
  static EIGEN_DONT_INLINE void run(
    int size, int otherSize,
    const Scalar* _tri, int triStride,
    Scalar* _other, int otherStride)
  {
    int cols = otherSize;
    ei_const_blas_data_mapper<Scalar, TriStorageOrder> tri(_tri,triStride);
    ei_blas_data_mapper<Scalar, ColMajor> other(_other,otherStride);

    typedef ei_product_blocking_traits<Scalar> Blocking;
    enum {
      SmallPanelWidth   = EIGEN_ENUM_MAX(Blocking::mr,Blocking::nr),
      IsLowerTriangular = (Mode&LowerTriangular) == LowerTriangular
    };

    int kc = std::min<int>(Blocking::Max_kc/4,size); // cache block size along the K direction
    int mc = std::min<int>(Blocking::Max_mc,size);   // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    ei_conj_if<Conjugate> conj;
    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<Conjugate,false> > gebp_kernel;
    ei_gemm_pack_lhs<Scalar,Blocking::mr,TriStorageOrder> pack_lhs;

    for(int k2=IsLowerTriangular ? 0 : size;
        IsLowerTriangular ? k2<size : k2>0;
        IsLowerTriangular ? k2+=kc : k2-=kc)
    {
      const int actual_kc = std::min(IsLowerTriangular ? size-k2 : k2, kc);

      // We have selected and packed a big horizontal panel R1 of rhs. Let B be the packed copy of this panel,
      // and R2 the remaining part of rhs. The corresponding vertical panel of lhs is split into
      // A11 (the triangular part) and A21 the remaining rectangular part.
      // Then the high level algorithm is:
      //  - B = R1                    => general block copy (done during the next step)
      //  - R1 = L1^-1 B              => tricky part
      //  - update B from the new R1  => actually this has to be performed continuously during the above step
      //  - R2 = L2 * B               => GEPP

      // The tricky part: compute R1 = L1^-1 B while updating B from R1
      // The idea is to split L1 into multiple small vertical panels.
      // Each panel can be split into a small triangular part A1 which is processed without optimization,
      // and the remaining small part A2 which is processed using gebp with appropriate block strides
      {
        // for each small vertical panels of lhs
        for (int k1=0; k1<actual_kc; k1+=SmallPanelWidth)
        {
          int actualPanelWidth = std::min<int>(actual_kc-k1, SmallPanelWidth);
          // tr solve
          for (int k=0; k<actualPanelWidth; ++k)
          {
            // TODO write a small kernel handling this (can be shared with trsv)
            int i  = IsLowerTriangular ? k2+k1+k : k2-k1-k-1;
            int s  = IsLowerTriangular ? k2+k1 : i+1;
            int rs = actualPanelWidth - k - 1; // remaining size

            Scalar a = (Mode & UnitDiagBit) ? Scalar(1) : Scalar(1)/conj(tri(i,i));
            for (int j=0; j<cols; ++j)
            {
              if (TriStorageOrder==RowMajor)
              {
                Scalar b = 0;
                const Scalar* l = &tri(i,s);
                Scalar* r = &other(s,j);
                for (int i3=0; i3<k; ++i3)
                  b += conj(l[i3]) * r[i3];

                other(i,j) = (other(i,j) - b)*a;
              }
              else
              {
                int s = IsLowerTriangular ? i+1 : i-rs;
                Scalar b = (other(i,j) *= a);
                Scalar* r = &other(s,j);
                const Scalar* l = &tri(s,i);
                for (int i3=0;i3<rs;++i3)
                  r[i3] -= b * conj(l[i3]);
              }
            }
          }

          int lengthTarget = actual_kc-k1-actualPanelWidth;
          int startBlock   = IsLowerTriangular ? k2+k1 : k2-k1-actualPanelWidth;
          int blockBOffset = IsLowerTriangular ? k1 : lengthTarget;

          // update the respective rows of B from other
          ei_gemm_pack_rhs<Scalar, Blocking::nr, ColMajor, true>()
            (blockB, _other+startBlock, otherStride, -1, actualPanelWidth, cols, actual_kc, blockBOffset);

          // GEBP
          if (lengthTarget>0)
          {
            int startTarget  = IsLowerTriangular ? k2+k1+actualPanelWidth : k2-actual_kc;

            pack_lhs(blockA, &tri(startTarget,startBlock), triStride, actualPanelWidth, lengthTarget);

            gebp_kernel(_other+startTarget, otherStride, blockA, blockB, lengthTarget, actualPanelWidth, cols,
                        actualPanelWidth, actual_kc, 0, blockBOffset*Blocking::PacketSize);
          }
        }
      }

      // R2 = A2 * B => GEPP
      {
        int start = IsLowerTriangular ? k2+kc : 0;
        int end   = IsLowerTriangular ? size : k2-kc;
        for(int i2=start; i2<end; i2+=mc)
        {
          const int actual_mc = std::min(mc,end-i2);
          if (actual_mc>0)
          {
            pack_lhs(blockA, &tri(i2, IsLowerTriangular ? k2 : k2-kc), triStride, actual_kc, actual_mc);

            gebp_kernel(_other+i2, otherStride, blockA, blockB, actual_mc, actual_kc, cols);
          }
        }
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

/* Optimized triangular solver with multiple left hand sides and the trinagular matrix on the right
 */
template <typename Scalar, int Mode, bool Conjugate, int TriStorageOrder>
struct ei_triangular_solve_matrix<Scalar,OnTheRight,Mode,Conjugate,TriStorageOrder,ColMajor>
{
  static EIGEN_DONT_INLINE void run(
    int size, int otherSize,
    const Scalar* _tri, int triStride,
    Scalar* _other, int otherStride)
  {
    int rows = otherSize;
//     ei_const_blas_data_mapper<Scalar, TriStorageOrder> rhs(_tri,triStride);
//     ei_blas_data_mapper<Scalar, ColMajor> lhs(_other,otherStride);

    Map<Matrix<Scalar,Dynamic,Dynamic,TriStorageOrder> > rhs(_tri,size,size);
    Map<Matrix<Scalar,Dynamic,Dynamic,ColMajor> > lhs(_other,rows,size);

    typedef ei_product_blocking_traits<Scalar> Blocking;
    enum {
      RhsStorageOrder   = TriStorageOrder,
      SmallPanelWidth   = EIGEN_ENUM_MAX(Blocking::mr,Blocking::nr),
      IsLowerTriangular = (Mode&LowerTriangular) == LowerTriangular
    };

    int kc = std::min<int>(/*Blocking::Max_kc/4*/32,size); // cache block size along the K direction
    int mc = std::min<int>(/*Blocking::Max_mc*/32,size);   // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*size*Blocking::PacketSize);

    ei_conj_if<Conjugate> conj;
    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<false,Conjugate> > gebp_kernel;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder> pack_rhs;
    ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder,true> pack_rhs_panel;
    ei_gemm_pack_lhs<Scalar, Blocking::mr, ColMajor, false, true> pack_lhs_panel;
    ei_gemm_pack_lhs<Scalar, Blocking::mr, ColMajor, false> pack_lhs;

    for(int k2=IsLowerTriangular ? size : 0;
        IsLowerTriangular ? k2>0 : k2<size;
        IsLowerTriangular ? k2-=kc : k2+=kc)
    {
      const int actual_kc = std::min(IsLowerTriangular ? k2 : size-k2, kc);
      int actual_k2 = IsLowerTriangular ? k2-actual_kc : k2 ;

      int startPanel = IsLowerTriangular ? 0 : k2+actual_kc;
      int rs = IsLowerTriangular ? actual_k2 : size - actual_k2 - actual_kc;        
      Scalar* geb = blockB+actual_kc*actual_kc*Blocking::PacketSize;

      if (rs>0) pack_rhs(geb, &rhs(actual_k2,startPanel), triStride, -1, actual_kc, rs);

      // triangular packing (we only pack the panels off the diagonal,
      // neglecting the blocks overlapping the diagonal
      {
        for (int j2=0; j2<actual_kc; j2+=SmallPanelWidth)
        {
          int actualPanelWidth = std::min<int>(actual_kc-j2, SmallPanelWidth);
          int actual_j2 = actual_k2 + j2;
          int panelOffset = IsLowerTriangular ? j2+actualPanelWidth : 0;
          int panelLength = IsLowerTriangular ? actual_kc-j2-actualPanelWidth : j2;

//           std::cerr << "$ " << k2 << " " << j2 << " " << actual_j2 << " " << panelOffset << " " << panelLength << "\n";

          if (panelLength>0)
          pack_rhs_panel(blockB+j2*actual_kc*Blocking::PacketSize,
                         &rhs(actual_k2+panelOffset, actual_j2), triStride, -1,
                         panelLength, actualPanelWidth,
                         actual_kc, panelOffset);
        }
      }

      for(int i2=0; i2<rows; i2+=mc)
      {
        const int actual_mc = std::min(mc,rows-i2);

        // triangular solver kernel
        {
          // for each small block of the diagonal (=> vertical panels of rhs)
          for (int j2 = IsLowerTriangular
                      ? (actual_kc - ((actual_kc%SmallPanelWidth) ? (actual_kc%SmallPanelWidth)
                                                                  : SmallPanelWidth))
                      : 0;
               IsLowerTriangular ? j2>=0 : j2<actual_kc;
               IsLowerTriangular ? j2-=SmallPanelWidth : j2+=SmallPanelWidth)
          {
            int actualPanelWidth = std::min<int>(actual_kc-j2, SmallPanelWidth);
            int absolute_j2 = actual_k2 + j2;
            int panelOffset = IsLowerTriangular ? j2+actualPanelWidth : 0;
            int panelLength = IsLowerTriangular ? actual_kc - j2 - actualPanelWidth : j2;

            // GEBP
            //if (lengthTarget>0)
            if(panelLength>0)
            {
              gebp_kernel(&lhs(i2,absolute_j2), otherStride,
                          blockA, blockB+j2*actual_kc*Blocking::PacketSize,
                          actual_mc, panelLength, actualPanelWidth,
                          actual_kc, actual_kc, // strides
                          panelOffset, panelOffset*Blocking::PacketSize); // offsets
            }

            // unblocked triangular solve
            for (int k=0; k<actualPanelWidth; ++k)
            {
              int j = IsLowerTriangular ? absolute_j2+actualPanelWidth-k-1 : absolute_j2+k;

              Scalar a = (Mode & UnitDiagBit) ? Scalar(1) : Scalar(1)/conj(rhs(j,j));
              for (int i=0; i<actual_mc; ++i)
              {
                int absolute_i = i2+i;
                Scalar b = 0;
                for (int k3=0; k3<k; ++k3)
                  if(IsLowerTriangular)
                    b += lhs(absolute_i,j+1+k3) * conj(rhs(j+1+k3,j));
                  else
                    b += lhs(absolute_i,absolute_j2+k3) * conj(rhs(absolute_j2+k3,j));
                lhs(absolute_i,j) = (lhs(absolute_i,j) - b)*a;
              }
            }

            // pack the just computed part of lhs to A
            pack_lhs_panel(blockA, _other+absolute_j2*otherStride+i2, otherStride,
                           actualPanelWidth, actual_mc,
                           actual_kc, j2);
          }
        }
        
        if (rs>0)
          gebp_kernel(_other+i2+startPanel*otherStride, otherStride, blockA, geb,
                      actual_mc, actual_kc, rs);
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*size*Blocking::PacketSize);
  }
};

#endif // EIGEN_TRIANGULAR_SOLVER_MATRIX_H
