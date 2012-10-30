
#ifndef EIGEN_SPARSELU_GEMM_KERNEL_H
#define EIGEN_SPARSELU_GEMM_KERNEL_H

namespace Eigen {

namespace internal {


/** \internal
  * A general matrix-matrix product kernel optimized for the SparseLU factorization.
  *  - A, B, and C must be column major
  *  - lda and ldc must be multiples of the respective packet size
  *  - C must have the same alignment as A
  */
template<typename Scalar>
EIGEN_DONT_INLINE
void sparselu_gemm(int m, int n, int d, const Scalar* A, int lda, const Scalar* B, int ldb, Scalar* C, int ldc)
{
  using namespace Eigen::internal;
  
  typedef typename packet_traits<Scalar>::type Packet;
  enum {
    PacketSize = packet_traits<Scalar>::size,
    PM = 8,                   // peeling in M
    RN = 2,                   // register blocking
    RK = 4,                   // register blocking
    BM = 4096/sizeof(Scalar), // number of rows of A-C per chunk
    SM = PM*PacketSize        // step along M
  };
  int d_end = (d/RK)*RK;    // number of columns of A (rows of B) suitable for full register blocking
  int n_end = (n/RN)*RN;    // number of columns of B-C suitable for processing RN columns at once
  int i0 = internal::first_aligned(A,m);
  
  eigen_internal_assert(((lda%PacketSize)==0) && ((ldc%PacketSize)==0) && (i0==internal::first_aligned(C,m)));
  
  // handle the non aligned rows of A and C without any optimization:
  for(int i=0; i<i0; ++i)
  {
    for(int j=0; j<n; ++j)
    {
      Scalar c = C[i+j*ldc];
      for(int k=0; k<d; ++k)
        c += B[k+j*ldb] * A[i+k*lda];
      C[i+j*ldc] = c;
    }
  }
  // process the remaining rows per chunk of BM rows
  for(int ib=i0; ib<m; ib+=BM)
  {
    int actual_b = std::min<int>(BM, m-ib);                 // actual number of rows
    int actual_b_end1 = (actual_b/SM)*SM;                   // actual number of rows suitable for peeling
    int actual_b_end2 = (actual_b/PacketSize)*PacketSize;   // actual number of rows suitable for vectorization
    
    // Let's process two columns of B-C at once
    for(int j=0; j<n_end; j+=RN)
    {
      const Scalar* Bc0 = B+(j+0)*ldb;
      const Scalar* Bc1 = B+(j+1)*ldb;
      
      for(int k=0; k<d_end; k+=RK)
      {
        
        // load and expand a RN x RK block of B
        Packet b00, b10, b20, b30, b01, b11, b21, b31;
        b00 = pset1<Packet>(Bc0[0]);
        b10 = pset1<Packet>(Bc0[1]);
        b20 = pset1<Packet>(Bc0[2]);
        b30 = pset1<Packet>(Bc0[3]);
        b01 = pset1<Packet>(Bc1[0]);
        b11 = pset1<Packet>(Bc1[1]);
        b21 = pset1<Packet>(Bc1[2]);
        b31 = pset1<Packet>(Bc1[3]);
        
        Packet a0, a1, a2, a3, c0, c1, t0, t1;
        
        const Scalar* A0 = A+ib+(k+0)*lda;
        const Scalar* A1 = A+ib+(k+1)*lda;
        const Scalar* A2 = A+ib+(k+2)*lda;
        const Scalar* A3 = A+ib+(k+3)*lda;
        
        Scalar* C0 = C+ib+(j+0)*ldc;
        Scalar* C1 = C+ib+(j+1)*ldc;
        
        a0 = pload<Packet>(A0);
        a1 = pload<Packet>(A1);
        a2 = pload<Packet>(A2);
        a3 = pload<Packet>(A3);
        
#define KMADD(c, a, b, tmp) tmp = b; tmp = pmul(a,tmp); c = padd(c,tmp);
#define WORK(I)  \
          c0 = pload<Packet>(C0+i+(I)*PacketSize);   \
          c1 = pload<Packet>(C1+i+(I)*PacketSize);   \
          KMADD(c0, a0, b00, t0);       \
          KMADD(c1, a0, b01, t1);       \
          a0 = pload<Packet>(A0+i+(I+1)*PacketSize); \
          KMADD(c0, a1, b10, t0);       \
          KMADD(c1, a1, b11, t1);       \
          a1 = pload<Packet>(A1+i+(I+1)*PacketSize); \
          KMADD(c0, a2, b20, t0);       \
          KMADD(c1, a2, b21, t1);       \
          a2 = pload<Packet>(A2+i+(I+1)*PacketSize); \
          KMADD(c0, a3, b30, t0);       \
          KMADD(c1, a3, b31, t1);       \
          a3 = pload<Packet>(A3+i+(I+1)*PacketSize); \
          pstore(C0+i+(I)*PacketSize, c0);           \
          pstore(C1+i+(I)*PacketSize, c1)
        
        // process rows of A' - C' with aggressive vectorization and peeling 
        for(int i=0; i<actual_b_end1; i+=PacketSize*8)
        {
          EIGEN_ASM_COMMENT("SPARSELU_GEMML_KERNEL1");
          _mm_prefetch((const char*)(A0+i+(5)*PacketSize), _MM_HINT_T0);
          _mm_prefetch((const char*)(A1+i+(5)*PacketSize), _MM_HINT_T0);
          _mm_prefetch((const char*)(A2+i+(5)*PacketSize), _MM_HINT_T0);
          _mm_prefetch((const char*)(A3+i+(5)*PacketSize), _MM_HINT_T0);
          WORK(0);
          WORK(1);
          WORK(2);
          WORK(3);
          WORK(4);
          WORK(5);
          WORK(6);
          WORK(7);
        }
        // process the remaining rows with vectorization only
        for(int i=actual_b_end1; i<actual_b_end2; i+=PacketSize)
        {
          WORK(0);
        }
        // process the remaining rows without vectorization
        for(int i=actual_b_end2; i<actual_b; ++i)
        {
          C0[i] += A0[i]*Bc0[0]+A1[i]*Bc0[1]+A2[i]*Bc0[2]+A3[i]*Bc0[3];
          C1[i] += A0[i]*Bc1[0]+A1[i]*Bc1[1]+A2[i]*Bc1[2]+A3[i]*Bc1[3];
        }
        
        Bc0 += RK;
        Bc1 += RK;
#undef WORK
      } // peeled loop on k
    } // peeled loop on the columns j
    // process the last column (we now perform a matrux-vector product)
    if((n-n_end)>0)
    {
      const Scalar* Bc0 = B+(n-1)*ldb;
      
      for(int k=0; k<d_end; k+=RK)
      {
        
        // load and expand a RN x RK block of B
        Packet b00, b10, b20, b30;
        b00 = pset1<Packet>(Bc0[0]);
        b10 = pset1<Packet>(Bc0[1]);
        b20 = pset1<Packet>(Bc0[2]);
        b30 = pset1<Packet>(Bc0[3]);
        
        Packet a0, a1, a2, a3, c0, t0/*, t1*/;
        
        const Scalar* A0 = A+ib+(k+0)*lda;
        const Scalar* A1 = A+ib+(k+1)*lda;
        const Scalar* A2 = A+ib+(k+2)*lda;
        const Scalar* A3 = A+ib+(k+3)*lda;
        
        Scalar* C0 = C+ib+(n_end)*ldc;
        
        a0 = pload<Packet>(A0);
        a1 = pload<Packet>(A1);
        a2 = pload<Packet>(A2);
        a3 = pload<Packet>(A3);
        
#define WORK(I) \
          c0 = pload<Packet>(C0+i+(I)*PacketSize);   \
          KMADD(c0, a0, b00, t0);       \
          a0 = pload<Packet>(A0+i+(I+1)*PacketSize); \
          KMADD(c0, a1, b10, t0);       \
          a1 = pload<Packet>(A1+i+(I+1)*PacketSize); \
          KMADD(c0, a2, b20, t0);       \
          a2 = pload<Packet>(A2+i+(I+1)*PacketSize); \
          KMADD(c0, a3, b30, t0);       \
          a3 = pload<Packet>(A3+i+(I+1)*PacketSize); \
          pstore(C0+i+(I)*PacketSize, c0);
        
        // agressive vectorization and peeling
        for(int i=0; i<actual_b_end1; i+=PacketSize*8)
        {
          EIGEN_ASM_COMMENT("SPARSELU_GEMML_KERNEL2");
          WORK(0);
          WORK(1);
          WORK(2);
          WORK(3);
          WORK(4);
          WORK(5);
          WORK(6);
          WORK(7);
        }
        // vectorization only
        for(int i=actual_b_end1; i<actual_b_end2; i+=PacketSize)
        {
          WORK(0);
        }
        // remaining scalars
        for(int i=actual_b_end2; i<actual_b; ++i)
        {
          C0[i] += A0[i]*Bc0[0]+A1[i]*Bc0[1]+A2[i]*Bc0[2]+A3[i]*Bc0[3];
        }
        
        Bc0 += RK;
#undef WORK
      }
    }
    
    // process the last columns of A, corresponding to the last rows of B
    int rd = d-d_end;
    if(rd>0)
    {
      for(int j=0; j<n; ++j)
      {
        typedef Map<Matrix<Scalar,Dynamic,1>, Aligned > MapVector;
        typedef Map<const Matrix<Scalar,Dynamic,1>, Aligned > ConstMapVector;
        if(rd==1)      MapVector (C+j*ldc+ib,actual_b) += B[0+d_end+j*ldb] * ConstMapVector(A+(d_end+0)*lda+ib, actual_b);
        
        else if(rd==2)  MapVector(C+j*ldc+ib,actual_b) += B[0+d_end+j*ldb] * ConstMapVector(A+(d_end+0)*lda+ib, actual_b)
                                                        + B[1+d_end+j*ldb] * ConstMapVector(A+(d_end+1)*lda+ib, actual_b);
        
        else            MapVector(C+j*ldc+ib,actual_b) += B[0+d_end+j*ldb] * ConstMapVector(A+(d_end+0)*lda+ib, actual_b)
                                                        + B[1+d_end+j*ldb] * ConstMapVector(A+(d_end+1)*lda+ib, actual_b)
                                                        + B[2+d_end+j*ldb] * ConstMapVector(A+(d_end+2)*lda+ib, actual_b);
      }
    }
  
  } // blocking on the rows of A and C
}
#undef KMADD

} // namespace internal

} // namespace Eigen

#endif // EIGEN_SPARSELU_GEMM_KERNEL_H
