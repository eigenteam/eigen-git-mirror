
#define EIGEN_INTERNAL_DEBUG_CACHE_QUERY
#include <iostream>
#include "../Eigen/Core"

using namespace Eigen;
using namespace std;

#define DUMP_CPUID(CODE) {\
  int abcd[4]; \
  abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;\
  EIGEN_CPUID(abcd, CODE, 0); \
  std::cout << "The code " << CODE << " gives " \
              << (int*)(abcd[0]) << " " << (int*)(abcd[1]) << " " \
              << (int*)(abcd[2]) << " " << (int*)(abcd[3]) << " " << std::endl; \
  }
  
int main()
{
  cout << "Eigen's L1    = " << ei_queryL1CacheSize() << endl;
  cout << "Eigen's L2/L3 = " << ei_queryTopLevelCacheSize() << endl;
  int l1, l2, l3;
  ei_queryCacheSizes(l1, l2, l3);
  cout << "Eigen's L1, L2, L3       = " << l1 << " " << l2 << " " << l3 << endl;
  
  #ifdef EIGEN_CPUID

  ei_queryCacheSizes_intel(l1, l2, l3);
  cout << "Eigen's intel L1, L2, L3 = " << l1 << " " << l2 << " " << l3 << endl;
  ei_queryCacheSizes_amd(l1, l2, l3);
  cout << "Eigen's amd L1, L2, L3   = " << l1 << " " << l2 << " " << l3 << endl;

  int abcd[4];
  int string[8];
  char* string_char = (char*)(string);

  // vendor ID
  EIGEN_CPUID(abcd,0x0,0);
  string[0] = abcd[1];
  string[1] = abcd[3];
  string[2] = abcd[2];
  string[3] = 0;
  cout << endl;
  cout << "vendor id = " << string_char << endl;
  cout << endl;

  // dump Intel direct method
  {
    l1 = l2 = l3 = 0;
    int cache_id = 0;
    int cache_type = 0;
    do {
      abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
      EIGEN_CPUID(abcd,0x4,cache_id);
      cache_type  = (abcd[0] & 0x0F) >> 0;
      int cache_level = (abcd[0] & 0xE0) >> 5;  // A[7:5]
      int ways        = (abcd[1] & 0xFFC00000) >> 22; // B[31:22]
      int partitions  = (abcd[1] & 0x003FF000) >> 12; // B[21:12]
      int line_size   = (abcd[1] & 0x00000FFF) >>  0; // B[11:0]
      int sets        = (abcd[2]);                    // C[31:0]
      int cache_size = (ways+1) * (partitions+1) * (line_size+1) * (sets+1);
      
      cout << "cache[" << cache_id << "].type       = " << cache_type << "\n";
      cout << "cache[" << cache_id << "].level      = " << cache_level << "\n";
      cout << "cache[" << cache_id << "].ways       = " << ways << "\n";
      cout << "cache[" << cache_id << "].partitions = " << partitions << "\n";
      cout << "cache[" << cache_id << "].line_size  = " << line_size << "\n";
      cout << "cache[" << cache_id << "].sets       = " << sets << "\n";
      cout << "cache[" << cache_id << "].size       = " << cache_size << "\n";
      
      cache_id++;
    } while(cache_type>0);
  }

  // manual method for intel
  {
    l1 = l2 = l3 = 0;
    abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
    EIGEN_CPUID(abcd,0x00000002,0);
    unsigned char * bytes = reinterpret_cast<unsigned char *>(abcd)+2;
    for(int i=0; i<14; ++i)
    {
      switch(bytes[i])
      {
        case 0x0A: l1 = 8; break;   // 0Ah   data L1 cache, 8 KB, 2 ways, 32 byte lines
        case 0x0C: l1 = 16; break;  // 0Ch   data L1 cache, 16 KB, 4 ways, 32 byte lines
        case 0x0E: l1 = 24; break;  // 0Eh   data L1 cache, 24 KB, 6 ways, 64 byte lines
        case 0x10: l1 = 16; break;  // 10h   data L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
        case 0x15: l1 = 16; break;  // 15h   code L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
        case 0x2C: l1 = 32; break;  // 2Ch   data L1 cache, 32 KB, 8 ways, 64 byte lines
        case 0x30: l1 = 32; break;  // 30h   code L1 cache, 32 KB, 8 ways, 64 byte lines
  // 56h   L0 data TLB, 4M pages, 4 ways, 16 entries
  // 57h   L0 data TLB, 4K pages, 4 ways, 16 entries
  // 59h   L0 data TLB, 4K pages, fully, 16 entries
        case 0x60: l1 = 16; break;  // 60h   data L1 cache, 16 KB, 8 ways, 64 byte lines, sectored
        case 0x66: l1 = 8; break;   // 66h   data L1 cache, 8 KB, 4 ways, 64 byte lines, sectored
        case 0x67: l1 = 16; break;  // 67h   data L1 cache, 16 KB, 4 ways, 64 byte lines, sectored
        case 0x68: l1 = 32; break;  // 68h   data L1 cache, 32 KB, 4 ways, 64 byte lines, sectored
  // 77h   code L1 cache, 16 KB, 4 ways, 64 byte lines, sectored (IA-64)
  // 96h   data L1 TLB, 4K...256M pages, fully, 32 entries (IA-64)


        case 0x1A: l2 = 96; break;   // code and data L2 cache, 96 KB, 6 ways, 64 byte lines (IA-64)
        case 0x22: l3 = 512; break;   // code and data L3 cache, 512 KB, 4 ways (!), 64 byte lines, dual-sectored
        case 0x23: l3 = 1024; break;   // code and data L3 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x25: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x29: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x39: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 64 byte lines, sectored
        case 0x3A: l2 = 192; break;   // code and data L2 cache, 192 KB, 6 ways, 64 byte lines, sectored
        case 0x3B: l2 = 128; break;   // code and data L2 cache, 128 KB, 2 ways, 64 byte lines, sectored
        case 0x3C: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 64 byte lines, sectored
        case 0x3D: l2 = 384; break;   // code and data L2 cache, 384 KB, 6 ways, 64 byte lines, sectored
        case 0x3E: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines, sectored
        case 0x40: l2 = 0; break;   // no integrated L2 cache (P6 core) or L3 cache (P4 core)
        case 0x41: l2 = 128; break;   // code and data L2 cache, 128 KB, 4 ways, 32 byte lines
        case 0x42: l2 = 256; break;   // code and data L2 cache, 256 KB, 4 ways, 32 byte lines
        case 0x43: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 32 byte lines
        case 0x44: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 32 byte lines
        case 0x45: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 4 ways, 32 byte lines
        case 0x46: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines
        case 0x47: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 8 ways, 64 byte lines
        case 0x48: l2 = 3072; break;   // code and data L2 cache, 3072 KB, 12 ways, 64 byte lines
        case 0x49: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 16 ways, 64 byte lines (P4) or
        case 0x4A: l3 = 6144; break;   // code and data L3 cache, 6144 KB, 12 ways, 64 byte lines
        case 0x4B: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 16 ways, 64 byte lines
        case 0x4C: l3 = 12288; break;   // code and data L3 cache, 12288 KB, 12 ways, 64 byte lines
        case 0x4D: l3 = 16384; break;   // code and data L3 cache, 16384 KB, 16 ways, 64 byte lines
        case 0x4E: l2 = 6144; break;   // code and data L2 cache, 6144 KB, 24 ways, 64 byte lines
        case 0x78: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 4 ways, 64 byte lines
        case 0x79: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x7A: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x7B: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x7C: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
        case 0x7D: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 64 byte lines
        case 0x7E: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 128 byte lines, sect. (IA-64)
        case 0x7F: l2 = 512; break;   // code and data L2 cache, 512 KB, 2 ways, 64 byte lines
        case 0x80: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 64 byte lines
        case 0x81: l2 = 128; break;   // code and data L2 cache, 128 KB, 8 ways, 32 byte lines
        case 0x82: l2 = 256; break;   // code and data L2 cache, 256 KB, 8 ways, 32 byte lines
        case 0x83: l2 = 512; break;   // code and data L2 cache, 512 KB, 8 ways, 32 byte lines
        case 0x84: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 32 byte lines
        case 0x85: l2 = 2048; break;   // code and data L2 cache, 2048 KB, 8 ways, 32 byte lines
        case 0x86: l2 = 512; break;   // code and data L2 cache, 512 KB, 4 ways, 64 byte lines
        case 0x87: l2 = 1024; break;   // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines
        case 0x88: l3 = 2048; break;   // code and data L3 cache, 2048 KB, 4 ways, 64 byte lines (IA-64)
        case 0x89: l3 = 4096; break;   // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines (IA-64)
        case 0x8A: l3 = 8192; break;   // code and data L3 cache, 8192 KB, 4 ways, 64 byte lines (IA-64)
        case 0x8D: l3 = 3072; break;   // code and data L3 cache, 3072 KB, 12 ways, 128 byte lines (IA-64)
        case 0x9B: l2 = 1024; break;   // data L2 TLB, 4K...256M pages, fully, 96 entries (IA-64)
                
        default: break;
      }
    }
    cout << endl;
    cout << "tedious way l1 = " << l1 << endl;
    cout << "tedious way l2 = " << l2 << endl;
    cout << "tedious way l3 = " << l3 << endl;
  }
  
  // dump everything
  std::cout << endl <<"Raw dump:" << endl;
  DUMP_CPUID(0x0);
  DUMP_CPUID(0x1);
  DUMP_CPUID(0x2);
  DUMP_CPUID(0x3);
  DUMP_CPUID(0x4);
  DUMP_CPUID(0x5);
  DUMP_CPUID(0x6);
  DUMP_CPUID(0x80000000);
  DUMP_CPUID(0x80000001);
  DUMP_CPUID(0x80000002);
  DUMP_CPUID(0x80000003);
  DUMP_CPUID(0x80000004);
  DUMP_CPUID(0x80000005);
  DUMP_CPUID(0x80000006);
  DUMP_CPUID(0x80000007);
  DUMP_CPUID(0x80000008);
  #else
  cout << "EIGEN_CPUID is not defined" << endl;
  #endif
  return 0;
}
