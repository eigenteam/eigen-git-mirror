
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
  cout << "vendor id = " << string_char << endl;

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
#if 0
#define FOO(CODE) { \
  EIGEN_CPUID(abcd, CODE); \
  std::cout << "The code " << CODE << " gives " \
              << (int*)(abcd[0]) << " " << (int*)(abcd[1]) << " " \
              << (int*)(abcd[2]) << " " << (int*)(abcd[3]) << " " << std::endl; \
 }


 int abcd[5];
  abcd[4] = 0;
//   for (int a = 0; a < 5; a++) {
//     cpuid(abcd,a);
//     std::cout << "The code " << a << " gives "
//               << abcd[0] << " " << abcd[1] << " "
//               << abcd[2] << " " << abcd[3] << " " << abcd[4] << std::endl;
//   }

  FOO(0x1); std::cerr << (char*)abcd << "\n";
  FOO(0x2);
  FOO(0x2);
  FOO(0x2);
  FOO(0x3); std::cerr << (char*)abcd << "\n";
  FOO(0x80000002); std::cerr << (char*)abcd << "\n";
  FOO(0x80000003); std::cerr << (char*)abcd << "\n";
  FOO(0x80000004); std::cerr << (char*)abcd << "\n";
  FOO(0x80000005); std::cerr << (char*)abcd << "\n";
  FOO(0x80000006); std::cerr << (char*)(abcd) << "\n";
  std::cerr << "L2 : " << (abcd[2] >> 16) << "KB\n";
//   FOO(0x80000019); std::cerr << (char*)(abcd) << "\n";
  FOO(0x8000001A); std::cerr << (char*)(abcd) << "\n";

  #endif