
#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif

int xerbla_(char * msg, int *info, int)
{
  std::cerr << "Eigen BLAS ERROR #" << *info << ": " << msg << "\n";
}

#ifdef __cplusplus
}
#endif
