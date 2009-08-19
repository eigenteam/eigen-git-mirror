
include(EigenTesting)

enable_testing()

find_package(Adolc)

include_directories(../../test)

if(ADOLC_FOUND)
  include_directories(${ADOLC_INCLUDES})
  ei_add_property(EIGEN_TESTED_BACKENDS "Adolc")
  ei_add_test(forward_adolc " " ${ADOLC_LIBRARIES})
else(ADOLC_FOUND)
  ei_add_property(EIGEN_MISSING_BACKENDS "Adolc")
endif(ADOLC_FOUND)

# temporary : 
include_directories(/home/orzel/tmp/cminpack-1.0.2/)
LINK_LIBRARIES(/home/orzel/tmp/cminpack-1.0.2/libminpack.a)

ei_add_test(NonLinear)
ei_add_test(autodiff)
ei_add_test(BVH)
#ei_add_test(matrixExponential)
ei_add_test(alignedvector3)
