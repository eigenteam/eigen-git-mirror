# $Id: Makefile.am,v 1.4 2005/03/09 11:06:21 opetzold Exp $

if CONFIG_CPPUNIT

TESTS = driver

check_PROGRAMS = driver

driver_SOURCES = \
        main.cc \
        SelfTest.h SelfTest.cc \
	TestNumericTraits.h TestNumericTraits.cc \
	TestNumericTraitsComplex.h TestNumericTraitsComplex.cc \
        TestConstruction.h TestConstruction.cc \
        TestComplexMatrix.h TestComplexMatrix.cc \
        TestComplexVector.h TestComplexVector.cc \
        TestDimension.h TestDimension.cc \
        TestMathOps.h TestMathOps.cc \
        TestMatrixEval.h TestMatrixEval.cc \
        TestMatrixFunctions.h TestMatrixFunctions.cc \
        TestMatrixOperators.h TestMatrixOperators.cc \
        TestSTL.h TestSTL.cc \
        TestUnFunc.h TestUnFunc.cc \
        TestVectorEval.h TestVectorEval.cc \
        TestVectorFunctions.h TestVectorFunctions.cc \
        TestVectorOperators.h TestVectorOperators.cc \
        TestXpr.h TestXpr.cc \
        TestXprMatrixFunctions.h TestXprMatrixFunctions.cc \
        TestXprMatrixOperators.h TestXprMatrixOperators.cc \
        TestXprVectorFunctions.h TestXprVectorFunctions.cc \
        TestXprVectorOperators.h TestXprVectorOperators.cc \
        TestUnloops.h TestUnloops.cc \
	TestMatrixElementwise.h TestMatrixElementwise.cc

AM_CXXFLAGS = $(CXX_DEBUG_FLAGS) $(CXX_WARN_FLAGS) \
        -I$(top_srcdir)/include -I$(top_builddir)/include \
        $(CPPUNIT_CFLAGS)

driver_LDADD = $(CPPUNIT_LIBS)

endif
