EXTRA_DIST = README t.cc

EXTRA_PROGRAMS	= \
	swap \
	bad_temps \
	bad_xpr_constRef \
	tvmet

AM_CXXFLAGS= -I$(top_srcdir)/include -I$(top_builddir)/include

swap_SOURCES = swap.cc
bad_temps_SOURCES = bad_temps.cc
bad_xpr_constRef_SOURCES = bad_xpr_constRef.cc
tvmet_SOURCES = tvmet.cc

tests: $(EXTRA_PROGRAMS)
