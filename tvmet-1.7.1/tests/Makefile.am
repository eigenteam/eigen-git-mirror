EXTRA_DIST = README t.cc

EXTRA_PROGRAMS	= \
	swap \
	bad_temps \
	bad_xpr_const_ref \
	tvmet

AM_CXXFLAGS= -I$(top_srcdir)/include -I$(top_builddir)/include

swap_SOURCES = swap.cc
bad_temps_SOURCES = bad_temps.cc
bad_xpr_const_ref_SOURCES = bad_xpr_const_ref.cc
tvmet_SOURCES = tvmet.cc

tests: $(EXTRA_PROGRAMS)
