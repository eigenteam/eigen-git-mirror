# $Id: Makefile.am,v 1.12 2004/04/28 21:59:40 opetzold Exp $

AM_CXXFLAGS = -I$(top_srcdir)/include -I$(top_builddir)/include

EXTRA_PROGRAMS	= \
	xpr_print_v1 \
	xpr_print_v2 \
	xpr_print_v3 \
	xpr_print_v4 \
	xpr_print_m1 \
	xpr_print_m2 \
	xpr_print_m3 \
	xpr_print_m4 \
	xpr_print_mv1 \
	xpr_print_mv2 \
	mv \
	mm \
	cmv \
	cmm \
	matrix_col \
	diag \
	ray \
	hspiess \
	redwards \
	frob_matrix_norm \
	alias \
	aliasing

DISTCLEANFILES = $(EXTRA_PROGRAMS)

xpr_print_v1_SOURCES = xpr_print_v1.cc
xpr_print_v2_SOURCES = xpr_print_v2.cc
xpr_print_v3_SOURCES = xpr_print_v3.cc
xpr_print_v4_SOURCES = xpr_print_v4.cc
xpr_print_m1_SOURCES = xpr_print_m1.cc
xpr_print_m2_SOURCES = xpr_print_m2.cc
xpr_print_m3_SOURCES = xpr_print_m3.cc
xpr_print_m4_SOURCES = xpr_print_m4.cc
xpr_print_mv1_SOURCES = xpr_print_mv1.cc
xpr_print_mv2_SOURCES = xpr_print_mv2.cc

mv_SOURCES = mv.cc
mm_SOURCES = mm.cc
cmv_SOURCES = cmv.cc
cmm_SOURCES = cmm.cc
matrix_col_SOURCES = matrix_col.cc
diag_SOURCES = diag.cc

ray_SOURCES = ray.cc

redwards_SOURCES = redwards.cc
hspiess_SOURCES = hspiess.cc

frob_matrix_norm_SOURCES = frob_matrix_norm.cc

alias_SOURCES = alias.cc
aliasing_SOURCES = aliasing.cc

examples: $(EXTRA_PROGRAMS)
