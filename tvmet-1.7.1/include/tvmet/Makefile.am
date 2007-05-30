# $Id: Makefile.am,v 1.10 2004/04/24 11:59:55 opetzold Exp $

SUBDIRS  = \
	config \
	meta \
	loop \
	util \
	xpr

libtvmetincludedir = $(includedir)/$(PACKAGE)

libtvmetinclude_HEADERS = \
	config.h \
	tvmet.h \
	AliasProxy.h \
	BinaryFunctionals.h \
	CompileTimeError.h \
	CommaInitializer.h \
	Extremum.h \
	Functional.h \
	Io.h \
	Matrix.h \
	MatrixImpl.h \
	MatrixFunctions.h \
	MatrixBinaryFunctions.h \
	MatrixUnaryFunctions.h \
	MatrixOperators.h \
	MatrixEval.h \
	NumericTraits.h \
	TypePromotion.h \
	RunTimeError.h \
	TvmetBase.h \
	UnaryFunctionals.h \
	Vector.h \
	VectorImpl.h \
	VectorEval.h \
	VectorFunctions.h \
	VectorBinaryFunctions.h \
	VectorUnaryFunctions.h \
	VectorOperators.h

DISTCLEANFILES = config.h

dist-hook:
	rm -f $(distdir)/config.h
