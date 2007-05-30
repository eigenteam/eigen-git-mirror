# $Id: Makefile.am,v 1.22 2004/09/16 08:12:13 opetzold Exp $

AM_CXXFLAGS = -O \
	-I@top_srcdir@/include -I@top_builddir@/include

DOXY_DOC = \
	benchmark.dox \
	build.dox \
	compiler.dox \
	faq.dox \
	intro.dox \
	license.dox \
	links.dox \
	misc.dox \
	notes.dox \
	projects.dox \
	usage.dox \
	works.dox

HTML_MISC = \
	tvmet.css \
	header.html footer.html.in

LTEX_MISC = \
	tvmet.sty.in

BENCH_FIG = \
	axpy.png \
	matrix_matrix.png \
	matrix_vector.png \
	aat.png \
	ata.png

EXTRA_DIST = \
	$(DOXY_DOC) \
	$(HTML_MISC) \
	$(LTEX_MISC) \
	$(BENCH_FIG)

DISTCLEANFILES = \
	Doxyfile doxygen-warning \
	footer.html tvmet.sty

# some depencies
Doxyfile: 	Doxyfile.in
footer.html:	footer.html.in
tvmet.sty: 	tvmet.sty.in

# non-install programs for generating doxygen files
noinst_PROGRAMS	= dox_operators dox_functions
dox_operators_SOURCES = dox_operators.cc Util.h
dox_functions_SOURCES = dox_functions.cc Util.h

#
# rules for documentation, if ordered
#
if CONFIG_DOC

#
# file created on the fly
#
DOXY_BUILD = \
	changelog.dox \
	news.dox \
	credits.dox \
	install.dox \
	operators.dox \
	functions.dox

DOXY_HTML_SRC = \
	$(DOXY_DOC) \
	$(DOXY_BUILD) \
	$(patsubst %.in, %, $(HTML_MISC))

DOXY_LTEX_SRC = \
	$(patsubst %.in, %, $(LTEX_MISC))

# file creating rules; not needed every time
operators.dox: dox_operators
	./dox_operators > operators.dox

functions.dox: dox_functions
	./dox_functions > functions.dox

# file creating rules
changelog.dox: @top_srcdir@/ChangeLog
	@echo "/** \page changelog ChangeLog" > $@
	@echo "\verbatim" >> $@
	cat $< >> $@
	@echo "\endverbatim" >> $@
	@echo "*/" >> $@

news.dox: @top_srcdir@/NEWS
	@echo "/** \page news News" > $@
	@echo "\verbatim" >> $@
	cat $< >> $@
	@echo "\endverbatim" >> $@
	@echo "*/" >> $@

credits.dox: @top_srcdir@/THANKS
	@echo "/** \page credits Credits" > $@
	@echo "\verbatim" >> $@
	cat $< >> $@
	@echo "\endverbatim" >> $@
	@echo "*/" >> $@

install.dox: @top_srcdir@/INSTALL
	@echo "/** \page basic_install INSTALL" > $@
	@echo "\verbatim" >> $@
	cat $< >> $@
	@echo "\endverbatim" >> $@
	@echo "*/" >> $@


#
# doxygen API
#
DOC_API_PDF = $(PACKAGE)-$(VERSION).pdf

all-local: doxygen-api

# need's LaTeX style, since we use an own style for LaTeX
# which is required for generating formulas in HTML too.
# Doxygen LaTeX batchmode doesn't solves the problem right here.
doxygen-html-dir:
	@if test ! -d ./html; then mkdir ./html; fi

doxygen-api: doxygen-html-dir Doxyfile $(DOXY_HTML_SRC)
	@echo "Making HTML manual"
	@cp @builddir@/tvmet.sty ./html/
	@DOXYGEN@ $(DOXYGEN_OPTS)
	@$(RM) -f ./html/tvmet.sty

#
# doxygen LaTeX API
#
TEXINPUTS_PATH=$(TEXINPUTS):$(top_builddir)

noinst_DATA = $(DOC_API_PDF)

all-local: $(noinst_DATA)

dvi-local:
ps-local:
pdf-local: $(DOC_API_PDF)

$(DOC_API_PDF): Doxyfile $(DOXY_LTEX_SRC)
	@if test -d ./latex; then \
		echo "Making PDF manual"; \
		if test -f $@; then \
			$(RM) $@; \
		fi; \
		TEXINPUTS=$(TEXINPUTS_PATH) $(MAKE) -C ./latex refman.pdf; \
		$(LN_S) ./latex/refman.pdf $@; \
	fi


#
# misc hooks
#
distclean-local:
	@rm -f $(DOXY_BUILD)
	@if test -f $(DOC_API_PDF); then \
		$(RM) -f $(DOC_API_PDF); \
	fi
	@if test -d html; then \
		$(RM) -rf html; \
	fi
	@if test -d latex; then \
		$(RM) -rf latex; \
	fi
	@if test -d rtf; then \
		$(RM) -rf rtf; \
	fi
	@if test -d man; then \
		$(RM) -rf man; \
	fi

doc-dist: doxygen-api
	echo "please wait while archiving the html docs."; \
	tar cf - -C ./html . | bzip2 --best -c > $(PACKAGE)-docs-$(VERSION).tar.bz2

# where to install all html documents
TVMET_DOC_DIR = $(datadir)/doc/$(PACKAGE)-$(VERSION)

# Automake's "distcheck" is sensitive to having files left over
# after "make uninstall", so we have to clean up the install hook.
uninstall-local:
	@if test -d $(TVMET_DOC_DIR); then \
		rm -rf $(TVMET_DOC_DIR); \
	fi

# Install hooks
make-install-dirs:
	@if test '!' -d $(TVMET_DOC_DIR); then \
		$(mkinstalldirs) $(TVMET_DOC_DIR); \
	fi

install-data-hook: make-install-dirs
	@echo Installing documentations into $(TVMET_DOC_DIR)
	@echo "install html"
	@$(INSTALL_DATA) @top_builddir@/doc/html/* $(TVMET_DOC_DIR)

else
doc-dist:
install-data-hook:
endif	# CONFIG_DOC


# --------------------------------------------------------
# in progress:
# make refman twoside
twoside:
	class=`cat latex/refman.tex | grep documentclass`
	newclass=`echo $$class | sed -e 's/twoside,//' -e 's/\[/\[twoside,/'`
	@echo "class:" $$class
	@echo "newclass:" $$newclass
	cat latex/refman.tex | sed 's/$$class/$$newclass/' >foo
