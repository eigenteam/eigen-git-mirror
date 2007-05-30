# $Id: Makefile.am,v 1.25 2004/09/16 08:12:28 opetzold Exp $

ACLOCAL_AMFLAGS  = -I config

SUBDIRS = include examples testsuite doc tests benchmark

m4sources = \
	config/ac_c_long_long.m4 \
	config/ac_create_prefix_config_h.m4 \
	config/ac_cxx_have_complex.m4 \
	config/ac_cxx_have_complex_math1.m4 \
	config/ac_cxx_have_complex_math2.m4 \
	config/ac_cxx_have_ieee_math.m4 \
	config/ac_cxx_have_mutable.m4 \
	config/ac_cxx_have_namespaces.m4 \
	config/ac_cxx_have_sysv_math.m4 \
	config/ac_cxx_partial_specialization.m4 \
	config/ac_cxx_typename.m4 \
	config/ac_set_compiler.m4 \
	config/ac_prog_cxx_intel.m4 \
	config/ac_prog_cxx_kai.m4 \
	config/ac_prog_cxx_pgi.m4 \
	config/op_doxygen_doc.m4

# m4 automake macros
m4datadir = $(datadir)/aclocal
m4data_DATA = tvmet.m4

# setup/config scripts
bin_SCRIPTS = tvmet-config

# manual for script
man_MANS = tvmet-config.1

EXTRA_DIST = \
	$(m4sources) \
	$(m4data_DATA) \
	$(man_MANS) \
	ChangeLog.1 \
	LICENSE

DISTCLEANFILES = \
	$(PACKAGE)-docs-$(VERSION).tar.bz2 \
	$(PACKAGE)-$(VERSION).pdf.bz2

# Targets
ChangeLog:
	@if test -d CVS -a -x `which cvs2log`; then \
		echo "Create ChangeLog from CVS"; \
		`which cvs2log`; \
	fi

dist-hook: ChangeLog
	cp tvmet.spec $(distdir)
	$(MAKE) doc-dist
	@if test -f $(top_builddir)/doc/$(PACKAGE)-$(VERSION).ps; then \
		$(LN_S) $(top_builddir)/doc/$(PACKAGE)-$(VERSION).ps $(PACKAGE)-$(VERSION).ps; \
		cat $(top_builddir)/doc/$(PACKAGE)-$(VERSION).ps | bzip2 --best > $(PACKAGE)-$(VERSION).ps.bz2; \
	fi
	@if test -f $(top_builddir)/doc/$(PACKAGE)-$(VERSION).pdf; then \
		$(LN_S) $(top_builddir)/doc/$(PACKAGE)-$(VERSION).pdf $(PACKAGE)-$(VERSION).pdf; \
		cat $(top_builddir)/doc/$(PACKAGE)-$(VERSION).pdf | bzip2 --best > $(PACKAGE)-$(VERSION).pdf.bz2; \
	fi

.PHONY: release snapshot rpm docs doc-dist

release:
	rm -rf .deps */.deps
	$(MAKE) distcheck

snapshot:
	$(MAKE) dist distdir=$(PACKAGE)-$(VERSION)-`date +"%y%m%d"`

docs:
	$(MAKE) -C doc

if CONFIG_DOC
doc-dist: docs
	$(MAKE) -C doc doc-dist
	mv -f doc/$(PACKAGE)-docs-$(VERSION).tar.bz2 .
else
doc-dist:
endif

example:
	$(MAKE) -C examples example

rpm: dist
	@if test -f $(PACKAGE).spec; then \
		echo "Build rpm distribution"; \
		$(RM) $(PACKAGE).spec && make $(PACKAGE).spec; \
	fi
	rpm -ta --target noarch $(PACKAGE)-$(VERSION).tar.bz2
	cp ~/RPM/SRPMS/$(PACKAGE)-$(VERSION)-1.src.rpm .
	cp ~/RPM/RPMS/noarch/$(PACKAGE)-$(VERSION)-1.noarch.rpm .

md5:
	md5sum `find . -maxdepth 1 -name "*.rpm" -or -name "*.gz" -or -name "*.bz2" -xtype f` > MD5sums

sign:	md5
	cat MD5sums | gpg --clearsign > .MD5sums && mv .MD5sums MD5sums
