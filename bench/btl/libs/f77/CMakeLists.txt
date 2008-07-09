
enable_language(Fortran)
btl_add_bench(btl_f77 main.cpp dmxv.f smxv.f dmxm.f smxm.f daxpy.f saxpy.f data.f sata.f daat.f saat.f)
