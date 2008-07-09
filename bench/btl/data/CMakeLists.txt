
ADD_CUSTOM_TARGET(copy_scripts)

SET(script_files go_mean aat.hh ata.hh axpy.hh order_lib mk_mean_script.sh mk_new_gnuplot.sh mk_gnuplot_script.sh matrix_matrix.hh matrix_vector.hh atv.hh)

FOREACH(script_file ${script_files})
ADD_CUSTOM_COMMAND(
  TARGET copy_scripts
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${script_file} ${CMAKE_CURRENT_BINARY_DIR}/
  ARGS
)
ENDFOREACH(script_file)

add_executable(smooth smooth.cxx)
add_executable(regularize regularize.cxx)
add_executable(main mean.cxx)
add_dependencies(main copy_scripts)
