macro(add_benchmark benchmark_name benchmark_file)
  add_executable(${benchmark_name} ${benchmark_file})
#  trajopt_target_compile_options(${benchmark_name} PRIVATE)
  target_link_libraries(${benchmark_name} PUBLIC
      ArrayFire::af
      benchmark::benchmark
      "${TORCH_LIBRARIES}"
      )
  target_include_directories(${benchmark_name} PRIVATE
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
  target_include_directories(${benchmark_name} SYSTEM PUBLIC
      ${EIGEN3_INCLUDE_DIRS}
      ${Boost_INCLUDE_DIRS})
  add_dependencies(run_benchmarks ${benchmark_name})
endmacro()


# This macro add a custom target that will run the benchmarks after they are finished building.
# This is added to allow ability do disable the running of benchmarks as part of the build for CI which calls make test
# TODO: Fix this
macro(add_run_benchmarks_target)
  if(ENABLE_RUN_BENCHMARKING)
    add_custom_target(run_benchmarks ALL
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_CTEST_COMMAND} -V -O "/tmp/${PROJECT_NAME}_ctest.log" -C $<CONFIGURATION>)
  else()
    add_custom_target(run_benchmarks)
  endif()
endmacro()
