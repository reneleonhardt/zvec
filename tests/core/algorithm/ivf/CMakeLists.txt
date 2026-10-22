include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)

file(GLOB_RECURSE ALL_TEST_SRCS *_test.cc)

foreach(CC_SRCS ${ALL_TEST_SRCS})
  get_filename_component(CC_TARGET ${CC_SRCS} NAME_WE)
  cc_gtest(
      NAME ${CC_TARGET}
      STRICT
      LIBS zvec_ailego core_framework core_utility core_metric core_quantizer core_knn_cluster core_knn_flat core_knn_ivf zvec_turbo
      SRCS ${CC_SRCS}
      INCS . ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/core ${PROJECT_ROOT_DIR}/src/core/algorithm/ivf
    )
  if(CC_TARGET STREQUAL "ivf_index_provider_test" OR CC_TARGET STREQUAL "ivf_searcher_test")
    target_include_directories(${CC_TARGET} PRIVATE ${PROJECT_ROOT_DIR}/src/turbo)
  endif()
endforeach()
