
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

#define N_TARGET 1
#define MAX_N_CLASS 3

union Entry {
  int missing;
  double fvalue;
  int qvalue;
};

__declspec(dllexport) int32_t hand_get_num_target(void);
__declspec(dllexport) void hand_get_num_class(int32_t* out);
__declspec(dllexport) int32_t hand_get_num_feature(void);
__declspec(dllexport) const char* hand_get_threshold_type(void);
__declspec(dllexport) const char* hand_get_leaf_output_type(void);
__declspec(dllexport) void hand_predict(union Entry* data, int pred_margin, double* result);
void hand_postprocess(double* result);
void hand_predict_unit0(union Entry* data, double* result);
void hand_predict_unit1(union Entry* data, double* result);
void hand_predict_unit2(union Entry* data, double* result);
void hand_predict_unit3(union Entry* data, double* result);

