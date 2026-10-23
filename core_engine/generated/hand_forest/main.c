
#include "header.h"


static const int32_t num_class[] = {  3, };

int32_t hand_get_num_target(void) {
  return N_TARGET;
}
void hand_get_num_class(int32_t* out) {
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t hand_get_num_feature(void) {
  return 24;
}
const char* hand_get_threshold_type(void) {
  return "float64";
}
const char* hand_get_leaf_output_type(void) {
  return "float64";
}

void hand_predict(union Entry* data, int pred_margin, double* result) {
  unsigned int tmp;
  hand_predict_unit0(data, result);
  hand_predict_unit1(data, result);
  hand_predict_unit2(data, result);
  hand_predict_unit3(data, result);
  
  // Average tree outputs
  result[0] /= 100;
  result[1] /= 100;
  result[2] /= 100;
  
  // Apply base_scores
  result[0] += 0;
  result[1] += 0;
  result[2] += 0;
  
  // Apply postprocessor
  if (!pred_margin) { hand_postprocess(result); }
}

void hand_postprocess(double* result) {
  // Do nothing
}

