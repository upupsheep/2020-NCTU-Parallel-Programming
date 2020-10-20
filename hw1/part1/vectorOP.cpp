#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // All ones
        maskAll = _pp_init_ones();

        // All zeros
        maskIsNegative = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll);  // x = values[i];

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll);  // if (x < 0) {

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative);  //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative);  // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative);  //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
    __pp_vec_float x;
    __pp_vec_int y;
    __pp_vec_float result;  // output[i]
    __pp_vec_float threshold = _pp_vset_float(9.999999f);
    __pp_vec_int zero_int = _pp_vset_int(0);
    __pp_vec_int one_int = _pp_vset_int(1);
    __pp_mask maskAll, maskIsNegative;
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //

    // All ones
    maskAll = _pp_init_ones();  // 1111

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __pp_mask maskThreshold = _pp_init_ones();
        maskThreshold = _pp_mask_not(maskAll);  //0000

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll);   // x = values[i];
        _pp_vload_int(y, exponents + i, maskAll);  // y = exponents[i];

        // Set mask according to predicate
        _pp_vgt_int(maskIsNegative, y, zero_int, maskAll);  // if (y > 0) {

        _pp_vset_float(result, 1.f, maskAll);  // output[i] = 1.f;

        _pp_vgt_int(maskIsNegative, y, zero_int, maskAll);  // count = y - 1;
        while (_pp_cntbits(maskIsNegative)) {               // while(count > 0) {
            _pp_vgt_int(maskIsNegative, y, zero_int, maskAll);
            _pp_vsub_int(y, y, one_int, maskIsNegative);  // count--;

            _pp_vmult_float(result, result, x, maskIsNegative);  // result *= x;
        }

        // Set mask
        // All zeros
        // maskThreshold = _pp_mask_not(maskAll);  // 0000
        _pp_vgt_float(maskThreshold, result, threshold, maskAll);

        // Execute "if"
        _pp_vset_float(result, 9.999999f, maskThreshold);

        // // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //

    for (int i = 0; i < N; i += VECTOR_WIDTH) {
    }

    return 0.0;
}