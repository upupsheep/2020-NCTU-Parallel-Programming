__kernel void convolution(const __global float * input, 
	__global float * output,
	__global float * filter,
    int imageWidth,
    int filterWidth) 
{
    int halffilterSize = filterWidth / 2;
    
    int rowOffset = get_global_id(1) * imageWidth * 4;
    int my = 4 * get_global_id(0) + rowOffset;

    float sum = 0.0;
    int r, c;
    int fIndex = 0;
    for (r = -halffilterSize; r <= halffilterSize; r++) {
		int curRow = my + r * (imageWidth);
		for (c = -halffilterSize; c <= halffilterSize; c++) {				
			sum += input[ curRow + c] * filter[fIndex];
            fIndex++; 
		}
	}
    output[my] = sum;
}
