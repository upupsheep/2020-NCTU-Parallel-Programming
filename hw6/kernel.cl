__kernel void convolution(const __global float * input, 
	__global float * output,
	__global float * filter,
    int filterWidth) 
{	
	int imageWidth = get_global_size(0);
	int imageHeight = get_global_size(1);

    int halffilterSize = (int)filterWidth / 2;

	int dy = get_global_id(1);
	int dx = get_global_id(0);
    
    int rowOffset = dy * imageWidth;
    int my = dx + rowOffset;

    float sum = 0.0;
    int fIndex = 0;
    for (int r = -halffilterSize; r <= halffilterSize; r++) {
		int curRow = my + r * (imageWidth);
		for (int c = -halffilterSize; c <= halffilterSize; c++) {				
			sum += input[ curRow + c] * filter[fIndex];
            fIndex++; 
		}
	}
    output[my] = sum;
}
