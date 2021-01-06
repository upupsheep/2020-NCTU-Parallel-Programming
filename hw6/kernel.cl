__kernel void convolution(const __global float * input, 
	__global float * output,
	__global float * filter,
    int filterWidth) 
{	
	int imageWidth = get_global_size(0);
	int imageHeight = get_global_size(1);

    int halffilterSize = (int)filterWidth / 2;

	int i = get_global_id(1);
	int j = get_global_id(0);
    
    int rowOffset = i * imageWidth;
    int my = j + rowOffset;

    float sum = 0.0;
    int fIndex = 0;
    for (int r = -halffilterSize; r <= halffilterSize; r++) {
		int curRow = my + r * (imageWidth);
		for (int c = -halffilterSize; c <= halffilterSize; c++) {				
			sum += input[(i + r) * imageWidth + j + c] * filter[(r + halffilterSize) * filterWidth + c + halffilterSize];
		}
	}
    output[i * imageWidth + j] = sum;
}
