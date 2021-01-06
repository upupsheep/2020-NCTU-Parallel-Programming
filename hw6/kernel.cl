__kernel void convolution(const __global float * input, 
	__global float * output,
	__global float * filter,
    int filterWidth,
	int imageHeight,
	int imageWidth) 
{	

    int halffilterSize = filterWidth / 2;

	int index = (get_global_id(0) * imageHeight) + get_global_id(1);
	int i = index / imageWidth;
	int j = index - (i * imageWidth);
    

    float sum = 0.0;
    
	for (int r = -halffilterSize; r <= halffilterSize; r++) {
		for (int c = -halffilterSize; c <= halffilterSize; c++) {
			if (i + r >= 0 && i + r < imageHeight &&
				j + c >= 0 && j + c < imageWidth)
			{
				sum += input[(i + r) * imageWidth + j + c] * filter[(r + halffilterSize) * filterWidth + c + halffilterSize];
			}
		}
	}
    output[index] = sum;
}
