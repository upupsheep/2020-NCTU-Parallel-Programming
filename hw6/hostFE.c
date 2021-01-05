#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;

    // create a command queue
    cl_commandqueue myqueue;
    myqueue = clCreateCommandQueue(context, device, 0, &status);

    // create buffers on device
    int image_data_size = imageHeight * imageWidth * sizeof(float);
    int filter_data_size = filterSize * sizeof(float);
    cl_mem d_ip = clCreateBuffer(context, CL_MEM_READ_ONLY, image_data_size, NULL, &status);
    cl_mem d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_data_size, NULL, &status);
    cl_mem d_op = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_data_size, NULL, &status);

    // transfer input data to the device
    status = clEnqueueWriteBuffer(myqueue, d_ip, CL_TRUE, 0, image_data_size, inputImage, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(myqueue, d_filter, CL_TRUE, 0, filter_data_size, filter, 0, NULL, NULL);

    // create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "convolution", &status);

    // set arguments of the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_ip);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_op);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&filterWidth);

    // set local and global workgroup sizes
    size_t localws = {16, 16};
    size_t globalws = {ImageWidth, ImageHeight};

    // execute kernel
    clEnqueueNDRangeKernel(myqueue, kernel, 2, 0, globalws, localws, 0, NULL, NULL);

    // copy results from device back to host
    clEnqueueReadBuffer(context, d_op, CL_TRUE, 0, image_data_size, outputImage, NULL, NULL, NULL); // CL_TRUE: blocking read back

    // clean up
    status = clFlush(myqueue);
    status = clFinish(myqueue);
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(d_in);
    status = clReleaseMemObject(d_filter);
    status = clReleaseMemObject(d_out);
    status = clReleaseCommandQueue(myqueue);
    status = clReleaseContext(context);
}