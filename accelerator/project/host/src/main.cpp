#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include "darknet.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;

using namespace aocl_utils;

cl_int status;

cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device;
cl_context context = NULL;
cl_command_queue queue;
cl_program program = NULL;

cl_kernel conv3x3, conv1x1, bn, pool;


cl_mem d_sample, d_conv0_weight, d_conv0_out;
cl_mem d_bn0_weight, d_bn0_bias, d_bn0_mean, d_bn0_var, d_bn0_out;
cl_mem d_pool1_out;
cl_mem d_conv2_weight, d_conv2_out;
cl_mem d_bn2_weight, d_bn2_bias, d_bn2_mean, d_bn2_var, d_bn2_out;
cl_mem d_pool3_out;
cl_mem d_conv4_weight, d_conv4_out;
cl_mem d_bn4_weight, d_bn4_bias, d_bn4_mean, d_bn4_var, d_bn4_out;
cl_mem d_pool5_out;
cl_mem d_conv6_weight, d_conv6_out;
cl_mem d_bn6_weight, d_bn6_bias, d_bn6_mean, d_bn6_var, d_bn6_out;
cl_mem d_pool7_out;
cl_mem d_conv8_weight, d_conv8_out;
cl_mem d_bn8_weight, d_bn8_bias, d_bn8_mean, d_bn8_var, d_bn8_out;
cl_mem d_pool9_out;
cl_mem d_conv10_weight, d_conv10_out;
cl_mem d_bn10_weight, d_bn10_bias, d_bn10_mean, d_bn10_var, d_bn10_out;
cl_mem d_pool11_out;
cl_mem d_conv12_weight, d_conv12_out;
cl_mem d_bn12_weight, d_bn12_bias, d_bn12_mean, d_bn12_var, d_bn12_out;
cl_mem d_pool13_out;
cl_mem d_conv14_weight, d_conv14_out;
cl_mem d_bn14_weight, d_bn14_bias, d_bn14_mean, d_bn14_var, d_bn14_out;

cl_event conv0_event, bn0_event, pool1_event;
cl_event conv2_event, bn2_event, pool3_event;
cl_event conv4_event, bn4_event, pool5_event;
cl_event conv6_event, bn6_event, pool7_event;
cl_event conv8_event, bn8_event, pool9_event;
cl_event conv10_event, bn10_event, pool11_event;
cl_event conv12_event, bn12_event, pool13_event;
cl_event conv14_event, bn14_event;

float* conv0_weight = (float* )malloc(32*output_channels_0*sizeof(float));
float* bn0_weight = (float* )malloc(output_channels_0*sizeof(float));
float* bn0_mean = (float* )malloc(output_channels_0*sizeof(float));
float* bn0_var = (float* )malloc(output_channels_0*sizeof(float));
float* bn0_bias = (float* )malloc(output_channels_0*sizeof(float));

float* conv2_weight = (float* )malloc(input_channels_2*3*3*output_channels_2*sizeof(float));
float* bn2_weight = (float* )malloc(output_channels_2*sizeof(float));
float* bn2_mean = (float* )malloc(output_channels_2*sizeof(float));
float* bn2_var = (float* )malloc(output_channels_2*sizeof(float));
float* bn2_bias = (float* )malloc(output_channels_2*sizeof(float));

float* conv4_weight = (float* )malloc(input_channels_4*3*3*output_channels_4*sizeof(float));
float* bn4_weight = (float* )malloc(output_channels_4*sizeof(float));
float* bn4_mean = (float* )malloc(output_channels_4*sizeof(float));
float* bn4_var = (float* )malloc(output_channels_4*sizeof(float));
float* bn4_bias = (float* )malloc(output_channels_4*sizeof(float));

float* conv6_weight = (float* )malloc(input_channels_6*3*3*output_channels_6*sizeof(float));
float* bn6_weight = (float* )malloc(output_channels_6*sizeof(float));
float* bn6_mean = (float* )malloc(output_channels_6*sizeof(float));
float* bn6_var = (float* )malloc(output_channels_6*sizeof(float));
float* bn6_bias = (float* )malloc(output_channels_6*sizeof(float));

float* conv8_weight = (float* )malloc(input_channels_8*3*3*output_channels_8*sizeof(float));
float* bn8_weight = (float* )malloc(output_channels_8*sizeof(float));
float* bn8_mean = (float* )malloc(output_channels_8*sizeof(float));
float* bn8_var = (float* )malloc(output_channels_8*sizeof(float));
float* bn8_bias = (float* )malloc(output_channels_8*sizeof(float));

float* conv10_weight = (float* )malloc(input_channels_10*3*3*output_channels_10*sizeof(float));
float* bn10_weight = (float* )malloc(output_channels_10*sizeof(float));
float* bn10_mean = (float* )malloc(output_channels_10*sizeof(float));
float* bn10_var = (float* )malloc(output_channels_10*sizeof(float));
float* bn10_bias = (float* )malloc(output_channels_10*sizeof(float));

float* conv12_weight = (float* )malloc(input_channels_12*3*3*output_channels_12*sizeof(float));
float* bn12_weight = (float* )malloc(output_channels_12*sizeof(float));
float* bn12_mean = (float* )malloc(output_channels_12*sizeof(float));
float* bn12_var = (float* )malloc(output_channels_12*sizeof(float));
float* bn12_bias = (float* )malloc(output_channels_12*sizeof(float));

float* conv14_weight = (float* )malloc(input_channels_14*1*1*output_channels_14*sizeof(float));
float* bn14_weight = (float* )malloc(output_channels_14*sizeof(float));
float* bn14_mean = (float* )malloc(output_channels_14*sizeof(float));
float* bn14_var = (float* )malloc(output_channels_14*sizeof(float));
float* bn14_bias = (float* )malloc(output_channels_14*sizeof(float));

char image_path[] = "cat.jpg";

char conv0_wt_file[] = "conv0_weight.txt";
char bn0_wt_file[]   = "bn0_weight.txt";
char bn0_mean_file[] = "bn0_mean.txt";
char bn0_var_file[]  = "bn0_var.txt";
char bn0_bias_file[] = "bn0_bias.txt";

char conv2_wt_file[] = "conv2_weight.txt";
char bn2_wt_file[]   = "bn2_weight.txt";
char bn2_mean_file[] = "bn2_mean.txt";
char bn2_var_file[]  = "bn2_var.txt";
char bn2_bias_file[] = "bn2_bias.txt";

char conv4_wt_file[] = "conv4_weight.txt";
char bn4_wt_file[]   = "bn4_weight.txt";
char bn4_mean_file[] = "bn4_mean.txt";
char bn4_var_file[]  = "bn4_var.txt";
char bn4_bias_file[] = "bn4_bias.txt";

char conv6_wt_file[] = "conv6_weight.txt";
char bn6_wt_file[]   = "bn6_weight.txt";
char bn6_mean_file[] = "bn6_mean.txt";
char bn6_var_file[]  = "bn6_var.txt";
char bn6_bias_file[] = "bn6_bias.txt";

char conv8_wt_file[] = "conv8_weight.txt";
char bn8_wt_file[]   = "bn8_weight.txt";
char bn8_mean_file[] = "bn8_mean.txt";
char bn8_var_file[]  = "bn8_var.txt";
char bn8_bias_file[] = "bn8_bias.txt";

char conv10_wt_file[] = "conv10_weight.txt";
char bn10_wt_file[]   = "bn10_weight.txt";
char bn10_mean_file[] = "bn10_mean.txt";
char bn10_var_file[]  = "bn10_var.txt";
char bn10_bias_file[] = "bn10_bias.txt";

char conv12_wt_file[] = "conv12_weight.txt";
char bn12_wt_file[]   = "bn12_weight.txt";
char bn12_mean_file[] = "bn12_mean.txt";
char bn12_var_file[]  = "bn12_var.txt";
char bn12_bias_file[] = "bn12_bias.txt";

char conv14_wt_file[] = "conv14_weight.txt";
char bn14_wt_file[]   = "bn14_weight.txt";
char bn14_mean_file[] = "bn14_mean.txt";
char bn14_var_file[]  = "bn14_var.txt";
char bn14_bias_file[] = "bn14_bias.txt";

const float eps = 0.00001;

float* image = (float *)malloc(input_size_0*input_size_0*input_channels_0*sizeof(float));
float* image_data = (float *)malloc(input_size_0*input_size_0*input_channels_0* sizeof(float));

float *bn14_result = (float *)malloc(output_channels_14* sizeof(float));
float* softmax_out = (float *)malloc(output_channels_14* sizeof(float));

char label[200];
float accuracy;

void load_image();
void softmax();
void get_results();
void cleanup();

int main() {

	printf("Initializing OpenCL\n");

	if(!setCwdToExeDir()) {
		return 1;
	}

	platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
	if(platform == NULL) {
		printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform. \n");
		return 1;
	}

	device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
	printf("Platform: %s\n", getPlatformName(platform).c_str());
	printf("Using %d device(s)\n", num_devices);
    for(unsigned int i = 0; i < num_devices; ++i) {
		printf("  %s\n", getDeviceName(device[i]).c_str());
	}

	context = clCreateContext(NULL, num_devices, device, NULL, NULL, &status);
	checkError(status, "Failed to create context");

	std::string binary_file = getBoardBinaryFile("darknet", device[0]);
	printf("Using AOCX: %s\n", binary_file.c_str());
	program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	checkError(status, "Failed to build program");

	queue = clCreateCommandQueue(context, device[0], CL_QUEUE_PROFILING_ENABLE, &status);
	checkError(status, "Failed to create command queue");


	conv3x3 = clCreateKernel(program, "conv" , &status);
	checkError(status, "Failed to create kernel conv3x3");

	conv1x1 = clCreateKernel(program, "conv1x1" , &status);
	checkError(status, "Failed to create kernel conv1x1");

	bn = clCreateKernel(program, "batchnorm" , &status);
	checkError(status, "Failed to create kernel batchnorm");

	pool = clCreateKernel(program, "pool", &status);
	checkError(status, "Failed to create kernel pool");

// Create device buffers

	// dinput image
	d_sample = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_size_0*input_size_0*input_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_sample buffer");
	// conv3x3 0
	d_conv0_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				32*output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv0_weight buffer");
	d_conv0_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_0*output_size_0*output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv0_out buffer");
	// bn 0
	d_bn0_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn0_weight buffer");
	d_bn0_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn0_bias buffer");
	d_bn0_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn0_mean buffer");
	d_bn0_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn0_var buffer");
	d_bn0_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		    output_size_0*output_size_0*output_channels_0*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn0_out buffer");
	// pool 1
	d_pool1_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_1*output_size_1*input_channels_1*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool1_out buffer");
	// conv3x3 2
	d_conv2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_2*3*3*output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv2_weight buffer");
	d_conv2_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_2*output_size_2*output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv2_out buffer");
	// bn 2
	d_bn2_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn2_weight buffer");
	d_bn2_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn2_bias buffer");
	d_bn2_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn2_mean buffer");
	d_bn2_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_2*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn2_var buffer");
	d_bn2_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
					 output_size_2*output_size_2*output_channels_2*sizeof(float), NULL, &status);
					 checkError(status, "Failed to create d_bn2_out buffer");
	// pool 3
	d_pool3_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_3*output_size_3*input_channels_3*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool3_out buffer");

	// conv3x3 4
	d_conv4_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_4*3*3*output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv4_weight buffer");
	d_conv4_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_4*output_size_4*output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv4_out buffer");
	// bn 4
	d_bn4_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn4_weight buffer");
	d_bn4_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn4_bias buffer");
	d_bn4_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn4_mean buffer");
	d_bn4_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_4*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn4_var buffer");
	d_bn4_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
						output_size_4*output_size_4*output_channels_4*sizeof(float), NULL, &status);
						checkError(status, "Failed to create d_bn4_out buffer");
	// pool 5
	d_pool5_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_5*output_size_5*input_channels_5*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool5_out buffer");

	// conv3x3 6
	d_conv6_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_6*3*3*output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv6_weight buffer");
	d_conv6_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_6*output_size_6*output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv6_out buffer");
	// bn 6
	d_bn6_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn6_weight buffer");
	d_bn6_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn6_bias buffer");
	d_bn6_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn6_mean buffer");
	d_bn6_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn6_var buffer");
	d_bn6_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_6*output_size_6*output_channels_6*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn6_out buffer");
	// pool 7
	d_pool7_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_7*output_size_7*input_channels_7*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool7_out buffer");

	// conv3x3 8
	d_conv8_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_8*3*3*output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv8_weight buffer");
	d_conv8_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_8*output_size_8*output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv8_out buffer");
				// bn 8
	d_bn8_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn8_weight buffer");
	d_bn8_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn8_bias buffer");
	d_bn8_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn8_mean buffer");
	d_bn8_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn8_var buffer");
	d_bn8_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_8*output_size_8*output_channels_8*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn8_out buffer");
	// pool 9
	d_pool9_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_9*output_size_9*input_channels_9*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool9_out buffer");

	// conv3x3 10
	d_conv10_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_10*3*3*output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv10_weight buffer");
	d_conv10_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_10*output_size_10*output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv10_out buffer");
	// bn 10
	d_bn10_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn10_weight buffer");
	d_bn10_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn10_bias buffer");
	d_bn10_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn10_mean buffer");
	d_bn10_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn10_var buffer");
	d_bn10_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_10*output_size_10*output_channels_10*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn10_out buffer");
	// pool 11
	d_pool11_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_11*output_size_11*input_channels_11*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool11_out buffer");

	// conv3x3 12
	d_conv12_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				input_channels_12*3*3*output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv12_weight buffer");
	d_conv12_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_12*output_size_12*output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv12_out buffer");
	// bn 12
	d_bn12_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn12_weight buffer");
	d_bn12_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn12_bias buffer");
	d_bn12_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn12_mean buffer");
	d_bn12_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
				output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn12_var buffer");
	d_bn12_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_12*output_size_12*output_channels_12*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn12_out buffer");
	// pool 13
	d_pool13_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				output_size_13*output_size_13*input_channels_13*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_pool13_out buffer");

  // conv3x3 14
  d_conv14_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
			  input_channels_14*1*1*output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv14_weight buffer");
  d_conv14_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			  output_size_14*output_size_14*output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_conv14_out buffer");
	// bn 14
  d_bn14_weight = clCreateBuffer(context, CL_MEM_READ_ONLY,
			  output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn14_weight buffer");
  d_bn14_bias = clCreateBuffer(context, CL_MEM_READ_ONLY,
			  output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn14_bias buffer");
  d_bn14_mean = clCreateBuffer(context, CL_MEM_READ_ONLY,
			  output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn14_mean buffer");
  d_bn14_var = clCreateBuffer(context, CL_MEM_READ_ONLY,
			  output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn14_var buffer");
  d_bn14_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
			  output_size_14*output_size_14*output_channels_14*sizeof(float), NULL, &status);
				checkError(status, "Failed to create d_bn14_out buffer");

	printf("Reading conv0 weights...");
	FILE* fp = fopen(conv0_wt_file, "r");
	unsigned int k=0;
	float wt;
  	for(unsigned i=0;i<output_channels_0;i++){
	  	for(unsigned j=0;j<32;j++){
		  	if(j<kernel_size_0*kernel_size_0*input_channels_0&& i<output_channels_0)
		  	{
			  	fscanf(fp,"%f",&wt);
			  	conv0_weight[k++]=wt;
		  	}
		  	else 	// Padding to make it multiple of CONV_BLOCK_SIZE
			  	conv0_weight[k++]=0.0;
		}
  	}	
	fclose(fp);


	printf("\nReading bn0 weights...");
	fp = fopen(bn0_wt_file, "r");
	for(int i=0; i<output_channels_0; i++)
	{
		fscanf(fp, "%f", &bn0_weight[i]);
	}
	fclose(fp);

	fp = fopen(bn0_mean_file, "r");
	for(int i=0; i<output_channels_0; i++)
	{
		fscanf(fp, "%f", &bn0_mean[i]);
	}
	fclose(fp);

	fp = fopen(bn0_var_file, "r");
	for(int i=0; i<output_channels_0; i++)
	{
		fscanf(fp, "%f", &bn0_var[i]);
	}
	fclose(fp);

	fp = fopen(bn0_bias_file, "r");
	for(int i=0; i<output_channels_0; i++)
	{
		fscanf(fp, "%f", &bn0_bias[i]);
	}
	fclose(fp);

	printf("\nReading conv2 weights...");
	fp = fopen(conv2_wt_file, "r");
	for(int i=0; i<(input_channels_2*3*3*output_channels_2); i++)
	{
		fscanf(fp, "%f", &conv2_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn2 weights...");
	fp = fopen(bn2_wt_file, "r");
	for(int i=0; i<output_channels_2; i++)
	{
		fscanf(fp, "%f", &bn2_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 2 means
	fp = fopen(bn2_mean_file, "r");
	for(int i=0; i<output_channels_2; i++)
	{
		fscanf(fp, "%f", &bn2_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 2 variance
	fp = fopen(bn2_var_file, "r");
	for(int i=0; i<output_channels_2; i++)
	{
		fscanf(fp, "%f", &bn2_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 2 bias
	fp = fopen(bn2_bias_file, "r");
	for(int i=0; i<output_channels_2; i++)
	{
		fscanf(fp, "%f", &bn2_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv4 weights...");
	fp = fopen(conv4_wt_file, "r");
	for(int i=0; i<(input_channels_4*3*3*output_channels_4); i++)
	{
		fscanf(fp, "%f", &conv4_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn4 weights...");
	fp = fopen(bn4_wt_file, "r");
	for(int i=0; i<output_channels_4; i++)
	{
		fscanf(fp, "%f", &bn4_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 4 means
	fp = fopen(bn4_mean_file, "r");
	for(int i=0; i<output_channels_4; i++)
	{
		fscanf(fp, "%f", &bn4_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 4 variance
	fp = fopen(bn4_var_file, "r");
	for(int i=0; i<output_channels_4; i++)
	{
		fscanf(fp, "%f", &bn4_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 4 bias
	fp = fopen(bn4_bias_file, "r");
	for(int i=0; i<output_channels_4; i++)
	{
		fscanf(fp, "%f", &bn4_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv6 weights...");
	fp = fopen(conv6_wt_file, "r");
	for(int i=0; i<(input_channels_6*3*3*output_channels_6); i++)
	{
		fscanf(fp, "%f", &conv6_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn6 weights...");
	fp = fopen(bn6_wt_file, "r");
	for(int i=0; i<output_channels_6; i++)
	{
		fscanf(fp, "%f", &bn6_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 6 means
	fp = fopen(bn6_mean_file, "r");
	for(int i=0; i<output_channels_6; i++)
	{
		fscanf(fp, "%f", &bn6_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 6 variance
	fp = fopen(bn6_var_file, "r");
	for(int i=0; i<output_channels_6; i++)
	{
		fscanf(fp, "%f", &bn6_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 6 bias
	fp = fopen(bn6_bias_file, "r");
	for(int i=0; i<output_channels_6; i++)
	{
		fscanf(fp, "%f", &bn6_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv8 weights...");
	fp = fopen(conv8_wt_file, "r");
	for(int i=0; i<(input_channels_8*3*3*output_channels_8); i++)
	{
		fscanf(fp, "%f", &conv8_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn8 weights...");
	fp = fopen(bn8_wt_file, "r");
	for(int i=0; i<output_channels_8; i++)
	{
		fscanf(fp, "%f", &bn8_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 8 means
	fp = fopen(bn8_mean_file, "r");
	for(int i=0; i<output_channels_8; i++)
	{
		fscanf(fp, "%f", &bn8_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 8 variance
	fp = fopen(bn8_var_file, "r");
	for(int i=0; i<output_channels_8; i++)
	{
		fscanf(fp, "%f", &bn8_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 8 bias
	fp = fopen(bn8_bias_file, "r");
	for(int i=0; i<output_channels_8; i++)
	{
		fscanf(fp, "%f", &bn8_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv10 weights...");
	fp = fopen(conv10_wt_file, "r");
	for(int i=0; i<(input_channels_10*3*3*output_channels_10); i++)
	{
		fscanf(fp, "%f", &conv10_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn10 weights...");
	fp = fopen(bn10_wt_file, "r");
	for(int i=0; i<output_channels_10; i++)
	{
		fscanf(fp, "%f", &bn10_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 10 means
	fp = fopen(bn10_mean_file, "r");
	for(int i=0; i<output_channels_10; i++)
	{
		fscanf(fp, "%f", &bn10_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 10 variance
	fp = fopen(bn10_var_file, "r");
	for(int i=0; i<output_channels_10; i++)
	{
		fscanf(fp, "%f", &bn10_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 10 bias
	fp = fopen(bn10_bias_file, "r");
	for(int i=0; i<output_channels_10; i++)
	{
		fscanf(fp, "%f", &bn10_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv12 weights...");
	fp = fopen(conv12_wt_file, "r");
	for(int i=0; i<(input_channels_12*3*3*output_channels_12); i++)
	{
		fscanf(fp, "%f", &conv12_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn12 weights...");
	fp = fopen(bn12_wt_file, "r");
	for(int i=0; i<output_channels_12; i++)
	{
		fscanf(fp, "%f", &bn12_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 12 means
	fp = fopen(bn12_mean_file, "r");
	for(int i=0; i<output_channels_12; i++)
	{
		fscanf(fp, "%f", &bn12_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 12 variance
	fp = fopen(bn12_var_file, "r");
	for(int i=0; i<output_channels_12; i++)
	{
		fscanf(fp, "%f", &bn12_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 12 bias
	fp = fopen(bn12_bias_file, "r");
	for(int i=0; i<output_channels_12; i++)
	{
		fscanf(fp, "%f", &bn12_bias[i]);
	}
	fclose(fp);


	printf("\nReading conv14 weights...");
	fp = fopen(conv14_wt_file, "r");
	for(int i=0; i<(input_channels_14*1*1*output_channels_14); i++)
	{
		fscanf(fp, "%f", &conv14_weight[i]);
	}
	fclose(fp);

	printf("\nReading bn14 weights...");
	fp = fopen(bn14_wt_file, "r");
	for(int i=0; i<output_channels_14; i++)
	{
		fscanf(fp, "%f", &bn14_weight[i]);
	}
	fclose(fp);

	// Reading batchnorm 14 means
	fp = fopen(bn14_mean_file, "r");
	for(int i=0; i<output_channels_14; i++)
	{
		fscanf(fp, "%f", &bn14_mean[i]);
	}
	fclose(fp);

	// Reading batchnorm 14 variance
	fp = fopen(bn14_var_file, "r");
	for(int i=0; i<output_channels_14; i++)
	{
		fscanf(fp, "%f", &bn14_var[i]);
	}
	fclose(fp);

	// Reading batchnorm 14 bias
	fp = fopen(bn14_bias_file, "r");
	for(int i=0; i<output_channels_14; i++)
	{
		fscanf(fp, "%f", &bn14_bias[i]);
	}
	fclose(fp);

// Transfer data to the buffers
	

	status = clEnqueueWriteBuffer(queue, d_conv0_weight, CL_TRUE, 0,
				32*output_channels_0*sizeof(float), conv0_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv0 weights");

	status = clEnqueueWriteBuffer(queue, d_bn0_weight, CL_TRUE, 0,
				output_channels_0*sizeof(float), bn0_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn0_bias, CL_TRUE, 0,
				output_channels_0*sizeof(float), bn0_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn0_mean, CL_TRUE, 0,
				output_channels_0*sizeof(float), bn0_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn0_var, CL_TRUE, 0,
				output_channels_0*sizeof(float), bn0_var, 0, NULL, NULL);

	status = clEnqueueWriteBuffer(queue, d_conv2_weight, CL_TRUE, 0,
				input_channels_2*3*3*output_channels_2*sizeof(float), conv2_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv2 weights");
	status = clEnqueueWriteBuffer(queue, d_bn2_weight, CL_TRUE, 0,
				output_channels_2*sizeof(float), bn2_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn2_bias, CL_TRUE, 0,
				output_channels_2*sizeof(float), bn2_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn2_mean, CL_TRUE, 0,
				output_channels_2*sizeof(float), bn2_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn2_var, CL_TRUE, 0,
				output_channels_2*sizeof(float), bn2_var, 0, NULL, NULL);

  	status = clEnqueueWriteBuffer(queue, d_conv4_weight, CL_TRUE, 0,
			  input_channels_4*3*3*output_channels_4*sizeof(float), conv4_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv4 weights");
  	status = clEnqueueWriteBuffer(queue, d_bn4_weight, CL_TRUE, 0,
			  output_channels_4*sizeof(float), bn4_weight, 0, NULL, NULL);
 	 status = clEnqueueWriteBuffer(queue, d_bn4_bias, CL_TRUE, 0,
			  output_channels_4*sizeof(float), bn4_bias, 0, NULL, NULL);
  	status = clEnqueueWriteBuffer(queue, d_bn4_mean, CL_TRUE, 0,
			  output_channels_4*sizeof(float), bn4_mean, 0, NULL, NULL);
  	status = clEnqueueWriteBuffer(queue, d_bn4_var, CL_TRUE, 0,
			  output_channels_4*sizeof(float), bn4_var, 0, NULL, NULL);

  	status = clEnqueueWriteBuffer(queue, d_conv6_weight, CL_TRUE, 0,
				input_channels_6*3*3*output_channels_6*sizeof(float), conv6_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv6 weights");
	status = clEnqueueWriteBuffer(queue, d_bn6_weight, CL_TRUE, 0,
				output_channels_6*sizeof(float), bn6_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn6_bias, CL_TRUE, 0,
				output_channels_6*sizeof(float), bn6_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn6_mean, CL_TRUE, 0,
				output_channels_6*sizeof(float), bn6_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn6_var, CL_TRUE, 0,
				output_channels_6*sizeof(float), bn6_var, 0, NULL, NULL);

	status = clEnqueueWriteBuffer(queue, d_conv8_weight, CL_TRUE, 0,
				input_channels_8*3*3*output_channels_8*sizeof(float), conv8_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv8 weights");
	status = clEnqueueWriteBuffer(queue, d_bn8_weight, CL_TRUE, 0,
				output_channels_8*sizeof(float), bn8_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn8_bias, CL_TRUE, 0,
				output_channels_8*sizeof(float), bn8_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn8_mean, CL_TRUE, 0,
				output_channels_8*sizeof(float), bn8_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn8_var, CL_TRUE, 0,
				output_channels_8*sizeof(float), bn8_var, 0, NULL, NULL);

	status = clEnqueueWriteBuffer(queue, d_conv10_weight, CL_TRUE, 0,
				input_channels_10*3*3*output_channels_10*sizeof(float), conv10_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv10 weights");
	status = clEnqueueWriteBuffer(queue, d_bn10_weight, CL_TRUE, 0,
				output_channels_10*sizeof(float), bn10_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn10_bias, CL_TRUE, 0,
				output_channels_10*sizeof(float), bn10_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn10_mean, CL_TRUE, 0,
				output_channels_10*sizeof(float), bn10_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn10_var, CL_TRUE, 0,
				output_channels_10*sizeof(float), bn10_var, 0, NULL, NULL);

	status = clEnqueueWriteBuffer(queue, d_conv12_weight, CL_TRUE, 0,
				input_channels_12*3*3*output_channels_12*sizeof(float), conv12_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv12 weights");
	status = clEnqueueWriteBuffer(queue, d_bn12_weight, CL_TRUE, 0,
				output_channels_12*sizeof(float), bn12_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn12_bias, CL_TRUE, 0,
				output_channels_12*sizeof(float), bn12_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn12_mean, CL_TRUE, 0,
				output_channels_12*sizeof(float), bn12_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn12_var, CL_TRUE, 0,
				output_channels_12*sizeof(float), bn12_var, 0, NULL, NULL);

	status = clEnqueueWriteBuffer(queue, d_conv14_weight, CL_TRUE, 0,
				input_channels_14*1*1*output_channels_14*sizeof(float), conv14_weight, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv14 weights");
	status = clEnqueueWriteBuffer(queue, d_bn14_weight, CL_TRUE, 0,
				output_channels_14*sizeof(float), bn14_weight, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn14_bias, CL_TRUE, 0,
				output_channels_14*sizeof(float), bn14_bias, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn14_mean, CL_TRUE, 0,
				output_channels_14*sizeof(float), bn14_mean, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(queue, d_bn14_var, CL_TRUE, 0,
				output_channels_14*sizeof(float), bn14_var, 0, NULL, NULL);

	load_image();

	printf("\n\nAccelerator starts...\n\n");

	// CONV 0
	unsigned int N_elem_0 = 32;
	size_t conv0_work_size[] = {256*256,16};
	size_t conv0_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_sample);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv0_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv0_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_0);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_0);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_0);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_0);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_0);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_0);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_0);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_0);
	checkError(status, "Setting conv3x3 0 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv0_work_size, conv0_local_work_size, 0, NULL, &conv0_event);
	checkError(status, "Enqueueing conv3x3 0");

	// BATCHNORM 0
	size_t bn0_work_size[] = {16,256*256};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv0_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn0_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn0_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn0_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn0_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn0_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_0);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_0);
	checkError(status, "Setting batchnorm 0 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn0_work_size, NULL, 1, &conv0_event, &bn0_event);
	checkError(status, "Enqueueing batchnorm 0");

	// POOL 1
	size_t pool1_work_size = 16;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn0_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool1_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_1);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_1);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_1);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_1);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_1);


	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool1_work_size, NULL, 1, &bn0_event, &pool1_event);
	checkError(status, "Enqueueing pool 1");

	// CONV 2
	unsigned int N_elem_2 = input_channels_2*kernel_size_2*kernel_size_2;
	size_t conv2_work_size[] = {128*128,32};
	size_t conv2_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool1_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv2_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv2_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_2);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_2);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_2);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_2);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_2);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_2);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_2);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_2);
	checkError(status, "Setting conv3x3 2 arguments");


	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv2_work_size, conv2_local_work_size, 1, &pool1_event, &conv2_event);
	checkError(status, "Enqueueing conv3x3 2");

	// BATCHNORM 2
	size_t bn2_work_size[] = {32,128*128};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv2_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn2_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn2_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn2_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn2_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn2_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_2);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_2);
	checkError(status, "Setting batchnorm 2 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn2_work_size, NULL, 1, &conv2_event, &bn2_event);
	checkError(status, "Enqueueing batchnorm 2");

	// POOL 3
	size_t pool3_work_size = 32;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn2_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool3_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_3);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_3);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_3);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_3);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_3);


	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool3_work_size, NULL, 1, &bn2_event, &pool3_event);
	checkError(status, "Enqueueing maxpool 3");


	// CONV 4
	unsigned int N_elem_4 = input_channels_4*kernel_size_4*kernel_size_4;

	size_t conv4_work_size[] = {64*64,64};
	size_t conv4_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool3_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv4_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv4_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_4);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_4);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_4);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_4);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_4);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_4);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_4);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_4);
	checkError(status, "Setting conv3x3 4 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv4_work_size, conv4_local_work_size, 1, &pool3_event, &conv4_event);
	checkError(status, "Enqueueing conv3x3 4");

	// BATCHNORM 4
	size_t bn4_work_size[] = {64, 64*64};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv4_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn4_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn4_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn4_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn4_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn4_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_4);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_4);
	checkError(status, "Setting batchnorm 4 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn4_work_size, NULL, 1, &conv4_event, &bn4_event);
	checkError(status, "Enqueueing batchnorm 4");

	// POOL 5
	size_t pool5_work_size = 64;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn4_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool5_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_5);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_5);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_5);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_5);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_5);

	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool5_work_size, NULL, 1, &bn4_event, &pool5_event);
	checkError(status, "Enqueueing maxpool 5");

	// CONV 6
	unsigned int N_elem_6 = input_channels_6*kernel_size_6*kernel_size_6;

	size_t conv6_work_size[] = {32*32,128};
	size_t conv6_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool5_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv6_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv6_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_6);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_6);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_6);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_6);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_6);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_6);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_6);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_6);
	checkError(status, "Setting conv3x3 6 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv6_work_size, conv6_local_work_size, 1, &pool5_event, &conv6_event);
	checkError(status, "Enqueueing conv3x3 6");

	// BATCHNORM 6
	size_t bn6_work_size[] = {128,32*32};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv6_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn6_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn6_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn6_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn6_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn6_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_6);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_6);
	checkError(status, "Setting batchnorm 6 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn6_work_size, NULL, 1, &conv6_event, &bn6_event);
	checkError(status, "Enqueueing batchnorm 6");

	// POOL 7
	size_t pool7_work_size = 128;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn6_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool7_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_7);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_7);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_7);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_7);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_7);

	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool7_work_size, NULL, 1, &bn6_event, &pool7_event);
	checkError(status, "Enqueueing maxpool 7");

	// CONV 8
	unsigned int N_elem_8 = input_channels_8*kernel_size_8*kernel_size_8;

	size_t conv8_work_size[] = {16*16,256};
	size_t conv8_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool7_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv8_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv8_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_8);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_8);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_8);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_8);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_8);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_8);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_8);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_8);
	checkError(status, "Setting conv3x3 8 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv8_work_size, conv8_local_work_size, 1, &pool7_event, &conv8_event);
	checkError(status, "Enqueueing conv3x3 8");

	// BATCHNORM 8
	size_t bn8_work_size[] = {256,16*16};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv8_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn8_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn8_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn8_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn8_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn8_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_8);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_8);
	checkError(status, "Setting batchnorm 8 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn8_work_size, NULL, 1, &conv8_event, &bn8_event);
	checkError(status, "Enqueueing batchnorm 8");

	// POOL 9
	size_t pool9_work_size = 256;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn8_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool9_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_9);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_9);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_9);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_9);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_9);

	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool9_work_size, NULL, 1, &bn8_event, &pool9_event);
	checkError(status, "Enqueueing maxpool 9");

	// CONV 10
	unsigned int N_elem_10 = input_channels_10*kernel_size_10*kernel_size_10;

	size_t conv10_work_size[] = {8*8,512};
	size_t conv10_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool9_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv10_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv10_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_10);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_10);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_10);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_10);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_10);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_10);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_10);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_10);
	checkError(status, "Setting conv3x3 10 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv10_work_size, conv10_local_work_size, 1, &pool9_event, &conv10_event);
	checkError(status, "Enqueueing conv3x3 10");

	// BATCHNORM 10
	size_t bn10_work_size[] = {512,8*8};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv10_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn10_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn10_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn10_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn10_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn10_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_10);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_10);
	checkError(status, "Setting batchnorm 10 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn10_work_size, NULL, 1, &conv10_event, &bn10_event);
	checkError(status, "Enqueueing batchnorm 10");

	// POOL 11
	size_t pool11_work_size = 512;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn10_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool11_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_11);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_11);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_11);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_11);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_11);

	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool11_work_size, NULL, 1, &bn10_event, &pool11_event);
	checkError(status, "Enqueueing maxpool 11");

	// conv3x3 12
	unsigned int N_elem_12 = input_channels_12*kernel_size_12*kernel_size_12;

	size_t conv12_work_size[] = {4*4,1024};
	size_t conv12_local_work_size[] = {8,8};

	status |= clSetKernelArg(conv3x3, 0, sizeof(cl_mem), &d_pool11_out);
	status |= clSetKernelArg(conv3x3, 1, sizeof(cl_mem), &d_conv12_weight);
	status |= clSetKernelArg(conv3x3, 2, sizeof(cl_mem), &d_conv12_out);
	status |= clSetKernelArg(conv3x3, 3, sizeof(int), &N_elem_12);
	status |= clSetKernelArg(conv3x3, 4, sizeof(int), &kernel_size_12);
	status |= clSetKernelArg(conv3x3, 5, sizeof(int), &stride_12);
	status |= clSetKernelArg(conv3x3, 6, sizeof(int), &pad_12);
	status |= clSetKernelArg(conv3x3, 7, sizeof(int), &input_channels_12);
	status |= clSetKernelArg(conv3x3, 8, sizeof(int), &input_size_12);
	status |= clSetKernelArg(conv3x3, 9, sizeof(int), &input_size_sq_12);
	status |= clSetKernelArg(conv3x3, 10, sizeof(int), &output_size_12);
	checkError(status, "Setting conv3x3 12 arguments");

	status = clEnqueueNDRangeKernel(queue, conv3x3, 2, NULL, conv12_work_size, conv12_local_work_size, 1, &pool11_event, &conv12_event);
	checkError(status, "Enqueueing conv3x3 12");

	// BATCHNORM 12
	size_t bn12_work_size[] = {1024,4*4};

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv12_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn12_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn12_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn12_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn12_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn12_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_12);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_12);
	checkError(status, "Setting batchnorm 12 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn12_work_size, NULL, 1, &conv12_event, &bn12_event);
	checkError(status, "Enqueueing batchnorm 12");

	// POOL 13
	size_t pool13_work_size = 1024;

	status |= clSetKernelArg(pool, 0, sizeof(cl_mem), &d_bn12_out);
	status |= clSetKernelArg(pool, 1, sizeof(cl_mem), &d_pool13_out);
	status |= clSetKernelArg(pool, 2, sizeof(int), &input_size_13);
	status |= clSetKernelArg(pool, 3, sizeof(int), &kernel_size_13);
	status |= clSetKernelArg(pool, 4, sizeof(int), &stride_13);
	status |= clSetKernelArg(pool, 5, sizeof(int), &output_size_13);
	status |= clSetKernelArg(pool, 6, sizeof(int), &pool_type_13);

	status = clEnqueueNDRangeKernel(queue, pool, 1, NULL, &pool13_work_size, NULL, 1, &bn12_event, &pool13_event);
	checkError(status, "Enqueueing maxpool 13");

	// CONV 14
	size_t conv14_work_size[] = {1000,1};

	status |= clSetKernelArg(conv1x1, 0, sizeof(cl_mem), &d_pool13_out);
	status |= clSetKernelArg(conv1x1, 1, sizeof(cl_mem), &d_conv14_weight);
	status |= clSetKernelArg(conv1x1, 2, sizeof(cl_mem), &d_conv14_out);
	status |= clSetKernelArg(conv1x1, 3, sizeof(int), &input_channels_14);
	status |= clSetKernelArg(conv1x1, 4, sizeof(int), &input_size_14);
	status |= clSetKernelArg(conv1x1, 5, sizeof(int), &pad_14);
	status |= clSetKernelArg(conv1x1, 6, sizeof(int), &stride_14);
	status |= clSetKernelArg(conv1x1, 7, sizeof(int), &output_size_14);
	checkError(status, "Setting conv1x1 14 arguments");

	status = clEnqueueNDRangeKernel(queue, conv1x1, 2, NULL, conv14_work_size, NULL, 1, &pool13_event, &conv14_event);
	checkError(status, "Enqueueing conv1x1 14"); 

  // BATCHNORM 14
	size_t bn14_work_size[] = {1000,1*1};
	float eps1 = 0.0f;

	status |= clSetKernelArg(bn, 0, sizeof(cl_mem), &d_conv14_out);
	status |= clSetKernelArg(bn, 1, sizeof(cl_mem), &d_bn14_weight);
	status |= clSetKernelArg(bn, 2, sizeof(cl_mem), &d_bn14_bias);
	status |= clSetKernelArg(bn, 3, sizeof(cl_mem), &d_bn14_mean);
	status |= clSetKernelArg(bn, 4, sizeof(cl_mem), &d_bn14_var);
	status |= clSetKernelArg(bn, 5, sizeof(cl_mem), &d_bn14_out);
	status |= clSetKernelArg(bn, 6, sizeof(int), &input_size_sq_14);
	status |= clSetKernelArg(bn, 7, sizeof(float), &eps1);
	status |= clSetKernelArg(bn, 8, sizeof(int), &relu_type_14);
	checkError(status, "Setting batchnorm 14 arguments");

	status = clEnqueueNDRangeKernel(queue, bn, 2, NULL, bn14_work_size, NULL, 1, &conv14_event, &bn14_event);
	checkError(status, "Enqueueing batchnorm 14");

	clWaitForEvents(1,&bn14_event);

	get_results(); // Finds the label and accuracy of the image

	printf("Conv 0  time: %0.3f ms\n", getStartEndTime(conv0_event)*1e-6);
	printf("Conv 2  time: %0.3f ms\n", getStartEndTime(conv2_event)*1e-6);
	printf("Conv 4  time: %0.3f ms\n", getStartEndTime(conv4_event)*1e-6);
	printf("Conv 6  time: %0.3f ms\n", getStartEndTime(conv6_event)*1e-6);
	printf("Conv 8  time: %0.3f ms\n", getStartEndTime(conv8_event)*1e-6);
	printf("Conv 10 time: %0.3f ms\n", getStartEndTime(conv10_event)*1e-6);
	printf("Conv 12 time: %0.3f ms\n", getStartEndTime(conv12_event)*1e-6);
	printf("Conv 14 time: %0.3f ms\n", getStartEndTime(conv14_event)*1e-6);

	cl_ulong conv_time = getStartEndTime(conv0_event)+getStartEndTime(conv2_event)+getStartEndTime(conv4_event)+
		getStartEndTime(conv6_event)+getStartEndTime(conv8_event)+getStartEndTime(conv10_event)+
		getStartEndTime(conv12_event)+getStartEndTime(conv14_event);

	printf("Total Convolution time: %0.3f ms\n\n", double(conv_time)*1e-6);

	printf("Batchnorm 0   time: %0.3f ms\n", getStartEndTime(bn0_event)*1e-6);
	printf("Batchnorm 2   time: %0.3f ms\n", getStartEndTime(bn2_event)*1e-6);
	printf("Batchnorm 4   time: %0.3f ms\n", getStartEndTime(bn4_event)*1e-6);
	printf("Batchnorm 6   time: %0.3f ms\n", getStartEndTime(bn6_event)*1e-6);
	printf("Batchnorm 8   time: %0.3f ms\n", getStartEndTime(bn8_event)*1e-6);
	printf("Batchnorm 10  time: %0.3f ms\n", getStartEndTime(bn10_event)*1e-6);
	printf("Batchnorm 12  time: %0.3f ms\n", getStartEndTime(bn12_event)*1e-6);
	printf("Batchnorm 14  time: %0.3f ms\n", getStartEndTime(bn14_event)*1e-6);

	cl_ulong bn_time = getStartEndTime(bn0_event)+getStartEndTime(bn2_event)+getStartEndTime(bn4_event)+
		getStartEndTime(bn6_event)+getStartEndTime(bn8_event)+getStartEndTime(bn10_event)+
		getStartEndTime(bn12_event)+getStartEndTime(bn14_event);

	printf("Total Batchnorm time: %0.3f ms\n\n", double(bn_time)*1e-6);

	printf("Maxpool 1  time: %0.3f ms\n", getStartEndTime(pool1_event)*1e-6);
	printf("Maxpool 3  time: %0.3f ms\n", getStartEndTime(pool3_event)*1e-6);
	printf("Maxpool 5  time: %0.3f ms\n", getStartEndTime(pool5_event)*1e-6);
	printf("Maxpool 7  time: %0.3f ms\n", getStartEndTime(pool7_event)*1e-6);
	printf("Maxpool 9  time: %0.3f ms\n", getStartEndTime(pool9_event)*1e-6);
	printf("Maxpool 11  time: %0.3f ms\n", getStartEndTime(pool11_event)*1e-6);
	printf("Maxpool 13  time: %0.3f ms\n", getStartEndTime(pool13_event)*1e-6);

	cl_ulong pool_time = getStartEndTime(pool1_event)+getStartEndTime(pool3_event)+getStartEndTime(pool5_event)+
		getStartEndTime(pool7_event)+getStartEndTime(pool9_event)+getStartEndTime(pool11_event)+
		getStartEndTime(pool13_event);

	printf("Total Pooling time: %0.3f ms\n\n", double(pool_time)*1e-6);

	cl_ulong total_time = conv_time + bn_time + pool_time;

	printf("Total Time: %0.3f\n\n", double(total_time)*1e-6);

	printf("Label   : %s\n", label);
	printf("Accuracy: %0.3f \n", accuracy);

	clReleaseEvent(conv0_event);
	clReleaseEvent(bn0_event);
	clReleaseEvent(pool1_event);
	clReleaseEvent(conv2_event);
	clReleaseEvent(bn2_event);
	clReleaseEvent(pool3_event);
	clReleaseEvent(conv4_event);
	clReleaseEvent(bn4_event);
	clReleaseEvent(pool5_event);
	clReleaseEvent(conv6_event);
	clReleaseEvent(bn6_event);
	clReleaseEvent(pool7_event);
	clReleaseEvent(conv8_event);
	clReleaseEvent(bn8_event);
	clReleaseEvent(pool9_event);
	clReleaseEvent(conv10_event);
	clReleaseEvent(bn10_event);
	clReleaseEvent(pool11_event);
	clReleaseEvent(conv12_event);
	clReleaseEvent(bn12_event);
	clReleaseEvent(pool13_event);
	clReleaseEvent(conv14_event);
	clReleaseEvent(bn14_event);

	cleanup();
}

void load_image()
{
	printf("\nImage preprocessing...");
	Mat img = imread(image_path);
	Mat img1;

	resize(img, img1, Size(256,256));
	img1.convertTo(img1, CV_32FC3, 1.0/255, 0);

	float* image_data = (float*)img1.data;

	unsigned int w,h,c;
	unsigned int k=0;
	for(h=0; h<input_size_0; h++){
		for(w=0; w<input_size_0; w++){
			for(c=0; c<input_channels_0; c++){
        		image[c*input_size_0*input_size_0+h*input_size_0+w]=image_data[k];
        		k++;
			}
		}
	}

	status = clEnqueueWriteBuffer(queue, d_sample, CL_TRUE, 0,
	 			input_size_0*input_size_0*input_channels_0*sizeof(float), image, 0, NULL, NULL);
    checkError(status, "Failed to transfer input data");

}

void softmax()
{
    unsigned int i;
	float data_max=0.0;
	float data_exp;
	float sum_exp=0.0;
	for(i=0; i<1000; i++)
	{
	  if(data_max<bn14_result[i])
	  data_max = bn14_result[i];
	}

    for(i=0; i<1000; i++)
	{
	   data_exp = exp(bn14_result[i]-data_max);
	   sum_exp += data_exp;
	}

    for(i=0; i<1000; i++)
	{
	   data_exp = exp(bn14_result[i]-data_max);
	   softmax_out[i] = (data_exp / sum_exp)*100.0;
	}
}

void get_results()
{
    unsigned int i, index=0;
	float data_max = 0.0;

	status = clEnqueueReadBuffer(queue, d_bn14_out, CL_TRUE, 0, sizeof(float)*(1000), bn14_result, 0, NULL, NULL);
    checkError(status, "Failed to read d)bn14_out");

	softmax();

	for(i=0; i<1000; i++)
	{
	  if(data_max<bn14_result[i]) {
	  	data_max = bn14_result[i];
	  	index = i;
	  }
	}
	accuracy = data_max;

	FILE* fp1;
    fp1 = fopen("labels.txt", "r");

    for(i = 0; i < index + 1; i++) {
    fgets(label, sizeof(label), fp1);
    }

    fclose(fp1);
}

void cleanup()
{
	clReleaseKernel(conv3x3);
	clReleaseKernel(conv1x1);
	clReleaseKernel(bn);
	clReleaseKernel(pool);

	clReleaseCommandQueue(queue);

	clReleaseMemObject(d_sample);

	clReleaseMemObject(d_conv0_weight);
	clReleaseMemObject(d_conv0_out);
	clReleaseMemObject(d_bn0_weight);
	clReleaseMemObject(d_bn0_bias);
	clReleaseMemObject(d_bn0_mean);
	clReleaseMemObject(d_bn0_var);
	clReleaseMemObject(d_bn0_out);
	clReleaseMemObject(d_pool1_out);

	clReleaseMemObject(d_conv2_weight);
	clReleaseMemObject(d_conv2_out);
	clReleaseMemObject(d_bn2_weight);
	clReleaseMemObject(d_bn2_bias);
	clReleaseMemObject(d_bn2_mean);
	clReleaseMemObject(d_bn2_var);
	clReleaseMemObject(d_bn2_out);
	clReleaseMemObject(d_pool3_out);

	clReleaseMemObject(d_conv4_weight);
	clReleaseMemObject(d_conv4_out);
	clReleaseMemObject(d_bn4_weight);
	clReleaseMemObject(d_bn4_bias);
	clReleaseMemObject(d_bn4_mean);
	clReleaseMemObject(d_bn4_var);
	clReleaseMemObject(d_bn4_out);
	clReleaseMemObject(d_pool5_out);

	clReleaseMemObject(d_conv6_weight);
	clReleaseMemObject(d_conv6_out);
	clReleaseMemObject(d_bn6_weight);
	clReleaseMemObject(d_bn6_bias);
	clReleaseMemObject(d_bn6_mean);
	clReleaseMemObject(d_bn6_var);
	clReleaseMemObject(d_bn6_out);
	clReleaseMemObject(d_pool7_out);

	clReleaseMemObject(d_conv8_weight);
	clReleaseMemObject(d_conv8_out);
	clReleaseMemObject(d_bn8_weight);
	clReleaseMemObject(d_bn8_bias);
	clReleaseMemObject(d_bn8_mean);
	clReleaseMemObject(d_bn8_var);
	clReleaseMemObject(d_bn8_out);
	clReleaseMemObject(d_pool9_out);

	clReleaseMemObject(d_conv10_weight);
	clReleaseMemObject(d_conv10_out);
	clReleaseMemObject(d_bn10_weight);
	clReleaseMemObject(d_bn10_bias);
	clReleaseMemObject(d_bn10_mean);
	clReleaseMemObject(d_bn10_var);
	clReleaseMemObject(d_bn10_out);
	clReleaseMemObject(d_pool11_out);

	clReleaseMemObject(d_conv12_weight);
	clReleaseMemObject(d_conv12_out);
	clReleaseMemObject(d_bn12_weight);
	clReleaseMemObject(d_bn12_bias);
	clReleaseMemObject(d_bn12_mean);
	clReleaseMemObject(d_bn12_var);
	clReleaseMemObject(d_bn12_out);
	clReleaseMemObject(d_pool13_out);

	clReleaseMemObject(d_conv14_weight);
	clReleaseMemObject(d_conv14_out);
	clReleaseMemObject(d_bn14_weight);
	clReleaseMemObject(d_bn14_bias);
	clReleaseMemObject(d_bn14_mean);
	clReleaseMemObject(d_bn14_var);
	clReleaseMemObject(d_bn14_out);


	clReleaseProgram(program);
	clReleaseContext(context);

	free(bn14_result);
	free(softmax_out);

}

