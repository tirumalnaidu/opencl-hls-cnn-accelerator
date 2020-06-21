// Convolution layer 0

	unsigned int input_channels_0 = 3;
	unsigned int input_size_0 = 256;
	unsigned int input_size_sq_0 = 256*256;
	unsigned int kernel_size_0 = 3;
	unsigned int pad_0 = 1;
	unsigned int stride_0 = 1;
	unsigned int output_size_0 = 256;
	unsigned int output_channels_0 = 16;
	unsigned int relu_type_0 = 1; // leaky relu

// Pooling layer 1

	unsigned int input_channels_1 = 16;
	unsigned int input_size_1 = 256;
	unsigned int kernel_size_1 = 2;
	unsigned int stride_1 = 2;
	unsigned int output_size_1 = 128;
	unsigned int pool_type_1 = 0; // max pool

// Convolution layer 2

	unsigned int input_channels_2 = 16;
	unsigned int input_size_2 = 128;
	unsigned int input_size_sq_2 = 128*128;
	unsigned int kernel_size_2 = 3;
	unsigned int pad_2 = 1;
	unsigned int stride_2 = 1;
	unsigned int output_size_2 = 128;
	unsigned int output_channels_2 = 32;
	unsigned int relu_type_2 = 1; // leaky relu


// Pooling layer 3

  unsigned int input_channels_3 = 32;
	unsigned int input_size_3 = 128;
	unsigned int kernel_size_3 = 2;
	unsigned int stride_3 = 2;
	unsigned int output_size_3 = 64;
	unsigned int pool_type_3 = 0; // max pool

// Convolution layer 4

	unsigned int input_channels_4 = 32;
	unsigned int input_size_4 = 64;
	unsigned int input_size_sq_4 = 64*64;
	unsigned int kernel_size_4 = 3;
	unsigned int pad_4 = 1;
	unsigned int stride_4 = 1;
	unsigned int output_size_4 = 64;
	unsigned int output_channels_4 = 64;
	unsigned int relu_type_4 = 1; // leaky relu


// Pooling layer 5

  unsigned int input_channels_5 = 64;
	unsigned int input_size_5 = 64;
	unsigned int kernel_size_5 = 2;
	unsigned int stride_5 = 2;
	unsigned int output_size_5 = 32;
	unsigned int pool_type_5 = 0; // max pool

// Convolution layer 6

	unsigned int input_channels_6 = 64;
	unsigned int input_size_6 = 32;
	unsigned int input_size_sq_6 = 32*32;
	unsigned int kernel_size_6 = 3;
	unsigned int pad_6 = 1;
	unsigned int stride_6 = 1;
	unsigned int output_size_6 = 32;
	unsigned int output_channels_6 = 128;
	unsigned int relu_type_6 = 1; // leaky relu


// Pooling layer 7

  unsigned int input_channels_7 = 128;
	unsigned int input_size_7 = 32;
	unsigned int kernel_size_7 = 2;
	unsigned int stride_7 = 2;
	unsigned int output_size_7 = 16;
	unsigned int pool_type_7 = 0; // max pool

// Convolution layer 8

	unsigned int input_channels_8 = 128;
	unsigned int input_size_8 = 16;
	unsigned int input_size_sq_8 = 16*16;
	unsigned int kernel_size_8 = 3;
	unsigned int pad_8 = 1;
	unsigned int stride_8 = 1;
	unsigned int output_size_8 = 16;
	unsigned int output_channels_8 = 256;
	unsigned int relu_type_8 = 1; // leaky relu


// Pooling layer 9

  unsigned int input_channels_9 = 256;
	unsigned int input_size_9 = 16;
	unsigned int kernel_size_9 = 2;
	unsigned int stride_9 = 2;
	unsigned int output_size_9 = 8;
	unsigned int pool_type_9 = 0; // max pool

// Convolution layer 10

	unsigned int input_channels_10 = 256;
	unsigned int input_size_10 = 8;
	unsigned int input_size_sq_10 = 8*8;
	unsigned int kernel_size_10 = 3;
	unsigned int pad_10 = 1;
	unsigned int stride_10 = 1;
	unsigned int output_size_10 = 8;
	unsigned int output_channels_10 = 512;
	unsigned int relu_type_10 = 1; // leaky relu


// Pooling layer 11

  unsigned int input_channels_11 = 512;
	unsigned int input_size_11 = 8;
	unsigned int kernel_size_11 = 2;
	unsigned int stride_11 = 2;
	unsigned int output_size_11 = 4;
	unsigned int pool_type_11 = 0; // max pool

// Convolution layer 12

	unsigned int input_channels_12 = 512;
	unsigned int input_size_12 = 4;
	unsigned int input_size_sq_12 = 4*4;
	unsigned int kernel_size_12 = 3;
	unsigned int pad_12 = 1;
	unsigned int stride_12 = 1;
	unsigned int output_size_12 = 4;
	unsigned int output_channels_12 = 1024;
	unsigned int relu_type_12 = 1; // leaky relu


// Pooling layer 13

  unsigned int input_channels_13 = 1024;
	unsigned int input_size_13 = 4;
	unsigned int kernel_size_13 = 4;
	unsigned int stride_13 = 1;
	unsigned int output_size_13 = 1;
	unsigned int pool_type_13 = 1; // avg pool

// Convolution layer 14

	unsigned int input_channels_14 = 1024;
	unsigned int input_size_14 = 1;
	unsigned int input_size_sq_14 = 1;
	unsigned int kernel_size_14 = 1;
	unsigned int pad_14 = 0;
	unsigned int stride_14 = 1;
	unsigned int output_size_14 = 1;
	unsigned int output_channels_14 = 1000;
	unsigned int relu_type_14 = 0; // no relu
