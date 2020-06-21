
__kernel void conv3x3(
	__global float *restrict input_im,
	__global const float *restrict filter_weight,
	__global float *restrict output_im,
	const int input_channels, const int input_size,
	const int pad, const int stride,
	const int output_size
	)
{
	int filter_index = get_global_id(0);
	int i =  get_global_id(1);

	filter_weight += filter_index * input_channels * 9;
	output_im += filter_index * output_size * output_size;

	//loop over output feature map
	//for(int i = 0; i < output_size; i++)
	{
		for(int j = 0; j < output_size; j++)
		{
			float tmp = 0;

			for(int k = 0; k < input_channels; k++)
			{
				#pragma unroll
				for(int l = 0; l < 3; l++)
				{
					int h = i * stride + l - pad;
					for(int m = 0; m < 3; m++)
					{
						int w = j * stride + m - pad;
						if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
						{
							tmp += input_im[k * input_size * input_size + h * input_size + w] \
                               * filter_weight[9 * k + 3 * l + m];
						}
					}
				}
			}

			output_im[i * output_size + j] = tmp;
                
                
		}
	}
}

                
__kernel void batchnorm(__global float *restrict in_data,
						__global float *restrict bn_weights,
						__global float *restrict bn_biases,
						__global float *restrict bn_running_mean,
						__global float *restrict bn_running_var,
						__global float *restrict out_data,
						const int in_size,
						const float eps,
						const int relu_type
						)
{
    int filter_index = get_global_id(0);
    int pixel_index = get_global_id(1);
    int index = filter_index*in_size + pixel_index;
    float out;

    out = (bn_weights[filter_index] * ((in_data[index] - bn_running_mean[filter_index])/sqrt(bn_running_var[filter_index] + eps))) + bn_biases[filter_index] ;

    if(relu_type == 1)
        out_data[index] = (out > 0.0f) ? out : (out * 0.1f); // leaky relu
    else
        out_data[index] = out;
}


__kernel void pool(__global float *restrict in_data,
				   __global float *restrict out_data,
				   const int in_size,
				   const int kernel_size,
				   const int stride,
				   const int out_size,
                   const int pool_type
				   )
{
	int filter_index = get_global_id(0);

	for(int row=0;row<out_size;row++)
	{
		for(int col=0;col<out_size;col++)
		{
            float tmp = -100;
                
			if(pool_type==1)
                tmp = 0;
                

			for(int i=0; i<(kernel_size*kernel_size); i++)
			{
				int k_row = i/kernel_size;
				int k_col = i - (k_row * kernel_size);

				float data = in_data[filter_index*in_size*in_size+(row*stride+k_row)*in_size+(col*stride+k_col)];
                
                if(pool_type==1)
                    tmp += data/16;                    
                else
                {
                    if(tmp < data)
                        tmp = data;                    
                }
			}
			out_data[filter_index*out_size*out_size+row*out_size+col]=tmp;
		}
	}
}

        
__kernel void conv1x1(
	__global float *restrict input_im,
	__global const float *restrict filter_weight,
	__global float *restrict output_im,
	const int input_channels, const int input_size,
	const int pad, const int stride,
	const int output_size
	)
{
	int filter_index = get_global_id(0);
	int i =  get_global_id(1);

	filter_weight += filter_index * input_channels;
	output_im += filter_index * output_size * output_size;

	//for(int i = 0; i < output_size; i++)
	{
		for(int j = 0; j < output_size; j++)
		{
			float tmp = 0;

			for(int k = 0; k < input_channels; k++)
			{
				int h = i * stride - pad;
                int w = j * stride - pad;
                    
				if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size))
				{
                    tmp += input_im[k * input_size * input_size + h * input_size + w] \
                            * filter_weight[k];
				}
			}

			output_im[i * output_size + j] = tmp;
                
                
		}
	}
}