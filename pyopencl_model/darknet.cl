#define CONV_BLOCK_SIZE 8

__kernel void conv(__global float *restrict in_data,
		__global float *restrict conv_wt,
		__global float *restrict out_data,
		const int N_elem,
		const int K_conv,
		const int S_conv,
		const int P_conv,
		const int N_Fin,			
		const int N_Fin_dim,		
		const int N_Fin_sq_pad,	
		const int N_Fout_dim	
		)
{
	__local float weight[CONV_BLOCK_SIZE][CONV_BLOCK_SIZE];
    __local float input[CONV_BLOCK_SIZE][CONV_BLOCK_SIZE];

    int block_y = get_group_id(1);

    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

	int global_y = get_global_id(1);
	int global_x = get_global_id(0);

	int K_conv_sq = K_conv*K_conv;

    int a_start = N_elem * CONV_BLOCK_SIZE * block_y;
    int a_end   = a_start + N_elem - 1;

    float out=0.0;
    
    for (int a = a_start, b = 0; a <= a_end; a += CONV_BLOCK_SIZE, b += CONV_BLOCK_SIZE)
    {
        weight[local_y][local_x] = conv_wt[a + N_elem * local_y + local_x];

   		ushort gx = b+local_y;
		ushort gy = global_x;
		ushort ch_no = gx/K_conv_sq;
		ushort i_k = gx-(ch_no*K_conv_sq);
		ushort k_row_no = i_k/K_conv;
		short k_row = k_row_no - P_conv;	
		short k_col = i_k - (k_row_no*K_conv) - P_conv;
		ushort out_feat_row = gy/N_Fout_dim;
		short row = (out_feat_row)*S_conv + k_row;
		short col = (gy-(out_feat_row*N_Fout_dim))*S_conv + k_col;
		unsigned location =  ch_no*N_Fin_sq_pad + row*N_Fin_dim + col;
		
		float data ;
		if(gx > N_Fin*K_conv_sq || gy > N_Fout_dim*N_Fout_dim || 
			row<0 || col<0 || row >= N_Fin_dim || col >= N_Fin_dim)	
			data=0.0;
		else
			data = in_data[location];

		input[local_x][local_y] = data;

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < CONV_BLOCK_SIZE; ++k)
        {
			out += (weight[local_y][k] * input[local_x][k]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }
    out_data[get_global_id(1) * get_global_size(0) + get_global_id(0)] = out;
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
        out_data[index] = out; // no activation
}


__kernel void maxpool(__global float *restrict in_data,
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
            float tmp1 = 0.0;
            float tmp2 = -100.0;
                
			for(int i=0; i<(kernel_size*kernel_size); i++)
			{
				int k_row = i/kernel_size;
				int k_col = i - (k_row * kernel_size);

				float data = in_data[filter_index*in_size*in_size+(row*stride+k_row)*in_size+(col*stride+k_col)];
                
                if(pool_type==1) 
                    tmp1 += data/(kernel_size*kernel_size);  // average pooling                                   
                else {
                    if(tmp2 < data) tmp2 = data;  // max pooling
                }
			}
			out_data[filter_index*out_size*out_size+row*out_size+col]= pool_type ? tmp1 : tmp2;
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

	//loop over output feature map
	//for(int i = 0; i < output_size; i++)
	{
		for(int j = 0; j < output_size; j++)
		{
			float tmp = 0;

			for(int k = 0; k < input_channels; k++)
			{
				int h = i * stride - pad;
                int w = j * stride - pad;
                    
				if((h >= 0) && (h < input_size) && (w >= 0) && (w < input_size)) {
                    tmp += input_im[k * input_size * input_size + h * input_size + w] \
                            * filter_weight[k];
				}
			}
			output_im[i * output_size + j] = tmp;                              
		}
	}
}
            

    
