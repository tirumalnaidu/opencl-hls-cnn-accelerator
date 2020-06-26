## About
We designed a Neural Network Accelerator for [Darknet Reference Model](https://pjreddie.com/darknet/imagenet/#reference) (which  is 2.9 times faster than AlexNet and attains the same top-1 and top-5 performance as AlexNet but with 1/10th the parameters). 

## Board
- [Terasic DE10-Nano Development Kit (Cyclone V SoC FPGA)](https://software.intel.com/content/www/us/en/develop/articles/terasic-de10-nano-get-started-guide.html)

## Requirements
- [Intel® FPGA SDK for OpenCL™ 18.1](https://fpgasoftware.intel.com/opencl/18.1/?edition=standard)
- [Intel® SoC FPGA Embedded Development Suite 18.1 (SoC EDS)](https://fpgasoftware.intel.com/soceds/18.1/?edition=standard)
- [PuTTY](https://www.putty.org/)
- PyTorch
- PyOpenCL 

## Darknet Refernce Model

<center>

|    | Layer    | Filters | Kernel Size | Stride | Pad |   Input Size  |  Output Size   |   
|---:|---------:|--------:|------------:|-------:|----:|-------------:|--------------:| 
| 1  |conv      | 16      | 3 x 3       | 1      | 1   |256 x 256 x 3  | 256 x 256 x 16 | 
| 2  |max       |         | 2 x 2       | 2      | 0   |256 x 256 x 16 | 128 x 128 x 16 | 
| 3  |conv      | 32      | 3 x 3       | 1      | 1   |128 x 128 x 16 | 128 x 128 x 32 |
| 4  |max       |         | 2 x 2       | 2      | 0   |128 x 128 x 32 | 64 x 64 x 32   |
| 5  |conv      | 64      | 3 x 3       | 1      | 1   |64 x 64 x 32   | 64 x 64 x 64   | 
| 6  |max       |         | 2 x 2       | 2      | 0   |64 x 64 x 64   | 32 x 32 x 64   | 
| 7  |conv      | 128     | 3 x 3       | 1      | 1   |32 x 32 x 64   | 32 x 32 x 128  | 
| 8  |max       |         | 2 x 2       | 2      | 0   |32 x 32 x 128  | 16 x 16 x 128  | 
| 9  |conv      | 256     | 3 x 3       | 1      | 1   |16 x 16 x 128  | 16 x 16 x 256  | 
| 10 |max       |         | 2 x 2       | 2      | 0   |16 x 16 x 256  | 8 x 8 x 256    |
| 11 |conv      | 512     | 3 x 3       | 1      | 1   |8 x 8 x 256    | 8 x 8 x 512    |
| 12 |max       |         | 2 x 2       | 2      | 0   |8 x 8 x 512    | 4 x 4 x 512    | 
| 13 |conv      | 1024    | 3 x 3       | 1      | 1   |4 x 4 x 512    | 4 x 4 x 1024   | 
| 14 |avg       |         | 4 x 4       | 1      | 0   |4 x 4 x 1024   | 1 x 1 x 1024   | 
| 15 |conv      | 1000    | 1 x 1       | 1      | 0   |1 x 1 x 1024   | 1 x 1 x 1000   | 

</center>

