
# Pre-requisites:
	• A Linux operating system
	• A NVIDIA GPU with a compute capability >= 6.0
	
# Installation steps:
	Install CUDA library by following the official guide in https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#introduction
	Install dependencies
		○ sudo apt-get install libboost-all-dev
		○ sudo apt-get install libpng-dev

# Declaration of mapping function (deformation):
	replace the code inside // replace the code inside "mapping" in "src/MC_estimation_cuda.cu" with the desired mapping function to use (refere to the examples folder)

# Compilation & build:
	Run "make" command [(optional name) ==> make TARGET="new_name"]
	Verify that the compilation is successful and you can run "./cuSpeckle" on the terminal that returns the help of the program
	
# Run Simple command: 
	./cuSpeckle img_out.png -width 100 -height 100
	
  Check the output image "img_out.png"
