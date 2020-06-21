ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif


# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config --arm)
AOCL_LINK_CONFIG := $(shell aocl link-config --arm)

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif

# Compiler. ARM cross-compiler.
CXX := arm-linux-gnueabihf-g++

# Target
TARGET := host
TARGET_DIR := bin

# Directories
INC_DIRS := host/inc common/inc ../common/inc /model C:\opencv\build\include
LIB_DIRS := 

# Files
INCS := $(wildcard host/inc/*.h)
SRCS := $(wildcard host/src/*.cpp ../common/src/AOCLUtils/*.cpp)
LIBS := rt pthread

# Make it all!
all : $(TARGET_DIR)/$(TARGET)

# Host executable target.
$(TARGET_DIR)/$(TARGET) : Makefile $(SRCS) $(INCS) $(TARGET_DIR)
	$(ECHO)$(CXX) $(CPPFLAGS) $(CXXFLAGS) -fPIC $(foreach D,$(INC_DIRS),-I$D) \
			$(AOCL_COMPILE_CONFIG) $(SRCS) $(AOCL_LINK_CONFIG) \
			$(foreach D,$(LIB_DIRS),-L$D) \
			$(foreach L,$(LIBS),-l$L) \
			-o $(TARGET_DIR)/$(TARGET)

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)
	
# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)

.PHONY : all clean
