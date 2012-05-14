
DEFINES		:= 
CFLAGS		+= -std=c99 -O3 -I/opt/local/cuda/include -fopenmp $(DEFINES)
LDFLAGS		:= -L/addons/cuda/NVIDIA-Linux-x86_64-270.41.19/
LDLIBS		:= -lOpenCL -lpthread

SOURCES		:= $(wildcard *.cl)
EXEC		:= $(SOURCES:.cl=)

target: clean $(EXEC)

clean:
	rm -rf $(EXEC) *.o
