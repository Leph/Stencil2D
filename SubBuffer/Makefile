
CFLAGS		+= -std=c99 -O3 -I/opt/local/cuda/include
LDFLAGS		:= -L/addons/cuda/NVIDIA-Linux-x86_64-270.41.19/
LDLIBS		:= -lOpenCL

SOURCES		:= $(wildcard *.cl)
EXEC		:= $(SOURCES:.cl=)

target: $(EXEC)

clean:
	rm -rf $(EXEC) *.o
