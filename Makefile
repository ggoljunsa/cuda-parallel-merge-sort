# Makefile for the CUDA Parallel Merge Sort capstone project.
#
# Override the toolkit location if CUDA lives elsewhere, e.g.:
#     make NVCC=/usr/local/cuda-12.2/bin/nvcc

NVCC      ?= /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda

SRC_DIR = src
BIN_DIR = bin
TARGET  = $(BIN_DIR)/cuda_merge_sort
SRC     = $(SRC_DIR)/cuda_merge_sort.cu

NVCCFLAGS = -std=c++14 -O3 -I$(CUDA_PATH)/include
LDFLAGS   = -L$(CUDA_PATH)/lib64 -lcudart

.PHONY: all clean help

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

help:
	@echo "Available targets:"
	@echo "  make        - Build $(TARGET)"
	@echo "  make clean  - Remove the built binary"
	@echo "  make help   - Show this message"
	@echo ""
	@echo "Override the CUDA toolkit path with:"
	@echo "  make NVCC=/path/to/nvcc CUDA_PATH=/path/to/cuda"
