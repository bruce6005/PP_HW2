# Makefile: 會 cd 進 src/ 資料夾，編譯並執行 HW2.cpp

CXX = g++
CXXFLAGS = -std=c++17 -Wall -I./xsimd/include
TARGET = a.out
SRC = HW2.cpp
DIR = src

.PHONY: all clean

all:
	cd $(DIR) && $(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) && ./$(TARGET)

clean:
	rm -f $(DIR)/$(TARGET)
