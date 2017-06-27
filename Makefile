# the compiler: gcc for C program, define as g++ for C++
CXX = clang++

CFLAGS = -O2 -std=c++14 -Wall -lhdf5 -I HighFive/include

# the build target executable:
TARGET = mincall/tools/align_ref

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET)
