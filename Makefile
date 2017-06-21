# the compiler: gcc for C program, define as g++ for C++
CXX = clang++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -std=c++14 -O2 -Wall -lhdf5 -I HighFive/include

# the build target executable:
TARGET = mincall/tools/align_ref

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

clean:
	$(RM) $(TARGET)
