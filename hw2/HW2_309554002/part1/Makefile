LIBS=-pthread

TARGET2=pi.out
CPP2=g++
CPP_FLAGS2=-std=c++11 -O2
SOURCE2=pi.cpp

all: pi.cpp
	$(CPP2) $(SOURCE2) -o $(TARGET2) $(LIBS)

.PHONY: clean
clean:
	rm  *.out
