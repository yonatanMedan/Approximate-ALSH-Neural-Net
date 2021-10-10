SRCS2=methods/random.cc methods/pri_queue.cc methods/util.cc methods/qalsh.cc  methods/h2_alsh.cc \
	methods/main2.cc

OBJS2=${SRCS2:.cc=.o}

CXX=g++ -std=c++11
CPPFLAGS=-w -O0 -Wall -Wextra

.PHONY: clean

all: ${OBJS2}
	${CXX} ${CPPFLAGS} -o aproximate-nn-minst ${OBJS2}


clean:
	-rm ${OBJS} alsh
