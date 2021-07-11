SRCS=random.cc pri_queue.cc util.cc qalsh.cc srp_lsh.cc l2_alsh.cc \
	l2_alsh2.cc xbox.cc simple_lsh.cc sign_alsh.cc h2_alsh.cc \
	amips.cc pre_recall.cc main.cc

SRCS2=random.cc pri_queue.cc util.cc qalsh.cc srp_lsh.cc l2_alsh.cc \
	l2_alsh2.cc xbox.cc simple_lsh.cc sign_alsh.cc h2_alsh.cc \
	amips.cc pre_recall.cc main2.cc

OBJS=${SRCS:.cc=.o}
OBJS2=${SRCS2:.cc=.o}

CXX=g++ -std=c++11
CPPFLAGS=-w -O0 -Wall -Wextra

.PHONY: clean

all: ${OBJS}
	${CXX} ${CPPFLAGS} -o alsh ${OBJS}

project: ${OBJS2}
	${CXX} ${CPPFLAGS} -o alsh2 ${OBJS2}


clean:
	-rm ${OBJS} alsh
