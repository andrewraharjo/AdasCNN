#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=MinGW-Windows
CND_DLIB_EXT=dll
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/layerdefs/ConvLayer.o \
	${OBJECTDIR}/layerdefs/FCLayer.o \
	${OBJECTDIR}/layerdefs/PoolLayer.o \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/netdef/ConvNet.o \
	${OBJECTDIR}/utility/matops.o \
	${OBJECTDIR}/utility/readpics.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/adascnn.exe

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/adascnn.exe: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/adascnn ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/layerdefs/ConvLayer.o: layerdefs/ConvLayer.cpp 
	${MKDIR} -p ${OBJECTDIR}/layerdefs
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/layerdefs/ConvLayer.o layerdefs/ConvLayer.cpp

${OBJECTDIR}/layerdefs/FCLayer.o: layerdefs/FCLayer.cpp 
	${MKDIR} -p ${OBJECTDIR}/layerdefs
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/layerdefs/FCLayer.o layerdefs/FCLayer.cpp

${OBJECTDIR}/layerdefs/PoolLayer.o: layerdefs/PoolLayer.cpp 
	${MKDIR} -p ${OBJECTDIR}/layerdefs
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/layerdefs/PoolLayer.o layerdefs/PoolLayer.cpp

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

${OBJECTDIR}/netdef/ConvNet.o: netdef/ConvNet.cpp 
	${MKDIR} -p ${OBJECTDIR}/netdef
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/netdef/ConvNet.o netdef/ConvNet.cpp

${OBJECTDIR}/utility/matops.o: utility/matops.cpp 
	${MKDIR} -p ${OBJECTDIR}/utility
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utility/matops.o utility/matops.cpp

${OBJECTDIR}/utility/readpics.o: utility/readpics.cpp 
	${MKDIR} -p ${OBJECTDIR}/utility
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/utility/readpics.o utility/readpics.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/adascnn.exe

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
