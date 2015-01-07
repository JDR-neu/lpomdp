################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/losm_lpomdp.cpp \
../src/losm_state.cpp \
../src/lpomdp.cpp 

OBJS += \
./src/losm_lpomdp.o \
./src/losm_state.o \
./src/lpomdp.o 

CPP_DEPS += \
./src/losm_lpomdp.d \
./src/losm_state.d \
./src/lpomdp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++1y -I"/home/axiom/Development/losm/losm" -I"/home/axiom/Development/librbr/librbr" -I"/home/axiom/Development/lpomdp/lvi_cuda" -O3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


