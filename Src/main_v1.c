#include <assert.h>

#include "main.h"
#include "stm32h7xx_hal.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"

#define CONF 1
#include "intq-mobilenet-v1.h"

//Layer 1	Conv / s2
static const uint8_t conv1_wt[CONV1_KER_DIM * CONV1_KER_DIM * CONV1_IM_CH * CONV1_OUT_CH] = CONV1_WT;
static const int32_t conv1_bias[CONV1_OUT_CH] = CONV1_BIAS;

//Layer 2	Conv dw/ s1
static const uint8_t conv2_wt[CONV2_KER_DIM * CONV2_KER_DIM * 1 * CONV2_OUT_CH] = CONV2_WT;
static const int32_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

//Layer 3	Conv Point/ s1
static const uint8_t conv3_wt[CONV3_KER_DIM * CONV3_KER_DIM * CONV3_IM_CH * CONV3_OUT_CH] = CONV3_WT;
static const int32_t conv3_bias[CONV3_OUT_CH] = CONV3_BIAS;

//Layer 4	Conv dw/ s2
static const uint8_t conv4_wt[CONV4_KER_DIM * CONV4_KER_DIM * 1 * CONV4_OUT_CH] = CONV4_WT;
static const int32_t conv4_bias[ CONV4_OUT_CH] = CONV4_BIAS;

//Layer 5	Conv Point/ s1
static const uint8_t conv5_wt[CONV5_KER_DIM * CONV5_KER_DIM * CONV5_IM_CH * CONV5_OUT_CH] = CONV5_WT;
static const int32_t conv5_bias[CONV5_OUT_CH] = CONV5_BIAS;

//Layer 6	Conv dw/ s1
static const uint8_t conv6_wt[CONV6_KER_DIM * CONV6_KER_DIM * 1 * CONV6_OUT_CH] = CONV6_WT;
static const int32_t conv6_bias[CONV6_OUT_CH] = CONV6_BIAS;

//Layer 7	Conv Point/ s1
static const uint8_t conv7_wt[CONV7_KER_DIM * CONV7_KER_DIM * CONV7_IM_CH * CONV7_OUT_CH] = CONV7_WT;
static const int32_t conv7_bias[CONV7_OUT_CH] = CONV7_BIAS;

//Layer 8	Conv dw/ s2
static const uint8_t conv8_wt[CONV8_KER_DIM * CONV8_KER_DIM * 1 * CONV8_OUT_CH] = CONV8_WT;
static const int32_t conv8_bias[CONV8_OUT_CH] = CONV8_BIAS;

//Layer 9	Conv Point/ s1
static const uint8_t conv9_wt[CONV9_KER_DIM * CONV9_KER_DIM * CONV9_IM_CH * CONV9_OUT_CH] = CONV9_WT;
static const int32_t conv9_bias[CONV9_OUT_CH] = CONV9_BIAS;

//Layer 10	Conv dw/ s1
static const uint8_t conv10_wt[CONV10_KER_DIM * CONV10_KER_DIM * 1 * CONV10_OUT_CH] = CONV10_WT;
static const int32_t conv10_bias[CONV10_OUT_CH] = CONV10_BIAS;

//Layer 11	Conv Point/ s1
static const uint8_t conv11_wt[CONV11_KER_DIM * CONV11_KER_DIM * CONV11_IM_CH * CONV11_OUT_CH] = CONV11_WT;
static const int32_t conv11_bias[CONV11_OUT_CH] = CONV11_BIAS;

//Layer 12	Conv dw/ s2
static const uint8_t conv12_wt[CONV12_KER_DIM * CONV12_KER_DIM * 1 * CONV12_OUT_CH] = CONV12_WT;
static const int32_t conv12_bias[CONV12_OUT_CH] = CONV12_BIAS;

//Layer 13	Conv Point/ s1
static const uint8_t conv13_wt[CONV13_KER_DIM * CONV13_KER_DIM * CONV13_IM_CH * CONV13_OUT_CH] = CONV13_WT;
static const int32_t conv13_bias[CONV13_OUT_CH] = CONV13_BIAS;

//Layer 14	Conv dw/ s1
static const uint8_t conv14_wt[CONV14_KER_DIM * CONV14_KER_DIM * 1 * CONV14_OUT_CH] = CONV14_WT;
static const int32_t conv14_bias[CONV14_OUT_CH] = CONV14_BIAS;

//Layer 15	Conv Point/ s1
static const uint8_t conv15_wt[CONV15_KER_DIM * CONV15_KER_DIM * CONV15_IM_CH * CONV15_OUT_CH>>1] = CONV15_WT;
static const int32_t conv15_bias[CONV15_OUT_CH] = CONV15_BIAS;

//Layer 16	Conv dw/ s1
static const uint8_t conv16_wt[CONV16_KER_DIM * CONV16_KER_DIM * 1 * CONV16_OUT_CH] = CONV16_WT;
static const int32_t conv16_bias[CONV16_OUT_CH] = CONV16_BIAS;

//Layer 17	Conv Point/ s1
static const uint8_t conv17_wt[CONV17_KER_DIM * CONV17_KER_DIM * CONV17_IM_CH * CONV17_OUT_CH>>1] = CONV17_WT;
static const int32_t conv17_bias[CONV17_OUT_CH] = CONV17_BIAS;

//Layer 18	Conv dw/ s1
static const uint8_t conv18_wt[CONV18_KER_DIM * CONV18_KER_DIM * 1 * CONV18_OUT_CH] = CONV18_WT;
static const int32_t conv18_bias[CONV18_OUT_CH] = CONV18_BIAS;

//Layer 19	Conv Point/ s1
static const uint8_t conv19_wt[CONV19_KER_DIM * CONV19_KER_DIM * CONV19_IM_CH * CONV19_OUT_CH>>1] = CONV19_WT;
static const int32_t conv19_bias[CONV19_OUT_CH] = CONV19_BIAS;

//Layer 20	Conv dw/ s1
static const uint8_t conv20_wt[CONV20_KER_DIM * CONV20_KER_DIM * 1 * CONV20_OUT_CH] = CONV20_WT;
static const int32_t conv20_bias[CONV20_OUT_CH] = CONV20_BIAS;

//Layer 21	Conv Point/ s1
static const uint8_t conv21_wt[CONV21_KER_DIM * CONV21_KER_DIM * CONV21_IM_CH * CONV21_OUT_CH>>1] = CONV21_WT;
static const int32_t conv21_bias[CONV21_OUT_CH] = CONV21_BIAS;

//Layer 22	Conv dw/ s1
static const uint8_t conv22_wt[CONV22_KER_DIM * CONV22_KER_DIM * 1 * CONV22_OUT_CH] = CONV22_WT;
static const int32_t conv22_bias[CONV22_OUT_CH] = CONV22_BIAS;

//Layer 23	Conv Point/ s1
static const uint8_t conv23_wt[CONV23_KER_DIM * CONV23_KER_DIM * CONV23_IM_CH * CONV23_OUT_CH] = CONV23_WT;
static const int32_t conv23_bias[CONV23_OUT_CH] = CONV23_BIAS;

//Layer 24	Conv dw/ s2
static const uint8_t conv24_wt[CONV24_KER_DIM * CONV24_KER_DIM * 1 * CONV24_OUT_CH] = CONV24_WT;
static const int32_t conv24_bias[CONV24_OUT_CH] = CONV24_BIAS;

//Layer 25	Conv Point/ s1
static const uint8_t conv25_wt[CONV25_KER_DIM * CONV25_KER_DIM * CONV25_IM_CH * CONV25_OUT_CH>>1] = CONV25_WT;
static const int32_t conv25_bias[CONV25_OUT_CH] = CONV25_BIAS;

//Layer 26	Conv dw/ s1
static const uint8_t conv26_wt[CONV26_KER_DIM * CONV26_KER_DIM * 1 * CONV26_OUT_CH] = CONV26_WT;
static const int32_t conv26_bias[CONV26_OUT_CH] = CONV26_BIAS;

//Layer 27	Conv Point/ s1
static const uint8_t conv27_wt[CONV27_KER_DIM * CONV27_KER_DIM * CONV27_IM_CH * CONV27_OUT_CH>>2] = CONV27_WT;
static const int32_t conv27_bias[CONV27_OUT_CH] = CONV27_BIAS;

//Layer 29 fully connected/ s1
static const uint8_t fc28_wt[FC28_IM_CH*FC28_OUT_CH>>2] = FC28_WT;
static const int32_t fc28_bias[FC28_OUT_CH] = FC28_BIAS;

// Input / Output
#include "160_input_image.h"
static const uint8_t image_data[CONV1_IM_CH * CONV1_IM_DIM * CONV1_IM_DIM] = INPUT_IMAGE;
//uint8_t output_data[IP1_OUT] = {0};

// Tensors Scratch
#define L2_TENSOR_IO_SIZE   (512000) //524288

#define L1_TENSOR_SIZE      (60000) //65536
uint8_t __attribute__((section (".L2RAM"))) l2_tensor_scratch[L2_TENSOR_IO_SIZE];
int16_t                                     l1_tensor_scratch[L1_TENSOR_SIZE];

void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/******** Cycle counter defines  **********/
#define MOBILENET_PROFILE 1
volatile unsigned int cpu_cycles[30];
volatile unsigned int *DWT_CYCCNT = (volatile unsigned int *)0xE0001004; 	// Cycle Counter Register
volatile unsigned int *DWT_CONTROL= (volatile unsigned int *)0xE0001000;	// Counter Control Register
volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC;

#if MOBILENET_PROFILE
#define USR_MEM_BARRIER() \
	do { \
		asm volatile("" ::: "memory");\
	} while (0)

#define USR_CC_ENABLE() \
  do { \
	asm volatile("" ::: "memory"); \
	*SCB_DEMCR = *SCB_DEMCR | 0x01000000;\
	*DWT_CYCCNT = 0; \
	*DWT_CONTROL = *DWT_CONTROL | 1; \
	asm volatile("" ::: "memory"); \
  } while (0)

#define USR_CC_RESET() \
  do { \
	asm volatile("" ::: "memory"); \
	*DWT_CYCCNT = 0; \
	asm volatile("" ::: "memory"); \
  } while (0)

#define USR_GET_CC_TIMESTAMP(x) \
  do { \
	asm volatile("" ::: "memory"); \
	x = (*(volatile unsigned int *) DWT_CYCCNT); \
	asm volatile("" ::: "memory"); \
  } while (0)
#else
#define USR_MEM_BARRIER()
#define USR_CC_ENABLE()
#define USR_CC_RESET()
#define USR_GET_CC_TIMESTAMP(x)
#endif

#define MOBILENET_CHECKSUM 0
#if MOBILENET_CHECKSUM
volatile unsigned int checksums[30];
#define USR_CHECKSUM(x,nb_elements,id) \
  do { \
	asm volatile("" ::: "memory"); \
	checksums[id] = 0; \
	for(int i=0; i< nb_elements;i++) \
		checksums[id]+=x[i]; \
	asm volatile("" ::: "memory"); \
  } while (0)
#else
#define USR_CHECKSUM(x,nb_elements,id)
#endif

int main(void)
{
	SCB_EnableICache(); //Enable I-Cache
	SCB_EnableDCache(); //Enable D-Cache
	HAL_Init();
	SystemClock_Config();
	MX_GPIO_Init();

	while(1) {
		int 	layerId = 0;
		int 	tensorId = 0;
		uint8_t *tensorIn  = l2_tensor_scratch;
		uint8_t *tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH>>1);

		/*Layer 1	Conv / s2
		*Convolutional
		*Cycle =c
		*/
		USR_CC_ENABLE();
		USR_CC_RESET();
		arm_convolve_HWC_u8_u4_u8_icn(image_data, CONV1_IM_DIM, CONV1_IM_CH, conv1_wt, CONV1_OUT_CH, CONV1_KER_DIM, CONV1_L_PADDING, CONV1_R_PADDING, CONV1_T_PADDING, CONV1_B_PADDING, CONV1_STRIDE, conv1_bias, tensorOut, CONV1_OUT_DIM, CONV1_IN_Z, CONV1_W_Z, CONV1_OUT_Z, CONV1_M_ZERO, CONV1_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV2_OUT_DIM*CONV2_OUT_DIM*CONV2_OUT_CH>>1);
		tensorId = !tensorId;

		/*Layer 2	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u4_u4_u8_icn(tensorIn, CONV2_IM_DIM, CONV2_IM_CH, conv2_wt, CONV2_OUT_CH, CONV2_KER_DIM, CONV2_L_PADDING, CONV2_R_PADDING, CONV2_T_PADDING, CONV2_B_PADDING, CONV2_STRIDE, conv2_bias, tensorOut, CONV2_OUT_DIM, CONV2_IN_Z, CONV2_W_Z, CONV2_OUT_Z, CONV2_M_ZERO, CONV2_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV2_OUT_DIM*CONV2_OUT_DIM*CONV2_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV3_OUT_DIM*CONV3_OUT_DIM*CONV3_OUT_CH>>2);
		tensorId = !tensorId;

		/*Layer 3	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u4_u2_u8_icn(tensorIn, CONV3_IM_DIM, CONV3_IM_CH, conv3_wt, CONV3_OUT_CH, CONV3_KER_DIM, CONV3_L_PADDING, CONV3_R_PADDING, CONV3_T_PADDING, CONV3_B_PADDING, CONV3_STRIDE, conv3_bias, tensorOut, CONV3_OUT_DIM, CONV3_IN_Z, CONV3_W_Z, CONV3_OUT_Z, CONV3_M_ZERO, CONV3_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV3_OUT_DIM*CONV3_OUT_DIM*CONV3_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV4_OUT_DIM*CONV4_OUT_DIM*CONV4_OUT_CH);
		tensorId = !tensorId;

		/*Layer 4	Conv dw/ s2
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u2_u8_u8(tensorIn, CONV4_IM_DIM, CONV4_IM_CH, conv4_wt, CONV4_OUT_CH, CONV4_KER_DIM, CONV4_L_PADDING, CONV4_R_PADDING, CONV4_T_PADDING, CONV4_B_PADDING, CONV4_STRIDE, conv4_bias, tensorOut, CONV4_OUT_DIM, CONV4_IN_Z, CONV4_W_Z, CONV4_OUT_Z, CONV4_M_ZERO, CONV4_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV4_OUT_DIM*CONV4_OUT_DIM*CONV4_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV5_OUT_DIM*CONV5_OUT_DIM*CONV5_OUT_CH);
		tensorId = !tensorId;

		/*Layer 5	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u4_u8_icn(tensorIn, CONV5_IM_DIM, CONV5_IM_CH, conv5_wt, CONV5_OUT_CH, CONV5_KER_DIM, CONV5_L_PADDING, CONV5_R_PADDING, CONV5_T_PADDING, CONV5_B_PADDING, CONV5_STRIDE, conv5_bias, tensorOut, CONV5_OUT_DIM, CONV5_IN_Z, CONV5_W_Z, CONV5_OUT_Z, CONV5_M_ZERO, CONV5_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV5_OUT_DIM*CONV5_OUT_DIM*CONV5_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV6_OUT_DIM*CONV6_OUT_DIM*CONV6_OUT_CH);
		tensorId = !tensorId;

		/*Layer 6	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u4_u4_u8_icn(tensorIn, CONV6_IM_DIM, CONV6_IM_CH, conv6_wt, CONV6_OUT_CH, CONV6_KER_DIM, CONV6_L_PADDING, CONV6_R_PADDING, CONV6_T_PADDING, CONV6_B_PADDING, CONV6_STRIDE, conv6_bias, tensorOut, CONV6_OUT_DIM, CONV6_IN_Z, CONV6_W_Z, CONV6_OUT_Z, CONV6_M_ZERO, CONV6_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV6_OUT_DIM*CONV6_OUT_DIM*CONV6_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV7_OUT_DIM*CONV7_OUT_DIM*CONV7_OUT_CH);
		tensorId = !tensorId;

		/*Layer 7	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u4_u4_u8_icn(tensorIn, CONV7_IM_DIM, CONV7_IM_CH, conv7_wt, CONV7_OUT_CH, CONV7_KER_DIM, CONV7_L_PADDING, CONV7_R_PADDING, CONV7_T_PADDING, CONV7_B_PADDING, CONV7_STRIDE, conv7_bias, tensorOut, CONV7_OUT_DIM, CONV7_IN_Z, CONV7_W_Z, CONV7_OUT_Z, CONV7_M_ZERO, CONV7_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV7_OUT_DIM*CONV7_OUT_DIM*CONV7_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV8_OUT_DIM*CONV8_OUT_DIM*CONV8_OUT_CH);
		tensorId = !tensorId;

		/*Layer 8	Conv dw/ s2
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u4_u8_u8(tensorIn, CONV8_IM_DIM, CONV8_IM_CH, conv8_wt, CONV8_OUT_CH, CONV8_KER_DIM, CONV8_L_PADDING, CONV8_R_PADDING, CONV8_T_PADDING, CONV8_B_PADDING, CONV8_STRIDE, conv8_bias, tensorOut, CONV8_OUT_DIM, CONV8_IN_Z, CONV8_W_Z, CONV8_OUT_Z, CONV8_M_ZERO, CONV8_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV8_OUT_DIM*CONV8_OUT_DIM*CONV8_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV9_OUT_DIM*CONV9_OUT_DIM*CONV9_OUT_CH);
		tensorId = !tensorId;

		/*Layer 9	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u8(tensorIn, CONV9_IM_DIM, CONV9_IM_CH, conv9_wt, CONV9_OUT_CH, CONV9_KER_DIM, CONV9_L_PADDING, CONV9_R_PADDING, CONV9_T_PADDING, CONV9_B_PADDING, CONV9_STRIDE, conv9_bias, tensorOut, CONV9_OUT_DIM, CONV9_IN_Z, CONV9_W_Z, CONV9_OUT_Z, CONV9_M_ZERO, CONV9_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV9_OUT_DIM*CONV9_OUT_DIM*CONV9_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV10_OUT_DIM*CONV10_OUT_DIM*CONV10_OUT_CH);
		tensorId = !tensorId;

		/*Layer 10	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV10_IM_DIM, CONV10_IM_CH, conv10_wt, CONV10_OUT_CH, CONV10_KER_DIM, CONV10_L_PADDING, CONV10_R_PADDING, CONV10_T_PADDING, CONV10_B_PADDING, CONV10_STRIDE, conv10_bias, tensorOut, CONV10_OUT_DIM, CONV10_IN_Z, CONV10_W_Z, CONV10_OUT_Z, CONV10_M_ZERO, CONV10_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV10_OUT_DIM*CONV10_OUT_DIM*CONV10_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV11_OUT_DIM*CONV11_OUT_DIM*CONV11_OUT_CH);
		tensorId = !tensorId;

		/*Layer 11	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u8(tensorIn, CONV11_IM_DIM, CONV11_IM_CH, conv11_wt, CONV11_OUT_CH, CONV11_KER_DIM, CONV11_L_PADDING, CONV11_R_PADDING, CONV11_T_PADDING, CONV11_B_PADDING, CONV11_STRIDE, conv11_bias, tensorOut, CONV11_OUT_DIM, CONV11_IN_Z, CONV11_W_Z, CONV11_OUT_Z, CONV11_M_ZERO, CONV11_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV11_OUT_DIM*CONV11_OUT_DIM*CONV11_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV12_OUT_DIM*CONV12_OUT_DIM*CONV12_OUT_CH);
		tensorId = !tensorId;

		/*Layer 12	Conv dw/ s2
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV12_IM_DIM, CONV12_IM_CH, conv12_wt, CONV12_OUT_CH, CONV12_KER_DIM, CONV12_L_PADDING, CONV12_R_PADDING, CONV12_T_PADDING, CONV12_B_PADDING, CONV12_STRIDE, conv12_bias, tensorOut, CONV12_OUT_DIM, CONV12_IN_Z, CONV12_W_Z, CONV12_OUT_Z, CONV12_M_ZERO, CONV12_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV12_OUT_DIM*CONV12_OUT_DIM*CONV12_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV13_OUT_DIM*CONV13_OUT_DIM*CONV13_OUT_CH);
		tensorId = !tensorId;

		/*Layer 13	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u8(tensorIn, CONV13_IM_DIM, CONV13_IM_CH, conv13_wt, CONV13_OUT_CH, CONV13_KER_DIM, CONV13_L_PADDING, CONV13_R_PADDING, CONV13_T_PADDING, CONV13_B_PADDING, CONV13_STRIDE, conv13_bias, tensorOut, CONV13_OUT_DIM, CONV13_IN_Z, CONV13_W_Z, CONV13_OUT_Z, CONV13_M_ZERO, CONV13_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV13_OUT_DIM*CONV13_OUT_DIM*CONV13_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV14_OUT_DIM*CONV14_OUT_DIM*CONV14_OUT_CH);
		tensorId = !tensorId;

		/*Layer 14	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV14_IM_DIM, CONV14_IM_CH, conv14_wt, CONV14_OUT_CH, CONV14_KER_DIM, CONV14_L_PADDING, CONV14_R_PADDING, CONV14_T_PADDING, CONV14_B_PADDING, CONV14_STRIDE, conv14_bias, tensorOut, CONV14_OUT_DIM, CONV14_IN_Z, CONV14_W_Z, CONV14_OUT_Z, CONV14_M_ZERO, CONV14_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV14_OUT_DIM*CONV14_OUT_DIM*CONV14_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV15_OUT_DIM*CONV15_OUT_DIM*CONV15_OUT_CH);
		tensorId = !tensorId;

		/*Layer 15	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u4_icn(tensorIn, CONV15_IM_DIM, CONV15_IM_CH, conv15_wt, CONV15_OUT_CH, CONV15_KER_DIM, CONV15_L_PADDING, CONV15_R_PADDING, CONV15_T_PADDING, CONV15_B_PADDING, CONV15_STRIDE, conv15_bias, tensorOut, CONV15_OUT_DIM, CONV15_IN_Z, CONV15_W_Z, CONV15_OUT_Z, CONV15_M_ZERO, CONV15_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV15_OUT_DIM*CONV15_OUT_DIM*CONV15_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV16_OUT_DIM*CONV16_OUT_DIM*CONV16_OUT_CH);
		tensorId = !tensorId;

		/*Layer 16	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV16_IM_DIM, CONV16_IM_CH, conv16_wt, CONV16_OUT_CH, CONV16_KER_DIM, CONV16_L_PADDING, CONV16_R_PADDING, CONV16_T_PADDING, CONV16_B_PADDING, CONV16_STRIDE, conv16_bias, tensorOut, CONV16_OUT_DIM, CONV16_IN_Z, CONV16_W_Z, CONV16_OUT_Z, CONV16_M_ZERO, CONV16_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV16_OUT_DIM*CONV16_OUT_DIM*CONV16_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV17_OUT_DIM*CONV17_OUT_DIM*CONV17_OUT_CH);
		tensorId = !tensorId;

		/*Layer 17	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u4_icn(tensorIn, CONV17_IM_DIM, CONV17_IM_CH, conv17_wt, CONV17_OUT_CH, CONV17_KER_DIM, CONV17_L_PADDING, CONV17_R_PADDING, CONV17_T_PADDING, CONV17_B_PADDING, CONV17_STRIDE, conv17_bias, tensorOut, CONV17_OUT_DIM, CONV17_IN_Z, CONV17_W_Z, CONV17_OUT_Z, CONV17_M_ZERO, CONV17_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV17_OUT_DIM*CONV17_OUT_DIM*CONV17_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV18_OUT_DIM*CONV18_OUT_DIM*CONV18_OUT_CH);
		tensorId = !tensorId;

		/*Layer 18	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV18_IM_DIM, CONV18_IM_CH, conv18_wt, CONV18_OUT_CH, CONV18_KER_DIM, CONV18_L_PADDING, CONV18_R_PADDING, CONV18_T_PADDING, CONV18_B_PADDING, CONV18_STRIDE, conv18_bias, tensorOut, CONV18_OUT_DIM, CONV18_IN_Z, CONV18_W_Z, CONV18_OUT_Z, CONV18_M_ZERO, CONV18_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV18_OUT_DIM*CONV18_OUT_DIM*CONV18_OUT_CH,layerId); layerId++;
		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV19_OUT_DIM*CONV19_OUT_DIM*CONV19_OUT_CH);
		tensorId = !tensorId;

		/*Layer 19	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u4_icn(tensorIn, CONV19_IM_DIM, CONV19_IM_CH, conv19_wt, CONV19_OUT_CH, CONV19_KER_DIM, CONV19_L_PADDING, CONV19_R_PADDING, CONV19_T_PADDING, CONV19_B_PADDING, CONV19_STRIDE, conv19_bias, tensorOut, CONV19_OUT_DIM, CONV19_IN_Z, CONV19_W_Z, CONV19_OUT_Z, CONV19_M_ZERO, CONV19_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV19_OUT_DIM*CONV19_OUT_DIM*CONV19_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV20_OUT_DIM*CONV20_OUT_DIM*CONV20_OUT_CH);
		tensorId = !tensorId;

		/*Layer 20	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV20_IM_DIM, CONV20_IM_CH, conv20_wt, CONV20_OUT_CH, CONV20_KER_DIM, CONV20_L_PADDING, CONV20_R_PADDING, CONV20_T_PADDING, CONV20_B_PADDING, CONV20_STRIDE, conv20_bias, tensorOut, CONV20_OUT_DIM, CONV20_IN_Z, CONV20_W_Z, CONV20_OUT_Z, CONV20_M_ZERO, CONV20_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV20_OUT_DIM*CONV20_OUT_DIM*CONV20_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV21_OUT_DIM*CONV21_OUT_DIM*CONV21_OUT_CH);
		tensorId = !tensorId;

		/*Layer 21	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u4_icn(tensorIn, CONV21_IM_DIM, CONV21_IM_CH, conv21_wt, CONV21_OUT_CH, CONV21_KER_DIM, CONV21_L_PADDING, CONV21_R_PADDING, CONV21_T_PADDING, CONV21_B_PADDING, CONV21_STRIDE, conv21_bias, tensorOut, CONV21_OUT_DIM, CONV21_IN_Z, CONV21_W_Z, CONV21_OUT_Z, CONV21_M_ZERO, CONV21_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV21_OUT_DIM*CONV21_OUT_DIM*CONV21_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV22_OUT_DIM*CONV22_OUT_DIM*CONV22_OUT_CH);
		tensorId = !tensorId;

		/*Layer 22	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV22_IM_DIM, CONV22_IM_CH, conv22_wt, CONV22_OUT_CH, CONV22_KER_DIM, CONV22_L_PADDING, CONV22_R_PADDING, CONV22_T_PADDING, CONV22_B_PADDING, CONV22_STRIDE, conv22_bias, tensorOut, CONV22_OUT_DIM, CONV22_IN_Z, CONV22_W_Z, CONV22_OUT_Z, CONV22_M_ZERO, CONV22_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV22_OUT_DIM*CONV22_OUT_DIM*CONV22_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV23_OUT_DIM*CONV23_OUT_DIM*CONV23_OUT_CH);
		tensorId = !tensorId;

		/*Layer 23	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u8(tensorIn, CONV23_IM_DIM, CONV23_IM_CH, conv23_wt, CONV23_OUT_CH, CONV23_KER_DIM, CONV23_L_PADDING, CONV23_R_PADDING, CONV23_T_PADDING, CONV23_B_PADDING, CONV23_STRIDE, conv23_bias, tensorOut, CONV23_OUT_DIM, CONV23_IN_Z, CONV23_W_Z, CONV23_OUT_Z, CONV23_M_ZERO, CONV23_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV23_OUT_DIM*CONV23_OUT_DIM*CONV23_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV24_OUT_DIM*CONV24_OUT_DIM*CONV24_OUT_CH);
		tensorId = !tensorId;

		/*Layer 24	Conv dw/ s2
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV24_IM_DIM, CONV24_IM_CH, conv24_wt, CONV24_OUT_CH, CONV24_KER_DIM, CONV24_L_PADDING, CONV24_R_PADDING, CONV24_T_PADDING, CONV24_B_PADDING, CONV24_STRIDE, conv24_bias, tensorOut, CONV24_OUT_DIM, CONV24_IN_Z, CONV24_W_Z, CONV24_OUT_Z, CONV24_M_ZERO, CONV24_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV24_OUT_DIM*CONV24_OUT_DIM*CONV24_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV25_OUT_DIM*CONV25_OUT_DIM*CONV25_OUT_CH);
		tensorId = !tensorId;

		/*Layer 25	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u4_icn(tensorIn, CONV25_IM_DIM, CONV25_IM_CH, conv25_wt, CONV25_OUT_CH, CONV25_KER_DIM, CONV25_L_PADDING, CONV25_R_PADDING, CONV25_T_PADDING, CONV25_B_PADDING, CONV25_STRIDE, conv25_bias, tensorOut, CONV25_OUT_DIM, CONV25_IN_Z, CONV25_W_Z, CONV25_OUT_Z, CONV25_M_ZERO, CONV25_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV25_OUT_DIM*CONV25_OUT_DIM*CONV25_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV26_OUT_DIM*CONV26_OUT_DIM*CONV26_OUT_CH);
		tensorId = !tensorId;

		/*Layer 26	Conv dw/ s1
		*dephtwise
		*Cycle =
		*/
		arm_depthwise_separable_conv_HWC_u8_u8_u8(tensorIn, CONV26_IM_DIM, CONV26_IM_CH, conv26_wt, CONV26_OUT_CH, CONV26_KER_DIM, CONV26_L_PADDING, CONV26_R_PADDING, CONV26_T_PADDING, CONV26_B_PADDING, CONV26_STRIDE, conv26_bias, tensorOut, CONV26_OUT_DIM, CONV26_IN_Z, CONV26_W_Z, CONV26_OUT_Z, CONV26_M_ZERO, CONV26_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV26_OUT_DIM*CONV26_OUT_DIM*CONV26_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (CONV27_OUT_DIM*CONV27_OUT_DIM*CONV27_OUT_CH);
		tensorId = !tensorId;

		/*Layer 27	Conv Point/ s1
		*pointwise
		*Cycle =
		*/
		arm_convolve_HWC_u8_u8_u2_icn(tensorIn, CONV27_IM_DIM, CONV27_IM_CH, conv27_wt, CONV27_OUT_CH, CONV27_KER_DIM, CONV27_L_PADDING, CONV27_R_PADDING, CONV27_T_PADDING, CONV27_B_PADDING, CONV27_STRIDE, conv27_bias, tensorOut, CONV27_OUT_DIM, CONV27_IN_Z, CONV27_W_Z, CONV27_OUT_Z, CONV27_M_ZERO, CONV27_N_ZERO, l1_tensor_scratch, NULL);
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,CONV27_OUT_DIM*CONV27_OUT_DIM*CONV27_OUT_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (FC28_IM_CH);
		tensorId = !tensorId;

		/*Layer 28	Avgpool */
//		arm_avepool_asym_uint8_HWC(tensorIn, CONV27_OUT_DIM, CONV27_OUT_CH, 5, 0, 5, 1, l1_tensor_scratch, tensorOut);

		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,FC28_IM_CH,layerId); layerId++;

		tensorIn  = tensorOut;
		if (!tensorId)
			tensorOut = l2_tensor_scratch;
		else
			tensorOut = l2_tensor_scratch + L2_TENSOR_IO_SIZE - (FC28_OUT_CH);
		tensorId = !tensorId;

		/*Layer 29	FC
		*Cycle =
		*/
//		arm_fully_connected_in_out_wt_icn(tensorIn, fc28_wt, FC28_IM_CH, FC28_OUT_CH, FC28_W_Z, FC28_IN_Z, FC28_OUT_Z, FC28_M_ZERO, FC28_N_ZERO, fc28_bias, tensorOut, l1_tensor_scratch);

		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);
		USR_CHECKSUM(tensorOut,FC28_OUT_CH,layerId); layerId++;

		USR_MEM_BARRIER();
		int top5[5];
		int top5_score[5];
		for (int i=0; i<5; i++){
			int max=0;
			for(int ii=0; ii<FC28_OUT_CH; ii++)
				if (tensorOut[ii] > max){
					top5[i] = ii;
					top5_score[i] = tensorOut[ii];
					max= tensorOut[ii];
				}
			tensorOut[top5[i]]=0;
		}
		USR_GET_CC_TIMESTAMP(cpu_cycles[layerId]);

		for (int i = 0; i < 5; i++){
		      printf("%d: %d\n", top5[i], top5_score[i]);
		}
	}
  return 0;
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  MODIFY_REG(PWR->CR3, PWR_CR3_SCUEN, 0);
  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while ((PWR->D3CR & (PWR_D3CR_VOSRDY)) != PWR_D3CR_VOSRDY)
  {

  }
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 50;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pins : PE2 PE3 PE4 PE5
                           PE6 PE7 PE8 PE9
                           PE10 PE11 PE12 PE13
                           PE14 PE15 PE0 PE1 */
  GPIO_InitStruct.Pin = GPIO_PIN_2|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5
                          |GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9
                          |GPIO_PIN_10|GPIO_PIN_11|GPIO_PIN_12|GPIO_PIN_13
                          |GPIO_PIN_14|GPIO_PIN_15|GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pins : PC13 PC14 PC15 PC0
                           PC1 PC2 PC3 PC4
                           PC5 PC6 PC7 PC8
                           PC9 PC10 PC11 PC12 */
  GPIO_InitStruct.Pin = GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15|GPIO_PIN_0
                          |GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3|GPIO_PIN_4
                          |GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8
                          |GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11|GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : PF0 PF1 PF2 PF3
                           PF4 PF5 PF6 PF7
                           PF8 PF9 PF10 PF11
                           PF12 PF13 PF14 PF15 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7
                          |GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11
                          |GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /*Configure GPIO pins : PH0 PH1 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

  /*Configure GPIO pins : PA0 PA1 PA2 PA3
                           PA4 PA5 PA6 PA7
                           PA8 PA9 PA10 PA11
                           PA12 PA15 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7
                          |GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11
                          |GPIO_PIN_12|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 PB1 PB2 PB10
                           PB11 PB12 PB13 PB14
                           PB15 PB3 PB4 PB5
                           PB6 PB7 PB8 PB9 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_10
                          |GPIO_PIN_11|GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14
                          |GPIO_PIN_15|GPIO_PIN_3|GPIO_PIN_4|GPIO_PIN_5
                          |GPIO_PIN_6|GPIO_PIN_7|GPIO_PIN_8|GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : PG0 PG1 PG2 PG3
                           PG4 PG5 PG6 PG7
                           PG8 PG9 PG10 PG11
                           PG12 PG13 PG14 PG15 */
  GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7
                          |GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11
                          |GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pins : PD8 PD9 PD10 PD11
                           PD12 PD13 PD14 PD15
                           PD0 PD1 PD2 PD3
                           PD4 PD5 PD6 PD7 */
  GPIO_InitStruct.Pin = GPIO_PIN_8|GPIO_PIN_9|GPIO_PIN_10|GPIO_PIN_11
                          |GPIO_PIN_12|GPIO_PIN_13|GPIO_PIN_14|GPIO_PIN_15
                          |GPIO_PIN_0|GPIO_PIN_1|GPIO_PIN_2|GPIO_PIN_3
                          |GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  while(1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
