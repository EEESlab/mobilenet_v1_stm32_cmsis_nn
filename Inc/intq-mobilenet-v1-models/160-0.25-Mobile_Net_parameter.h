#ifndef __MOBILE_NET_PARAMS__
#define __MOBILE_NET_PARAMS__

/*
 * Layer 1
 * Conv / s2
 * Padding: 0
 * INPUT: 160 x 160 x 3
 * KERNEL: 3 x 3 x 3 x 8
 * OUTPUT: 80 x 80 x 8
 */
#define CONV1_IM_DIM 160
#define CONV1_IM_CH 3
#define CONV1_KER_DIM 3
#define CONV1_L_PADDING 0
#define CONV1_R_PADDING 0
#define CONV1_T_PADDING 0
#define CONV1_B_PADDING 0
#define CONV1_STRIDE 2
#define CONV1_OUT_CH 8
#define CONV1_OUT_DIM 80

 /*
 * Layer 2
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 80 x 80 x 8
 * KERNEL: 3 x 3 x 8
 * OUTPUT: 80 x 80 x 8
 */
#define CONV2_IM_DIM 80
#define CONV2_IM_CH 8
#define CONV2_KER_DIM 3
#define CONV2_L_PADDING 1
#define CONV2_R_PADDING 1
#define CONV2_T_PADDING 1
#define CONV2_B_PADDING 1
#define CONV2_STRIDE 1
#define CONV2_OUT_CH 8
#define CONV2_OUT_DIM 80
 /*
 * Layer 3
 * Conv / s1
 * Padding: 0
 * INPUT: 80 x 80 x 8
 * KERNEL: 1 x 1 x 8 x 16
 * OUTPUT: 80 x 80 x 16
 */
#define CONV3_IM_DIM 80
#define CONV3_IM_CH 8
#define CONV3_KER_DIM 1
#define CONV3_L_PADDING 0
#define CONV3_R_PADDING 0
#define CONV3_T_PADDING 0
#define CONV3_B_PADDING 0
#define CONV3_STRIDE 1
#define CONV3_OUT_CH 16
#define CONV3_OUT_DIM 80
 /*
 * Layer 4
 * Conv dw/ s2
 * Padding: 0
 * INPUT: 80 x 80 x 16
 * KERNEL: 3 x 3 x 16
 * OUTPUT: 40 x 40 x 16
 */
#define CONV4_IM_DIM 80
#define CONV4_IM_CH 16
#define CONV4_KER_DIM 3
#define CONV4_L_PADDING 0
#define CONV4_R_PADDING 0
#define CONV4_T_PADDING 0
#define CONV4_B_PADDING 0
#define CONV4_STRIDE 2
#define CONV4_OUT_CH 16
#define CONV4_OUT_DIM 40
 /*
 * Layer 5
 * Conv / s1
 * Padding: 0
 * INPUT: 40 x 40 x 16
 * KERNEL: 1 x 1 x 16 x 32
 * OUTPUT: 40 x 40 x 32
 */
#define CONV5_IM_DIM 40
#define CONV5_IM_CH 16
#define CONV5_KER_DIM 1
#define CONV5_L_PADDING 0
#define CONV5_R_PADDING 0
#define CONV5_T_PADDING 0
#define CONV5_B_PADDING 0
#define CONV5_STRIDE 1
#define CONV5_OUT_CH 32
#define CONV5_OUT_DIM 40
 /*
 * Layer 6
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 40 x 40 x 32
 * KERNEL: 3 x 3 x 32
 * OUTPUT: 40 x 40 x 32
 */
#define CONV6_IM_DIM 40
#define CONV6_IM_CH 32
#define CONV6_KER_DIM 3
#define CONV6_L_PADDING 1
#define CONV6_R_PADDING 1
#define CONV6_T_PADDING 1
#define CONV6_B_PADDING 1
#define CONV6_STRIDE 1
#define CONV6_OUT_CH 32
#define CONV6_OUT_DIM 40
 /*
 * Layer 7
 * Conv / s1
 * Padding: 0
 * INPUT: 40 x 40 x 32
 * KERNEL: 1 x 1 x 32 x 32
 * OUTPUT: 40 x 40 x 32
 */
#define CONV7_IM_DIM 40
#define CONV7_IM_CH 32
#define CONV7_KER_DIM 1
#define CONV7_L_PADDING 0
#define CONV7_R_PADDING 0
#define CONV7_T_PADDING 0
#define CONV7_B_PADDING 0
#define CONV7_STRIDE 1
#define CONV7_OUT_CH 32
#define CONV7_OUT_DIM 40
 /*
 * Layer 8
 * Conv dw/ s2
 * Padding: 0
 * INPUT: 40 x 40 x 32
 * KERNEL: 3 x 3 x 32
 * OUTPUT: 20 x 20 x 32
 */
#define CONV8_IM_DIM 40
#define CONV8_IM_CH 32
#define CONV8_KER_DIM 3
#define CONV8_L_PADDING 0
#define CONV8_R_PADDING 0
#define CONV8_T_PADDING 0
#define CONV8_B_PADDING 0
#define CONV8_STRIDE 2
#define CONV8_OUT_CH 32
#define CONV8_OUT_DIM 20
 /*
 * Layer 9
 * Conv / s1
 * Padding: 0
 * INPUT: 20 x 20 x 32
 * KERNEL: 1 x 1 x 32 x 64
 * OUTPUT: 20 x 20 x 64
 */
#define CONV9_IM_DIM 20
#define CONV9_IM_CH 32
#define CONV9_KER_DIM 1
#define CONV9_L_PADDING 0
#define CONV9_R_PADDING 0
#define CONV9_T_PADDING 0
#define CONV9_B_PADDING 0
#define CONV9_STRIDE 1
#define CONV9_OUT_CH 64
#define CONV9_OUT_DIM 20
 /*
 * Layer 10
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 20 x 20 x 64
 * KERNEL: 3 x 3 x 64
 * OUTPUT: 20 x 20 x 64
 */
#define CONV10_IM_DIM 20
#define CONV10_IM_CH 64
#define CONV10_KER_DIM 3
#define CONV10_L_PADDING 1
#define CONV10_R_PADDING 1
#define CONV10_T_PADDING 1
#define CONV10_B_PADDING 1
#define CONV10_STRIDE 1
#define CONV10_OUT_CH 64
#define CONV10_OUT_DIM 20
 /*
 * Layer 11
 * Conv / s1
 * Padding: 0
 * INPUT: 20 x 20 x 64
 * KERNEL: 1 x 1 x 64 x 64
 * OUTPUT: 20 x 20 x 64
 */
#define CONV11_IM_DIM 20
#define CONV11_IM_CH 64
#define CONV11_KER_DIM 1
#define CONV11_L_PADDING 0
#define CONV11_R_PADDING 0
#define CONV11_T_PADDING 0
#define CONV11_B_PADDING 0
#define CONV11_STRIDE 1
#define CONV11_OUT_CH 64
#define CONV11_OUT_DIM 20
 /*
 * Layer 12
 * Conv dw/ s2
 * Padding: 0
 * INPUT: 20 x 20 x 64
 * KERNEL: 3 x 3 x 64
 * OUTPUT: 10 x 10 x 64
 */
#define CONV12_IM_DIM 20
#define CONV12_IM_CH 64
#define CONV12_KER_DIM 3
#define CONV12_L_PADDING 0
#define CONV12_R_PADDING 0
#define CONV12_T_PADDING 0
#define CONV12_B_PADDING 0
#define CONV12_STRIDE 2
#define CONV12_OUT_CH 64
#define CONV12_OUT_DIM 10
 /*
 * Layer 13
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 64
 * KERNEL: 1 x 1 x 64 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV13_IM_DIM 10
#define CONV13_IM_CH 64
#define CONV13_KER_DIM 1
#define CONV13_L_PADDING 0
#define CONV13_R_PADDING 0
#define CONV13_T_PADDING 0
#define CONV13_B_PADDING 0
#define CONV13_STRIDE 1
#define CONV13_OUT_CH 128
#define CONV13_OUT_DIM 10
 /*
 * Layer 14
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV14_IM_DIM 10
#define CONV14_IM_CH 128
#define CONV14_KER_DIM 3
#define CONV14_L_PADDING 1
#define CONV14_R_PADDING 1
#define CONV14_T_PADDING 1
#define CONV14_B_PADDING 1
#define CONV14_STRIDE 1
#define CONV14_OUT_CH 128
#define CONV14_OUT_DIM 10
 /*
 * Layer 15
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 1 x 1 x 128 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV15_IM_DIM 10
#define CONV15_IM_CH 128
#define CONV15_KER_DIM 1
#define CONV15_L_PADDING 0
#define CONV15_R_PADDING 0
#define CONV15_T_PADDING 0
#define CONV15_B_PADDING 0
#define CONV15_STRIDE 1
#define CONV15_OUT_CH 128
#define CONV15_OUT_DIM 10
 /*
 * Layer 16
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV16_IM_DIM 10
#define CONV16_IM_CH 128
#define CONV16_KER_DIM 3
#define CONV16_L_PADDING 1
#define CONV16_R_PADDING 1
#define CONV16_T_PADDING 1
#define CONV16_B_PADDING 1
#define CONV16_STRIDE 1
#define CONV16_OUT_CH 128
#define CONV16_OUT_DIM 10
 /*
 * Layer 17
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 1 x 1 x 128 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV17_IM_DIM 10
#define CONV17_IM_CH 128
#define CONV17_KER_DIM 1
#define CONV17_L_PADDING 0
#define CONV17_R_PADDING 0
#define CONV17_T_PADDING 0
#define CONV17_B_PADDING 0
#define CONV17_STRIDE 1
#define CONV17_OUT_CH 128
#define CONV17_OUT_DIM 10
 /*
 * Layer 18
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV18_IM_DIM 10
#define CONV18_IM_CH 128
#define CONV18_KER_DIM 3
#define CONV18_L_PADDING 1
#define CONV18_R_PADDING 1
#define CONV18_T_PADDING 1
#define CONV18_B_PADDING 1
#define CONV18_STRIDE 1
#define CONV18_OUT_CH 128
#define CONV18_OUT_DIM 10
 /*
 * Layer 19
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 1 x 1 x 128 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV19_IM_DIM 10
#define CONV19_IM_CH 128
#define CONV19_KER_DIM 1
#define CONV19_L_PADDING 0
#define CONV19_R_PADDING 0
#define CONV19_T_PADDING 0
#define CONV19_B_PADDING 0
#define CONV19_STRIDE 1
#define CONV19_OUT_CH 128
#define CONV19_OUT_DIM 10
 /*
 * Layer 20
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV20_IM_DIM 10
#define CONV20_IM_CH 128
#define CONV20_KER_DIM 3
#define CONV20_L_PADDING 1
#define CONV20_R_PADDING 1
#define CONV20_T_PADDING 1
#define CONV20_B_PADDING 1
#define CONV20_STRIDE 1
#define CONV20_OUT_CH 128
#define CONV20_OUT_DIM 10
 /*
 * Layer 21
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 1 x 1 x 128 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV21_IM_DIM 10
#define CONV21_IM_CH 128
#define CONV21_KER_DIM 1
#define CONV21_L_PADDING 0
#define CONV21_R_PADDING 0
#define CONV21_T_PADDING 0
#define CONV21_B_PADDING 0
#define CONV21_STRIDE 1
#define CONV21_OUT_CH 128
#define CONV21_OUT_DIM 10
 /*
 * Layer 22
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV22_IM_DIM 10
#define CONV22_IM_CH 128
#define CONV22_KER_DIM 3
#define CONV22_L_PADDING 1
#define CONV22_R_PADDING 1
#define CONV22_T_PADDING 1
#define CONV22_B_PADDING 1
#define CONV22_STRIDE 1
#define CONV22_OUT_CH 128
#define CONV22_OUT_DIM 10
 /*
 * Layer 23
 * Conv / s1
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 1 x 1 x 128 x 128
 * OUTPUT: 10 x 10 x 128
 */
#define CONV23_IM_DIM 10
#define CONV23_IM_CH 128
#define CONV23_KER_DIM 1
#define CONV23_L_PADDING 0
#define CONV23_R_PADDING 0
#define CONV23_T_PADDING 0
#define CONV23_B_PADDING 0
#define CONV23_STRIDE 1
#define CONV23_OUT_CH 128
#define CONV23_OUT_DIM 10
 /*
 * Layer 24
 * Conv dw/ s2
 * Padding: 0
 * INPUT: 10 x 10 x 128
 * KERNEL: 3 x 3 x 128
 * OUTPUT: 5 x 5 x 128
 */
#define CONV24_IM_DIM 10
#define CONV24_IM_CH 128
#define CONV24_KER_DIM 3
#define CONV24_L_PADDING 0
#define CONV24_R_PADDING 0
#define CONV24_T_PADDING 0
#define CONV24_B_PADDING 0
#define CONV24_STRIDE 2
#define CONV24_OUT_CH 128
#define CONV24_OUT_DIM 5
 /*
 * Layer 25
 * Conv / s1
 * Padding: 0
 * INPUT: 5 x 5 x 128
 * KERNEL: 1 x 1 x 128 x 256
 * OUTPUT: 5 x 5 x 256
 */
#define CONV25_IM_DIM 5
#define CONV25_IM_CH 128
#define CONV25_KER_DIM 1
#define CONV25_L_PADDING 0
#define CONV25_R_PADDING 0
#define CONV25_T_PADDING 0
#define CONV25_B_PADDING 0
#define CONV25_STRIDE 1
#define CONV25_OUT_CH 256
#define CONV25_OUT_DIM 5
 /*
 * Layer 26
 * Conv dw/ s1
 * Padding: 1
 * INPUT: 5 x 5 x 256
 * KERNEL: 3 x 3 x 256
 * OUTPUT: 5 x 5 x 256
 */
#define CONV26_IM_DIM 5
#define CONV26_IM_CH 256
#define CONV26_KER_DIM 3
#define CONV26_L_PADDING 1
#define CONV26_R_PADDING 1
#define CONV26_T_PADDING 1
#define CONV26_B_PADDING 1
#define CONV26_STRIDE 1
#define CONV26_OUT_CH 256
#define CONV26_OUT_DIM 5
 /*
 * Layer 27
 * Conv / s1
 * Padding: 0
 * INPUT: 5 x 5 x 256
 * KERNEL: 1 x 1 x 256 x 256
 * OUTPUT: 5 x 5 x 256
 */
#define CONV27_IM_DIM 5
#define CONV27_IM_CH 256
#define CONV27_KER_DIM 1
#define CONV27_L_PADDING 0
#define CONV27_R_PADDING 0
#define CONV27_T_PADDING 0
#define CONV27_B_PADDING 0
#define CONV27_STRIDE 1
#define CONV27_OUT_CH 256
#define CONV27_OUT_DIM 5
 /*
 * Layer 28
 * Avg Pool /s1
 * Padding: 
 * INPUT: 5 x 5 x 256
 * KERNEL: Pool 7 x 7
 * OUTPUT: 1 x 1 x 256
 */
#define POOL1_IM_DIM 5
#define POOL1_IM_CH 256
#define POOL1_KER_DIM 7
#define POOL1_L_PADDING 0
#define POOL1_R_PADDING 0
#define POOL1_T_PADDING 0
#define POOL1_B_PADDING 0
#define POOL1_STRIDE 1
#define POOL1_OUT_CH 256
#define POOL1_OUT_DIM 1
 /*
 * Layer 29
 * FC /s1
 * Padding: 
 * INPUT: 1 x 1 x 256
 * KERNEL: 256 x 1000
 * OUTPUT: 1 x 1 x 1000
 */
#define IP1_DIM 1*1*256
#define IP1_IM_DIM 1
#define IP1_IM_CH 256
#define IP1_OUT 1000

#endif /*__MOBILE_NET_PARAMS__*/
