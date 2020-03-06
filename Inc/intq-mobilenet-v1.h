/*
 * intq-mobile_net.h
 *
 *  Created on: Feb 7, 2019
 *      Author: Alessandro Capotondi
 */

#ifndef __INQ_MOBILE_NET_V1_MODELS_H__
#define __INQ_MOBILE_NET_V1_MODELS_H__

#if CONF==1
#include "intq-mobilenet-v1-models/224_1_0_parameters.h"
#include "intq-mobilenet-v1-models/224_1_0_weights_bias.h"
#endif

#if CONF==2
#include "intq-mobilenet-v1-models/224_0_75_parameters.h"
#include "intq-mobilenet-v1-models/224_0_75_weights_bias.h"
#endif

#if CONF==3
#include "intq-mobilenet-v1-models/224_0_5_parameters.h"
#include "intq-mobilenet-v1-models/224_0_5_weights_bias.h"
#endif

#if CONF==4
#include "intq-mobilenet-v1-models/224_0_25_parameters.h"
#include "intq-mobilenet-v1-models/224_0_25_weights_bias.h"
#endif

#if CONF==5
#include "intq-mobilenet-v1-models/192_0_5_parameters.h"
#include "intq-mobilenet-v1-models/192_0_5_weights_bias.h"
#endif

#if CONF==6
#include "intq-mobilenet-v1-models/192_0_25_parameters.h"
#include "intq-mobilenet-v1-models/192_0_25_weights_bias.h"
#endif

#if CONF==7
#include "intq-mobilenet-v1-models/160_0_5_parameters.h"
#include "intq-mobilenet-v1-models/160_0_5_weights_bias.h"
#endif

#if CONF==8
#include "intq-mobilenet-v1-models/160_0_25_parameters.h"
#include "intq-mobilenet-v1-models/160_0_25_weights_bias.h"
#endif

#if CONF==9
#include "intq-mobilenet-v1-models/128_0_5_parameters.h"
#include "intq-mobilenet-v1-models/128_0_5_weights_bias.h"
#endif

#if CONF==10
#include "intq-mobilenet-v1-models/128_0_25_parameters.h"
#include "intq-mobilenet-v1-models/128_0_25_weights_bias.h"
#endif

#endif /* MOBILE_NET_V1_MODELS_INTQ_MOBILE_NET_H_ */

