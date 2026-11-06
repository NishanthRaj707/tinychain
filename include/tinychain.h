#ifndef TINY_CHAIN
#define TINY_CHAIN

#include<stdint.h>
#include<stddef.h>
#include<stdbool.h>

#ifndef cplusplus
extern "C" {
#endif

typedef enum{
    TC_OK=0,
    TC_ERR_NULL_PTR=-1,
    TC_ERR_INVALID_CFG=-2
}tc_status_t;

typedef enum{
    TC_OP_DISABLED,
    TC_OP_ARGMAX,
    TC_OP_GREATER,
    TC_OP_LESSER,
    TC_OP_RANGE
}tc_op_t;

typedef void (*tc_node_cb)(uint16_t node_id,float val,void* user_data);

typedef struct{
    
}







#ifndef __cplusplus
}
#endif

#endif