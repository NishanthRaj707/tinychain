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
    tc_op_t op;
    float threshold_min;
    float threshold max;
    tc_node_cb callback;
    void* user_ctx;
    uint16_t required_streak;
    bool latch_once;

    uint16_t _current_streak;
    bool _has_fired;
}tc_node_t;

typedef struct{
    tc_node_t *nodes;
    size_t node_count;
    uint16_t _last_argmax_idx;
    uint16_t _eval_cycles;
}tc_handle_t;









#ifndef __cplusplus
}
#endif

#endif