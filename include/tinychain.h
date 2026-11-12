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
    TC_DTYPE_F32=0,
    TC_DTYPE_I8
}tc_dtype_t;

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

    union {
        const void* raw;
        const float* flt32;
        const int8_t* i8;
    }tensor;

    float scale;
    int32_t zero_point;
    tc_dtype_t dtype

    size_t node_count;
    uint16_t _last_argmax_idx;
    uint16_t _eval_cycles;
    bool _has_argmax;
}tc_handle_t;

#define TC_NODE_IGNORE {.op=(uint8_t)TC_OP_DISABLED,.threshold_min=0,.threshold_max=0,.callback=NULL,.user_ctx=NULL,.required_streak=0,.latch_once=false}

#define TC_NODE_BINARY(thresh, streak, latch, fn, ctx) \
    { .action = (fn), .user_ctx = (ctx), .threshold_min = (thresh), .threshold_max = 0.0f, \
      .op = (uint8_t)TC_OP_GREATER, .required_streak = (streak), ._current_streak = 0U, \
      .latch_once = (latch), ._has_fired = false }

#define TC_NODE_DROP(thresh, streak, latch, fn, ctx) \
    { .action = (fn), .user_ctx = (ctx), .threshold_min = (thresh), .threshold_max = 0.0f, \
      .op = (uint8_t)TC_OP_LESS, .required_streak = (streak), ._current_streak = 0U, \
      .latch_once = (latch), ._has_fired = false }

#define TC_NODE_ARGMAX(min_conf, streak, latch, fn, ctx) \
    { .action = (fn), .user_ctx = (ctx), .threshold_min = (min_conf), .threshold_max = 0.0f, \
      .op = (uint8_t)TC_OP_ARGMAX, .required_streak = (streak), ._current_streak = 0U, \
      .latch_once = (latch), ._has_fired = false }

#define TC_NODE_RANGE(min_v, max_v, streak, fn, ctx) \
    { .action = (fn), .user_ctx = (ctx), .threshold_min = (min_v), .threshold_max = (max_v), \
      .op = (uint8_t)TC_OP_RANGE, .required_streak = (streak), ._current_streak = 0U, \
      .latch_once = false, ._current_streak = 0U, ._has_fired = false }


#define TC_INIT_TFLM_F32(handle_ptr, tflm_tensor_ptr, nodes_arr, count) \
    tc_init((handle_ptr), (const void *)((tflm_tensor_ptr)->data.f), \
            TC_DTYPE_F32, 1.0f, 0, (nodes_arr), (count))

#define TC_INIT_TFLM_I8(handle_ptr, tflm_tensor_ptr, nodes_arr, count) \
    tc_init((handle_ptr), (const void *)((tflm_tensor_ptr)->data.int8), \
            TC_DTYPE_I8, (tflm_tensor_ptr)->params.scale, \
            (tflm_tensor_ptr)->params.zero_point, (nodes_arr), (count))

tc_status_t tc_init(tc_handle_t *h,
                    const void *tensor_ptr,
                    tc_dtype_t dtype,
                    float scale,
                    int32_t zero_point,
                    tc_node_t *nodes_array,
                    uint16_t num_nodes);

tc_status_t tc_run(tc_handle_t *h);
tc_status_t tc_reset(tc_handle_t *h);

#ifndef __cplusplus
}
#endif

#endif