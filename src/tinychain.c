#include "tinychain.h"

static inline float tc_read_val(tc_handle_t *h,uint16_t idx)
{
    switch(h->dtype)
    {
        case TC_DTYPE_F32:
            return h->tensor.f32[idx];
        case TC_DTYPE_I8:
            return ((float)h->tensor.i8[idx] - (float)h->zero_point)*h->scale;
        default:
            return 0.0f;
    };
}

static inline bool tc_check_op(tc_node_t* n,float val,uint16_t max_idx)
{
    switch(n->op)
    {
        case TC_OP_GREATER:
            return (val>n->threshold_min);
        case TC_OP_LESSER:
            return (val<n->threshold_max);
        case TC_OP_RANGE:
            return ((val>n->threshold_min)&&(val<n->threshold_max));
        case TC_OP_ARGMAX:
            return (node_idx == max_idx) && (val >= node->threshold_min);

        default:
            return false;
    };
}

static inline tc_status_t tc_reset(tc_handle_t *h)
{
    if(h == NULL || h->nodes == NULL)
    {
        return TC_ERR_NULL_PTR;
    }
    
    for (uint16_t i = 0U; i < handle->num_nodes; i++) {
        handle->nodes[i]._current_streak = 0U;
        handle->nodes[i]._has_fired      = false;
    }

    handle->last_argmax_idx = TC_INDEX_NONE;
    return TC_OK;
}