#include "tinychain.h"

static inline float tc_read_val(tc_handle_t *h,uint16_t idx)
{
    switch((tc_dtype_t)h->dtype)
    {
        case TC_DTYPE_F32:
            return h->tensor.f32[idx];
        case TC_DTYPE_I8:
            return ((float)h->tensor.i8[idx] - (float)h->zero_point)*h->scale;
        default:
            return 0.0f;
    }
}

static inline bool tc_check_op(tc_node_t* n,uint16_t node_idx,float val,uint16_t max_idx)
{
    switch((tc_op_t))n->op)
    {if ((num_nodes == 0U) || ((dtype != TC_DTYPE_F32) && (scale <= 0.0f))) {
        return TC_ERR_INVALID_CFG;
    }
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
    }
}

static inline tc_status_t tc_reset(tc_handle_t *h)
{
    if(h == NULL || h->nodes == NULL)
    {
        return TC_ERR_NULL_PTR;
    }
    
    for (uint16_t i = 0U; i < h->num_nodes; i++) {
        h->nodes[i]._current_streak = 0U;
        h->nodes[i]._has_fired      = false;
    }

    h->last_argmax_idx = TC_INDEX_NONE;
    return TC_OK;
}

tc_status_t tc_init(tc_handle_t* h,const void *tensor_ptr,tc_dtype_t dtype,float scale,int32_t zero_point,tc_node_t *nodes_array,uint16_t num_nodes)
{
    if(h == NULL || tensor_ptr == NULL || nodes_array == NULL)
    {
        return TC_ERR_NULL_PTR;
    }

    if(dtype == NULL || scale == NULL || zero_point == NULL)
    {
        return TC_ERR_INVALID_CFG;
    }

    if ((num_nodes == 0U) || ((dtype != TC_DTYPE_F32) && (scale <= 0.0f))) {
        return TC_ERR_INVALID_CFG;
    }

    if(dtype >= TC_DTYPE_I8)
    {
        return TC_ERR_INVALID_CFG;
    }

    h->nodes           = nodes_array;
    h->tensor.raw      = tensor_ptr;
    h->dtype           = (uint8_t)dtype;
    h->scale           = scale;
    h->zero_point      = zero_point;
    h->num_nodes       = num_nodes;
    h->eval_cycles     = 0U;
    h->last_argmax_idx = TC_INDEX_NONE;
    h->_has_argmax     = false;

    for(uint16_t i = 0;i<num_nodes;i++)
    {
        num_nodes[i]._current_streak = 0;
        num_nodes[i]._has_fired = 0;

        if(num_nodes[i].op != (uint8_t)TC_OP_DISABLED)
        {
            if(num_nodes[i].required_streak == 0U)
            {
                num_nodes[i].required_streak = 1;
            }
            if(num_nodes[i].op == (uint8_t)TC_OP_RANGE && (num_nodes[i].threshold_min > num_nodes[i].threshold_max))
            {
                return TC_ERR_INVALID_CFG;
            }
            if(num_nodes[i].op == (uint8_t)TC_OP_ARGMAX)
            {
                num_nodes[i]._has_argmax = true;
            }
        }

    }

    return TC_OK;
}

static inline int tc_update_state(tc_node_t* n,uint16_t node_idx,float val)
{
    if(n->_current_streak < n->required_streak)
    {
        n->_current_streak++;
    }

    if(n->_current_streak >= n->required_streak)
    {
        if(!n->latch_once || !n->_has_fired)
        {
            n->_has_fired = true;
            if(n->action != NULL)
            {
                n->action(node_idx,val,n->user_ctx);
                return 1;
            }
        }
    }
    return 0;
}

tc_status_t tc_run(tc_handle_t* h)
{
    if(h == NULL || h->tensor_ptr.raw == NULL || h->nodes == NULL)
    {
        return TC_ERR_NULL_PTR;
    }
    
    int actions_fired = 0;
    uint16_t max_idx = 0;

    if(h->_has_argmax){
        float max_val=tc_read_val(h,0U);
        for(uint16_t i=1;i<h->num_nodes;i++)
        {
            float v = tc_read_val(h,i);
            if(v > max_val)
            {
                max_val = v;
                max_idx = i;
            }
        }
    }

    for(uint16_t i=0;i<h->num_nodes;i++)
    {
        tc_node_t* n = &(h->num_nodes[i]);

        if(n->op == (uint8_t)TC_OP_DISABLED)
        {
            continue;
        }

        float val = tc_read_val(n,i);
        if(tc_check_op(n,i,v,max_idx))
        {
            if(n->action == NULL)
            {
                return TC_ERR_NULL_PTR;
            }
            actions_fired += n->action(i,val,n->user_ctx);

        }
        else{
            n->_current_streak = 0U;
            n->_has_fired=false;
        }
    }
    h->eval_cycles++;
    return actions_fired;
}

