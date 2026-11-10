# TinyChain

TinyChain is a zero-allocation, ANSI C99 dispatch library that routes 1D edge machine learning inference output tensors directly to hardware actuation callbacks using static rule tables, streak-based hysteresis, and latching.

## Overview & Problem Statement

### Embedded ML Actuation Challenges
Edge machine learning inference engines—such as TensorFlow Lite for Microcontrollers—produce raw numerical arrays (1D tensors) representing class probabilities, anomaly scores, or continuous regression values. Connecting these tensor outputs to physical hardware actuators (relays, GPIO pins, PWM timers, motor drivers) introduces recurring engineering issues inside the microcontroller super-loop or RTOS task:

1. **Signal Chatter (Threshold Oscillation):** Output values near decision thresholds fluctuate rapidly across consecutive inference cycles due to sensor noise and marginal predictions. Directly mapping these values to actuators causes rapid, uncontrolled toggling (contact chatter, relay wear, and acoustic noise).
2. **Boilerplate State Management:** Suppressing chatter requires debounce counters, timers, and hysteresis logic. When written ad-hoc with `if-else` blocks and global variables, this logic creates tightly coupled, non-reusable state tracking code across the application.
3. **Repeated Callback Execution:** Unfiltered super-loops repeatedly invoke callbacks on every frame while a condition remains active. Embedded applications typically require single-shot execution for event-driven triggers (such as alarms or gesture detections) while maintaining continuous stream execution for proportional control (such as motor velocity or PWM duty cycles).
4. **Competitive Class Selection Overhead:** Multiclass classification models require identifying the tensor element with the maximum probability (`argmax`) and verifying that it exceeds a minimum confidence floor before triggering an action.

### The TinyChain Solution
TinyChain eliminates ad-hoc control logic by introducing a static, 1:1 index-mapped routing table (`tc_node_t` array). Each entry `nodes[i]` corresponds directly to output element `tensor[i]`. In a single sequential evaluation pass (`tc_evaluate_f32`), TinyChain evaluates mathematical conditions, increments or resets consecutive frame streak counters (debouncing), enforces single-shot or continuous latching, and invokes bound C function pointers (`tc_action_fn_t`) with a user-defined hardware context pointer (`void *user_ctx`).

### Target Environment & Constraints
* **Language Standard:** Strict ANSI C99.
* **Memory Model:** Zero dynamic memory allocation (`malloc` and `free` are not used). All rule tables and execution handles are statically allocated by the caller at compile or initialization time.
* **Target Platforms:** Bare-metal microcontrollers and Real-Time Operating Systems (RTOS), including Zephyr RTOS, FreeRTOS, ESP-IDF, and STM32Cube.

### Supported Model Types (v1.0)
* **Linear Regression:** Bounded range checks (`TC_OP_RANGE` / `TC_NODE_REGRESSION`) where `threshold_min <= val <= threshold_max`. Configured for continuous execution (`latch_once = false`) to update proportional actuators (e.g., PWM duty cycles, DAC outputs).
* **Binary Classification & Anomaly Detection:** Upper and lower threshold evaluations (`TC_OP_GREATER`, `TC_OP_LESS` / `TC_NODE_BINARY`) to detect target states or anomalous sensor behavior.
* **Multiclass Classification (MCC):** Competitive global maximum checks (`TC_OP_ARGMAX` / `TC_NODE_ARGMAX`) where node `i` must hold the highest numerical value across the entire tensor and satisfy `val >= threshold_min`.

## Execution Flow

The following diagram illustrates the signal path from model output tensor to hardware callback dispatch:

```
+-----------------------------------------------------------------------+
|                    Inference Engine Output Tensor                     |
|              const float tensor[N] = { y_0, y_1, ..., y_N-1 }         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|              TinyChain Evaluation Pass: tc_evaluate_f32()             |
|                                                                       |
|   For each index i in [0, N-1] with node = &handle->nodes[i]:         |
|                                                                       |
|   +---------------------------------------------------------------+   |
|   | 1. Condition Evaluation                                       |   |
|   |    Evaluate tensor[i] against node->op:                       |   |
|   |    - TC_OP_DISABLED : Skip node                               |   |
|   |    - TC_OP_GREATER  : tensor[i] > node->threshold_min         |   |
|   |    - TC_OP_LESS     : tensor[i] < node->threshold_min         |   |
|   |    - TC_OP_RANGE    : min <= tensor[i] <= max                 |   |
|   |    - TC_OP_ARGMAX   : i == argmax(tensor) && val >= min       |   |
|   +---------------------------------------------------------------+   |
|                                   |                                   |
|                  +----------------+----------------+                  |
|                  |                                 |                  |
|             Condition Met                  Condition Not Met          |
|                  |                                 |                  |
|                  v                                 v                  |
|   +------------------------------+  +-----------------------------+   |
|   | node->current_streak++       |  | node->current_streak = 0    |   |
|   +------------------------------+  | node->latched = false       |   |
|                  |                  +-----------------------------+   |
|                  v                                                    |
|   +---------------------------------------------------------------+   |
|   | 2. Streak Hysteresis Filter                                   |   |
|   |    Is node->current_streak >= node->required_streak ?         |   |
|   +---------------------------------------------------------------+   |
|                  |                                 |                  |
|                 Yes                                No                 |
|                  |                                 |                  |
|                  v                                 v                  |
|   +-------------------------------+         (Suppress Dispatch)       |
|   | 3. Latch State Check          |                                   |
|   |    Is (!node->latched ||      |                                   |
|   |        !node->latch_once) ?   |                                   |
|   +-------------------------------+                                   |
|                  |                |                                   |
|                 Yes               No (Already Latched)                |
|                  |                |                                   |
|                  v                v                                   |
|   +--------------------+     (Suppress Dispatch)                      |
|   | Update Latch:      |                                              |
|   | If latch_once:     |                                              |
|   |   latched = true   |                                              |
|   +--------------------+                                              |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Hardware Action Callback Dispatch                  |
|               node->action(i, tensor[i], handle->user_ctx)            |
+-----------------------------------------------------------------------+
```

## Core Mechanics

### 1:1 Index Mapping
TinyChain operates directly on flat 1D output tensors produced by edge neural network runtimes. The routing table is declared as an array of `tc_node_t` structures whose length matches the tensor dimension ($N$). Index $i$ of the node array maps directly to index $i$ of the tensor:

```
node[0]   <---> tensor[0]
node[1]   <---> tensor[1]
...
node[N-1] <---> tensor[N-1]
```

During execution, `tc_evaluate_f32()` traverses the table in a single deterministic pass ($O(N)$ overall time complexity, $O(1)$ constant time per element). There are no dynamic lookups, string identifiers, binary search trees, or heap allocations. Unused or irrelevant tensor outputs are marked with `TC_OP_DISABLED` (`TC_NODE_IGNORE()`), allowing the evaluation loop to advance to the next index with minimal instruction overhead.

### Streak Hysteresis (`required_streak`)
Raw inference values frequently oscillate around decision boundaries across consecutive sampling intervals. Direct thresholding transforms this numeric oscillation into mechanical actuator chatter.

TinyChain resolves this using consecutive frame accumulation:
* Each node maintains an internal counter, `current_streak`.
* When the node condition evaluates to true during an inference pass, `current_streak` increments by 1.
* If `current_streak < required_streak`, callback dispatch is inhibited.
* If the condition evaluates to false at any time, `current_streak` immediately resets to 0.
* Callback execution occurs only after the condition holds true for `required_streak` consecutive inference passes.

This mechanism enforces a temporal debounce filter: an event must remain valid over a continuous time window of $(T_{\text{inference}} \times \text{required\_streak})$ before hardware actuators respond.

### State Latching (`latch_once`)
Embedded actuators require two distinct operational modes:
* **Single-Shot Latching (`latch_once = true`):** The action callback executes exactly once when `current_streak` reaches `required_streak`. The internal flag `latched` is set to `true`. While the input condition continues to evaluate to true in subsequent inference cycles, further callback dispatches are suppressed. The latch is released (`latched = false`, `current_streak = 0`) only when the input value breaks the condition. This mode is used for discrete state triggers, relay pulses, alarm notifications, and event-driven interrupts.
* **Continuous Execution (`latch_once = false`):** The action callback executes on every inference frame for which `current_streak >= required_streak`. If `required_streak = 1`, the callback fires continuously across all frames where the value remains within the bounded interval. This mode is used for proportional control loops, linear regression outputs, and continuous actuator modulation (such as updating a PWM timer compare register or DAC voltage).

### Competitive Selection (`TC_OP_ARGMAX`)
In multiclass classification models (e.g., softmax categorical distributions), class outputs are mutually competitive. An action must fire only if its corresponding class holds the highest confidence score across the entire tensor, rather than merely crossing an independent scalar threshold.

When evaluating nodes configured with `TC_OP_ARGMAX`:
1. TinyChain computes the global maximum value and its corresponding index across the tensor:
   $$\text{argmax}(T) = \arg\max_{j} T[j]$$
2. For any node $i$ configured with `TC_OP_ARGMAX`, the condition is met if and only if:
   $$i == \arg\max(T) \quad \text{and} \quad T[i] \ge \text{threshold\_min}$$
3. If the highest confidence score across all classes fails to meet `threshold_min`, the condition fails for all argmax nodes, suppressing execution during ambiguous or low-confidence states.
4. For all losing indices ($j \ne \arg\max(T)$), the condition evaluates to false, resetting `current_streak` to 0 and clearing any latches.

## Repository Structure

```
tinychain/
├── CMakeLists.txt
├── LICENSE
├── README.md
├── include/
│   └── tinychain.h
├── src/
│   └── tinychain.c
├── examples/
│   ├── multiclass_gesture.c
│   └── regression_actuator.c
└── tests/
    └── test_tinychain.c
```


