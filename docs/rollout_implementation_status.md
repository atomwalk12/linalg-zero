# Rollout Function Implementation Status

## ✅ Completed Implementation

### Core Rollout Function
- **File**: `linalg_zero/grpo/openpipe_art/main.py`
- **Function**: `rollout(model: art.Model, scenario: LinearAlgebraScenario) -> Any`
- **Status**: ✅ **WORKING** - Basic integration complete

### Key Features Implemented
1. **Environment Creation**: Uses `create_linalg_environment(run_config)`
2. **Agent Creation**: Uses `create_linalg_agent()` factory function
3. **Episode Execution**: Calls `agent.solve()` with proper parameters
4. **Error Handling**: Graceful fallback with empty trajectories
5. **Configuration Management**: Global config passing (temporary solution)
6. **Trajectory Conversion**: Mock implementation with proper structure

### Integration Points Working
- ✅ **LinAlgEnvironment**: Fully functional with tool execution
- ✅ **LinAlgAgent**: Complete agent framework with placeholder model client
- ✅ **Tool System**: All 6 mathematical tools working
- ✅ **Reward System**: Dual evaluation (format + accuracy) functional
- ✅ **Configuration**: RunConfig and LinearAlgebraTrainingConfig integration

## 🚧 Outstanding Work (TODO Items)

### 1. **CRITICAL: art.Model Integration**
**Location**: `linalg_zero/grpo/openpipe_art/linalg_agent.py:_call_model()`
**Priority**: **HIGH** - Required for actual training

```python
# Current status: NotImplementedError
def _call_model(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
    # TODO: Implement art.Model integration
    raise NotImplementedError("Model integration not yet implemented")
```

**Required Implementation**:
1. Accept `art.Model` instance in LinAlgAgent constructor
2. Convert conversation messages to art.Model format
3. Call `model.generate()` or equivalent method
4. Parse response for tool calls vs text responses
5. Return structured response: `{"message": {...}, "cost": float}`

### 2. **art.Trajectory Conversion**
**Location**: `linalg_zero/grpo/openpipe_art/main.py:_convert_solve_result_to_trajectory()`
**Priority**: **HIGH** - Required for GRPO training

```python
# Current status: MockTrajectory placeholder
def _convert_solve_result_to_trajectory(solve_result, scenario) -> Any:
    # TODO: Implement proper art.Trajectory conversion
    return MockTrajectory(trajectory_data)
```

**Required Implementation**:
1. Understand `art.Trajectory` format requirements
2. Convert `SolveResult.messages` to trajectory format
3. Structure rewards and metadata properly
4. Ensure compatibility with GRPO training loop

### 3. **Configuration Refactoring**
**Location**: `linalg_zero/grpo/openpipe_art/main.py` (global variable)
**Priority**: **MEDIUM** - Improve architecture

```python
# Current status: Global variable workaround
_global_run_config: RunConfig | None = None
```

**Better Solution**:
1. Modify art framework to pass config to rollout
2. Use dependency injection pattern
3. Create environment factory with config binding

### 4. **Dataset Integration**
**Location**: Task 3.x in implementation plan
**Priority**: **MEDIUM** - Currently using sample tasks

**Required Implementation**:
1. Load tasks from `atomwalk12/linalgzero` dataset
2. Integrate with preprocessing pipeline
3. Support different difficulty levels and problem types

## 🧪 Testing Status

### ✅ Integration Test Results
```bash
$ uv run python test_rollout_integration.py
✅ Rollout completed successfully!
   Trajectory type: MockTrajectory
   Trajectory data keys: ['messages', 'reward', 'info', 'scenario_step', 'total_cost']
   Reward: 1.0
```

### Test Coverage
- ✅ **Rollout Function**: Basic execution path working
- ✅ **Environment Integration**: LinAlgEnvironment creation and usage
- ✅ **Agent Integration**: LinAlgAgent creation and solve method
- ✅ **Error Handling**: Graceful failure with empty trajectories
- ❌ **Model Integration**: Placeholder only (uses mock responses)
- ❌ **Trajectory Format**: Mock format only

## 🎯 Next Development Steps

### Immediate (Week 1)
1. **Implement art.Model Integration**
   - Modify LinAlgAgent constructor to accept art.Model
   - Implement `_call_model()` method with proper art.Model calls
   - Test with actual model inference

2. **Research art.Trajectory Format**
   - Examine existing art.Trajectory usage in codebase
   - Understand required fields and structure
   - Implement proper conversion from SolveResult

### Short Term (Week 2)
3. **End-to-End Training Test**
   - Run actual GRPO training with implemented rollout
   - Verify trajectory collection and model updates
   - Debug any integration issues

4. **Dataset Integration**
   - Implement task loading from HuggingFace datasets
   - Add task filtering and selection logic
   - Test with real linear algebra problems

### Medium Term (Week 3-4)
5. **Production Readiness**
   - Refactor configuration management
   - Add comprehensive error handling
   - Implement performance optimizations
   - Add monitoring and logging

## 🔗 Key Integration Points

### Current Architecture Flow
```
main() → rollout() → LinAlgAgent.solve() → LinAlgEnvironment.step() → LinAlgTool.invoke()
   ↓         ↓              ↓                      ↓                        ↓
RunConfig → art.Model → [PLACEHOLDER] → Mathematical Tools → lib.py functions
```

### Missing Links
1. **art.Model → LinAlgAgent**: Need `_call_model()` implementation
2. **SolveResult → art.Trajectory**: Need proper conversion
3. **Global Config**: Need better config passing mechanism

## 📊 Implementation Progress

| Component | Status | Functionality | Integration |
|-----------|--------|---------------|-------------|
| Rollout Function | ✅ Complete | Episode execution | ✅ Working |
| Environment | ✅ Complete | Tool execution, rewards | ✅ Working |
| Agent Framework | ✅ Complete | Conversation management | 🚧 Placeholder model |
| Tool System | ✅ Complete | Mathematical operations | ✅ Working |
| Reward System | ✅ Complete | Format + accuracy eval | ✅ Working |
| Model Integration | ❌ Missing | art.Model calls | ❌ NotImplemented |
| Trajectory Format | 🚧 Mock | GRPO compatibility | ❌ Placeholder |
| Dataset Loading | ❌ Missing | Real problem tasks | ❌ Sample tasks only |

**Overall Status**: 🟡 **Functional with Placeholders** - Ready for model integration
