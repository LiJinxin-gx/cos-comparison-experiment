# Behavior Composition Memory Agent + Universal Executor

## 1. Behavior as Data (formalized in tests)

Von Neumann architecture: behavior definitions (target/args/defaults/enabled/stats)
stored in DB, instruction cycle fetch -> decode -> reflect-execute -> write-back.

- Dynamic reconfiguration (zero code): configure(enabled/target/defaults)
- External behaviors loaded via interface.api (CallDict + importlib reflection)
- Delegation slots: loader_func / DatabaseToolWrap(check_same_thread=False) /
  ExecutorDriver worker / ControlFlowDriver+Sequence / Monitor
- **19 offline tests passed**

## 2. Universal Executor (atomic instruction triples)

```
instructions table: (pc, func, args, kwargs)
run_program: fetch(DB) -> $N result reference decode -> reflect-execute -> write-back register
Composite: func='program:<subprogram>' recursive (register copy isolation)
Plugin: any module.func reflection + error isolation
```

## 3. Prompt-Driven (external config + prompt changes behavior)

- prompt_plan: prompt keywords -> stage templates (stage_behaviors/stage_keywords
  stored in external JSON config) -> behavior sequence into DB
- Same fixed code executes ordinary tasks and designated tasks (only prompt differs)

## 4. Demonstrated Results

- Behavior Agent framework: 11->19 test evolution
- Web surfing task: instruction-based + reflection + feedback + hierarchical memory
  multiple versions (see web_knowledge.md)
- Dynamic execution modification: change DB config (top/swap implementation/disable)
  takes effect immediately

## Conclusion

Fixed code + DB instructions + delegation injection = three elements of a
universal executor; logic generation/modification requires no code changes.
