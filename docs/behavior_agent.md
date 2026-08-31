# 行为组合记忆 Agent + 通用执行器

## 1. 行为即数据 (tests/behavior_agent.py, 已正式化)

冯·诺依曼结构: 行为定义 (target/args/defaults/enabled/统计) 存 DB,
指令周期 取指→译码→反射执行→写回.

- 动态重配 (零代码): configure(enabled/target/defaults)
- 外部行为经 interface.api (CallDict + importlib 反射) 加载
- 委托槽: loader_func / DatabaseToolWrap(check_same_thread=False) /
  ExecuterDriver worker / ControlFlowDriver+Sequence / Monitor
- **test_behavior_agent: 19 项离线测试通过**

## 2. 通用执行器 (原子指令三元组)

```
instructions 表: (pc, func, args, kwargs)
run_program: 取指(DB) -> $N 结果引用译码 -> 反射执行 -> 写回寄存器
复合封装: func='program:<子程序>' 递归 (寄存器拷贝隔离)
插件式: 任意 module.func 反射 + 错误隔离
```

## 3. 提示词驱动 (外部配置+提示词改变行为)

- prompt_plan: 提示词关键词 → 阶段模板 (stage_behaviors/stage_keywords
  存外部 JSON behavior_agent_config.json) → 行为序列入 DB
- 同一固定代码执行普通任务与指定任务 (仅提示词不同)

## 4. 演示成果

- 行为 Agent 框架: 11→19 项测试演进
- 冲浪任务: 指令化 + 反射 + 反馈 + 层次记忆多版本 (见 web_knowledge.md)
- 动态修改执行: 改 DB 配置 (top/换实现/禁用) 即时生效

## 结论

固定代码+DB 指令+委托注入 = 通用执行器的三要素; 逻辑生成/修改无需改代码。
