"""Toy ReAct agent loop — stdlib only.

Implements the five ingredients from docs/en.md:
  1. message buffer
  2. tool registry
  3. stop condition
  4. turn budget
  5. observation formatter

ToyLLM is a scripted policy so the loop runs offline and deterministic. Swap
ToyLLM for a real provider client and the control flow is identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCall:
    """表示一个具体的外部工具调用请求。

    name: 被调用的工具名称
    args: 传递给工具的参数字典
    """
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    """表示 Agent 交互环路中的单个轮次（步骤）。

    kind: 轮次类型，例如 "user" (用户输入), "thought" (模型思考), "action" (模型行动), "final" (最终回复)
    content: 对应轮次的文本内容（如思考内容或回复内容）
    tool_call: 若包含工具调用，则记录 ToolCall 实例
    observation: 工具调用后返回的观察结果（Observation）
    """
    kind: str
    content: str
    tool_call: ToolCall | None = None
    observation: str | None = None


class ToolRegistry:
    """工具注册表，用于管理可用的外部工具并统一分发执行。"""
    def __init__(self) -> None:
        # 内部存储：工具名称 -> 可调用函数（返回值为字符串形式的结果）
        self._tools: dict[str, Callable[..., str]] = {}

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """注册一个新工具"""
        self._tools[name] = fn

    def names(self) -> list[str]:
        """返回所有已注册的工具名称列表（按字母排序）"""
        return sorted(self._tools)

    def dispatch(self, call: ToolCall) -> str:
        """分发执行指定的 ToolCall 任务，并妥善处理参数或执行报错"""
        fn = self._tools.get(call.name)
        if fn is None:
            return f"error: unknown tool {call.name!r}"
        try:
            return fn(**call.args)
        except TypeError as e:
            # 参数不匹配报错
            return f"error: bad args for {call.name}: {e}"
        except Exception as e:
            # 其他执行时异常报错
            return f"error: {type(e).__name__}: {e}"


def calculator(expr: str) -> str:
    """简易计算器工具（仅允许安全数学字符，防止恶意代码注入）"""
    allowed = set("0123456789+-*/(). ")
    if not set(expr).issubset(allowed):
        return "error: illegal character in expr"
    try:
        # 在受限环境内执行数学算式 eval
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


class KVStore:
    """模拟外部数据库/键值存储工具"""
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str:
        """读取数据，若不存在则提示 missing"""
        return self._store.get(key, f"missing:{key}")

    def set(self, key: str, value: str) -> str:
        """写入数据"""
        self._store[key] = value
        return f"stored {key}"


class ToyLLM:
    """脚本化 ReAct 策略模型（用写好的脚本列表模拟大模型的输出）。

    每个脚本条目要么包含一个思考（thought）和行动（action），要么直接结束并输出最终答案（finish）。
    这使得我们可以离线且确定性地测试控制环路。
    """

    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = script
        self.cursor = 0  # 脚本指令执行游标

    def respond(self, history: list[Turn]) -> dict[str, Any]:
        """模拟大模型根据当前交互历史（history）作出的下一步响应"""
        if self.cursor >= len(self.script):
            return {"kind": "finish", "content": "no more actions"}
        entry = self.script[self.cursor]
        self.cursor += 1
        return entry


@dataclass
class AgentLoop:
    """ReAct 核心控制循环。

    管理：
      1. 消息历史缓冲区 (history)
      2. 工具注册表 (tools)
      3. 终止条件 / 回复判断 (finish)
      4. 运行步数限制 (max_turns)
    """
    llm: ToyLLM
    tools: ToolRegistry
    max_turns: int = 12
    history: list[Turn] = field(default_factory=list)

    def run(self, user_message: str) -> str:
        """执行 Agent 控制循环，直到给出最终回答或轮次超支。"""
        # 添加用户初始消息
        self.history.append(Turn(kind="user", content=user_message))
        for step in range(self.max_turns):
            # 1. 询问 LLM（获取下一个行动计划或结束指令）
            reply = self.llm.respond(self.history)
            
            # 终止条件：LLM 表示已搜集齐信息，给出最终回答
            if reply["kind"] == "finish":
                self.history.append(Turn(kind="final", content=reply["content"]))
                return reply["content"]
            
            # 2. 记录模型当前的想法（Thought）
            thought = reply.get("thought", "")
            self.history.append(Turn(kind="thought", content=thought))
            
            # 3. 解析并调用对应工具（Action），然后产生观察结果（Observation）
            call = ToolCall(name=reply["action"], args=reply.get("args", {}))
            observation = self.tools.dispatch(call)
            
            # 4. 将 Action 及其 Observation 作为新轮次写入历史，供模型下一步决策参考
            self.history.append(
                Turn(kind="action", content=call.name,
                     tool_call=call, observation=observation)
            )
            
        # 超出最大迭代次数（Turn Budget）限制，强行退出
        self.history.append(Turn(kind="final",
                                 content="budget exhausted"))
        return "budget exhausted"


def pretty_trace(history: list[Turn]) -> None:
    """美化打印整个交互执行轨迹，直观展示 ReAct 的 Thought -> Action -> Observation 链条"""
    for i, turn in enumerate(history):
        tag = f"[{i:02d} {turn.kind:>7}]"
        if turn.kind == "user":
            print(f"{tag} {turn.content}")
        elif turn.kind == "thought":
            print(f"{tag} {turn.content}")
        elif turn.kind == "action":
            call = turn.tool_call
            assert call is not None
            print(f"{tag} {call.name}({call.args}) -> {turn.observation}")
        elif turn.kind == "final":
            print(f"{tag} {turn.content}")


def build_demo_agent() -> AgentLoop:
    """构造并初始化演示 Agent（预注册 calculator 和 KVStore 工具，并写入固定脚本）"""
    # 注册工具
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    kv = KVStore()
    tools.register("kv_get", kv.get)
    tools.register("kv_set", kv.set)

    # 模拟大模型在逐步推理过程中的决策输出：
    # 存基础价格 -> 计算税费 -> 存税费 -> 计算总和 -> 确认获取 -> 最终回答
    script: list[dict[str, Any]] = [
        {"kind": "action", "thought": "store the base price",
         "action": "kv_set", "args": {"key": "base", "value": "120"}},
        {"kind": "action", "thought": "compute 15% tax",
         "action": "calculator", "args": {"expr": "120 * 0.15"}},
        {"kind": "action", "thought": "store the tax",
         "action": "kv_set", "args": {"key": "tax", "value": "18.0"}},
        {"kind": "action", "thought": "compute total",
         "action": "calculator", "args": {"expr": "120 + 18.0"}},
        {"kind": "action", "thought": "confirm stored values",
         "action": "kv_get", "args": {"key": "base"}},
        {"kind": "finish", "content": "the total including 15% tax is 138.0"},
    ]
    return AgentLoop(llm=ToyLLM(script), tools=tools, max_turns=10)


def main() -> None:
    print("=" * 70)
    print("TOY REACT LOOP — Phase 14, Lesson 01")
    print("=" * 70)
    agent = build_demo_agent()
    # 启动控制环路
    final = agent.run("What is 120 plus 15% tax, stored in kv?")
    print()
    # 打印运行轨迹
    pretty_trace(agent.history)
    print()
    print(f"final answer: {final}")
    print(f"turns used:   {len([t for t in agent.history if t.kind == 'action'])}")
    print(f"tools used:   {agent.tools.names()}")


if __name__ == "__main__":
    main()
