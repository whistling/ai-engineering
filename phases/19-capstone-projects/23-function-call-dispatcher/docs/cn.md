# Function Call Dispatcher

> 分发器是让 harness 为 schema 做出的每一个承诺买单的地方。超时、重试、去重、错误映射。所有这些都在同一缝隙上处理。

**Type:** 构建  
**Languages:** Python  
**Prerequisites:** Phase 13 第01-07课，Phase 14 第01课  
**Time:** ~90 分钟

## 学习目标
- 为每次调用在处理器外包一层超时，超时返回一个有类型的错误而不是让循环挂起。
- 应用带抖动的指数退避重试，并设置最大尝试次数。
- 基于幂等键对重试请求去重，以避免与仍在进行的原始请求并发执行同一个操作。
- 将处理器异常和传输故障映射到循环已经能理解的单一错误信封。
- 使用并发限制来约束并发调度，避免像 40 个工具调用的扇出耗尽事件循环。

## 分发器的位置

位于 harness 循环（lesson twenty）和工具注册表（lesson twenty-one）之间。传输（lesson twenty-two）为循环提供输入。循环将工具调用交给分发器。分发器查询注册表，运行处理器，并返回结果或符合 JSON-RPC 形状的错误信封。

```mermaid
flowchart TD
    loop[harness 循环]
    disp[分发器]
    reg[工具注册表]
    handler[处理器]
    loop --> disp
    disp -->|获取 名称|get name| reg
    disp -->|验证 参数|validate args| reg
    disp -->|使用 asyncio.wait_for 调用处理器 (args, timeout)| asyncio.wait_for handler args timeout | handler
    handler -->|成功| disp
    handler -->|TimeoutError -> 重试或失败| disp
    handler -->|Exception -> 映射到错误代码| disp
    disp -->|返回 Ok 结果 或 DispatchError| loop
```

分发器是唯一知道计时器、重试和幂等性的层。循环不知道。注册表不知道。处理器不知道。这种隔离就是设计要点。

## 超时

每个工具都有一个默认超时。注册表记录携带 `timeout_ms`。当 harness 传入 per-call 覆盖值时，分发器会覆盖它。我们使用 `asyncio.wait_for`。超时时，处理器任务会被取消，分发器返回 `DispatchError(kind="timeout")`。

默认情况下，超时并不是对非幂等工具可重试的错误。一次 `db.write` 在超时时可能已经提交也可能未提交。重试会导致写入重复。分发器会遵守注册表记录中的 `idempotent` 标志。幂等工具会重试，非幂等工具不会。

## 带指数退避的重试

重试策略最多三次尝试。退避是指数级的并带抖动。

```text
attempt 1  -> delay 0
attempt 2  -> delay 0.1s * (1 + random[0..0.5])
attempt 3  -> delay 0.4s * (1 + random[0..0.5])
```

只有 `timeout` 和 `transient` 错误会重试。`schema` 错误、`not_found` 或 `internal` 错误不会重试。Schema 错误是确定性的，重试不会改变结果且会浪费预算。

重试循环会遵守来自 harness 的预算。如果调用方的预算剩余工具调用次数为零，分发器会在第一次尝试时快速失败并返回 `kind="budget_exceeded"`。

## 幂等键去重

在原始调用仍在进行时触发的重试是生产环境中的一个真实 bug。第一次调用在 4.9 秒挂起（刚好低于超时），重试在 5 秒触发。现在两个请求同时竞争同一后端。如果工具是 `payments.charge`，你就可能被多扣一次款。

分发器接受一个可选的 `idempotency_key`。当一个具有相同键的调用到达且该键当前正在进行中时，分发器会等待正在进行的 future 并返回其结果。缓存会在完成后保留该键 60 秒，以吸收迟到的重试。

键由调用方负责。harness 从规划器派生它：`f"{step_id}:{tool_name}:{hash(args)}"`。分发器不主动生成键，因为仅从参数派生键会让两个语义不同的调用看起来相同。

## 错误信封

失败的 dispatch 返回单一形状。

```text
DispatchError
  kind        : "timeout" | "transient" | "schema" | "not_found" | "internal" | "budget_exceeded"
  message     : str
  attempts    : int
  jsonrpc_code: int   (one of -32601, -32602, -32603)
```

harness 循环将 `kind` 映射到下一个状态。`schema` 和 `not_found` 走 `on_error` 并触发重规划。`timeout` 和 `transient` 走 `on_error`，是否重规划取决于尝试次数。`budget_exceeded` 触发 `on_budget_exceeded`。

## 扇出时的并发限制

`gather(*calls)` 会同时运行所有协程。当有四十个工具调用时，就是四十个打开的套接字或四十个子进程管道。大多数后端不喜欢单个客户端发起 40 个并发连接。

分发器用一个信号量封装 `gather`。默认并发限制为 8。每个调用在分发之前获取信号量，并在完成时释放。调用方看到的是 `gather` 形状的输出，但实际调度是有界的。

## 单次调用的流程

```mermaid
flowchart TD
    start([调用者: dispatch 名称, args, opts])
    validate[registry.validate name, args]
    schema_err[DispatchError kind=schema]
    idem_check{幂等性 缓存？}
    in_flight[await 已存在的 future]
    cached[返回 缓存 结果]
    attempt[使用 asyncio.wait_for 调用处理器 (args, timeout)]
    success[缓存 + 返回 结果]
    timeout_branch{TimeoutError + 幂等？}
    retry[使用退避重试]
    fail[DispatchError]
    transient_branch{TransientError？}
    other[映射 Exception 到 kind，不重试]
    exhausted[DispatchError]

    start --> validate
    validate -->|出错| schema_err
    validate -->|OK| idem_check
    idem_check -->|命中：正在进行中| in_flight
    idem_check -->|命中：最近完成| cached
    idem_check -->|未命中| attempt
    attempt --> success
    attempt --> timeout_branch
    timeout_branch -->|是| retry
    timeout_branch -->|否| fail
    attempt --> transient_branch
    transient_branch -->|是，仍有尝试次数| retry
    transient_branch -->|耗尽| exhausted
    attempt --> other
    retry --> attempt
```

## 如何阅读代码

`code/main.py` 定义了 `Dispatcher`、`DispatchError` 和 `TransientError`。分发器在构造时接收一个注册表。异步方法 `dispatch(name, args, ...)` 是唯一的入口点。每次尝试的超时在 `_run_with_retries` 内联使用 `asyncio.wait_for` 应用。`gather_bounded(calls)` 使用并发限制来运行多个 dispatch。

`code/tests/test_dispatcher.py` 覆盖了超时触发、对 transient 错误重试、对 schema 错误不重试、幂等性去重（两个具有相同键的并发调用会合并为一次处理器调用）和并发限制（信号量的作用）。

测试使用 `asyncio.sleep(0)` 和基于确定性计数器的处理器，因此它们在毫秒级完成且不依赖于实际时钟时间。

## 进一步扩展

生产环境的分发器通常会加入两项扩展。首先，在每个转换点做结构化日志（尽管循环的事件流已经能提供很多，但分发器应当也发出 `dispatch.attempt` 和 `dispatch.retry` 事件）。其次，熔断器：在一个窗口期内失败超过 N 次后，某个工具进入冷却期，分发直接返回 `kind="circuit_open"`，而不是尝试调用处理器。这两项都可以在不改变契约的前提下叠加到当前分发器上。

Lesson twenty-four 将分发器与 plan-and-execute agent 连接起来，这样你就能看到所有四个部分一起运行。