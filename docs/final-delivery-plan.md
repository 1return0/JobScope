# JobScope 结课完整项目交付计划

## 技术选择

结课前端采用 **Vue 3 + TypeScript + Vite**，后端继续使用现有 FastAPI 服务。

不把 Streamlit 或 Gradio 作为最终主前端。它们适合快速构建模型 Demo，但 JobScope 需要展示文档治理、人工审核、Agent Trace、评测实验和多页面业务状态，Vue 更适合形成企业式前后端分离项目。必要时可保留 Python Demo 作为辅助调试入口。

## 组装前必须完成的后端能力

1. 固定 Planner 黄金评测集、参数规范化和不可覆盖报告。
2. 补齐岗位名称、招聘类型等查询工具参数与数据库过滤。
3. 正式组装 Planner、工具执行、最终回答和应用生命周期。
4. 增加结构化岗位记录的审核、激活和查询 API。
5. 为文档入库、混合检索、Grounded Answer、OCR 和评测提供稳定 API。
6. 配置明确的 CORS 来源、错误契约、Trace ID 和安全的密钥加载方式。

## Vue 页面范围

### 1. 系统总览

- FastAPI、PostgreSQL、Embedding、OCR 与回答模型状态
- Current Corpus、Snapshot、Chunk 和当前岗位记录数量
- 最近任务、失败状态与稳定错误码

### 2. 文档与语料中心

- 上传 PDF、Word、Excel、PPT 和 HTML
- 展示解析器、OCR降级、Fragment、Chunk与Snapshot状态
- 查看Manifest、文件哈希、来源与版本关系

### 3. RAG检索与可信问答

- BM25、Dense、Hybrid检索模式与Top-K结果
- Grounded Answer、Claim与Citation对应展示
- Evidence原文、来源位置和拒答原因

### 4. Job Agent工作台

- 输入招聘业务问题
- 展示Planner decision、reason、tool call、规范化参数和tool result
- 展示最终回答、引用、拒答与稳定错误码
- 展示单次Trace、延迟、Token与模型身份

### 5. 结构化岗位审核台

- 左侧显示Evidence，右侧显示抽取字段和逐字段引用
- 展示机械Grounding与语义Judge结果
- 人工确认、拒绝或激活record
- 查看同一Snapshot下的历史record及current切换

### 6. 评测看板

- Retrieval：Recall、MRR、nDCG
- Answer：回答、拒答与Grounding指标
- OCR：Exact Match、CER、WER与关键字段准确率
- Planner：decision、tool、arguments与case accuracy
- 展示dataset identity、模型身份、逐Case失败和实验对比

## 最终演示主线

```text
多格式招聘文件上传
  → 解析/OCR
  → 切块与Snapshot入库
  → Hybrid检索
  → Grounded Answer
  → 结构化岗位抽取与双层校验
  → 人工激活
  → Job Agent规划并调用岗位工具
  → Trace与评测看板复核
```

## 完成标准

- 前端展示的核心数据必须来自真实FastAPI接口，不使用静态假数据冒充完整链路。
- Agent执行过程要区分Planner、Tool、Final Answer和失败状态。
- 所有引用可回到Evidence与DocumentSnapshot。
- 演示中明确标注已实现能力与尚未达到的生产级边界。
- README提供本地启动、数据库迁移、模型配置、前后端联调和演示步骤。
