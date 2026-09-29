# JobScope Frontend

Vue 3 + TypeScript + Vite 产品化工作台。

```powershell
npm install
npm run dev
```

开发服务器运行在 `http://127.0.0.1:5173`，并将 `/health` 与 `/v1` 代理到
`http://127.0.0.1:8110` 的 JobScope FastAPI 服务。

## 已接入工作台

- 运行总览：读取健康状态和后端能力契约。
- 文档处理：多格式临时解析与切块预览。
- 检索问答：Hybrid RRF检索和带引用回答。
- Job Agent：Planner、注册工具调用和执行轨迹。
- 来源审核：保守的招聘来源可信度评估。
- 质量评测：只读展示黄金集与不可覆盖的实验报告。

依赖未启用时页面会明确显示不可用状态，不会伪造成功结果。
