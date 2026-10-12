# 当前检索链路本地性能检查（2026-10-12）

当前 PostgreSQL Corpus 含 267 个 Chunk。在一台配备 RTX 3060 Laptop GPU（6 GB 显存）的开发机上，使用已预热的 FastAPI `TestClient` 对同一条中文查询执行 60 次请求、并发 4、服务容量 4：60/60 成功，墙钟 5.417 秒，约 11.075 req/s，P50 356 ms、P95 410 ms、P99 419 ms。完整命令：

```powershell
.venv/Scripts/python.exe -m scripts.retrieval.benchmark_hybrid_api_concurrency --requests 60 --concurrency 4 --service-capacity 4 --warmup 5
```

这是进程内、单查询、小语料的诊断结果，不经过真实网络，也不包含 Agent 规划、回答模型或前端；不能当作线上 QPS/P95、SLA 或多实例容量写入简历。真实 PostgreSQL 可用时，同日运行 `.venv/Scripts/python.exe -m pytest tests -q` 得到 433 通过、5 跳过。检索质量和配对置信区间另见 [BEIR SciFact 外部实验](beir-scifact.md)，其英文科学文献领域同样不能替代招聘问答验收。
