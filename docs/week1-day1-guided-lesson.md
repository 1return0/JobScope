# 第1周第1天带学课：JobScope为什么先做骨架

> 建议用时：2小时  
> 今日验收：能运行服务和测试，并用自己的话解释接口、配置与能力边界。

## 第一段：先理解业务

JobScope要解决的不是“生成一份求职建议”，而是“从多公司、多岗位、多版本招聘材料中找到正确证据”。

例子：用户问“哪些当前校招Agent岗位明确要求MCP？”系统不能只凭模型记忆回答。它需要：

1. 限定校招和当前版本；
2. 检索岗位原文；
3. 区分必须项与加分项；
4. 返回公司、岗位、原文位置和抓取时间；
5. 没有足够证据时拒答。

因此JobScope首先是证据系统，其次才是问答系统。

## 第二段：理解配置

阅读`app/config.py`。

```python
@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "JobScope"
    version: str = "0.1.0"
    environment: str = "development"
    host: str = "127.0.0.1"
    port: int = 8110
```

- `@dataclass`自动生成初始化等样板代码；
- `frozen=True`防止程序运行中随意修改配置；
- `slots=True`防止动态增加拼错的字段；
- 类型标注让编辑器、测试和读代码的人知道预期类型。

```python
environment=os.getenv("JOBSCOPE_ENV", "development")
```

程序先读取环境变量，没有时才使用默认值。这样同一份代码可以运行在开发、测试和部署环境。

## 第三段：理解FastAPI接口

阅读`app/main.py`。

```python
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
```

Pydantic模型定义接口契约：字段名称、类型和输出结构。

```python
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
```

请求执行过程：

```text
GET /health
  → FastAPI路由
  → health函数
  → HealthResponse校验
  → JSON序列化
  → HTTP响应
```

`/health`返回`UP`只说明进程能响应，不能证明数据库、向量库、Embedding或检索已经可用。后续会增加`/readiness`检查依赖。

`/v1/meta`将能力分为：

- `implemented_capabilities`：代码和测试已经存在；
- `planned_capabilities`：已进入计划但尚未实现。

这同时训练简历诚实性：计划指标不能写成已取得结果。

## 第四段：理解测试

阅读`tests/system/test_system_api.py`。

```python
self.client = TestClient(app)
response = self.client.get("/health")
```

`TestClient`在进程内模拟HTTP请求，不需要先启动8110端口。

```python
self.assertEqual(200, response.status_code)
self.assertEqual({...}, response.json())
```

第一条检查请求是否成功，第二条检查接口契约是否被破坏。

另一个测试确认`hybrid-retrieval`仍处于计划列表。如果现在错误地把它写进已实现列表，测试会失败。

## 第五段：运行

```powershell
cd JobScope
py -3.12 -m unittest discover -s tests -v
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 8110
```

服务启动后依次打开：

- <http://127.0.0.1:8110/health>
- <http://127.0.0.1:8110/v1/meta>
- <http://127.0.0.1:8110/docs>

观察完成后回到PowerShell按`Ctrl+C`停止。

## 第六段：检查题

请用自己的话回答，不要求背术语。

1. 为什么JobScope首先是证据系统，而不是聊天系统？
2. `response_model`对接口有什么作用？
3. `/health`为什么不能证明混合检索已经可用？
4. 为什么配置从环境变量读取，而不是全部写死在`main.py`？
5. 为什么现在必须把`hybrid-retrieval`放在计划能力中？

## 第七段：第一次代码练习

在`ServiceMetaResponse`中增加：

```python
read_only: bool
```

在`service_meta()`返回值中增加：

```python
read_only=True
```

再在测试中添加：

```python
self.assertTrue(payload["read_only"])
```

运行测试。这个练习用于理解“接口契约变化必须同步实现与测试”。如果测试失败，不要先问AI修复，先阅读错误信息并记录你的判断。

## 第1天通过条件

- [ ] 2个基础测试通过；
- [ ] 浏览器能打开三个接口页面；
- [ ] 完成5道检查题；
- [ ] 完成`read_only`字段练习；
- [ ] 修改后测试仍然通过；
- [ ] 能解释当前项目距离最终RAG系统还缺哪些模块。

