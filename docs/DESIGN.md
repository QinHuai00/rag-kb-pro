# 设计说明（DESIGN）

这份文档只记录**取舍**，不重复 README 的功能清单。每条都写"为什么不选另一条路"，
以及"这条路的代价是什么"。

---

## 1. 为什么做混合检索（dense + BM25）

**问题**：上游只有一路向量检索，而且 `k` 被写死成 1（`similarity_threshold = 1`）。
在本项目的语料上，这有两个具体后果：

1. 中文短查询里，用户问「羊毛衫水温」时真正有区分度的词是「羊毛」「水温」，
   属于精确词面信号；稠密向量把整句压成一个点，容易把「洗涤养护」里另一段
   （比如「羽绒服水温」）排上来 —— 字面很像、答案错。
2. 单路检索没有"互相校验"：一路排错就错到底。

**取舍**：
- BM25 用**字符 bigram/trigram** 而不是 jieba 之类分词库。代价是精度略低、
  无法做词形归一；收益是零依赖、零词典、离线确定（并且和 HashingEmbedder 用同一口径，
  对比才公平）。
- 两路都取 10 条进候选池（`candidate_k=10`），再融合取 top-k。
  代价是多算一路的分数（实测 BM25 单次 <0.1 ms，可以忽略）。

**结果**（`eval/results.md`）：hybrid 相对 bm25 的主要增益在严格指标
「关键词命中@3」82.1% → 89.3%，Top-1 持平。也就是说**在这份语料上，混合检索的收益是"召回更稳"，
不是"排序更准"** —— 这个结论比"混合检索一定更好"更诚实，也更符合实测。

---

## 2. RRF 怎么算，为什么不用加权分数融合

给定两路结果列表 `L_dense`、`L_bm25`，对文档 `d`：

```
RRF(d) = Σ_i  w_i / (rrf_k + rank_i(d))        rank 从 1 开始；未出现则不计该项
```

本仓库取 `rrf_k = 60`、`w_i = 1`（两路等权）。举个可验证的例子
（`tests/test_retriever.py::test_rrf_fuse_sums_reciprocal_ranks`）：

```
L_dense = [a, b, c]   L_bm25 = [c, a]

RRF(a) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.032523
RRF(b) = 1/(60+2)                = 1/62      ≈ 0.016129
RRF(c) = 1/(60+3) + 1/(60+1) = 1/63 + 1/61 ≈ 0.032266
排序：a > c > b
```

**为什么不用 `α·cos + (1-α)·bm25_score` 直接加权？**

- 两路分数**量纲不同、分布不同**：余弦在 [-1,1]，BM25 是开区间上的无界分，
  且随语料规模漂移。要做加权就必须归一化（min-max / z-score），而归一化依赖
  "这一次查询的候选分布"，会让同一个片段在不同查询下分数不可比。
- 换向量库（Chroma 的 cosine 距离 → 内积 → L2）会直接改变分数尺度，
  加权融合的参数就得重调；**RRF 只用名次，换后端完全不用改**。

**代价**：RRF 丢掉了"这一条到底有多相关"的强度信息 —— 一条以 0.95 相似度排第一
和 0.31 排第一，在 RRF 里贡献相同。这在候选池小的时候影响不大，
但候选池里全是弱相关片段时，RRF 会把它们平等地推上来。
这正是下面 `dense_min_score` 存在的理由。

---

## 3. 为什么把"阈值"和"k"拆成两个参数

上游代码：

```python
# config_data.py
similarity_threshold = 1     # 注释：检索返回匹配的文档数量
# vector_stores.py
self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
```

名字叫「相似度阈值」，实际是「返回条数」。命名误导的后果不是审美问题：
它让人以为"低相关的结果会被过滤掉"，而实际行为是**无论多不相关都硬凑 k 条**。
向量检索必须能表达"没有相关内容"，否则跨领域提问也会拿到 k 条垃圾上下文，
再交给 LLM 生成就是幻觉的温床。

本仓库拆成：

| 参数 | 语义 | 默认 |
| --- | --- | --- |
| `top_k` | 最终返回几条 | 3 |
| `dense_min_score` | 余弦低于此值**不召回** | 0.0 |

`tests/test_retriever.py::test_dense_min_score_filters_unrelated_chunks` 锁死这条行为：
不相关查询返回空列表，而不是"最不相关的那几条"。

> 副作用提醒：`dense_min_score=0.0` 对**带符号哈希**的向量比较苛刻
> （负余弦意味着"字面负相关"），所以离线基线里 dense 的召回条数会少于 k。
> 换成真实 embedding 后这个下限通常要重新标定（例如 0.3~0.5）。

---

## 4. 为什么 Embedder 要可插拔

真实链路和测试链路的诉求是矛盾的：

| | 真实模型 | 测试/CI |
| --- | --- | --- |
| 网络 | 需要 | 不能有 |
| 成本 | 按 token 计费 | 免费 |
| 结果 | 非确定（模型/批次/浮点） | 必须确定才能断言 |
| 依赖 | SDK / torch | 希望零依赖 |

只保留一种实现，必然牺牲另一边：要么测试变成"跑不起来的集成测试"，
要么评测数字只能靠嘴说。所以定义 `Embedder` 接口（`embed_documents` / `embed_query`），
给两个实现：

- `DashScopeEmbedder`：OpenAI 兼容端点 `/compatible-mode/v1/embeddings`，
  用标准库 `urllib` 发请求（不引 `requests`），批量 10 条 + 指数退避重试
  （429/5xx 可重试，4xx 直接失败 —— `EmbeddingError.retryable` 区分）。
- `HashingEmbedder`：字符 n-gram → 带符号哈希桶 → 次线性词频 → L2 归一化。
  用 `blake2b` 保证跨进程/跨平台一致，只用 `hashlib` + `math`。

**代价（必须写清楚）**：`HashingEmbedder` **没有语义能力**。
"微胖"和"丰满"在它眼里毫无关系。因此它产出的 dense 指标**不能**当作
向量检索能力的证据 —— `eval/results.md` 与 README 都把这句话写在显眼位置。
它的正确用途是：可复现的回归基线 + 让 CI 能验证"检索框架"而不是"模型好坏"。

同样的理由也套在生成侧：`ChatClient` 接口 + `EchoChatClient`（把命中片段按引用编号
拼成回答）。它不做语言生成，但让端到端链路（入库 → 检索 → 组装回答）在离线状态下可测。

---

## 5. 为什么去掉 Streamlit

上游两个页面（上传页 / 聊天页）在**演示**上没问题，但在**工程**上有三个硬伤：

1. **每次交互重跑整个脚本**：Streamlit 的执行模型是"自上而下重跑"，
   状态只能塞进 `st.session_state`。上游因此不得不把 `RagService` 缓存在 session 里，
   对象生命周期由框架掌控，稍微复杂一点就会踩重复初始化的坑。
2. **没有稳定接口**：页面渲染出来的东西不能被其它服务调用，也不能被脚本断言。
   上游的"验证"只能靠截图（`assets/chat_demo*.png`）。
3. **依赖重**：`streamlit` 自带一整套前端运行时要装，而它只是一个壳。

换成 FastAPI 后：

- 出入参由 Pydantic 模型约束（`IngestRequest` / `QueryRequest` / `QueryResponse`），
  错误以 4xx 显式返回，而不是页面上的一行异常；
- `tests/test_api.py` 用 `TestClient` 直接断言状态码 + 命中片段的 `source`；
- 依赖变成可选（`pip install -e ".[api]"`），核心链路不需要它。

**代价**：丢掉了开箱即用的聊天 UI。这是明确的取舍 —— 本仓库的定位是"能被调用、
能被测试的检索服务"，UI 交给上层（或用 `curl` / OpenAPI 文档 `/docs`）。

---

## 6. 去重为什么做成两层

| 层级 | 键 | 捕获的场景 | 上游是否有 |
| --- | --- | --- | --- |
| 文档级 | `md5(整篇文本)` | 同一个文件重复上传 | 有（`md5.text` + `[Repeat]`） |
| 片段级 | `sha256(归一化片段)[:16]` | 不同文件包含同一段落 | 无（本次新增） |

上游只做第一层，第二种情况（两份文档都抄了同一段"通用养护小贴士"）会被写入两遍：
向量库里多一份冗余，检索时两条近似片段同时挤进 top-k，等于**用重复项挤掉了真正有
区分度的片段**。`tests/test_regression_dedup.py` 用两篇文档构造了这个场景，
断言公共段落只存一份。

**代价**：
- 片段级去重是**全局**的，同一段文字如果确实属于两个来源，会被归到第一个入库的来源上；
  需要精确归属时应该把 `chunk_hash` 的键加上 `source`（本仓库没这么做，因为
  "跨文档重复"正是要消掉的）。
- 归一化只压平空白、不压平换行：换行代表结构，不当噪音处理（见
  `test_regression_chunk_hash_ignores_whitespace_noise`）。
- 哈希级去重识别不了"语义相同但改写过的段落"。

---

## 7. 重排为什么不用交叉编码器

真正的 rerank 是 cross-encoder（`bge-reranker` / 云端 rerank API），把
`(query, passage)` 一起送进模型打分，效果显著但代价是 torch + 模型权重下载，
与"离线可跑、依赖最小"的硬约束冲突。

折中方案：在候选池内做**BM25 分数加权的轻量重排**：

```
fused_norm  = minmax(RRF scores within candidates)
bm25_norm   = minmax(BM25 scores within candidates)
final       = (1 - w) * fused_norm + w * bm25_norm        w = 0.4
```

它强化"精确词面命中"的候选，抑制"只靠向量相似"的候选。
`tests/test_retriever.py::test_rerank_promotes_strong_bm25_candidate` 用一个两条候选的
最小例子把这个数学关系钉死（`w=0.6` 时 BM25 强者反超）。

**代价如实说**：
1. 只在池内调序，**不重新召回** —— 池里没有的片段，重排救不回来；
2. 实测 Top-1 从 92.9% 掉到 89.3%（`w=0.4` 未调优），也就是**当前参数下是负收益**。
   保留它是因为它是可配置开关 + 有单测覆盖的机制，而不是因为它"更好"。
3. 归一化依赖候选池分布，池子大小变化会让分数尺度变化（这也是 README 里
   没把重排写成默认开启的原因）。

---

## 8. 依赖策略：为什么核心链路零依赖

上游 `requirements.txt` 把 `langchain 0.1.x` / `numpy<2` / `chromadb<0.5` 钉死，
在 Python 3.13+ 上没有可解的版本组合（`ResolutionImpossible`）。复现成本直接变成
"换 Python 版本 + 找老 wheel"，这是复现项目最不该有的门槛。

本仓库的做法：

- **核心链路只用标准库**：切分（`re`）、哈希向量（`hashlib` + `math`）、
  BM25（`collections.Counter`）、余弦（`zip` 循环）、HTTP 调用（`urllib`）。
  代价是性能不如 numpy 向量化（离线场景可忽略：17 个片段、单次查询 <1 ms）。
- **可选依赖显式降级**：`fastapi` / `pydantic` 缺失时 `service.py` 仍能 import，
  调用 `create_app` 才报明确错误；测试集用 `pytest.importorskip` 整模块跳过。
- **不静默降级**：`build_embedder("magic")` 直接 `ValueError`，
  而不是偷偷回落到 hashing —— 配置错误必须炸出来。

---

## 9. 度量口径的取舍

- **Top-1 / Top-3 / MRR** 都以 `source`（文件名）为准。粗，但和上游的元数据缺陷
  直接相关：上游元数据丢失时这三个指标**根本无法计算**。这也说明修那个 bug 是评测的前提。
- 因此额外加了 **关键词命中@3**：命中片段必须真的包含标注的目标文本。
  两条曲线一起看，才能区分"检索框架在跑"和"检索真的对了"。
- 平均耗时只作参考，不是性能声明（没做 profiling，也没排除解释器启动抖动之外的噪声 ——
  计时区间只包住 `retrieve`）。

---

## 10. 如果重来一次，会改什么

1. 切分先按 Markdown 标题层级，再退回字符切分（当前会把小标题和正文切散，见失败案例）。
2. `Chunk.metadata` 用 `TypedDict` 或 dataclass 约束，而不是裸 `dict`。
3. `rrf_k` / `rerank_bm25_weight` / `candidate_k` 做小网格搜索，把最优值写进默认配置
   —— 现在的默认值是"合理猜测"，不是"调过的"。
4. 给 `ChromaStore` 补一个 `pytest.importorskip("chromadb")` 的集成测试，
   让"未实测"这件事在 CI 里有明确位置。