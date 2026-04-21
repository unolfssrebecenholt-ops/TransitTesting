# TransitTesting

一个基于 `uv + httpx + rich` 的 API 质量检测探针工具，用来检查：

- 降智 / 模型偷换风险
- 原生 Prompt Caching 是否真实生效
- 当前结果与历史基线的相对接近程度

现在仓库里也包含一个可被 Agent 自动发现的 skill：`transit-probe/`。

## 项目结构

- `app.py`：主程序
- `probe.ini`：默认配置文件
- `run_probe.sh`：默认启动脚本
- `transit-probe/`：Skill 目录，包含 `SKILL.md`、`agents/openai.yaml`、包装脚本和配置模板

## Hermes / Agent Skill

如果你的 Hermes Agent 读取 Codex 风格的 skills 目录，这个 skill 已经可以直接使用。

本地安装方式：

- 仓库内 skill 源码：`transit-probe/`
- 本地发现路径：`~/.codex/skills/transit-probe`

skill 入口脚本：

```bash
python3 transit-probe/scripts/transit_probe.py --help
```

如果 Agent 通过 `$skill-name` 方式调用，使用名称：

```text
$transit-probe
```

## 1. 初始化环境

```bash
uv sync
```

如果只是第一次运行，也可以直接执行：

```bash
./run_probe.sh
```

`uv` 会自动使用项目环境。

## 2. 配置接口信息

编辑 `probe.ini`：

```ini
[probe]
provider = openai
base_url = https://your-transit.example.com/v1
api_key = sk-your-key
model = gpt-5.4
timeout = 120
cases_per_dimension = 5
context_records = 36
request_retries = 2
skip_cache = false
show_raw = false

[report]
report_out = reports/latest.json
report_label = flagship

[baseline]
baseline_profile = baselines/main.json
```

说明：

- `provider`：当前支持 `openai` 或 `anthropic`
- `base_url`：中转站根地址
- `api_key`：接口 Key
- `model`：可选；留空时自动从 `/models` 选择
- `cases_per_dimension`：每个维度跑多少题，默认 `5`
- `skip_cache`：设为 `true` 时只跑降智题组，不跑缓存探测
- `report_out`：可选；导出本次运行报告 JSON
- `baseline_profile`：可选；如果填写，会把本次结果和基线做距离比较

建议不要把真实 `api_key` 提交到仓库。

## 3. 启动探针

最简单的启动方式：

```bash
./run_probe.sh
```

如果你想临时覆盖配置文件里的某个值，也可以直接追加参数：

```bash
./run_probe.sh --model gpt-4o --cases-per-dimension 3
```

参数优先级：

1. 命令行参数
2. `probe.ini`
3. 程序内置默认值

## 4. 输出结果说明

程序会输出两部分：

- `Degradation Suite`
  说明模型在 `Format Control / JSON Schema / Constraint Binding / Context Retrieval` 四个维度上的表现
- `Prompt Caching Test`
  检查缓存字段和两次 TTFT 是否同时出现明显优化

如果配置了 `report_out`，会额外导出一份结构化 JSON 报告，便于后续做基线分析。

## 5. 构建基线

先分别导出多份报告，例如：

```bash
./run_probe.sh --report-label flagship --report-out reports/flagship-001.json
./run_probe.sh --report-label flagship --report-out reports/flagship-002.json
./run_probe.sh --report-label small --report-out reports/small-001.json
```

然后离线构建基线：

```bash
uv run app.py \
  --build-baseline reports \
  --baseline-out baselines/main.json \
  --baseline-name main-profile
```

## 6. 使用基线对比新结果

有了基线文件后，可以在 `probe.ini` 的 `[baseline]` 里配置：

```ini
[baseline]
baseline_profile = baselines/main.json
```

然后再次运行：

```bash
./run_probe.sh
```

输出里会新增 `Baseline Comparison`，用于判断当前结果更接近哪一类已知样本。

## 7. 常用命令

只跑降智题组：

```bash
./run_probe.sh --skip-cache
```

显示原始响应预览：

```bash
./run_probe.sh --show-raw
```

指定其他配置文件：

```bash
uv run app.py --config custom.ini
```
