# 快速上手:用 hust-eval-client-mmlu-pro 做一次 MMLU-Pro 测试

写给:第一次使用这个工具、要对一个**已经在运行的** vLLM 服务做 MMLU-Pro 全量测试的同事。
命令与程序的真实命令行对应(`tests/test_docs.py` 会检查文档里的参数都存在)。凡是**没有在真实环境跑过**的,文中标了「未验证」。

> 工具只做客户端:冻结计划、发请求、评分、算哈希、封印证据。**它不会启动、停止或配置服务**;服务由运维负责。

## 0. 开始前要准备好的

| 要有 | 说明 |
|---|---|
| 这个项目的代码 | 目前只在本地目录,没有提交或推送到任何远程;先向项目负责人要 |
| 一个在运行的 OpenAI 兼容服务 | 地址、端口(`--base-url` 不要带 `/v1`)、`served-model-name` |
| 运维提供的服务证据 | 启动日志的**副本**、服务的**实际启动命令**(argv)、镜像 digest(没有就留空) |
| 模型目录 | 用它的分词器和 chat template 来冻结提示 |
| MMLU-Pro 的两个 parquet | `test-00000-of-00001.parquet`、`validation-00000-of-00001.parquet`,HF 版本 `b189ec765aa7ed75c8acfea42df31fdae71f97be`;哈希不对工具会拒绝 |
| `TIGER-AI-Lab/MMLU-Pro` 的固定检出 | commit `f418b116db00b065c2aea046518d8fcf74d39872` |
| Python 3.9 及以上 | |

## 1. 安装

```bash
cd hust-eval-client-mmlu-pro
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip            # Mac 自带 Python 的 pip 太旧,不升级的话下一行会失败
pip install -e '.[prepare]'
pip install 'transformers==5.14.1' 'pyarrow==19.0.1'
hust-eval-mmlu-pro --version
```

`transformers` 的版本要和冻结提示时一致:提示是分词器按 chat template 渲染出的 token ID,版本不同可能渲染不同,
得到的就是另一份计划。升级 pip 之后的安装(`-e` 和普通安装)已验证;`[prepare]` 额外依赖没有在全新环境里从头装过(未验证)。

## 2. 取参考代码

```bash
curl -L -o mmlu-pro-ref.tgz https://codeload.github.com/TIGER-AI-Lab/MMLU-Pro/tar.gz/f418b116db00b065c2aea046518d8fcf74d39872
mkdir ref && tar xzf mmlu-pro-ref.tgz -C ref --strip-components=1
```

`github.com` 直连不通时可用 `codeload.github.com`。工具会检查 `ref/evaluate_from_local.py` 和
`ref/cot_prompt_lib/initial_prompt.txt` 的 SHA-256,不对就拒绝。

## 3. 冻结计划(全量 12,032 题)

```bash
hust-eval-mmlu-pro mmlu-pro prepare --assets <parquet目录> --model-path <模型目录> \
    --reference ref --output plan
```

- 最后应打印 `12032 tasks of 12032`。想先试个子集,加 `--per-category 5`。
- 输出目录必须不存在,已存在会报错。
- **核对 `plan/cases.json` 的 SHA-256**(`hust-eval-mmlu-pro hash-file plan/cases.json`)。同样的模型、数据和参考,
  我们第一次冻结的结果是 `f77a7e5e6fd96fbbf46be7ab11f360309554105e97921805f02d1c93fc05fcf3`。
  不一致多半是 `transformers` 版本不同,**先弄清楚再往下跑**。
- 可选:`--runtime-contract`,把服务环境的描述嵌进计划(由 `hust-eval-mmlu-pro contract --profile … --out …` 生成)。

## 4. 记录服务的证据

```bash
RUN=runs/mmlu-pro/mmlu-pro-<名称>-$(date +%Y%m%dT%H%M%S)
hust-eval-mmlu-pro receipt --run-dir $RUN --base-url http://<地址>:<端口> \
    --profile examples/profile.b0_35b.json \
    --startup-log <启动日志副本> --server-argv <实际argv文件> --image-id sha256:<digest>
```

- `examples/profile.b0_35b.json` 是「期望配置」的模板,**要按你们服务的实际配置改**。不给 `--profile` 则只记录
  `/v1/models`,不做比对。
- `--server-argv` 可以是 JSON 数组,也可以是 `/proc/<pid>/cmdline` 的内容。
- 读输出里的 `mismatches` 和 `unverified`:`mismatches` 非空会以退出码 2 结束,**先去问运维**;
  `unverified` 表示日志和命令行都没能证明那一项,不等于没问题。
- 同一个运行目录只能采集一次。

## 5. 冒烟,再全量

```bash
hust-eval-mmlu-pro mmlu-pro run --plan plan --base-url http://<地址>:<端口> \
    --served-model <服务名> --output $RUN/outputs/smoke --limit 20
```

看 `summary.json`:`transport_failures` 应为 0,`invalid_answers` 不该接近 20。冒烟的结果**不是成绩**(文件里有标注)。
没问题再全量,放后台:

```bash
nohup hust-eval-mmlu-pro mmlu-pro run --plan plan --base-url http://<地址>:<端口> \
    --served-model <服务名> --output $RUN/outputs/mmlu-pro-accuracy --concurrency 16 > $RUN/run.log 2>&1 &
tail -f $RUN/run.log         # 每 100 题一行
```

- 我们在一台 2 卡 910B2 的服务上约 4 到 5 小时;你们的服务会不同。
- 中断后用**同样的命令加 `--resume`** 继续,只复用已成功的请求。
- 失败的请求**不重试**,计为答错并保留在 `requests/` 里。
- 服务要鉴权的话,用环境变量放密钥,再加 `--api-key-env <变量名>`。

## 6. 看结果

```bash
hust-eval-mmlu-pro mmlu-pro summary --plan plan --accuracy-dir $RUN/outputs/mmlu-pro-accuracy --out $RUN/summary.report.json
```

没跑完时报告顶部有 `PARTIAL` 警告,**那个数字不是成绩**(计划按类别排序,没跑完只覆盖前几个类别)。
报告里有:总准确率、无效答案数、传输失败数、顶到输出上限的题数、分类别准确率、token 与时延分布,
以及「正常结束」和「被截断」两类题各自的正确率。

## 7. 压测(可选,**未验证**)

`mmlu-pro materialize` 和 `mmlu-pro bench` 还没有在真实服务上跑过,第一次请小规模试;`bench` 需要本机有 `vllm` 命令。

```bash
hust-eval-mmlu-pro mmlu-pro materialize --assets <parquet目录> --reference ref --model-path <模型目录> \
    --plan plan --output $RUN/inputs/materialized.200.jsonl --sample 200
hust-eval-mmlu-pro mmlu-pro bench --run-dir $RUN --name mmlu-pro --dataset $RUN/inputs/materialized.200.jsonl \
    --base-url http://<地址>:<端口> --served-model <服务名> --tokenizer <模型目录>
```

## 8. 封印并记下哈希

```bash
cp <完整服务日志> $RUN/receipt/serve.log     # 先把服务日志的副本放进运行目录(工具看不到服务是否还在写日志)
hust-eval-mmlu-pro seal evidence --run-dir $RUN
#   -> 得到 evidence_bundle_sha256,写进结果表的说明页(Provenance)
mkdir -p $RUN/results && cp <做好的结果表.xlsx> $RUN/results/
hust-eval-mmlu-pro seal final --run-dir $RUN --table $RUN/results/<结果表.xlsx> --ledger runs/ledger.jsonl
hust-eval-mmlu-pro seal check --run-dir $RUN
```

- `seal final` 打印的 `manifest_sha256`,要记在运行目录**之外**(台账、提交记录或聊天记录),它是整条证据链的锚点。
- **结果表目前要手工做**:工具只输出 `summary` 的 JSON,不生成 xlsx。
- 封印之后改动运行目录里任何文件,`seal check` 都会报出来。

## 9. 汇报时必须写明

- 这是**非官方配置**:提示是套了 chat template、关闭思考的 token ID,保留 5 个样例、没有 `Question:` 停止串、
  上下文 32K,与 MMLU-Pro 官方榜单设定不同。
- 提取不到答案字母时计为答错(官方脚本是随机猜一个)。
- 同一道题再跑一次,输出文本**可能不同**(我们在一台服务上观察到,温度 0 也如此,只试过 5 道题)。单次准确率带有波动,
  工具没有估算;想估算就把整个测试重复几次。
- 被输出上限截断的题,答案常常是兜底正则从正文里抓的一个字母,正确率明显更低,看 `summary` 里的拆分。
- 收据里**没有**服务进程的环境变量、版本、`npu-smi`:工具是纯客户端采不到,需要的话请运维另外提供。
- 镜像 digest、权重哈希没提供就如实写「未提供」。

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| `pip install -e` 报 editable mode 需要 setuptools | pip 太旧,先 `pip install -U pip` |
| `prepare` 报 `SHA-256 mismatch` | parquet 或参考检出的版本不对,换成固定版本 |
| `prepare` 报 `output … exists` | 输出目录必须是新的,换个名字 |
| 全量跑完的 `cases.json` 哈希对不上 | 多半是 `transformers` 版本不同 |
| `run` 报 `Scorer changed after contract freeze` | 冻结计划之后评分器文件被改动,重新 `prepare` 到新目录 |
| `run` 报 `output exists` | 重跑要换新目录;要继续同一次运行请加 `--resume` |
| `run` 报 `plan was made by the upstream adapter` | 旧计划,确认来源后加 `--allow-legacy-plan` |
| 连不上服务 | 工具不会拉起服务,找运维 |
| `receipt` 退出码 2 | 服务配置与期望不符,看 `effective.json` 的 `mismatches` |

更多说明见 `README.md`(英文)。
