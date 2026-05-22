# nanoGPT（中文版）

![nanoGPT](assets/nanogpt.jpg)

---

**2025年11月更新** nanoGPT 现在有一个新的、改进的“表亲”项目 [nanochat](https://github.com/karpathy/nanochat)。您很可能其实是想使用/找到 nanochat。nanoGPT（本仓库）现在非常陈旧且已废弃，但我会保留它作为纪念。

---

用于训练/微调中等规模 GPT 的最简单、最快的代码库。这是对 [minGPT](https://github.com/karpathy/minGPT) 的重写，优先注重实用性而非教学性。仍在积极开发中，但当前文件 `train.py` 可以在单个 8×A100 40GB 节点上，在约 4 天的训练时间内重现 OpenWebText 上的 GPT-2 (124M)。代码本身简洁可读：`train.py` 是一个约 300 行的样板训练循环，`model.py` 是一个约 300 行的 GPT 模型定义，并且可以选择从 OpenAI 加载 GPT-2 权重。仅此而已。

![repro124m](assets/gpt2_124M_loss.png)

由于代码非常简单，非常容易根据您的需求进行修改，从头训练新模型，或者微调预训练的检查点（例如，目前可作为起点的最大模型是 OpenAI 的 GPT-2 1.3B 模型）。

## 安装

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖项：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers` 用于 HuggingFace transformers <3（加载 GPT-2 检查点）
- `datasets` 用于 HuggingFace datasets <3（如果你需要下载并预处理 OpenWebText）
- `tiktoken` 用于 OpenAI 的快速 BPE 编码 <3
- `wandb` 用于可选日志记录 <3
- `tqdm` 用于进度条 <3

## 快速开始

如果你不是深度学习专业人士，只是想感受一下魔法并初步尝试，最快的方式是在莎士比亚的作品上训练一个字符级别的 GPT。首先，我们将其下载为一个单独的（1MB）文件，并将原始文本转换为一个大的整数流：

```sh
python data/shakespeare_char/prepare.py
```

这会在该数据目录下创建 `train.bin` 和 `val.bin`。现在是时候训练你的 GPT 了。模型大小在很大程度上取决于你系统的计算资源：

**我有一块 GPU**。太好了，我们可以使用 [config/train_shakespeare_char.py](config/train_shakespeare_char.py) 配置文件中提供的设置快速训练一个小型 GPT：

```sh
python train.py config/train_shakespeare_char.py
```

如果你查看文件内部，你会发现我们正在训练一个上下文大小最多为 256 个字符、特征通道数为 384 的 GPT，它是一个 6 层 Transformer，每层 6 个注意力头。在一块 A100 GPU 上，这次训练大约需要 3 分钟，最佳验证损失为 1.4697。根据配置，模型检查点会被写入 `--out_dir` 目录 `out-shakespeare-char`。因此训练结束后，我们可以通过将采样脚本指向该目录来从最佳模型中进行采样：

```sh
python sample.py --out_dir=out-shakespeare-char
```

这会生成一些样本，例如：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哈哈 `¯\_(ツ)_/¯`。对于一个在 GPU 上训练了 3 分钟的字符级模型来说还算不错。通过在这个数据集上微调预训练的 GPT-2 模型，很可能获得更好的结果（参见后面的微调部分）。

**我只有一台 MacBook**（或其他廉价电脑）。没关系，我们仍然可以训练 GPT，但需要适当降低要求。我建议在安装时选择最前沿的 PyTorch 夜间版（[在此处选择](https://pytorch.org/get-started/locally/)），因为它目前很可能使你的代码更高效。但即使没有它，一个简单的训练运行也可以如下所示：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

在这里，由于我们在 CPU 而非 GPU 上运行，我们必须设置 `--device=cpu`，并通过 `--compile=False` 关闭 PyTorch 2.0 的编译功能。然后评估时我们会得到更嘈杂但更快的估计（`--eval_iters=20`，从 200 降下来），我们的上下文大小只有 64 个字符而不是 256，每次迭代的批量大小只有 12 个样本而不是 64。我们还将使用一个更小的 Transformer（4 层，4 个头，128 嵌入维度），并将迭代次数减少到 2000（并相应地使用 `--lr_decay_iters` 将学习率衰减到大约 max_iters）。因为我们的网络非常小，我们也降低了正则化（`--dropout=0.0`）。这仍然在大约 3 分钟内运行完成，但损失只能达到 1.88，因此样本质量也更差，但仍然很有趣：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

生成如下样本：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

对于在 CPU 上运行约 3 分钟来说还不错，至少捕捉到了正确的字符形态。如果你愿意等待更长时间，可以随意调整超参数，增加网络大小、上下文长度（`--block_size`）、训练时长等。

最后，在 Apple Silicon MacBook 上使用最新的 PyTorch 版本时，请确保添加 `--device=mps`（“Metal Performance Shaders”的缩写）；PyTorch 将使用片上 GPU，这可以显著加速训练（2-3 倍）并允许你使用更大的网络。更多信息请参见 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)。

## 复现 GPT-2

更专业的深度学习从业者可能对复现 GPT-2 的结果更感兴趣。那么开始吧——我们首先对数据集进行分词，这里使用的是 [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)，这是 OpenAI（私有）WebText 的一个开放复现版本：

```sh
python data/openwebtext/prepare.py
```

这会下载并对 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集进行分词。它将创建 `train.bin` 和 `val.bin`，其中包含 GPT2 BPE 词元 ID 的一个长序列，存储为原始的 uint16 字节。然后我们就可以开始训练了。要复现 GPT-2 (124M)，你至少需要一个 8×A100 40GB 节点，并运行：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

这将使用 PyTorch 分布式数据并行（DDP）运行大约 4 天，损失降至约 2.85。现在，仅在 OWT 上评估的 GPT-2 模型验证损失约为 3.11，但如果进行微调，损失将降至约 2.85 的范围（由于明显的数据集领域差异），从而使两个模型大致匹配。

如果你在集群环境中，并且拥有多个 GPU 节点，你可以让 GPU 飞速运转，例如跨越 2 个节点：

```sh
# 在第一个（主）节点上运行，示例 IP 123.456.123.456：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# 在工作节点上运行：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

建议对你的互连进行基准测试（例如 iperf3）。特别是，如果你没有 InfiniBand，则需要在上述启动命令前加上 `NCCL_IB_DISABLE=1`。你的多节点训练可以工作，但很可能会_爬行_。默认情况下，检查点会定期写入 `--out_dir`。我们可以通过简单的 `python sample.py` 从模型中采样。

最后，要在单个 GPU 上训练，只需运行 `python train.py` 脚本。请查看其所有参数，该脚本力求非常可读、可修改和透明。你可能需要根据需求调整其中许多变量。

## 基线

OpenAI GPT-2 检查点让我们能够为 openwebtext 建立一些基线。我们可以通过以下方式获取数据：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

并观察到以下训练和验证损失：

| 模型 | 参数量 | 训练损失 | 验证损失 |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M | 3.11 | 3.12 |
| gpt2-medium | 350M | 2.85 | 2.84 |
| gpt2-large | 774M | 2.66 | 2.67 |
| gpt2-xl | 1558M | 2.56 | 2.54 |

然而，我们必须注意，GPT-2 是在（未公开的）WebText 上训练的，而 OpenWebText 只是该数据集的最佳开放复现。这意味着存在数据集领域差异。实际上，采用 GPT-2 (124M) 检查点并在 OWT 上直接微调一段时间，损失可以降至约 2.85。这便成为关于复现的更合适的基线。

## 微调

微调与训练没有区别，我们只需确保从预训练模型初始化，并以较小的学习率进行训练。关于如何在新的文本上微调 GPT 的示例，请转到 `data/shakespeare` 并运行 `prepare.py` 下载 tiny shakespeare 数据集，并使用 GPT-2 的 OpenAI BPE 分词器将其渲染为 `train.bin` 和 `val.bin`。与 OpenWebText 不同，这将在几秒钟内完成。微调可能只需要很少的时间，例如在单个 GPU 上只需几分钟。运行一个微调示例：

```sh
python train.py config/finetune_shakespeare.py
```

这将加载 `config/finetune_shakespeare.py` 中的配置参数覆盖（尽管我没有对它们进行太多调整）。基本上，我们使用 `init_from` 从 GPT2 检查点初始化，然后正常训练，只是训练时间更短且学习率较小。如果遇到内存不足，请尝试减小模型大小（可选值为 `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）或减小 `block_size`（上下文长度）。最佳检查点（最低验证损失）将位于 `out_dir` 目录中，例如根据配置文件，默认在 `out-shakespeare` 中。然后你可以运行 `sample.py --out_dir=out-shakespeare` 中的代码：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

哇，GPT 进入了一些黑暗领域。我并没有在配置中过多调整超参数，欢迎尝试！

## 采样 / 推理

使用脚本 `sample.py` 从 OpenAI 发布的预训练 GPT-2 模型或你自己训练的模型中进行采样。例如，以下是从最大的可用模型 `gpt2-xl` 采样的方法：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

如果你想从自己训练的模型中采样，请使用 `--out_dir` 相应地指向代码。你也可以用文件中的一些文本作为提示，例如 ```python sample.py --start=FILE:prompt.txt```。

## 效率说明

对于简单的模型基准测试和分析，`bench.py` 可能很有用。它与 `train.py` 训练循环核心部分发生的情况相同，但省略了许多其他复杂性。

请注意，代码默认使用 [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写本文时（2022年12月29日），这使得 `torch.compile()` 在夜间版本中可用。这一行代码带来的改进是显著的，例如将迭代时间从约 250ms/步减少到 135ms/步。PyTorch 团队干得漂亮！

## 待办事项

- 研究并添加 FSDP 替代 DDP
- 在标准评估（例如 LAMBADA？HELM？等）上评估零样本困惑度
- 微调微调脚本，我认为超参数不太好
- 计划在训练期间线性增加批量大小
- 整合其他位置编码（RoPE、ALiBi）
- 我认为在检查点中将优化器缓冲区与模型参数分开
- 增加关于网络健康状况的额外日志记录（例如梯度裁剪事件、梯度范数）
- 对更好的初始化等进行更多研究

## 故障排除

请注意，默认情况下本仓库使用 PyTorch 2.0（即 `torch.compile`）。这是相当新且实验性的，并且尚未在所有平台上可用（例如 Windows）。如果你遇到相关错误消息，请尝试通过添加 `--compile=False` 标志来禁用此功能。这会减慢代码速度，但至少可以运行。

关于本仓库、GPT 和语言建模的一些背景知识，观看我的 [Zero To Hero 系列](https://karpathy.ai/zero-to-hero.html)可能会有所帮助。具体来说，如果你有一些语言建模的背景知识，[GPT 视频](https://www.youtube.com/watch?v=kCc8FmEb1nY)很受欢迎。

如有更多问题/讨论，欢迎随时访问 Discord 上的 **#nanoGPT** 频道：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 致谢

所有 nanoGPT 实验均由 [Lambda labs](https://lambdalabs.com) 上的 GPU 提供支持，这是我最喜欢的云 GPU 供应商。感谢 Lambda labs 赞助 nanoGPT！








# nanoGPT（英文版）

![nanoGPT](assets/nanogpt.jpg)


---

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead. nanoGPT (this repo) is now very old and deprecated but I will leave it up for posterity.

---

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
