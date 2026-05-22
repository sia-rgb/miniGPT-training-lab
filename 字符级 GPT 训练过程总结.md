# 字符级 GPT 训练教程（以莎士比亚文本为例）

---

## 1. 数据准备与编码（Data Preprocessing / Tokenization）

**主要文件**：`data/shakespeare_char/prepare.py`

**目的**：将原始文本转为模型可处理的整数序列。

**操作步骤**：

1. 读取原始文本文件 `input.txt`。
2. 统计文本中所有字符，生成字符表（vocab）。
3. 将文本中每个字符编码为整数 ID。
4. 划分训练集（train）与验证集（val）。
5. 保存二进制文件 `train.bin` / `val.bin` 和元数据 `meta.pkl`。

**输出**：

* `train.bin`, `val.bin`：整数序列
* `meta.pkl`：字符映射表及相关信息

---

## 2. 批处理与数据加载（Batching）

**主要文件**：`train.py` 内部 `CharDataset` 类

**目的**：生成训练 mini-batch，便于 GPU 并行计算。

**操作步骤**：

1. 从二进制数据中读取整数序列。
2. 随机切片长度为 `block_size + 1` 的序列片段。
3. 前 `block_size` 个 token → 输入 X，后 `block_size` 个 token → 目标 Y。
4. 组合 `batch_size` 个片段形成 mini-batch。

**输出**：

* `X` 张量 `[batch_size, block_size]`
* `Y` 张量 `[batch_size, block_size]`

**备注**：Batch 生成可以与前一次 GPU 前向/反向并行进行，以节省数据加载时间。

---

## 3. 模型前向传播（Forward Pass）

**主要文件**：`model.py`（`GPT` 类）

**目的**：计算输入序列的预测 logits。

**操作步骤**：

1. 将 X 转为 embedding，加入位置编码。
2. 通过多层 Transformer block（Masked Self-Attention + MLP）。
3. 输出 linear projection → logits `[batch, block_size, vocab_size]`。

---

## 4. 损失计算（Loss Computation）

**主要文件**：`train.py`

**目的**：量化模型预测与真实目标的差距。

**操作步骤**：

1. 使用 logits 与目标 Y 计算交叉熵损失（CrossEntropyLoss）。
2. 得到标量 loss 用于反向传播。

**备注**：loss 计算必须在 forward 之后，顺序不可并行。

---

## 5. 反向传播与优化（Backward & Optimizer Step）

**主要文件**：`train.py`

**目的**：更新模型参数，最小化损失。

**操作步骤**：

1. `loss.backward()` → 计算梯度。
2. `optimizer.step()` → 根据梯度更新权重（AdamW）。
3. `optimizer.zero_grad()` → 清空梯度，为下一 batch 做准备。

**注意**：forward → loss → backward → optimizer → zero_grad 必须顺序执行。

---

## 6. 评估与验证（Evaluation / Validation）

**主要文件**：`train.py`

**目的**：监控训练效果，防止过拟合。

**操作步骤**：

1. 使用验证集数据，关闭梯度计算 (`torch.no_grad()`)。
2. 计算平均验证 loss。
3. 可周期性输出 perplexity 指标。

**可并行性**：可以异步进行，不阻塞训练主循环。

---

## 7. 模型保存 Checkpoint（Checkpointing）

**主要文件**：`train.py`

**目的**：保存训练状态，便于恢复和生成。

**操作步骤**：

1. 使用 `torch.save()` 保存模型权重和 optimizer 状态。
2. 可定期保存 checkpoint 或在训练结束保存最终模型。

**可并行性**：异步保存，确保保存时模型状态一致。

---

## 8. 文本生成 / 推理（Sampling / Inference）

**主要文件**：`sample.py`

**目的**：使用训练好的模型生成文本。

**操作步骤**：

1. 加载模型 checkpoint。
2. 输入初始 prompt（可选）。
3. 迭代生成：

   * forward → logits
   * 根据 temperature / top-k 采样下一个 token
   * 拼接到上下文继续生成
4. 最终解码为可读文本。

---

## 重点：train.py 训练循环八大功能关系

```text
+---------------------------+
| 1) 数据加载/批处理       | <- CharDataset
+---------------------------+
            ↓
+---------------------------+
| 2) Forward 传播          | <- model.py GPT 类
+---------------------------+
            ↓
+---------------------------+
| 3) Loss 计算             | <- train.py
+---------------------------+
            ↓
+---------------------------+
| 4) Backward + Optimizer   | <- train.py
+---------------------------+
            ↓
+---------------------------+
| 5) Zero grad             | <- train.py
+---------------------------+
            ↓
+---------------------------+
| 6) Eval / 验证           | <- train.py (异步可行)
+---------------------------+
            ↓
+---------------------------+
| 7) Checkpoint 保存        | <- train.py (异步可行)
+---------------------------+
            ↓
| 8) Sample / 推理          | <- sample.py
```

**说明**：

* **必须顺序执行**：Forward → Loss → Backward → Optimizer → Zero_grad
* **可异步/并行**：数据加载（提前 batch）、Eval、Checkpoint
* `train.py` 集中了 batch、forward、loss、backward、optimizer、eval、checkpoint 的一体化逻辑，保持训练简单易读。

---