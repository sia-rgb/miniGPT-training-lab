# NanoGPT 小型 GPT 训练

## 项目概览

* **项目内容**：基于 [nanoGPT](https://github.com/karpathy/nanoGPT) 复现小型 GPT 模型训练流程。
* **完整闭环**：从文本语料（莎士比亚文集章节） → **字符级编码** → **下一字符预测（next-token prediction）** → **模型训练（前向+反向传播）** → **文本生成**。
* **目标**：

  * 理解 GPT 内部训练机制
  * 掌握 mini-batch、迭代训练、验证集监控流程
  * 为模型训练或微调积累实践经验

---

## 核心知识点

* **数据处理**：

  * 字符级 tokenization（每个字符映射整数）
  * 训练/验证集划分（保证验证集为“新题”，评估泛化能力）
* **训练逻辑**：

  * mini-batch 生成：`batch_size=64`，`block_size=256`，X 为上下文，Y 为下一字符
  * 前向传播 + loss 计算 + 反向传播 + 参数更新
* **迭代与验证**：

  * 训练迭代（iteration）为最小权重更新单位
  * 每 250 次迭代在验证集上计算 loss，监控泛化
* **文本生成**：

  * `sample.py` 加载 checkpoint → 根据 prompt 生成文本
* **工程能力**：

  * checkpoint 管理
  * GPU 并行加速
  * 超参数理解（batch_size、block_size、learning rate）

---

## 本项目复刻的Shakespeare Mini-GPT 参数

| 参数类别           | 参数名称                          | 默认值                               |
| -------------- | ----------------------------- | --------------------------------- |
| **数据 & Batch** | `dataset`                     | 'shakespeare_char' → 使用莎士比亚字符级数据  |
|                | `batch_size`                  | 64 → 每个 mini-batch 的序列条数          |
|                | `block_size`                  | 256 → 上下文长度，每条序列最多 256 个字符        |
|                | `gradient_accumulation_steps` | 1 → 不累计梯度，每个 batch 更新一次           |
| **模型结构**       | `n_layer`                     | 6 层 Transformer Block             |
|                | `n_head`                      | 6 个注意力头                           |
|                | `n_embd`                      | 384 隐藏维度                          |
|                | `dropout`                     | 0.2 → 防止过拟合                       |
| **训练控制**       | `learning_rate`               | 1e-3 → 初始学习率                      |
|                | `max_iters`                   | 2500 → 最大训练迭代次数                   |
|                | `lr_decay_iters`              | 2500 → 学习率衰减步数                    |
|                | `min_lr`                      | 1e-4 → 学习率下限                      |
|                | `beta2`                       | 0.99 → AdamW 参数，适合小 batch token 数 |


---

## 模型迭代次数调整说明（从5000降低到2500）

在当前 Shakespeare Mini-GPT 小模型训练中，原项目默认迭代上限为5000，本项目在迭代 2500 次时停止训练，理由如下：

1. **训练与验证 loss 已接近收敛**

   * train loss 已下降到约 **1.00**
   * val loss 约 **1.47**
   * 迭代 loss 波动很小，说明模型已充分学习训练集与验证集的模式

2. **生成文本效果已经可用**

   * 当前 checkpoint 生成文本具有连贯性和合理性
   * 对小型实验或演示，进一步迭代提升有限

3. **官方迭代次数意义**

   * 默认 `max_iters = 5000` 是经验值，用于确保模型完全收敛并覆盖训练集中稀有模式
   * 对小数据量的小型模型来说，2500 次迭代已覆盖大部分规律

4. **算力与效率考虑**

   * 迭代 5000 次会花费额外时间，而效果提升有限
   * 在展示学习成果时，2500 次迭代的模型已能充分体现训练流程和理解程度

---

## Shakespeare Mini-GPT 生成样本

### 输入

1. **checkpoint**

   * 模型权重（如 `ckpt.pt`），包含训练得到的 embedding、Transformer 层、注意力头和 MLP 权重
   * 决定生成文本的风格和模式

2. **prompt（起始文本）**

   * 默认在脚本中定义为 `context = "\n"`（空行）
   * prompt 很短，相当于骨架或起点，模型从这里开始生成文本
   * 你也可以通过命令行参数 `--start "文本"` 自定义 prompt

3. **生成参数**

   * `num_samples` → 生成样本数量，例如 10
   * `max_new_tokens` → 每个样本生成的最大 token 数
   * `temperature`、`top_k` → 控制生成随机性和多样性

### 输出

* 模型基于 **checkpoint + prompt** 进行自回归生成，生成文本风格与训练数据一致
* 输出文本可能包含训练数据中未出现过的新组合词或台词
* 每个样本都是从同一个 prompt 出发的不同生成结果

> 总结：模型最终输出的 CAPULET、CAMILLO 等角色台词，是模型根据 **checkpoint 的训练模式** 自行生成的。prompt 只是起始骨架，模型依赖训练权重预测字符序列，而不是基于具体问答输入生成的回答。

CAPULET:
The news I shall well as the sea-share,
To love the enemies of the law was men.

CAMILLO:
It is a saint pair livering royal death,
He hath been so much proaches the state and so long.

BENVOLIO:
Take him of my lord: he shall not serve:
He is he would not sigh; on his side is false,
While we die for the merrimage of her son:
'Tis not the pitch, this proclamation respect me
To leave the state with his friend. Why, how looks you so fast?
And you was a good tyrant mortal mather's life,
To w
---------------

DUKE VINCENTIO:
Come, sir, you can I were not pass your foul.

ISABELLA:
Sir, I pray thee, perhappy now.

LUCIO:
I would not relish my guests when I behold him that
be done to be the true; we must not die: the friar is for my
office is not extramined to you the Antium of Lucio with
me: if we were not so four a hateful mother, Julietness and him
friends to us and to rue for him; but render son
this pity is a longer to keep and great modesty.

AUFIDIUS:
I have been a word of contraction like a par
---------------

All inclination, I rather now on.

CAMILLO:
We seek upon again.

CORIOLANUS:
Good, I would have my father:
Therefore, free we were strong-pench on't,
Within the extract of my wife: yet I see the faults,
Which thereto the malice of a thousand contains,
A little believe the air of the whole fight
Romeo, was once the vanity that I have power,
As the thralt of heavens for a days.

GREGORY:
It is the tale of the readies of the worst.

KING EDWARD IV:
The traitor, because that the was so little.

GLOU
---------------

Here in this into being our own brother's part;
Her noble lord, and that my request
Were little proclaim'd frown our brother in
the common confound of my air lawful best
Of what I did repent to be bold;
And yet I were in my brother's eyes.

KING RICHARD II:
You are so; and my leisure to attend me?

EDWARD:
Then stay out, and but the lord of the king,
If I shall not give my father's sake and weeping Edward's.

QUEEN ELIZABETH:
My husband post by the best hands for thy hands.

GLOUCESTER:
No, good
---------------

And all the rest with the old of York:
Alas, the gates of York to your suitors,
And you will not him for the souls of York before.
Plantagenet, his friend, lords, his honour souls
Sits from undertaked, whiles he seems to be ours.

CLIFFORD:
Belike the time that I may reply thy deeds.

WARWICK:
From this wit long will to command these wars
With the hands of heaven for deceived bloody thoughts.

WARWICK:
Ay, my lord; be rebels of thine, before George.

KING HENRY VI:
Exeter, brother, Montgomery:
T
---------------

About myself and whose churchyard humble hasty,
All birth, look'd to bear some word, Clifford,
With summer valiant and fortune's good contract
To warrant him for your consented marriage
That it was very man.
In God's king of my head, and my soul more!

CLIFFORD:
Thou shalt not be contented against thy house.

YORK:
What say'st thou grant then and heavy sweet sons?

QUEEN MARGARET:
Then stay, and go not like the tower soul to do thee.
Yea, that forsake my turns the reetirers from thee,
I will bea
---------------

IE should be the house of this business
Of royal voices, how the seals to his promise,
Are he inclining to his behalf war.

SICINIUS:
O woman!

MENENIUS:
O hare before our propery!

BRUTUS:
Not so shall she take me to be so much of charged
With all this power of great Aufidius time;
For nothing receives in countenance against his gods,
For the men and fall with a verged fellow.

SICINIUS:
Your good father,

VOLUMNIA:
Let me speak you:
If I will be satisfied
By your prating respect with strength
---------------

Now many years for my could be inquired.

Nurse:
My gracious lord, on the heavens hath fell in the state
Of Margaret's freedom of a discommanded black
A valour in his heart, doing but every body
Virtue to go by back again.

JULIET:
Peace, cousin, good friends! no! show we must die,
And tell me the thing lands, that refuge the world:
Thou wilt not tell me thy father to thy tears,
And bid him not speak thee that cure night
Of nice which upon the more and old tailors with him!

FRIAR LAURENCE:
O mo
---------------

By inch the rock, how I have not made to our souls?

DUKE OF AUMERLE:
Ay, she be told me not to die; and she
should I know the house that foul were estable.

HENRY BOLINGBROKE:
My lord, I say.

KING RICHARD II:
My lord, the liege is an early deed;
Which, little armour lives might join ouf purge.

DUKE OF YORK:
Must I take your majesty, your remembrance?
Your honour, go from any hands. For what?
My liege, lords, what for thy heads to you not the gracious deed?

QUEEN MARGARET:
Boldly son! dail, y
---------------

But I am a silent devil:
My son is a my soldier's lover.

GLOUCESTER:
So slaughter'd your crown for your hands
On gallies then far and your scope by the worst.

BUCKINGHAM:
Well, well, well well, sweet plays scarce thou shalt not
Be train'd by the follower.

QUEEN ELIZABETH:
Why, not light she hath sworn to bear the fatal grace
As princely for you will not be here any men.

HENRY BOLINGBROKE:
This name is the king my son were little no way,
Which then call me last the sea that was too resolved


---

## 训练过程中的问题与启发

| 问题            | 描述                          | 启发                                                |
| ------------- | --------------------------- | ------------------------------------------------- |
| CPU vs GPU 性能 | CPU 单线程训练速度过慢               | GPU 是训练效率的关键；理解 batch_size 和 block_size 对显存和速度的影响 |
| 显卡显存限制        | 小显卡无法承载大 batch 或 block_size | 学会调整 batch_size/block_size；理解显存与并行计算的关系           |
| 加速器 / NPU 支持  | 有些加速器不兼容 PyTorch 默认配置       | 掌握跨硬件适配能力，了解加速器兼容性                                |
| 大数据 vs 小数据    | 大数据训练慢，小数据容易 overfit        | 小数据可快速调试 pipeline；大数据训练需 checkpoint 管理与验证 loss 监控 |
| 脚本配置复杂        | train.py、config 文件参数多       | 理解每个参数作用（max_iters, eval_interval, learning_rate） |
| 训练中断问题        | 训练迭代中意外中断，重新运行无法接续          | 强化周期性 checkpoint 保存和可续训练机制意识                      |

---

## 模型对比说明

Shakespeare Mini-GPT 与 GPT-2（124M）复现的核心差异主要体现在以下几个方面：

1. **参数规模**

   * Mini-GPT：约 10.65M 参数
   * GPT-2：约 124M 参数
   * 体现了 **模型容量和表达能力的差异**

2. **训练迭代**

   * Mini-GPT：5000 迭代步
   * GPT-2：600000 迭代步
   * 显示了 **训练强度和数据处理量的差异**

3. **上下文长度（block_size）**

   * Mini-GPT：256 个字符
   * GPT-2：1024 个子词
   * 体现了 **模型理解上下文和生成连贯文本能力的差异**

### 背后原因与约束

* **算力设备差异**：Mini-GPT 可在单卡 GPU 或个人设备上训练；GPT-2 复现需要多卡高端 GPU（如 8 × A100 40GB）才能完成训练。
* **处理目标差异**：Mini-GPT 聚焦于理解 GPT 基础训练流程和生成小规模文本；GPT-2 针对大规模语言建模和泛化任务。
* 我目前没有复现 GPT-2，只做了概念理解。通过对比参数、训练迭代和上下文长度，可以清晰掌握两个模型的核心差异，同时认识到算力和训练规模是主要客观约束。

---

## Shakespeare Mini-GPT vs GPT-2 参数对比

| 参数项                         | Shakespeare Mini-GPT | GPT-2 (124M) 复现 |
| --------------------------- | -------------------- | --------------- |
| 数据集                         | shakespeare_char     | OpenWebText     |
| token 粒度                    | 字符级（char-level）      | 子词级（BPE）        |
| batch_size                  | 64                   | 12              |
| block_size                  | 256                  | 1024            |
| gradient_accumulation_steps | 1                    | 5 × 8           |
| Transformer 层数 n_layer      | 6                    | 12              |
| 注意力头数 n_head                | 6                    | 12              |
| 隐藏维度 n_embd                 | 384                  | 768             |
| dropout                     | 0.2                  | —               |
| 学习率 learning_rate           | 1e-3                 | 项目默认 / 大规模训练配置  |
| 最大迭代 max_iters              | 2500                 | 600000          |
| 学习率衰减 lr_decay_iters        | 2500                 | 600000          |
| 最低学习率 min_lr                | 1e-4                 | 项目默认 / 大规模训练配置  |
| eval_interval               | 250                  | 1000            |
| 参数量                         | 约 10.65M             | 124M            |
| 训练硬件                        | 单卡 GPU / 个人设备可跑      | 8 × A100 40GB   |
| 训练耗时量级                      | 约 3 分钟（A100）（本机为AMD Radeon™ 860M，耗时8小时）         | 约 4–5 天         |
| 主要定位                        | 理解 GPT 基础训练闭环        | 复现更接近真实大模型训练范式  |

---