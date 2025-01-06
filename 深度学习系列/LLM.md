# 前置知识

- Linux
- git
- Python
- PyTorch

# LLM博文

[万字长文——这次彻底了解LLM大语言模型-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/2368425)

[使用LLaMA Factory来训练智谱ChatGLM3-6B模型-CSDN博客](https://xiaoxiang113.blog.csdn.net/article/details/138772373)

[智谱ChatGLM3本地私有化部署（Linux）_chatglm 3 私有化部署-CSDN博客](https://xiaoxiang113.blog.csdn.net/article/details/138967468)

[2024 年 8 个顶级开源 LLM（大语言模型）_开源llm-CSDN博客](https://blog.csdn.net/yugongpeng/article/details/135084958)

# 书生大模型实战营第三期

## [部署书生大模型Demo](https://github.com/InternLM/Tutorial/tree/camp3/docs/L1/Demo)

这部分主要介绍了如何搭建 `InternLM2-Chat-1.8B`、`InternLM-XComposer2-VL-2B`、`InternVL2-2B`

### 模型介绍

<b>各个模型的介绍如下</b>

- InternLM2-Chat-1.8B：18亿参数的语言模型，可以进行问答、总结文本、生成文本、编写故事、分析情感、提供推荐、开发算法、编写代码等任何基于语言的任务。
- InternLM-XComposer2-VL-2B：20亿参数的视觉语言大模型（Vision Language），也就是我们经常说的多模态模型。能够同时从图像和文本中学习。这类模型也属于生成模型，输入为图像和文本，输出为文本。具有良好的零样本能力和泛化能力，能够处理包括文档、网页等在内的多种类型的图像。
- InternVL2-2B：20亿参数的视觉语言大模型，首个综合性能媲美国际闭源商业模型的开源多模态大模型。

简单说，里面有一个语言大模型（LLM）InternLM2-Chat-1.8B，两个视觉语言大模型。

### 模型部署的技术介绍

官方给出了三种模型部署的方式，分别是：无任何量化的 Cli（控制台）部署、Streamlit 的 Web 端部署、LMDeploy 的量化部署。

- Streamlit 是一个基于 Python 的 Web 应用程序框架
- LMDeploy 是一个用于压缩、部署、服务 LLM 的工具包，并且提供了一套完整的服务解决方案。
  - 高效的推理：LMDeploy 通过引入持久化批处理、块 KV 缓存、动态分割与融合、张量并行、高性能 CUDA 内核等关键技术，提供了比 vLLM 高 1.8 倍的推理性能。
  - 有效的量化：LMDeploy 支持仅权重量化和 k/v 量化，4bit 推理性能是 FP16 的 2.4 倍。量化后模型质量已通过 OpenCompass 评估确认。
  - 轻松的分发：利用请求分发服务，LMDeploy 可以在多台机器和设备上轻松高效地部署多模型服务。
  - 交互式推理模式：通过缓存多轮对话过程中注意力的 k/v，推理引擎记住对话历史，从而避免重复处理历史会话。
  - 优秀的兼容性：LMDeploy支持 KV Cache Quant，AWQ 和自动前缀缓存同时使用。

### 配置部署环境

这部分没什么好说的，就是搭建一个运行环境。

```shell
# 创建环境
conda create -n GPT python=3.9 -y
# 激活环境
conda activate GPT
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
pip install transformers==4.38
pip install sentencepiece==0.1.99
pip install einops==0.8.0
pip install protobuf==5.27.2
pip install accelerate==0.33.0
pip install streamlit==1.37.0
```

transformers、sentencepiece 库需要特殊说明。

transformers‌ 是由 [Hugging Face](https://www.baidu.com/s?sa=re_dqa_generate&wd=Hugging Face&rsv_pq=eacac621003cee60&oq=transformers 库介绍&rsv_t=8bacAmHWtyc375RXVhiQEj+v97NWktjcamKZ8Z+EPlytwUjekkxjIoZP7xOfq5PVyHlIOoA&tn=15007414_18_dg&ie=utf-8) 开发的一个 Python 库，专门用于[自然语言处理](https://www.baidu.com/s?sa=re_dqa_generate&wd=自然语言处理&rsv_pq=eacac621003cee60&oq=transformers 库介绍&rsv_t=8bacAmHWtyc375RXVhiQEj+v97NWktjcamKZ8Z+EPlytwUjekkxjIoZP7xOfq5PVyHlIOoA&tn=15007414_18_dg&ie=utf-8)(NLP)任务。它提供了大量的预训练模型。如果我们要加载 [HuggingFace 网站](https://huggingface.co/)的已经训练好的 torch 网络参数或 tensorflow 网络参数必须使用它。

sentencepiece ‌是用于文本分词和编码的开源工具。

### Cli部署InternLM2-Chat-1.8B

下面我们用 Cli 部署一个 InternLM2-Chat-1.8B 的语言模型。

- 任意创建一个工程目录
- 创建一个 cli_deploy.py
- 在 ~.py 中编写部署代码

部署的代码也非常简单。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)
```

代码非常简单，唯一需要解释的是 BFloat16。

BFloat16 (Brain Floating Point)是一种 16 bit 的浮点数格式，动态表达范围和 float 32 是一样的，但是精度低。BF16 的指数位与 FP32 相同，但小数位较少，旨在通过牺牲一定的精度来换取更大的数值空间（Dynamic Range）。

使用 `torch.bfloat16` 的主要优势包括：

- ‌**性能优化**‌：通过减少数据类型的大小，可以加快计算速度并减少内存使用，这对于大规模模型和数据处理非常有益。
- ‌**数值稳定性**‌：与FP16相比，BF16提供了更大的数值范围，这对于某些计算来说是必要的，尤其是在涉及大量数值范围变化的计算中。

`torch.bfloat16` 也有下面的缺点：

- ‌**精度问题**‌：由于牺牲了一定的精度，对于需要高精度计算的场景，BF16 可能不是最佳选择。
- ‌**兼容性问题**‌：不是所有的操作都支持 BF16 数据类型，因此在转换数据类型时可能会遇到不支持的操作，这可能导致运行时错误。
- ‌**混合数据类型处理**‌：在某些情况下，可能需要同时处理多种数据类型，如 FP32 和 BF16 的混合使用，这需要特别注意数值的转换和兼容性。

### streamlit Web部署InternLM2-Chat-1.8B

书生浦语官方已经给我们准备好了[代码](https://github.com/InternLM/Tutorial/blob/camp3/tools/streamlit_demo.py)。创建文件 web_depoly.py，将代码复制到该文件中。然后使用下面的命令启动 Web 服务。

```shell
cd /root/demo
streamlit run ./web_depoly.py --server.address 127.0.0.1 --server.port 6006
```

这样，web 服务就部署好了，我们可以使用 curl 来验证。

```shell
curl localhost:6006
```

如果想要在本地浏览器，通过（localhost）访问远程服务器部署的web服务，可以使用端口转发技术。

```shell
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 远程服务器的ssh端口号
```

在完成端口映射后，我们就可以通过浏览器访问 `http://localhost:6006` 来启动我们的 Demo。

### LMDeploy部署

部署前需要完善环境，安装 LMDeploy。

```shell
conda activate GPT
pip install lmdeploy[all]==0.5.1
pip install timm==1.0.7
```

<b>部署 InternLM-XComposer2-VL-1.8B</b>

使用 LMDeploy 启动一个与模型交互的 Gradio 服务（Web 服务），启动的时候指定模型文件的路径即可。

```shell
lmdeploy serve gradio /share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-1_8b --cache-max-entry-count 0.1
```

上面的命令中，我们需要注意的是 `--cache-max-entry-count 0.1`，这是用于设置 K/V 缓存比例的，会占用一定的显存，但是可以提升模型推理速度。如果模型显存非常富裕，可以设置的大一些。

<b>部署 InternVL2-2B模型</b>

部署方式同上，换一下模型参数的路径即可

```shell
lmdeploy serve gradio /share/new_models/OpenGVLab/InternVL2-2B --cache-max-entry-count 0.1
```

<b>InternVL2-2B 模型参数文件</b>

可以去 HuggingFace 下。

```shell
.
|-- README.md
|-- added_tokens.json
|-- config.json
|-- configuration_intern_vit.py
|-- configuration_internlm2.py
|-- configuration_internvl_chat.py
|-- conversation.py
|-- examples
|   |-- image1.jpg
|   |-- image2.jpg
|   `-- red-panda.mp4
|-- generation_config.json
|-- model.safetensors
|-- modeling_intern_vit.py
|-- modeling_internlm2.py
|-- modeling_internvl_chat.py
|-- preprocessor_config.json
|-- special_tokens_map.json
|-- tokenization_internlm2.py
|-- tokenization_internlm2_fast.py
|-- tokenizer.model
`-- tokenizer_config.json
```

### 练习部署其他模型

LMDeploy 部署 internlm2_5-7b-chat（24G 显存）

```shell
lmdeploy serve gradio /root/share/new_models/Shanghai_AI_Laboratory/internlm2_5-7b-chat --cache-max-entry-count 0.1
```

### 学习总结

学会了上面的部署方式，就可以部署任何 internlm 已经发布的 huggingface 格式的模型了。

后面就是学如何微调属于自己的、特定领域的模型了。

## 提示词工程

### 确定角色

先确定角色，然后在提问：你是一个作家，写一段描述春天的话。

 提示词策略

- 写清楚说明
- 复杂任务拆解成子任务
- 提供参考文本（样例）
- 反复修改

### 思维链提示

给他一个例子，写出完整的思考过程，告诉他该怎么做，怎么思考；然后再提问。

## RAG检索增强生成

InternLM + LlamaIndex RAG 实践。

检索增强生成（Retrieval Augmented Generation，RAG）。

2022 年训练的模型无法回答 2024 年的新问题？如何让 LLM 能够获得最新的知识？有两种方式，一种是微调，另一种是 RAG。微调算力成本和数据成本较高，而 RAG 则比较经济实惠，成本低，可以实时更新。就一种方式。

<b>RAG VS Finetune</b>

RAG

- 低成本
- 可实时更新
- 受基座模型影响大
- 单次回答知识有限

Finetune

- 可个性化微调
- 知识覆盖面广
- 成本高昂
- 无法实时更新

<b>RAG 检索增强生成的处理流程</b>

![image-20240815153312437](D:\CodeNotes\深度学习系列\image-20240815153312437.png)

```mermaid
graph 
SentenceTransformer-->文本向量化
Chroma向量数据库-->匹配相似文本段
用户输入-->文本向量化-->匹配相似文本段-->Prompt
用户输入-->Prompt-->InternLM-->最终输出



```



## XTuner微调模型



# Hugging Face
Hugging Face 是一家专注于自然语言处理和机器学习的公司，以其开源的Transformers库而闻名。该平台提供了丰富的预训练模型，支持多种语言任务，如文本生成、
翻译和情感分析。Hugging Face 还致力于推动A!的民主化，鼓励开发者和研究人员共享和合作。

作为 Hugging Face 最核心的项目，Transformers 无疑是这个社区的灵魂。

Transformers 提供 API 和工具，可轻松下载和训练最先进的预训练模型。使用预训练模型可以降低计算成本
并节省从头开始训练模型所需的时间和资源。这些模型支持不同模式的常见任务:

- 自然语言处理:文本分类、命名实体识别、问答、语言建模、摘要、翻译、多项选择和文本生成。
- 计算机视觉:图像分类、对象检测和分割。
- 音频:自动语音识别和音频分类。
- 多模态:表格问答、光学字符识别、扫描文档信息提取、视频分类和视觉问答。

此外，Hugging Face官方还提供免费的课程，如何利用社区生态(Transformers等项目)来进行NLP的学习

Hugging Face 中检索模型。

- Files and Versions 里包含了模型文件和模型的版本管理。我们如果想要使用模型，需要把里面所有的文件都下载过来。

## GitHub CodeSpace 的使用
GitHub Codespace 通过 GitHub 原生的完全配置、安全的云开发环境，可以更快地启动和写代码
它提供了一系列模板，我们在跑机器学习深度学习相关的实验的时候，可以选择它的 Jupyter NoteBook 模板

## 模型上传

Hugging Face 同样是跟 Git 相关联，对于大文件，我们需要安装 git-lfs，对大文件系统支持。
使用 huggingface-cli login 命令进行登录，登录过程中需要输入用户的 Access Tokens

## Spaces 的使用

Hugging Face Spaces 是一个允许我们轻松地托管、分享和发现基于机器学习模型的应用的平台。
Spaces 使得开发者可以快速将我们的模型部署为可交互的 web 应用，且无需担心后端基础设施或部署的复杂性。

# LLM

## LLM介绍

专用模型==>通用模型。通用大模型，一个模型应对多种任务，多种模态。

LLM（Large Language Model）大语言模型。大型语言模型 (LLM) 是一类基础模型，经过大量数据训练，使其能够理解和生成自然语言和其他类型的内容，以执行各种任务。

LLM 使用一种被称为无监督学习的方法来理解语言。这个过程要向机器学习模型提供大规模的数据集，其中包含数百亿个单词和短语，供模型学习和模仿。这种无监督的预训练学习阶段是开发 LLM（如 GPT-3（Generative Pre-trained Transformer ）和 BERT（Bidirectional Encoder Representations from Transformers）的基本步骤。 

<b>如何让模型落地</b>

```mermaid
graph LR
模型选型-->业务场景-->|复杂|算力情况-->|不足|部分参数微调-->交互环境-->|需要|构建智能体-->模型评测-->|量化|部署模型
交互环境-->|不需要|模型评测
算力情况-->|充足|全参数微调-->交互环境
业务场景-->|简单|交互环境
```

<b>框架选择：书生浦语（方便，开箱即用）</b>

```mermaid
graph TB
1(数据,2TB数据<br>涵盖多种模态与任务)
2(预训练 InternLM-Tain<br>并行训练,极致优化)
3(微调 XTuner<br>支持全参数微调<br>支持LoRA等低成本微调)
4(部署 LMDeploy<br>全链路部署)
5(评测 OpemCompass<br>全方位评测)
6(应用 Lagent AgentLego<br>支持多种智能体<br>支持代码解释器等多种工具)
```

<b>数据</b>

- 文本数据：50 亿个文档，数据量 1TB+
- 图像-文本数据：2200w 文件+，数据量 140GB+
- 视频数据：1000 文件+，数据量 900GB+

- 多模态融合
  万卷包含文本、图像和视频等多模态数据，涵盖科技、文学、媒体、教育和法律等
  多个领域。该数据集对模型的知识内容、逻辑推理和泛化能力的提升有显著效果。
- 精细化处理
  万卷经过语言筛选、文本提取、格式标准化、数据过滤和清洗(基于规则和模型)、多
  尺度去重和数据质量评估等精细数据处理环节，能够很好地适应后续模型训练的要求。
- 价值观对齐
  在万卷的构建过程中，研究人员注重将数据内容与主流中国价值观进行对齐，并通
  过算法和人工评估的结合提高语料库的纯净度。

<b>微调</b>

- 支持增量续训：让基座模型学习到一-些新知识，如某个垂类领域知识。（训练数据:文章、书籍、代码等）
- 有监督微调：让模型学会理解和遵循各种指令，或者注入少量领域知识（训练数据:高质量的对话、问答数据）这种方式要用到的数据量会比增量续训要小一些。
- 8G 显存就可以微调 7B 模型。（支持微调百川、通义千问等模型）

<b>部署</b>

- Python、gRPC、RESTful 接口
- 4bit 量化

<b>OpenCompass 评测体系</b>

<div align="center"><img src="./image-20240815151147641.png"></div>

<b>LLM==>智能体</b>

Lagent 是一个轻量级、开源的基于大语言模型的智能体(agent) 框架，用户可以快速地将一个大语言模型转变为多种类型的智能体。通过 Lagent 框架可以更好的发挥 InternLM 模型的全部性能。

## LLM 部署

常规部署

量化部署

## LLM 微调

不同的微调范式

微调后模型的量化、融合

## LLM RAG

RAG 检索增强生成。

# 工程化

## 向量数据库

[LangChain教程 - 支持的向量数据库列举_langchain支持的向量数据库-CSDN博客](https://blog.csdn.net/fenglingguitar/article/details/142436241)

市面上的向量数据库有很多，这里我们主要学习 LlamaIndex 和 Chroma。

### LlamaIndex

[LlamaIndex - LlamaIndex](https://docs.llamaindex.ai/en/stable/#introduction)

LlamaIndex 是一个上下文增强的 LLM 框架，旨在通过将其与特定上下文数据集集成，增强大型语言模型（LLMs）的能力。它允许您构建应用程序，既利用 LLMs 的优势，又融入您的私有或领域特定信息。LlamaIndex 支持文本、图片等多种数据的向量化存储和检索。

<b>什么是向量化存储?</b>

向量化存储就是指把文本、图像这种数据转换成为<b>向量/特征</b>，存储到特定的向量数据库中，便于快速检索。

<b>如何利用 LlamaIndex 构建向量数据库，并从向量数据库中检索相关信息，一并送入 LLM 中，增强 LLM 的能力</b>

### Chroma

Chroma 是一个开源且轻量级，易于本地部署，专门为检索增强生成 (RAG) 应用设计的向量数据库。不过目前 Chroma 的功能较为基础，缺乏大规模分布式支持；比较适合用于开发和测试阶段的小型项目或个人应用。

快速入门

```python
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()
texts = ["这是一个例子", "另一个例子"]
vectorstore = Chroma.from_texts(texts, embedding_model)
docs = vectorstore.similarity_search("这是查询")
print(docs)
```

## LangChain

[构建检索增强生成（RAG）应用：第一部分 | 🦜️🔗 LangChain 框架](https://python.langchain.ac.cn/docs/tutorials/rag/)

LangChain 提供了一个模块化和适应性强的框架，用于构建各种 NLP 应用程序，包括聊天机器人、内容生成工具和复杂的流程自动化系统。

通过 LangChain，我们可以方便快捷的调用 LLM 的 API 接口，并集成 RAG 增强 LLM。LangChain 内部也集成了一个非常简易的向量数据库 `InMemoryVectorStore`。

`InMemoryVectorStore` 将向量数据存储在内存中，提供了快速的访问速度，但不具备数据持久化的能力。这种实现方式适合于数据量较小且需要快速访问的场景，例如原型开发、测试或小型应用。由于数据存储在内存中，一旦程序终止，内存中的数据将会丢失。因此，`InMemoryVectorStore` 不适合需要长期存储和大规模数据处理的生产环境。

## 工程化实现

茴香豆[InternLM/HuixiangDou: HuixiangDou: Overcoming Group Chat Scenarios with LLM-based Technical Assistance](https://github.com/InternLM/HuixiangDou/tree/main)



 





































