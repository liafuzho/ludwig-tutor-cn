# [Ludwig](https://ludwig-ai.github.io/ludwig-docs/) 中文教程

<font color="silver">翻译自：</font><br>
[https://ludwig-ai.github.io/ludwig-docs/getting_started/](https://ludwig-ai.github.io/ludwig-docs/getting_started/)<br>
[https://ludwig-ai.github.io/ludwig-docs/examples/](https://ludwig-ai.github.io/ludwig-docs/examples/)<br>

## 目录
* [入门](#入门)
  * [简介](#简介)
  * [安装](#安装)
  * [基本原则](#基本原则)
  * [训练](#训练)
  * [分布式训练](#分布式训练)
  * [预测与评估](#预测与评估)
  * [编程接口](#编程接口)
  * [可扩展性](#可扩展性)
* [示例](#示例)
  * [文本分类](#文本分类)
  * [命名实体识别标签](#命名实体识别标签)
  * [自然语言理解](#自然语言理解)
  * [机器翻译](#机器翻译)
  * [基于 Sequence2Sequence 的聊天对话建模](#基于Sequence2Sequence的聊天对话建模)
  * [情绪分析](#情绪分析)
  * [图像分类](#图像分类)
  * [图像分类(MNIST)](#图像分类(MNIST))
  * [图像描述](#图像描述)
  * [单样本学习和孪生网络](#单样本学习和孪生网络)
  * [视觉问答](#视觉问答)
  * [数字语音识别](#数字语音识别)
  * [语音验证](#语音验证)
  * [Kaggle's Titanic: 预测幸存者](#KagglesTitanic预测幸存者)
  * [时间序列预测](#时间序列预测)
  * [时间序列预报(天气数据示例)](#时间序列预报(天气数据示例))
  * [电影分级预测](#电影分级预测)
  * [多标签分类](#多标签分类)
  * [多任务学习](#多任务学习)
  * [简单回归: 燃油效率预测](#简单回归燃油效率预测)
  * [二分类：欺诈交易识别](#二分类欺诈交易识别)
* [Ludwig 中文使用手册](https://github.com/liafuzho/ludwig-doc-cn)

## 入门<a id='入门'></a>
### 简介<a id='简介'></a>
Ludwig 是一个工具箱，它允许用户训练和测试深度学习模型，而不需要编写代码。它建立在 TensorFlow 之上。

为了训练您的模型，您必须提供一个包含您数据的文件，一个“列”列表作为输入，另外一个“列”列表作为输出，Ludwig 将完成剩下的工作。简单的命令可以本地也可以分布式地去训练模型，并使用它们来预测新数据。

您也可以使用 Python 的 Ludwig 编程接口。一套可视化工具允许您分析模型的训练和测试其性能，并对他们进行比较。

Ludwig 是基于可扩展性原则构建的，并且基于抽象的数据类型，不但可以轻松添加新的数据类型，而且对添加新的模型架构也变得非常容易。

它可以使从业人员用来做快速训练并且测试深度学习模型，也可以让研究人员通过实验设置，执行相同的数据处理和评估来获得强劲可靠的比较基准。

Ludwig 提供了一组模型架构，可以组合在一起为给定的用例创建端到端的模型。打个比方，如果深度学习函数库为您的建筑物提供了构建模块， Ludwig 则为您的城市提供了建筑物。您可以从可用的建筑物中进行选择，或者将自己的建筑物添加进来使其可用。

工具箱中的核心设计原则是:-不需要编码: 不需要编码技能来训练模型并使用它来获得预测。通用性: 一种新的基于数据类型的深度学习模型设计方法使得该工具可以跨越许多不同的用例使用。- 灵活性: 经验丰富的用户可以广泛的控制模型构建和训练，而新手将发现它易于使用。扩展性: 可以轻松添加新的模型架构和新的数据类型。可理解性: 深度学习模型内部通常被认为是黑箱，但是 Ludwig 提供标准的可视化工具来理解它们的性能并比较它们的预测。- 开源: Apache License 2.0

### 安装<a id='安装'></a>
Ludwig 要求您使用 python3.6+ ，如果您没有安装 python3，可以通过运行以下命令来安装:

```sh
sudo apt install python3  # on ubuntu
brew install python3      # on mac
```

您可能想虚拟一个单独的 Python 环境。

```shell
virtualenv -p python3 venv
```

安装 Ludwig 只需运行：

```shell
pip install ludwig
```

这将只安装基本的 Ludwig，不同的需求类型有着不同的依赖关系。我们将它们分为不同的附加组件，这样用户仅仅只安装他们真正需要的组件：`-ludwig[text]` 为文本依赖。`-ludwig[audio]` 为音频和语音的依赖。`-ludwig[image]` 为图像依赖。`-ludwig[hyperopt]` 为超参数优化依赖。`-ludwig[horovod]` 为分布式训练依赖。`-ludwig[serve]` 为服务依赖。`-ludwig[viz]` 为可视化依赖。`-ludwig[test]` 为测试依赖。

支持 Horovod 分布式训练， 可以通过 `pip install ludwig[horovod]`  或 `HOROVOD_GPU_OPERATIONS=NCCL pip install ludwig[horovod]` 用于 GPU 支持。详情请参阅 Horovod 的[安装指南](https://horovod.readthedocs.io/en/stable/install_include.html)，了解可用的安装选项。

可以使用 `pip install ludwig[extra1,extra2,...]` 同时安装任何额外的包，例如 `pip install ludwig[text,viz]`。完整的依赖关系集可以通过 `pip install ludwig[full]` 安装。

希望从源代码构建的开发人员：

```shell
git clone git@github.com:ludwig-ai/ludwig.git
cd ludwig
virtualenv -p python3 venv
source venv/bin/activate
pip install -e '.[test]'
```

注意: 如果您不使用 GPUs 运行，那么您可能希望使用仅支持 cpu 的 TensorFlow 版本，它占用的磁盘空间要少得多。要使用只支持 cpu 的 TensorFlow 版本，请卸载 TensorFlow 并在安装路德维希后用 TensorFlow-cpu 替换它。请确保在 requirements.txt 所示的兼容范围内安装一个版本。

**注意**：如果您不使用 GPU 运行，只想使用 CPU 版本的 TensorFlow， 它占用的磁盘空间则要少得多。要使用只支持 CPU 的 TensorFlow 版本， 请卸载 TensorFlow 并在安装 Ludwig 后用 tensorflow-cpu 替换它。请确保在 requirements.txt 所示的兼容范围内安装一个版本。

### 基本原则<a id='基本原则'></a>
Ludwig 提供了三个主要功能：训练模型和使用他们来预测和评估。它基于抽象的数据类型，因此将对共有数据类型的不同数据集执行相同的数据预处理和后处理，并且开发的相同编码和解码模型可以在多个任务中重复使用。

在 Ludwig 中训练一个模型非常简单: 您提供一个数据集文件和一个 YAML 文件定义的模型。

模型定义包含一个输入列的列表和一个输出列的列表，您所要做的就是指定数据集中作为输入到模型的列的名称以及它们的数据类型，还有数据集中作为输出的列的名称，模型将学会预测的目标变量。Ludwig 将为此构建一个深度学习模型，并为您训练它。

目前，Ludwig 中可用的数据类型是：

* binary
* numerical
* category
* set
* bag
* sequence
* text
* timeseries
* image
* audio
* date
* h3
* vector

通过为输入和输出选择不同的数据类型，用户可以解决许多不同的任务，例如：

* text input + category output = text classifier【文本分类】
* image input + category output = image classifier【图像分类】
* image input + text output = image captioning【图像描述】
* audio input + binary output = speaker verification【语音验证】
* text input + sequence output = named entity recognition / summarization【命名实体识别/摘要】
* category, numerical and binary inputs + numerical output = regression【回归】
* timeseries input + numerical output = forecasting model【预测模型】
* category, numerical and binary inputs + binary output = fraud detection【欺诈检测】

看看[示例](#示例)，了解如何使用 Ludwig 用于更多的任务。

模型定义可以包含额外的信息，特别是如何预处理数据中的每一列，为每一列使用哪个编码器和解码器，架构和训练参数，超参数进行优化。这使得新手可以轻松使用，专家可以灵活使用。

### 训练<a id='训练'></a>
例如，给定一个下面所示的文本分类数据集：

| doc_text | class | 
| :-------- | :----- |
| Former president Barack Obama ... | politics | 
| Juventus hired Cristiano Ronaldo ... | sport | 
| LeBron James joins the Lakers ... | sport | 
| ... | ... |

您希望学习一个使用 doc _ text 列的内容作为输入来预测 class 列中的值的模型。你可以使用下面的模型定义：

```json
{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}
```

然后在控制台输入以下命令开始训练：

```shell
ludwig train --dataset path/to/file.csv --config "{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}"
```

其中 path/to/file.csv 是 UTF-8 编码的 CSV 文件的路径，该文件包含先前表中的数据集(许多其他的数据格式也支持)。Ludwig 将:

1. 执行数据的随机分割。
2. 预处理数据集。
3. 构建一个 ParallelCNN 模型(文本的默认模型) ，该模型通过 softmax 分类器解码输出分类。
4. 在训练集上训练模型，直到验证集上的性能停止提升。

训练进度将显示在控制台，但 TensorBoard 也可以使用。

如果您更喜欢使用 RNN 编码器并增加要训练的周期数，您所要做的就是将模型定义改为：

```json
{input_features: [{name: doc_text, type: text, encoder: rnn}], output_features: [{name: class, type: category}], training: {epochs: 50}}
```

请参考[使用手册](https://github.com/liafuzho/ludwig-doc-cn)，找出模型定义中可用的所有选项，并查看[示例](#示例)，了解如何将 Ludwig 用于几个不同的任务。

训练结束后，Ludwig 将创建一个 `results ` 目录，其中包含训练模型及其超参数和训练过程的统计摘要。您可以使用 `visualize` 工具中的几个可视化选项之一来可视化它们，例如：

```shell
ludwig visualize --visualization learning_curves --training_statistics path/to/training_statistics.json
```

这个命令将显示一个如下图所示的图形，在这里您可以看到在训练过程中的损失和准确性：

![](/Users/kevinluo/Documents/ludwig/getting_started_learning_curves.png)

还有几个可视化效果可用，请参考[可视化](https://github.com/liafuzho/ludwig-doc-cn#%E5%8F%AF%E8%A7%86%E5%8C%96)以获得更多细节。

### 分布式训练<a id='分布式训练'></a>
您可以使用 [Horovod](https://github.com/horovod/horovod) 分布式训练，它允许在具有多个 GPUs 的单台机器上以及在具有多个 GPUs 的多台机器上进行培训。详情请参阅[使用手册](https://github.com/liafuzho/ludwig-doc-cn)。

### 预测与评估<a id='预测与评估'></a>
如果您想让以前训练过的模型预测新数据的目标输出值，您可在控制台输入以下命令：
```shell
ludwig predict --dataset path/to/data.csv --model_path /path/to/model
```

运行这个命令将返回模型预测。

如果数据集还包含目标输出的正确标注，则可以将它们与从模型获得的预测进行比较，以评估模型性能。

```shell
ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
```

这将产生评估性能统计数据，可以通过可视化工具进行可视化，也可用于比较不同模型的性能和其预测，例如：

```shell
ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
```

将返回一个不同指标的模型比较条形图：

![](/Users/kevinluo/Documents/ludwig/compare_performance.png)

一个方便的 `ludwig experiment` 命令，可以依次进行训练和预测。

### 编程接口<a id='编程接口'></a>
Ludwig 还提供了一个简单的编程接口，允许您训练或加载模型，并使用它获取对新数据的预测:

```python
from ludwig.api import LudwigModel

# train a model
config = {...}
model = LudwigModel(config)
train_stats = model.train(training_data)

# or load a model
model = LudwigModel.load(model_path)

# obtain predictions
predictions = model.predict(test_data)
```

`config` 包含提供给命令行的 YAML 文件的相同信息。更多的细节在[使用手册](https://github.com/liafuzho/ludwig-doc-cn)和接口文档中提供。

### 可扩展性<a id='可扩展性'></a>
Ludwig 是在考虑可扩展性的基础上建立起来的。通过添加特定于数据类型的抽象类实现(包含预处理、编码和解码数据的函数) ，可以很容易地添加目前不支持的附加数据类型。

此外，通过实现一个接受张量(根据数据类型，具有特定等级)作为输入并提供张量作为输出的类，可以很容易地添加具有特定超参数的新模型。这鼓励重用和与社区共享新的模型。更多细节请参考开发人员指南。

## 示例<a id='示例'></a>
本节包含几个示例，展示如何使用 Ludwig 为各种任务构建模型。对于每个任务，我们都显示一个示例数据集和一个示例模型定义，可以使用这些数据来训练模型。

### 文本分类<a id='文本分类'></a>
这个例子展示了如何用 Ludwig 构建一个文本分类器。它可以使用 [Reuters-21578](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/reuters-allcats-6.zip) 数据集执行，特别是在 [CMU's Text Analytics course website](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/) 上的版本。同一个网页上的其他数据集，比如 [OHSUMED](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/ohsumed-allcats-6.zip)，是一个著名的医学摘要数据集，而 [Epinions.com](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW2/epinions.zip), 网站，一个产品评论数据集，也可以使用，因为列的名称是相同的。

| text | class | 
| :-------- | :----- |
| Toronto Feb 26 - Standard Trustco said it expects earnings in 1987 to increase at least 15... | politics | 
| New York Feb 26 - American Express Co remained silent on market rumors... | acquisition|
| BANGKOK March 25 - Vietnam will resettle 300000 people on state farms known as new economic... | coffee |

```shell
ludwig experiment \
  --dataset text_classification.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: text
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: class
        type: category
```

### 命名实体识别标签<a id='命名实体识别标签'></a>
| utterance | tag | 
| :-------- | :----- |
| Blade Runner is a 1982 neo-noir science fiction film directed by Ridley Scott | Movie Movie O O Date O O O O O O Person Person |
| Harrison Ford and Rutger Hauer starred in it | Person Person O Person person O O O |
| Philip Dick 's novel Do Androids Dream of Electric Sheep ? was published in 1968 | Person Person O O Book Book Book Book Book Book Book O O O Date |

```shell
ludwig experiment \
  --dataset sequence_tags.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: utterance
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          word_tokenizer: space

output_features:
    -
        name: tag
        type: sequence
        decoder: tagger
```

### 自然语言理解<a id='自然语言理解'></a>
| utterance | intent | slots |
| :-------- | :----- | :----- |
| I want a pizza | order_food | O O O B-Food_type |
| Book a flight to Boston | book_flight | O O O O B-City |
| Book a flight at 7pm to London | book_flight | O O O O B-Departure_time O B-City |

```shell
ludwig experiment \
  --dataset nlu.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: utterance
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        bidirectional: true
        num_layers: 2
        reduce_output: null
        preprocessing:
          word_tokenizer: space

output_features:
    -
        name: intent
        type: category
        reduce_input: sum
        num_fc_layers: 1
        fc_size: 64
    -
        name: slots
        type: sequence
        decoder: tagger
```

### 机器翻译<a id='机器翻译'></a>
| english | intent |
| :-------- | :----- |
| Hello! How are you doing? | Ciao, come stai? |
| I got promoted today | Oggi sono stato promosso! |
| Not doing well today | Oggi non mi sento bene |

```shell
ludwig experiment \
  --dataset translation.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: english
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null
        preprocessing:
          word_tokenizer: english_tokenize

output_features:
    -
        name: italian
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        attention: bahdanau
        loss:
            type: sampled_softmax_cross_entropy
        preprocessing:
          word_tokenizer: italian_tokenize

training:
    batch_size: 96
```

### 基于 Sequence2Sequence 的聊天对话建模<a id='基于Sequence2Sequence的聊天对话建模'></a>
| user1 | user2 |
| :-------- | :----- |
| Hello! How are you doing? | Doing well, thanks! |
| I got promoted today | Congratulations |
| Not doing well today | I’m sorry, can I do something to help you? |

```shell
ludwig experiment \
  --dataset chitchat.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: user1
        type: text
        level: word
        encoder: rnn
        cell_type: lstm
        reduce_output: null

output_features:
    -
        name: user2
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        attention: bahdanau
        loss:
            type: sampled_softmax_cross_entropy

training:
    batch_size: 96
```

### 情绪分析<a id='情绪分析'></a>
| review | sentiment |
| :-------- | :----- |
| The movie was fantastic! | positive |
| Great acting and cinematography | positive |
| The acting was terrible!	| negative |

```shell
ludwig experiment \
  --dataset sentiment.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: review
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: sentiment
        type: category
```

### 图像分类<a id='图像分类'></a>
| image_path | class |
| :-------- | :----- |
| images/image_000001.jpg | car |
| images/image_000002.jpg | dog |
| images/image_000003.jpg | boat |

```shell
ludwig experiment \
  --dataset image_classification.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: class
        type: category
```

### 图像分类(MNIST)<a id='图像分类(MNIST)'></a>
这是一个在 MNIST 数据集上训练图像分类模型的完整例子。

#### 下载 MNIST 数据集

```shell
git clone https://github.com/myleott/mnist_png.git
cd mnist_png/
tar -xf mnist_png.tar.gz
cd mnist_png/
```

#### 创建训练和测试 csv
在同一个目录下打开 python shell 并运行以下命令：

```python
import os
for name in ['training', 'testing']:
    with open('mnist_dataset_{}.csv'.format(name), 'w') as output_file:
        print('=== creating {} dataset ==='.format(name))
        output_file.write('image_path,label\n')
        for i in range(10):
            path = '{}/{}'.format(name, i)
            for file in os.listdir(path):
                if file.endswith(".png"):
                    output_file.write('{},{}\n'.format(os.path.join(path, file), str(i)))
```

现在您应该有一个包含 60000 条训练数据的 `mnist_dataset_training.csv` 和包含 10000 条测试数据的 `mnist_dataset_testing.csv`，格式如下：

| image_path | label |
| :-------- | :----- |
| training/0/16585.png | 0 |
| training/0/24537.png | 0 |
| training/0/25629.png | 0 |

#### 训练模型
从安装 ludwig 的虚拟环境的目录中：

```shell
ludwig train \
  --training_set <PATH_TO_MNIST_DATASET_TRAINING_CSV> \
  --test_set <PATH_TO_MNIST_DATASET_TEST_CSV> \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
        conv_layers:
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
            -
                num_filters: 64
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: 0.4
        fc_layers:
            -
                fc_size: 128
                dropout: 0.4

output_features:
    -
        name: label
        type: category

training:
    early_stop: 5
```

### 图像描述<a id='图像描述'></a>
| image_path | caption |
| :-------- | :----- |
| imagenet/image_000001.jpg	| car driving on the street |
| imagenet/image_000002.jpg	 | dog barking at a cat |
| imagenet/image_000003.jpg	 | boat sailing in the ocean |

```shell
ludwig experiment \
--dataset image captioning.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: caption
        type: text
        level: word
        decoder: generator
        cell_type: lstm
```

### 单样本学习和孪生网络<a id='单样本学习和孪生网络'></a>
可以将此示例视为 [Omniglot](https://github.com/brendenlake/omniglot) 数据集上的单样本学习的简单基准。任务是，给定两个手写字符的图像，识别两个实例是否是相同字符。

| image_path_1 | image_path_2 | similarity |
| :-------- | :----- | :----- |
| balinese/character01/0108_13.png | balinese/character01/0108_18.png | 1 |
| balinese/character01/0108_13.png | balinese/character08/0115_12.png | 0 |
| balinese/character01/0108_04.png | balinese/character01/0108_08.png | 1 |
| balinese/character01/0108_11.png | balinese/character05/0112_02.png | 0 |

```shell
ludwig experiment \
--dataset balinese_characters.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path_1
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
    -
        name: image_path_2
        type: image
        encoder: stacked_cnn
        preprocessing:
          width: 28
          height: 28
          resize_image: true
        tied_weights: image_path_1

combiner:
    type: concat
    num_fc_layers: 2
    fc_size: 256

output_features:
    -
        name: similarity
        type: binary
```

### 视觉问答<a id='视觉问答'></a>
| image_path | question | answer |
| :-------- | :----- | :----- |
| imdata/image_000001.jpg | Is there snow on the mountains? | yes |
| imdata/image_000002.jpg | What color are the wheels | blue |
| imdata/image_000003.jpg | What kind of utensil is in the glass bowl | knife |

```shell
ludwig experiment \
--dataset vqa.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn
    -
        name: question
        type: text
        level: word
        encoder: parallel_cnn

output_features:
    -
        name: answer
        type: text
        level: word
        decoder: generator
        cell_type: lstm
        loss:
            type: sampled_softmax_cross_entropy
```

### 数字语音识别<a id='数字语音识别'></a>
这是在 “MNIST 语音识别数据集” 上训练数字语音识别模型的一个完整例子。

#### 下载免费数字语音
```shell
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
mkdir speech_recog_digit_data
cp -r free-spoken-digit-dataset/recordings speech_recog_digit_data
cd speech_recog_digit_data
```

#### 创建一个 CSV 数据集
```shell
echo "audio_path","label" >> "spoken_digit.csv"
cd "recordings"
ls | while read -r file_name; do
   audio_path=$(readlink -m "${file_name}")
   label=$(echo ${file_name} | cut -c1)
   echo "${audio_path},${label}" >> "../spoken_digit.csv"
done
cd "../"
```

现在您 `spoken_digit.csv` 文件中包含了 2000 个具有以下格式的示例：

| audio_path | label |
| :-------- | :----- |
| .../speech_recog_digit_data/recordings/0_jackson_0.wav | 0 |
| .../speech_recog_digit_data/recordings/0_jackson_10.wav | 0 |
| .../speech_recog_digit_data/recordings/0_jackson_11.wav | 0 |
| ... | ... |
| .../speech_recog_digit_data/recordings/1_jackson_0.wav | 1 |

#### 训练模型
从安装 ludwig 的虚拟环境的目录中：

```shell
ludwig experiment \
  --dataset <PATH_TO_SPOKEN_DIGIT_CSV> \
  --config_file config_file.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: audio_path
        type: audio
        encoder: stacked_cnn
        preprocessing:
            audio_feature:
                type: fbank
                window_length_in_s: 0.025
                window_shift_in_s: 0.01
                num_filter_bands: 80
            audio_file_length_limit_in_s: 1.0
            norm: per_file
        reduce_output: concat
        conv_layers:
            -
                num_filters: 16
                filter_size: 6
                pool_size: 4
                pool_stride: 4
                dropout: 0.4
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: 0.4
        fc_layers:
            -
                fc_size: 64
                dropout: 0.4

output_features:
    -
        name: label
        type: category

training:
    early_stop: 10
```

### 语音验证<a id='语音验证'></a>
这个例子描述了如何使用 Ludwig 为一个简单的说话人验证任务。我们假设有下列数据，标签0对应于未授权语音的音频文件，标签1对应于授权语音的音频文件。样本数据如下:

| audio_path | label |
| :-------- | :----- |
| audiodata/audio_000001.wav | 0 |
| audiodata/audio_000002.wav | 0 |
| audiodata/audio_000003.wav | 1 |
| audiodata/audio_000004.wav | 1 |

```shell
ludwig experiment \
--dataset speaker_verification.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: audio_path
        type: audio
        preprocessing:
            audio_file_length_limit_in_s: 7.0
            audio_feature:
                type: stft
                window_length_in_s: 0.04
                window_shift_in_s: 0.02
        encoder: cnnrnn

output_features:
    -
        name: label
        type: binary
```

### Kaggle's Titanic: 预测幸存者<a id='KagglesTitanic预测幸存者'></a>
这个例子描述了如何使用 Ludwig 为 [kaggle competition](https://www.kaggle.com/c/titanic/) 竞赛训练一个模型，来预测乘客在泰坦尼克号灾难中生存的概率。下面是数据的一个例子：

| Pclass | Sex | Age | SibSp | Parch | Fare | Survived | Embarked |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 3 | male | 22 | 1 | 0 | 7.2500 | 0 | S |
| 1 | female | 38 | 1 | 0 | 71.2833 | 1 | C |
| 3 | female | 26 | 0 | 0 | 7.9250 | 0 | S |
| 3 | male | 35 | 0 | 0 | 8.0500 | 0 | S |

完整的数据和列描述可以在[这里](https://www.kaggle.com/c/titanic/data)找到。

在下载数据之后，使用 Ludwig 在这个数据集上训练模型,

```shell
ludwig experiment \
  --dataset <PATH_TO_TITANIC_CSV> \
  --config_file config.yaml
```
`config.yaml`：

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: numerical
    -
        name: Parch
        type: numerical
    -
        name: Fare
        type: numerical
        preprocessing:
          missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

使用更精确的特征转换和预处理可以获得更好的结果，但这个例子的唯一目的是展示这种类型的任务和数据如何在 Ludwig 中使用。

### 时间序列预测<a id='时间序列预测'></a>
尽管直接时间序列预测尚在进行中，但 Ludwig 可以摄取时间序列输入特征数据并进行数值预测。下面例子是一个经过训练的模型，它可以在五个不同的层次上预测时间。

| timeseries_data | y1 | y2 | y3 | y4 | y5 |
| :--- | :-- | :-- | :-- | :-- | :-- |
| 15.07 14.89 14.45 ... | 16.92 | 16.67 | 16.48 | 17.00 | 17.02 |
| 14.89 14.45 14.30 ...	| 16.67 | 16.48 | 17.00 | 17.02 | 16.48 |
| 14.45 14.3 14.94 ... | 16.48 | 17.00 | 17.02 | 16.48 | 15.82 |

```shell
ludwig experiment \
--dataset timeseries_data.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: timeseries_data
        type: timeseries

output_features:
    -
        name: y1
        type: numerical
    -
        name: y2
        type: numerical
    -
        name: y3
        type: numerical
    -
        name: y4
        type: numerical
    -
        name: y5
        type: numerical
```

### 时间序列预报(天气数据示例)<a id='时间序列预报(天气数据示例)'></a>
这个例子说明了使用洛杉矶的历史温度数据进行单变量时间序列预测。

在 Kaggle 上下载并解压每小时可用的历史气象数据<br>
https://www.kaggle.com/selfishgene/historical-hourly-weather-data

运行下面的 python 脚本来准备训练数据：

```python
import pandas as pd
from ludwig.utils.data_utils import add_sequence_feature_column

df = pd.read_csv(
    '<PATH_TO_FILE>/temperature.csv',
    usecols=['Los Angeles']
).rename(
    columns={"Los Angeles": "temperature"}
).fillna(method='backfill').fillna(method='ffill')

# normalize
df.temperature = ((df.temperature-df.temperature.mean()) /
                  df.temperature.std())

train_size = int(0.6 * len(df))
vali_size = int(0.2 * len(df))

# train, validation, test split
df['split'] = 0
df.loc[
    (
        (df.index.values >= train_size) &
        (df.index.values < train_size + vali_size)
    ),
    ('split')
] = 1
df.loc[
    df.index.values >= train_size + vali_size,
    ('split')
] = 2

# prepare timeseries input feature colum
# (here we are using 20 preceding values to predict the target)
add_sequence_feature_column(df, 'temperature', 20)
df.to_csv('<PATH_TO_FILE>/temperature_la.csv')
```

```shell
ludwig experiment \
--dataset <PATH_TO_FILE>/temperature_la.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: temperature_feature
        type: timeseries
        encoder: rnn
        embedding_size: 32
        state_size: 32

output_features:
    -
        name: temperature
        type: numerical
```

### 电影分级预测<a id='电影分级预测'></a>
| year | duration | nominations | categories | rating |
| :--- | :-- | :-- | :-- | :-- |
| 1921 | 3240 | 0 | comedy drama | 8.4 |
| 1925 | 5700 | 1 | adventure comedy | 8.3 |
| 1927 | 9180 | 4 | drama comedy scifi | 8.4 |

```shell
ludwig experiment \
--dataset movie_ratings.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: year
        type: numerical
    -
        name: duration
        type: numerical
    -
        name: nominations
        type: numerical
    -
        name: categories
        type: set

output_features:
    -
        name: rating
        type: numerical
```

### 多标签分类<a id='多标签分类'></a>
| image_path | tags |
| :-------- | :----- |
| images/image_000001.jpg | car man |
| images/image_000002.jpg | happy dog tie |
| images/image_000003.jpg | boat water |

```shell
ludwig experiment \
--dataset image_data.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: image_path
        type: image
        encoder: stacked_cnn

output_features:
    -
        name: tags
        type: set
```

### 多任务学习<a id='多任务学习'></a>
这个例子的灵感来自 Collobert 等人的经典论文 [Natural Language Processing (Almost) from Scratch](https://arxiv.org/abs/1103.0398)。

| sentence | chunks | part_of_speech | named_entities |
| :--- | :-- | :-- | :-- |
| San Francisco is very foggy | B-NP I-NP B-VP B-ADJP I-ADJP | NNP NNP VBZ RB JJ | B-Loc I-Loc O O O |
| My dog likes eating sausage | B-NP I-NP B-VP B-VP B-NP | PRP NN VBZ VBG NN | O O O O O |
| Brutus Killed Julius Caesar | B-NP B-VP B-NP I-NP | NNP VBD NNP NNP | B-Per O B-Per I-Per |

```shell
ludwig experiment \
--dataset nl_data.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
    -
        name: sentence
        type: sequence
        encoder: rnn
        cell: lstm
        bidirectional: true
        reduce_output: null

output_features:
    -
        name: chunks
        type: sequence
        decoder: tagger
    -
        name: part_of_speech
        type: sequence
        decoder: tagger
    -
        name: named_entities
        type: sequence
        decoder: tagger
```

### 简单回归: 燃油效率预测<a id='简单回归燃油效率预测'></a>
这个例子复制了 https://www.tensorflow.org/tutorials/Keras/basic_regression 的 Keras 的例子，根据 [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 数据集中的特征来预测一辆汽车每加仑的行驶里程。

| MPG | Cylinders | Displacement | Horsepower | Weight | Acceleration | ModelYear | Origin |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 18.0 | 8 | 307.0 | 130.0 | 3504.0 | 12.0 | 70 | 1 |
| 15.0 | 8 | 350.0 | 165.0 | 3693.0 | 11.5 | 70 | 1 |
| 18.0 | 8 | 318.0 | 150.0 | 3436.0 | 11.0 | 70 | 1 |
| 16.0 | 8 | 304.0 | 150.0 | 3433.0 | 12.0 | 70 | 1 |

```shell
ludwig experiment \
--dataset auto_mpg.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
training:
    batch_size: 32
    epochs: 1000
    early_stop: 50
    learning_rate: 0.001
    optimizer:
        type: rmsprop
input_features:
    -
        name: Cylinders
        type: numerical
    -
        name: Displacement
        type: numerical
    -
        name: Horsepower
        type: numerical
    -
        name: Weight
        type: numerical
    -
        name: Acceleration
        type: numerical
    -
        name: ModelYear
        type: numerical
    -
        name: Origin
        type: category
output_features:
    -
        name: MPG
        type: numerical
        optimizer:
            type: mean_squared_error
        num_fc_layers: 2
        fc_size: 64
```

### 二分类：欺诈交易识别<a id='二分类欺诈交易识别'></a>

| transaction_id | card_id | customer_id | customer_zipcode | merchant_id | merchant_name | merchant_category | merchant_zipcode | merchant_country | transaction_amount | authorization_response_code | atm_network_xid | cvv_2_response_xflg | fraud_label |
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 469483 | 9003 | 1085 | 23039 | 893 | Wright Group | 7917 | 91323 | GB | 1962 | C | C | N | 0 |
| 926515 | 9009 | 1001 | 32218 | 1011 | Mums Kitchen	 | 5813 | 10001 | US | 1643 | C | D | M | 1 |
| 730021 | 9064 | 1174 | 9165 | 916 | Keller | 7582 | 38332 | DE | 1184 | D | B | M | 0 |

```shell
ludwig experiment \
--dataset transactions.csv \
  --config_file config.yaml
```

`config.yaml`：

```yaml
input_features:
  -
    name: customer_id
    type: category
  -
    name: card_id
    type: category
  -
    name: merchant_id
    type: category
  -
    name: merchant_category
    type: category
  -
    name: merchant_zipcode
    type: category
  -
    name: transaction_amount
    type: numerical
  -
    name: authorization_response_code
    type: category
  -
    name: atm_network_xid
    type: category
  -
    name: cvv_2_response_xflg
    type: category

combiner:
    type: concat
    num_fc_layers: 1
    fc_size: 48

output_features:
  -
    name: fraud_label
    type: binary
```
