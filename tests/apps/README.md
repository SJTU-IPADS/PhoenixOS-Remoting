# XPURemoting Application Examples

## Quick Start

### build the project
in root directory
```shell
cargo build
```

### download models\[optinal\]

if you want to run inference applications, you need to download the model first. The models are

- [gpt2](https://huggingface.co/openai-community/gpt2)
- [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

if you want to run ResNet application, please create a `checkpoint` folder under `tests/apps/` and run the training application first

### run applications

in root directory
```shell
NETWORK_CONFIG="path/to/config.toml" cargo run server
```

in tests/apps directory
```shell
NETWORK_CONFIG="path/to/config.toml" ./run.sh <target-python> <params>
# e.g. ./run.sh infer/gpt2/inference.py 1 4
```

information about configuratoins when running server can be found in integraton test part in [README](../../README.md)

## Correctness Test

### Parmas

epoch:1 , batch_size:4

### GPT2

remoting output: 
```
["Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I"]
```

local output: 
```
["Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I", "Hello, I'm a language model, not a programming language. I'm a language model. I"]
```

### BERT

remoting output: 
```
["The primary language of the United States is English.","The primary language of the United States is English.","The primary language of the United States is English.","The primary language of the United States is English."]
```

local output: 
```
["The primary language of the United States is English.","The primary language of the United States is English.","The primary language of the United States is English.","The primary language of the United States is English."]
```

### Stable Diffusion

remoting output:
![remoting_output](assets/astronaut_rides_horse_remoting.png)


local output:
![local_output](assets/astronaut_rides_horse_local.png)