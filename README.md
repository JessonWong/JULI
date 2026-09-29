# JULI: Jailbreak Large Language Models by Self-Introspection

This is the core implementation for [JULI: Jailbreak Large Language Models by Self-Introspection](https://arxiv.org/abs/2505.11790), accepted to ICLR 2026. It contains the code needed to cache token distributions, train BiasNet, and apply BiasNet during generation.

This release contains six method files:

- `modeling_biasnet.py`: BiasNet model and checkpoint I/O.
- `training/pre_logits_openweight.py`: cache full token distributions from an open-weight model.
- `training/pre_logits_gemini.py`: cache top-k token distributions returned by Gemini on Vertex AI.
- `training/train_biasnet.py`: train BiasNet from cached distributions.
- `inference_opensource.py`: apply BiasNet during open-weight model generation.
- `inference_gemini.py`: apply BiasNet during Gemini generation.

Benchmark data, target answers, generated results, and evaluator implementations are not bundled. Supply data that you are authorized to use and follow the terms for each model and API.

## Setup

Use Python 3.10 or newer. A CUDA-capable GPU is recommended for model caching, training, and local inference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the PyTorch build appropriate for your system first.
pip install transformers accelerate tqdm google-genai
```

Access to gated Hugging Face models may require authentication. The Gemini scripts use Vertex AI application-default credentials and read the project and location from CLI arguments or environment variables:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"
gcloud auth application-default login
```

No API key, cloud project, model path, or output path is embedded in the code.

## Input format

The caching scripts require a local JSON or JSONL file. By default, each record must contain string fields named `prompt` and `answer`:

```json
{"prompt": "A user prompt", "answer": "The target continuation"}
```

For a JSON file, use either one object or a top-level list of objects. For JSONL, write one object per line. Use `--prompt_field` and `--answer_field` when your field names differ. `--start_index` and `--max_samples` select a deterministic slice of the local records.

## 1. Cache token distributions

For an open-weight model:

```bash
python training/pre_logits_openweight.py \
  --input_file ./inputs/train.jsonl \
  --prompt_field prompt \
  --answer_field answer \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --output_dir ./cached_logits/llama3_8b \
  --max_samples 100
```

For Gemini through Vertex AI:

```bash
python training/pre_logits_gemini.py \
  --input_file ./inputs/train.jsonl \
  --prompt_field prompt \
  --answer_field answer \
  --model_name gemini-2.5-pro \
  --tokenizer_name google/gemma-3-1b-pt \
  --output_dir ./cached_logits/gemini \
  --max_samples 100
```

The Gemini API returns only its top-k candidates. The script stores those returned log probabilities and fills the remaining vocabulary with a value below the smallest returned log probability. Keep `--tokenizer_name` consistent across caching, BiasNet training, and inference so token IDs and vocabulary size remain aligned.

Each selected prompt/answer pair produces one `.pt` file containing `log_probs` and `labels`. Existing files with the same content-derived name are skipped.

## 2. Train BiasNet

```bash
python training/train_biasnet.py \
  --data_dir ./cached_logits/llama3_8b \
  --output_dir ./checkpoints/biasnet_llama3_8b \
  --base_model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --lm_head_init copy \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 1e-5
```

The checkpoint directory contains `config.json` and `pytorch_model.bin`. `--base_model_name_or_path` supplies the hidden and vocabulary sizes. You can instead provide `--hidden_size` and `--vocab_size` explicitly. Run `python training/train_biasnet.py --help` for the available initialization modes and training controls.

## 3. Apply BiasNet during generation

For an open-weight model:

```bash
python inference_opensource.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --biasnet_ckpt ./checkpoints/biasnet_llama3_8b \
  --prompt_file ./inputs/prompts.txt \
  --use_chat_template \
  --temperature 0 \
  --max_new_tokens 200 \
  --output_json ./outputs/llama3_8b.jsonl \
  --device_map auto
```

For Gemini through Vertex AI:

```bash
python inference_gemini.py \
  --model_name gemini-2.5-pro \
  --tokenizer_name google/gemma-3-1b-pt \
  --biasnet_ckpt ./checkpoints/biasnet_gemini \
  --prompt_file ./inputs/prompts.txt \
  --output_json ./outputs/gemini.jsonl \
  --max_new_tokens 100 \
  --overwrite
```

Both inference scripts accept user-supplied prompts and write records with `prompt` and `completion` fields. The Gemini path requests the next-token distribution at each generation step, applies BiasNet locally, and feeds the accepted prefix into the next request.

## Citation

```bibtex
@inproceedings{
  wang2026juli,
  title={{JULI}: Jailbreak Large Language Models by Self-Introspection},
  author={Jesson Wang and Zhanhao Hu and David Wagner},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=JDtIrWYB4o}
}
```

## License

JULI is licensed under the [Apache License 2.0](LICENSE). This core release does
not bundle benchmark data, evaluator implementations, model weights, or
generated results; see [THIRD_PARTY.md](THIRD_PARTY.md).
