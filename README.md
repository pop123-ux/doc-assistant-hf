# doc-assistant-hf

An AI-powered document intelligence tool built with Hugging Face Transformers and Gradio. Upload a PDF, DOCX or TXT file — or paste text directly — and get either a generated summary or an exact, extracted answer to a question about the document.

Everything lives in [`coded.ipynb`](coded.ipynb), a Colab notebook that covers the whole pipeline: loading data, fine-tuning the summarizer with LoRA, loading the QA model, parsing documents and launching the UI.

<a href="https://colab.research.google.com/github/pop123-ux/doc-assistant-hf/blob/main/coded.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Models

| Task | Base model | How it's used |
| --- | --- | --- |
| Summarization | [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) | Loaded in 4-bit and **fine-tuned with LoRA** on the SAMSum dialogue-summarization dataset. Adapter: [`pop123ux/lora_bart_samsum_results`](https://huggingface.co/pop123ux/lora_bart_samsum_results) |
| Question answering | [`allenai/longformer-large-4096-finetuned-triviaqa`](https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa) | Used as-is for extractive QA, with custom span decoding over the start/end logits |

Longformer's 4,096-token attention window replaces the shorter-context QA setup used earlier, and the summarizer is no longer stock BART — it carries a LoRA adapter trained in this notebook.

## Features

* **Multi-format extraction** — reads `.pdf` (pypdf), `.docx` (python-docx) and `.txt`, with per-file error handling that reports failures in the UI instead of crashing.
* **Extractive document QA** — answers are spans copied verbatim from your document, not generated text, so they can't hallucinate.
* **Fine-tuned summarization** — a QLoRA-trained adapter on top of BART-large-CNN, with beam search and repetition control at generation time.
* **Two-tab Gradio UI** — separate Summarizer and QA tabs, each accepting pasted text *or* a file upload, launched with a public share link.
* **Trains on a free T4** — 4-bit NF4 quantization plus LoRA keeps the whole fine-tune inside Colab's free tier.

## Fine-tuning the summarizer

The notebook fine-tunes BART-large-CNN on [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) (14,731 train / 818 validation / 819 test dialogue–summary pairs).

**Quantization + adapter**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type=TaskType.SEQ_2_SEQ_LM)
```

With gradient checkpointing and `prepare_model_for_kbit_training`, this trains **1,179,648 of 407,470,080 parameters — 0.29%**.

**Training setup**

| Setting | Value |
| --- | --- |
| Batch size | 4 × 4 gradient accumulation steps (effective 16) |
| Learning rate | 5e-5, weight decay 0.01 |
| Epochs | 1 (921 steps) |
| Precision | fp16 |
| Max input / target length | 512 / 128 tokens, dynamically padded by `DataCollatorForSeq2Seq` |
| Label padding | `-100`, so padding is ignored in the loss |

**Result of the run in the notebook:** 1,435 s (~24 min) on a T4, final `train_loss` 6.32, `eval_loss` 1.57. Trained for a single epoch as a demonstration of the QLoRA workflow — no ROUGE evaluation was run, so treat the summarizer as a working pipeline rather than a benchmarked model.

The adapter is saved three ways: pushed to the Hub with `trainer.push_to_hub()`, written to Google Drive, and reloaded with `PeftModel.from_pretrained` so inference can resume without retraining.

## How QA answers are picked

Rather than calling the `question-answering` pipeline, the notebook runs the model directly and decodes spans itself:

1. Take the top 20 start logits and top 20 end logits.
2. Reject any pair where `end < start` or the span is longer than 30 tokens.
3. Score each remaining pair as `start_logit + end_logit` and decode the best one.
4. Return `"No answer found."` if nothing valid survives.

## Running it

Open the notebook in Colab with a GPU runtime and run the cells in order. Dependencies:

```bash
pip install -q pypdf python-docx gradio
pip install -U "bitsandbytes>=0.46.1"
# transformers, peft, datasets, accelerate and torch are preinstalled in Colab
```

`!hf auth login` is only needed for the cell that pushes the adapter to the Hub. To skip training entirely, run the model-loading cells and point `adapter_path` at the published adapter instead of Google Drive.

The last cell calls `demo.launch(share=True)`, which prints a temporary public URL alongside the local one.

## Tech stack

* **Framework**: Python 3.12, Gradio (Blocks API)
* **AI/ML**: Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit NF4), Datasets, PyTorch, Accelerate
* **Parsers**: pypdf, python-docx
* **Environment**: Google Colab (T4), Google Drive for adapter storage

## Known limitations

* **QA input is truncated to 1,024 tokens**, even though Longformer supports 4,096 — raising `max_length` in `process_qa` would use the full window.
* **Summaries are capped by BART's 1,024-token input limit** and there's no chunk-and-merge pass, so only the beginning of a long document is summarized.
* **SAMSum is conversational**, so the adapter pushes the summarizer toward chat and meeting transcripts; on formal reports, stock BART-large-CNN may read better.
* **`.doc` is matched in the extension check but not actually supported** — python-docx reads only `.docx`. The upload widget filters to `.pdf`, `.docx` and `.txt` anyway.
* **Notebook-only** — there's no `app.py` or `requirements.txt` yet, and a GPU runtime is assumed.

## Roadmap

* Toggle between abstractive (rewriting) and extractive (bullet-point) summarization.
* Chunked summarization for documents past the 1,024-token limit.
* ROUGE evaluation against the SAMSum test split, and a longer training run.
* Export the app as a standalone script / Hugging Face Space.

## License

MIT — see [LICENSE](LICENSE).

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
