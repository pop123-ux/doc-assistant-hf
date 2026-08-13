# doc-assistant-hf

An AI-powered document intelligence tool built with Hugging Face Transformers and Gradio. Upload a PDF, DOCX or TXT file — or paste text directly — and get a bulleted summary, a prose summary, or an exact answer extracted verbatim from the document.

Everything lives in [`coded.ipynb`](coded.ipynb), a Colab notebook covering the whole pipeline: loading the dataset, fine-tuning the summarizer with QLoRA, loading the QA model, parsing documents and launching the UI.

<a href="https://colab.research.google.com/github/pop123-ux/doc-assistant-hf/blob/main/coded.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## What it does

The Gradio app exposes three tabs, each accepting pasted text *or* a file upload:

| Tab | Output |
| --- | --- |
| **Document Summarizer (Extractive)** | A 3–6 bullet summary, sentence-split with NLTK |
| **Document Summarizer (Abstractive)** | A continuous prose summary (50–256 new tokens, 4-beam search) |
| **Document QA System** | A span copied verbatim from the document, or `"No answer found."` |

## Models

| Task | Base model | How it's used |
| --- | --- | --- |
| Summarization | [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) | Loaded in 4-bit and **fine-tuned with LoRA** on SAMSum. Adapter: [`pop123ux/lora_bart_samsum_results`](https://huggingface.co/pop123ux/lora_bart_samsum_results) |
| Question answering | [`allenai/longformer-large-4096-finetuned-triviaqa`](https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa) | Used as-is for extractive QA, with custom span decoding over the start/end logits |

## Design decisions

**Why QLoRA instead of a full fine-tune.** BART-large-CNN is 407M parameters — full fine-tuning it needs optimizer state that won't fit a free Colab T4. Loading the base model in 4-bit NF4 with double quantization and attaching a rank-8 LoRA adapter brings the trainable count down to **1,179,648 of 407,470,080 parameters (0.29%)**, which fits comfortably alongside gradient checkpointing.

**Why extractive QA rather than a generative model.** Answers are decoded as spans of the source document, so the system physically cannot invent facts. For a document assistant, a wrong-but-grounded span is a better failure mode than a fluent hallucination.

**Why Longformer for QA.** Standard BERT-family QA models cap out at 512 tokens. Longformer's sliding-window attention supports 4,096, which matters when the answer sits deep in a long document. The notebook does not pass a `global_attention_mask`, and that is deliberate — `LongformerForQuestionAnswering` auto-computes global attention over the question tokens when none is supplied, which is exactly the pattern the TriviaQA checkpoint was trained with.

**Why hand-rolled span decoding instead of the `question-answering` pipeline.** Running the model directly and scoring candidate spans makes the answer-selection policy explicit and tunable (see below), rather than hidden behind pipeline defaults.

**Why SAMSum.** It is a clean, well-known 14.7k-example dialogue-summarization corpus — small enough to fine-tune on in one epoch on free-tier hardware while still demonstrating the full QLoRA workflow end to end.

## Fine-tuning the summarizer

Fine-tuned on [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) — 14,731 train / 818 validation / 819 test dialogue–summary pairs.

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

**Training setup**

| Setting | Value |
| --- | --- |
| Batch size | 4 × 4 gradient accumulation steps (effective 16) |
| Learning rate | 5e-5, linear decay, weight decay 0.01 |
| Epochs | 1 (921 optimizer steps) |
| Precision | fp16, gradient checkpointing enabled |
| Max input / target length | 512 / 128 tokens, dynamically padded by `DataCollatorForSeq2Seq` |
| Label padding | `-100`, so padding is ignored in the loss |

**Results from the run recorded in the notebook**

| Metric | Value |
| --- | --- |
| Train runtime | 1,435 s (~24 min) on a T4 |
| `train_loss` (mean over the epoch) | 6.32 |
| `eval_loss` (end of epoch) | 1.57 |
| Loss trend | ~8.9 (first 10 steps) → ~6.0 (last 50 steps), min 4.14 |

Three `nan` gradient norms appear at steps 1, 7 and 8 — ordinary fp16 loss-scaler warmup, after which the scaler settles and every subsequent step reports a finite norm.

Note the wide gap between the averaged `train_loss` (6.32) and the final `eval_loss` (1.57). Part of it is arithmetic — the average includes the very high early steps — but the per-step training loss still plateaus around 6, which is high. The label-masking issue described under [Known limitations](#known-limitations) is the most likely contributor. **No ROUGE evaluation was run**, so treat this as a working, demonstrative pipeline rather than a benchmarked model.

The adapter is persisted three ways: pushed to the Hub via `trainer.push_to_hub()`, saved to Google Drive, and reloaded with `PeftModel.from_pretrained` so inference can resume without retraining.

## How QA answers are picked

Rather than calling the `question-answering` pipeline, the notebook runs the model directly and decodes spans itself:

1. Take the top 20 start logits and top 20 end logits.
2. Reject any pair where `end < start`, or where the span exceeds 30 tokens.
3. Score each surviving pair as `start_logit + end_logit` and decode the highest.
4. Return `"No answer found."` if nothing valid survives.

## Known limitations

These are real constraints of the current notebook, documented rather than hidden.

* **The UI tab labels still quote the old 2,048-token limit.** The summarizer functions now clamp correctly to `min(tokenizer1.model_max_length, 1024)`, matching BART-large-CNN's 1,024 learned positional embeddings — but both tab titles still read *"LIMIT: 2048 tokens"*. Cosmetic only; the enforced cap is the correct one.
* **Bullet counts above 3 degrade quality.** `process_summary_extractive` scales generation length by the requested bullet count (`min_new_tokens=14*bpn`, `max_new_tokens=35*bpn`). At `bpn=6` that forces at least 84 new tokens, which is far longer than the 1–2 sentence summaries the SAMSum adapter was trained to produce — so the model pads the gap with repetition or invented detail. The count itself is also not guaranteed: NLTK splits whatever BART generated, and any shortfall is filled with the literal placeholder *"Key detail omitted by BART…"*. See [Tuning the extractive tab](#tuning-the-extractive-tab).
* **The lower bound still forbids 1–2 bullets.** `assert bpn > 2` means the control accepts 3–6 only, and an out-of-range value raises an `AssertionError` that surfaces as a Gradio error rather than a friendly message.
* **The end-of-sequence token is masked out of the training loss.** `tokenizer1.pad_token` is set to `tokenizer1.eos_token`, which makes `pad_token_id == eos_token_id == 2` (the trainer confirms this: *"Updated tokens: {'pad_token_id': 2}"*). The tokenize function then rewrites every label token equal to `pad_token_id` to `-100` — which masks the real `</s>` terminating each target summary, not just padding. The model consequently gets no gradient signal for *when to stop*, and generation leans on `max_new_tokens` / `early_stopping` instead. The manual rewrite is also redundant, since `DataCollatorForSeq2Seq(label_pad_token_id=-100)` already masks padding.
* **The "Extractive" tab is not extractive.** It runs the same abstractive BART generation as the other tab, then splits the *generated* text into sentences with NLTK and prefixes each with `- `. True extractive summarization selects sentences from the source document. The output is better described as bullet-formatted abstractive summary — and because the bullets come from generated text, they can still depart from the source. This is the root cause of the bullet-count problems above.
* **QA input is truncated to 1,024 tokens** even though Longformer supports 4,096 — raising `max_length` in `process_qa` would use the full window.
* **No chunk-and-merge pass.** Documents longer than the input cap are silently truncated, so only the opening section is summarized or searched.
* **SAMSum is conversational**, so the adapter pulls the summarizer toward chat and meeting transcripts. On formal reports, stock BART-large-CNN may well read better than the fine-tuned version.
* **`.doc` is matched in the extension check but not supported** — `python-docx` reads only `.docx`. In practice the upload widget filters to `.pdf`, `.docx` and `.txt`, so the branch is unreachable.
* **Notebook-only, and GPU-bound.** There is no `app.py` or `requirements.txt`, inference assumes a CUDA runtime, and `adapter_path` points at Google Drive — so re-running requires either that Drive mount or switching the path to the published Hub adapter.

## Tuning the extractive tab

Asking for more bullets tends to produce repetition, invented detail, or fewer bullets than requested. The cause is structural rather than a parameter that needs nudging: the tab generates **one** abstractive summary whose length is scaled by the bullet count, then slices it into sentences. Nothing ties the number of sentences BART produces to the number requested.

Two levers matter most:

* **`min_new_tokens=14*bpn` is the main driver of hallucination.** A minimum length is a hard ban on emitting `</s>` before the floor is reached, so at `bpn=6` the model is forbidden to stop for 84 tokens even when it has said everything the source supports. It fills the remainder by repeating or inventing. Prefer raising `length_penalty` (which biases beam *scoring* toward longer sequences) over raising `min_new_tokens` (which forbids stopping outright). The masked-`</s>` training issue above compounds this, because the adapter's stopping signal is already weak.
* **SAMSum summaries are 1–2 sentences.** The adapter was fine-tuned toward short, conversational output, so demanding six sentences works against the fine-tune. Stock `facebook/bart-large-cnn` produces longer multi-sentence summaries and behaves better at high bullet counts.

Three ways to get a reliable *N*, in increasing order of fidelity to the tab's name:

1. **Chunk-and-summarize** — split the source into `bpn` roughly equal chunks and generate one short summary per chunk (`max_new_tokens≈40`, no `min_new_tokens`). Exactly *N* bullets, each grounded in a different part of the document, and it reuses the existing model.
2. **Over-generate and select** — generate a single longer summary, sentence-split it, then keep the `bpn` best sentences (longest, or highest TF-IDF overlap with the source) instead of the first `bpn`. Removes the placeholder filler.
3. **Genuine extraction** — score the *source* sentences and select the top `bpn` (TF-IDF centroid, TextRank via `sumy`, or embedding centrality with `sentence-transformers`). Guarantees the exact count, cannot hallucinate, needs no generation at all, and makes the tab's name accurate.

Option 3 is the recommended fix, with option 1 as the smaller change. If the current approach is kept, drop `min_new_tokens`, keep `no_repeat_ngram_size=3`, and leave `repetition_penalty` near 1.1 — pushing it much past ~1.2 with beam search tends to cause degeneration rather than fix it.

## Running it

Open the notebook in Colab with a GPU runtime and run the cells in order.

```bash
pip install -q pypdf python-docx gradio nltk regex
pip install -U "bitsandbytes>=0.46.1"
# transformers, peft, datasets, accelerate and torch are preinstalled in Colab
```

`!hf auth login` is only needed for the cell that pushes the adapter to the Hub. To skip training entirely, run the model-loading cells and point `adapter_path` at the published adapter instead of Google Drive.

The final cell calls `demo.launch(share=True)`, which prints a temporary public URL alongside the local one.

## Tech stack

* **Framework**: Python 3.12, Gradio (Blocks API)
* **AI/ML**: Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit NF4), Datasets, PyTorch, Accelerate
* **Parsers**: pypdf, python-docx; NLTK for sentence segmentation
* **Environment**: Google Colab (T4), Google Drive for adapter storage

## Roadmap

* Correct the UI tab labels to quote the enforced 1,024-token limit.
* Stop masking `</s>` in the labels, then retrain and compare.
* True extractive summarization (score and select source sentences) rather than bulleting generated text.
* Chunk-and-merge summarization for long documents.
* Raise the QA window from 1,024 to Longformer's full 4,096.
* ROUGE evaluation against the SAMSum test split, and a longer training run.
* Export the app as a standalone script / Hugging Face Space.

## License

MIT — see [LICENSE](LICENSE).

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
