# doc-assistant-hf

A document intelligence tool built with Hugging Face Transformers and Gradio. Upload a PDF, DOCX or TXT file — or paste text directly — and get a bulleted summary, a prose summary, or an exact answer extracted verbatim from the document.

> **This is a learning and portfolio project, not a production service.** It was built to work through three things end to end on free-tier hardware: fine-tuning a seq2seq model with QLoRA, running extractive QA with custom span decoding, and — after the first approach failed — implementing a classical NLP ranking algorithm from scratch. The write-up below documents the reasoning and the dead ends as much as the result. Training was a single epoch with no ROUGE evaluation, and everything runs in one Colab notebook.

[`coded-test.ipynb`](coded-test.ipynb) covers the whole pipeline: loading the dataset, fine-tuning the summarizer with QLoRA, loading the QA model, parsing documents and launching the UI.

<a href="https://colab.research.google.com/github/pop123-ux/doc-assistant-hf/blob/main/coded-test.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## What it does

The Gradio app exposes three tabs, each accepting pasted text *or* a file upload:

| Tab | Output | Input limit |
| --- | --- | --- |
| **Document Summarizer (Extractive)** | *N* sentences selected verbatim from the document, ranked by TextRank | none |
| **Document Summarizer (Abstractive)** | A continuous prose summary (50–256 new tokens, 4-beam search) | 1,024 tokens |
| **Document QA System** | A span copied verbatim from the document, or `"No answer found."` | 1,024 tokens |

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

**Why TextRank for the extractive tab.** The tab originally generated an abstractive summary with BART and split it into bullets, which meant the bullet count was whatever BART happened to emit, and the bullets could still drift from the source. Scoring the *document's own* sentences instead makes the tab genuinely extractive: the requested number of bullets is guaranteed, the text is copied verbatim so it cannot hallucinate, and — because BART is never invoked — the tab has **no token limit at all** and handles arbitrarily long documents. It also needs no GPU.

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

Note the wide gap between the averaged `train_loss` (6.32) and the final `eval_loss` (1.57). Part of it is arithmetic — the average includes the very high early steps — but the per-step training loss still plateaus around 6, which is high. The label-masking issue described under [Limitations and scope](#limitations-and-scope) is the most likely contributor. **No ROUGE evaluation was run**, so treat this as a working, demonstrative pipeline rather than a benchmarked model.

The adapter is persisted three ways: pushed to the Hub via `trainer.push_to_hub()`, saved to Google Drive, and reloaded with `PeftModel.from_pretrained` so inference can resume without retraining.

## The extractive tab: what didn't work, and why

The first implementation — `process_summary_extractive`, still present in the notebook as a record of the attempt — was not extractive at all. It generated an abstractive summary with the fine-tuned BART, then split the *generated* text into sentences with NLTK and prefixed each with a bullet:

```python
summary_ids = model1.generate(
    **inputs,
    max_new_tokens=35 * bpn,
    min_new_tokens=14 * bpn,   # <- the problem
    ...
)
sentences = nltk.sent_tokenize(summary)
```

It behaved acceptably at three bullets and fell apart above that, in two distinct ways:

**It repeated itself and invented detail.** `min_new_tokens` is a hard ban on emitting the end-of-sequence token before the floor is reached. At six bullets that floor is 84 tokens, so once the model had said everything the source supported, it was *forbidden to stop* — and filled the remainder by repeating phrases or inventing content. Scaling the minimum length with the requested bullet count meant every extra bullet made this worse. The masked-`</s>` bug in the fine-tuning (see limitations) compounded it, since the adapter's stopping signal was already weak.

**It couldn't honour the requested count.** Nothing connected "sentences BART generated" to "sentences requested". NLTK split whatever came out, and any shortfall was padded with a literal placeholder string — `"- Key detail omitted by BART (expand input or adjust parameters)."` — which is a bug wearing a UI message as a disguise.

The deeper problem was conceptual: **the tab was named for a technique it wasn't using.** Extractive summarization selects sentences from the source; this generated new text and formatted it to look selected. That framing made the bullet count feel like a parameter to tune, when no generation parameter could have fixed it.

Replacing it with TextRank resolved both symptoms at once and made the name accurate. The lesson worth keeping: *when output quality degrades as you push a knob, check whether the knob is the right abstraction before tuning it.*

## How the extractive summarizer works

`textrank_summarization` implements TextRank — PageRank run over a graph of sentence similarities:

1. **Segment and clean.** Split the document into sentences with NLTK, then lowercase and strip punctuation to build a parallel "cleaned" copy. The original sentences are kept intact for output; only the cleaned copies are scored.
2. **Short-circuit.** If the document has no more sentences than requested, return them all — no ranking needed.
3. **Vectorize.** TF-IDF over the cleaned sentences, with English stop words removed so common function words don't dominate the similarity scores.
4. **Build the graph.** Cosine similarity between every sentence pair becomes the adjacency matrix, with the diagonal zeroed so a sentence cannot vote for itself.
5. **Rank.** `nx.pagerank(alpha=0.85)` scores each sentence by how central it is to the document — sentences similar to many *other* important sentences rank highest.
6. **Emit.** Take the top *N*, re-sort by original position so the summary reads in document order, and format as bullets.

Because the output is copied verbatim from the source, this tab cannot hallucinate, always returns exactly the requested number of bullets, and has no input-length ceiling.

## How QA answers are picked

Rather than calling the `question-answering` pipeline, the notebook runs the model directly and decodes spans itself:

1. Take the top 20 start logits and top 20 end logits.
2. Reject any pair where `end < start`, or where the span exceeds 30 tokens.
3. Score each surviving pair as `start_logit + end_logit` and decode the highest.
4. Return `"No answer found."` if nothing valid survives.

## Limitations and scope

This is a learning project, so some of these are bugs worth fixing and others are deliberate scope choices. Both are listed, labelled honestly.

**Known bugs**

* **The end-of-sequence token is masked out of the training loss.** `tokenizer1.pad_token` is set to `tokenizer1.eos_token`, which makes `pad_token_id == eos_token_id == 2` (the trainer confirms this: *"Updated tokens: {'pad_token_id': 2}"*). The tokenize function then rewrites every label token equal to `pad_token_id` to `-100` — which masks the real `</s>` terminating each target summary, not just padding. The model consequently gets no gradient signal for *when to stop*, and generation leans on `max_new_tokens` / `early_stopping` instead. The manual rewrite is also redundant, since `DataCollatorForSeq2Seq(label_pad_token_id=-100)` already masks padding.
* **QA input is truncated to 1,024 tokens** even though Longformer supports 4,096 — raising `max_length` in `process_qa` would use the full window.
* **`.doc` is matched in the extension check but not supported** — `python-docx` reads only `.docx`. The upload widget filters to `.pdf`, `.docx` and `.txt`, so the branch is unreachable.
* **The two summarizer paths use different bullet characters** — the short-document early return emits `-`, the ranked path emits `•`.

**Deliberate scope choices**

* **One epoch, no ROUGE.** The fine-tune demonstrates the QLoRA workflow rather than chasing a benchmark. `eval_loss` 1.57 is reported instead of a task metric, so the summarizer should be read as a working pipeline, not a tuned model.
* **`process_summary_extractive` is kept but unused.** It is the superseded first attempt, left in place as documentation of the iteration described above. Nothing calls it — the tab is wired to `textrank_summarization`.
* **No chunk-and-merge on the abstractive tab.** Documents past 1,024 tokens are truncated, so only the opening section is summarized. The extractive tab is unaffected since it never calls BART.
* **SAMSum is conversational**, so the adapter pulls the summarizer toward chat and meeting transcripts. On formal reports, stock BART-large-CNN may well read better than the fine-tuned version — the dataset was chosen for its size and cleanliness on free-tier hardware, not for domain fit.
* **Notebook-only, and GPU-bound.** There is no `app.py` or `requirements.txt`, inference assumes a CUDA runtime, and `adapter_path` points at Google Drive — so re-running requires either that Drive mount or switching the path to the published Hub adapter.

## Running it

Open the notebook in Colab with a GPU runtime and run the cells in order.

```bash
pip install -q pypdf python-docx gradio regex nltk
pip install -q -U transformers peft
pip install -q -U "bitsandbytes>=0.46.1"
# datasets, accelerate, torch, scikit-learn and networkx are preinstalled in Colab
```

`!hf auth login` is only needed for the cell that pushes the adapter to the Hub. To skip training entirely, run the model-loading cells and point `adapter_path` at the published adapter instead of Google Drive.

The final cell calls `demo.launch(share=True)`, which prints a temporary public URL alongside the local one.

## Tech stack

* **Framework**: Python 3.12, Gradio (Blocks API)
* **AI/ML**: Hugging Face Transformers, PEFT (LoRA), bitsandbytes (4-bit NF4), Datasets, PyTorch, Accelerate
* **Classical NLP**: scikit-learn (TF-IDF, cosine similarity), NetworkX (PageRank)
* **Parsers**: pypdf, python-docx; NLTK for sentence segmentation
* **Environment**: Google Colab (T4), Google Drive for adapter storage

## Roadmap

* Stop masking `</s>` in the labels, then retrain and compare.
* Chunk-and-merge summarization for the abstractive tab.
* Raise the QA window from 1,024 to Longformer's full 4,096.
* ROUGE evaluation against the SAMSum test split, and a longer training run.
* Export the app as a standalone script / Hugging Face Space.

## License

MIT — see [LICENSE](LICENSE).

## 🔗 More

- Author: [@pop123-ux](https://github.com/pop123-ux)
- Medium write-ups: [medium.com/@Pop123](https://medium.com/@Pop123)
