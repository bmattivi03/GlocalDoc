import re
import random
from datasets import load_dataset


# ECtHR Article A labels — used by ProtoClassifier (P-E-02 / P-J-03) to
# initialize label prototypes from label-text encodings instead of random.
# Order matches `coastalcph/lex_glue / ecthr_a` train["labels"] indices.
ECTHR_LABEL_TEXTS: list[str] = [
    "Article 2: right to life",
    "Article 3: prohibition of torture and inhuman or degrading treatment",
    "Article 5: right to liberty and security",
    "Article 6: right to a fair trial",
    "Article 8: right to respect for private and family life",
    "Article 9: freedom of thought, conscience and religion",
    "Article 10: freedom of expression",
    "Article 11: freedom of assembly and association",
    "Article 14: prohibition of discrimination",
    "Article 1 of Protocol 1: protection of property",
]


def load_ecthr(min_paragraphs: int = 5, max_paragraphs: int = 50):
    """Load ECtHR and filter to documents with 5–50 paragraphs."""
    dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
    return dataset.filter(lambda x: min_paragraphs <= len(x["text"]) <= max_paragraphs)


def mask_text(
    paragraph: str,
    sent_dropout: float = 0.20,
    span_rate: float = 0.15,
    span_len_min: int = 3,
    span_len_max: int = 5,
) -> str:
    """
    Two-stage masking applied to a single paragraph string.

    Stage 1 — sentence dropout (20%): split on sentence boundaries, drop each
    sentence independently with probability sent_dropout. If all sentences are
    dropped, return original paragraph unchanged.

    Stage 2 — span masking (15%, spans 3–5 words): sample word-level spans until
    ~15% of words are covered. Replace each span with a single '<mask>' token.
    Consecutive '<mask>' tokens are collapsed to one.

    '<mask>' is the RoBERTa mask token and is handled correctly by
    RobertaTokenizerFast without special treatment.
    """
    # Stage 1: sentence dropout
    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    kept = [s for s in sentences if random.random() > sent_dropout]
    if not kept:
        kept = sentences
    text = ' '.join(kept)

    # Stage 2: word-level span masking
    words = text.split()
    if not words:
        return paragraph

    n_to_mask = max(1, round(len(words) * span_rate))
    masked_positions: set = set()
    attempts = 0
    while len(masked_positions) < n_to_mask and attempts < len(words) * 3:
        attempts += 1
        start = random.randint(0, len(words) - 1)
        span = random.randint(span_len_min, span_len_max)
        for j in range(start, min(start + span, len(words))):
            masked_positions.add(j)

    result = []
    prev_mask = False
    for i, w in enumerate(words):
        if i in masked_positions:
            if not prev_mask:
                result.append('<mask>')
            prev_mask = True
        else:
            result.append(w)
            prev_mask = False

    return ' '.join(result)


def get_paragraph_mask(n_paragraphs: int, dropout_rate: float = 0.30) -> list:
    """Return sorted list of kept paragraph indices after dropping dropout_rate fraction."""
    n_keep = max(1, round(n_paragraphs * (1 - dropout_rate)))
    return sorted(random.sample(range(n_paragraphs), n_keep))


def mask_paragraphs(paragraphs: list, mask_ratio_min: float = 0.2, mask_ratio_max: float = 0.4):
    """Legacy API: paragraph-level dropout. Returns (kept_paragraphs, kept_indices)."""
    n = len(paragraphs)
    mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    n_keep = max(1, int(n * (1 - mask_ratio)))
    indices = sorted(random.sample(range(n), n_keep))
    return [paragraphs[i] for i in indices], indices


def sample_few_shot(dataset_split, n_per_class: int, seed: int, num_classes: int = 10):
    """
    Multi-label aware few-shot sampling. Ensures n_per_class examples per label.
    Deduplicates documents that satisfy multiple classes.
    """
    rng = random.Random(seed)
    per_class = [[] for _ in range(num_classes)]
    indices = list(range(len(dataset_split)))
    rng.shuffle(indices)
    selected_set = set()

    for idx in indices:
        labels = dataset_split[idx]["labels"]
        for label in labels:
            if len(per_class[label]) < n_per_class:
                per_class[label].append(idx)
                selected_set.add(idx)
        if all(len(pc) >= n_per_class for pc in per_class):
            break

    # Surface rare-class shortfalls — label 5 in ECtHR has only ~41 train examples,
    # so n_per_class=100 will silently undersample without this warning.
    shortfall = [(c, len(pc)) for c, pc in enumerate(per_class) if len(pc) < n_per_class]
    if shortfall:
        details = ", ".join(f"class {c}: {got}/{n_per_class}" for c, got in shortfall)
        print(f"[sample_few_shot] WARN seed={seed}: undersampled — {details}")

    return [dataset_split[i] for i in sorted(selected_set)]
