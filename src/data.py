import random
from datasets import load_dataset


def load_ecthr(min_paragraphs=5):
    """Load ECtHR dataset and filter out documents with fewer than min_paragraphs."""
    dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
    dataset = dataset.filter(lambda x: len(x["text"]) >= min_paragraphs)
    return dataset


def mask_paragraphs(paragraphs, mask_ratio_min=0.2, mask_ratio_max=0.4):
    """Randomly drop 20–40% of paragraphs. Always keeps at least 1."""
    n = len(paragraphs)
    mask_ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    n_keep = max(1, int(n * (1 - mask_ratio)))
    indices = sorted(random.sample(range(n), n_keep))
    return [paragraphs[i] for i in indices]


def sample_few_shot(dataset_split, n_per_class, seed, num_classes=10):
    """
    Sample indices such that each class has at least n_per_class examples.
    Documents can satisfy multiple classes (multi-label). Deduplicates.
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

    return [dataset_split[i] for i in sorted(selected_set)]
