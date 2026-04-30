"""Generate GlocalIB NLP project report as a .docx file."""
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

doc = Document()

# ── Page margins ──────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── Style helpers ─────────────────────────────────────────────────────────────
def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p

def h2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)
    return p

def h3(text):
    return doc.add_heading(text, level=3)

def body(text, bold_prefix=None):
    p = doc.add_paragraph()
    if bold_prefix:
        run = p.add_run(bold_prefix + " ")
        run.bold = True
    p.add_run(text)
    return p

def bullet(text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = p.add_run(bold_prefix + ": ")
        run.bold = True
    p.add_run(text)
    return p

def ref_entry(authors, year, title, venue, extra=""):
    p = doc.add_paragraph(style="List Number")
    r = p.add_run(f"{authors} ({year}). ")
    r.bold = True
    p.add_run(f"{title}. ")
    r2 = p.add_run(f"{venue}.")
    r2.italic = True
    if extra:
        p.add_run(f" {extra}")
    return p

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title_para.add_run(
    "GlocalIB for Low-Resource Legal NLP:\n"
    "IB-Regularized Pre-training with Learnable Compression\n"
    "for Few-Shot Document Classification"
)
run.bold = True
run.font.size = Pt(18)
run.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

doc.add_paragraph()
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(
    "MSc Computing for Data Science\n"
    "Free University of Bozen-Bolzano\n"
    "April 2026"
).font.size = Pt(12)

doc.add_page_break()

# ─────────────────────────────────────────────────────────────────────────────
# 1. INTRODUCTION / MOTIVATION
# ─────────────────────────────────────────────────────────────────────────────
h1("1. Introduction and Motivation")

body(
    "Supervised learning with pre-trained language models has transformed natural language "
    "processing over the last decade. Yet one of its most persistent failure modes remains "
    "largely unsolved: what happens when labeled data is scarce? In high-stakes domains — "
    "legal, clinical, administrative — annotations require domain expertise, take weeks to "
    "produce, and cost thousands of euros per thousand examples. Fine-tuning a 110-million "
    "parameter model on ten or fifty labeled documents reliably produces brittle classifiers "
    "that memorize surface patterns rather than learning the underlying concepts."
)

body(
    "This project addresses that failure mode through a different kind of pre-training. "
    "Instead of asking a language model to predict masked tokens (MLM), we ask it to "
    "reconstruct the global meaning of a document from an incomplete view of it — with an "
    "explicit information-theoretic constraint that forces it to throw away irrelevant noise. "
    "The intuition is simple: a representation that has already learned to discard surface "
    "details — because it was trained to do so before ever seeing a label — should generalize "
    "better when only a handful of labeled examples are available."
)

body(
    "The mechanism we adapt is the Global-Local Information Bottleneck (GlocalIB), originally "
    "proposed by Yang et al. (2025) for time series imputation. GlocalIB uses a teacher-student "
    "setup: a teacher encoder reads a complete input and produces a deterministic target "
    "representation; a student encoder reads a masked version of the same input and is trained "
    "to reconstruct the teacher's representation. The student's latent space is constrained by "
    "an Information Bottleneck (IB) penalty — a KL divergence term that penalizes representations "
    "for carrying more information about the input than is strictly necessary to align with the "
    "teacher."
)

body(
    "The GlocalIB architecture maps naturally onto legal documents: the teacher reads all "
    "paragraphs of a case; the student reads the same case with 20–40% of paragraphs randomly "
    "removed; the student must reconstruct the global meaning from partial evidence. This "
    "mirrors the human legal reasoning task — a lawyer reading an incomplete brief must still "
    "infer the likely outcome from the available facts. Crucially, the IB compression penalty "
    "explicitly discourages the model from memorizing surface patterns it will not have access "
    "to at inference time."
)

body(
    "We apply this framework to the European Court of Human Rights (ECtHR) dataset, a "
    "multi-label benchmark with 10 article violation classes, severe class imbalance (Article 3 "
    "appears 4,704 times; Article 5 appears 41 times), and a naturally low-resource fine-tuning "
    "regime due to the expert knowledge required for annotation. The ECtHR dataset is "
    "particularly well-suited to our method: cases consist of pre-split factual paragraphs "
    "that can be directly masked, and the paragraph-structured format allows our "
    "chunk-and-pool encoder to handle documents of arbitrary length without requiring a "
    "long-context model."
)

body(
    "The key contribution is a systematic empirical comparison under controlled conditions: "
    "three pre-training objectives (standard MLM, GlocalIB with compression disabled, "
    "GlocalIB with learnable compression) applied to the same base model, same unlabeled "
    "data, and same downstream fine-tuning procedure — varying only the number of labeled "
    "examples (N ∈ {10, 50, 100}). This design isolates the effect of the training objective "
    "from all other confounds and tests whether IB-regularized pre-training improves "
    "few-shot macro-F1 on a realistic low-resource legal classification task."
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBLEM STATEMENT
# ─────────────────────────────────────────────────────────────────────────────
h1("2. Problem Statement")

body(
    "Fine-tuning large pre-trained language models on small labeled datasets produces "
    "classifiers that overfit to idiosyncratic surface patterns in the training examples. "
    "In legal NLP, this problem is acute: the labeled pool is small by necessity, the "
    "documents are long and structurally complex, and the class distribution is severely "
    "imbalanced. We identify four specific sub-problems that motivate the design of "
    "our approach:"
)

bullet(
    "Labeling cost and scarcity. Legal document annotation requires qualified legal experts "
    "who must read lengthy case records and reason about multi-label article violations. "
    "Obtaining even 100 labeled examples per class is a significant institutional undertaking. "
    "Any practical system must operate in the N = 10–100 regime.",
    bold_prefix="P1"
)

bullet(
    "Surface-pattern memorization. When a language model is fine-tuned on ten examples, "
    "it will latch onto any statistical regularity in the training set — specific legal "
    "phrases, article numbers mentioned explicitly, or common n-grams in the training "
    "split — rather than learning the abstract legal concept being judged. This produces "
    "classifiers that fail catastrophically on the test set despite perfect training accuracy.",
    bold_prefix="P2"
)

bullet(
    "Missing IB constraint in existing SSL methods. Contrastive methods such as SimCSE "
    "(Gao et al., 2021) and DeCLUTR (Giorgi et al., 2021) improve sentence and document "
    "representations but do not explicitly constrain the amount of information retained in "
    "the latent space. Without a compression objective, the encoder can still memorize "
    "arbitrary input features that are irrelevant to the downstream task.",
    bold_prefix="P3"
)

bullet(
    "Document length and pooling. ECtHR cases average 23 paragraphs (max: 558). Standard "
    "transformer encoders are limited to 512 tokens, preventing direct encoding of full "
    "documents. Existing approaches either truncate or require specialized long-context "
    "models. A practical encoder for legal documents must handle arbitrary-length inputs "
    "with a standard encoder backbone.",
    bold_prefix="P4"
)

body(
    "These four problems jointly motivate an approach that: (a) pre-trains on unlabeled "
    "legal documents to reduce dependence on labeled data, (b) applies an explicit "
    "information bottleneck to force compression of irrelevant features during pre-training, "
    "(c) uses a global-local teacher-student structure to make the compression task "
    "well-defined and non-trivial, and (d) encodes documents via a paragraph-level "
    "chunk-and-pool strategy compatible with standard 512-token encoders."
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. RESEARCH QUESTION
# ─────────────────────────────────────────────────────────────────────────────
h1("3. Research Question")

rq_para = doc.add_paragraph()
rq_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
rq_run = rq_para.add_run(
    "Does a GlocalIB-style information bottleneck objective with a learnable β\n"
    "during domain pre-training improve few-shot document classification\n"
    "compared to standard MLM pre-training, under identical conditions?"
)
rq_run.bold = True
rq_run.italic = True
rq_run.font.size = Pt(12)

doc.add_paragraph()

body(
    "This research question is deliberately narrow and falsifiable. 'Identical conditions' "
    "means the same base model (RoBERTa-base), the same unlabeled pre-training corpus "
    "(ECtHR training split), the same labeled fine-tuning data (same N examples sampled "
    "with the same seeds), and the same evaluation protocol (macro-F1 on the ECtHR test split). "
    "The only variable is the pre-training objective."
)

body(
    "The question separates into two sub-questions addressed by the ablation design:"
)

bullet(
    "RQ1 (IB vs. MLM): Does GlocalIB full (with compression) outperform standard MLM pre-training? "
    "This tests whether the global-local teacher-student objective — with IB compression — "
    "produces better few-shot representations than token-level masked language modeling.",
    bold_prefix="RQ1"
)

bullet(
    "RQ2 (compression is the cause): Does GlocalIB full outperform GlocalIB β=0 (same "
    "architecture, compression disabled)? This tests whether the IB penalty is doing real "
    "work, or whether the improvement is simply from the teacher-student architecture itself.",
    bold_prefix="RQ2"
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPOTHESIS
# ─────────────────────────────────────────────────────────────────────────────
h1("4. Hypothesis")

body(
    "A RoBERTa encoder trained with GlocalIB on unlabeled ECtHR documents — where compression "
    "strength β is learned automatically with a lower-bound constraint (β ≥ 0.01) rather than "
    "fixed — produces more compressed and more semantically stable representations than one "
    "trained with MLM on the same data. This results in higher and more stable macro-F1 "
    "classification accuracy at low label counts (N ≤ 100) on the ECtHR test set."
)

body("The hypothesis decomposes into three testable predictions:")

bullet(
    "GlocalIB full > MLM baseline in macro-F1 at N = 10. The primary prediction. At the "
    "most extreme data scarcity regime, IB-regularized representations should exhibit the "
    "largest advantage because they have discarded the most noise during pre-training.",
    bold_prefix="H1"
)

bullet(
    "GlocalIB full > GlocalIB β=0 in macro-F1 at N = 10. The ablation prediction. The "
    "compression term is the active ingredient: disabling it (while keeping the same "
    "teacher-student architecture) should reduce performance toward or below the MLM baseline.",
    bold_prefix="H2"
)

bullet(
    "Variance reduction. GlocalIB full should exhibit lower standard deviation across seeds "
    "than the MLM baseline at N = 10. More compressed representations produce a smoother, "
    "more predictable optimization landscape for the linear classification head.",
    bold_prefix="H3"
)

body(
    "Falsification criteria: If GlocalIB full does not outperform the MLM baseline at N = 10 "
    "(H1 fails), the primary hypothesis is rejected. If GlocalIB full does not outperform "
    "GlocalIB β=0 (H2 fails), the improvement — if any — is attributable to the architecture "
    "rather than the IB compression, and the claim must be weakened to 'global-local "
    "pre-training improves few-shot legal NLP' rather than 'IB compression improves few-shot "
    "legal NLP'. Both outcomes are scientifically meaningful and publishable."
)

body(
    "Beta trajectory diagnostic: If the learned β collapses to its lower bound (0.01) "
    "throughout pre-training, the IB term is not doing real work even when nominally enabled. "
    "This will be reported explicitly and treated as a failure of H2, triggering investigation "
    "of the compression loss weight or the architecture design rather than silent acceptance "
    "of a false positive."
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. RELATED WORK
# ─────────────────────────────────────────────────────────────────────────────
h1("5. Related Work")

body(
    "Our work sits at the intersection of four research threads: Information Bottleneck theory "
    "and its variational extensions, self-supervised representation learning for NLP, "
    "few-shot and low-resource text classification, and legal NLP benchmarks. We review "
    "each in turn and situate our specific contributions within them."
)

# 5.1
h2("5.1 Information Bottleneck Theory")

body(
    "The Information Bottleneck (IB) principle was introduced by Tishby, Pereira, and Bialek "
    "(1999) as a formal framework for lossy compression. Given a source variable X and a "
    "relevance variable Y, the IB objective finds a compressed representation T that minimizes "
    "the mutual information I(X; T) — discarding as much of X as possible — while maximizing "
    "I(T; Y) — retaining everything predictive of Y. The trade-off is governed by a Lagrange "
    "multiplier β: the higher β, the stronger the compression. This yields a family of "
    "representations along the 'information curve' from fully compressed to fully informative."
)

body(
    "Applying the IB principle to deep neural networks required making the mutual information "
    "terms tractable. Alemi et al. (2017) introduced the Deep Variational Information Bottleneck "
    "(Deep VIB), which replaces the intractable I(X; T) with a KL divergence between the "
    "learned posterior q(T|X) and a fixed prior p(T) = N(0, I). The encoder is a probabilistic "
    "function mapping inputs to Gaussian distributions (mu, sigma); samples drawn via the "
    "reparameterization trick enable end-to-end backpropagation through the stochastic "
    "bottleneck. Alemi et al. showed that VIB representations exhibit improved robustness to "
    "adversarial perturbations and better out-of-distribution generalization compared to "
    "deterministic encoders — precisely the properties we seek for few-shot classification."
)

body(
    "Our student branch is a direct instantiation of the Deep VIB encoder: the CLS pooled "
    "document vector is projected to (mu, sigma) via two linear heads; the compression loss "
    "KL(N(mu, σ²) ∥ N(0,1)) is algebraically identical to the VIB penalty; and the "
    "reparameterization trick enables gradient flow through the sampling step. The key "
    "extension beyond VIB is (1) replacing the label-supervised I(T; Y) term with a "
    "self-supervised global alignment objective, and (2) making β a learnable parameter "
    "rather than a pre-set hyperparameter."
)

body(
    "The connection between the IB principle and self-supervised learning was formalized by "
    "Wang et al. (2022), who proved that standard contrastive learning objectives converge "
    "to the minimal sufficient representation of the augmentation distribution — i.e., they "
    "retain only the information shared between augmented views, which may not coincide with "
    "task-relevant information. Adding an explicit compression penalty (the KL term in our "
    "framework) addresses this limitation by forcing the representation to be maximally "
    "compact while still predictive of the teacher's global view."
)

# 5.2
h2("5.2 Self-Supervised Representation Learning for NLP")

body(
    "Self-supervised pre-training for NLP has been dominated by masked language modeling "
    "(Devlin et al., 2019; Liu et al., 2019), which trains encoders to predict masked tokens "
    "from context. Liu et al. (2019) demonstrated that RoBERTa — obtained by training BERT "
    "longer on more data with larger batches and without the next-sentence prediction objective "
    "— substantially outperforms the original BERT on downstream tasks. RoBERTa-base (125M "
    "parameters, 768-dimensional hidden states) is the backbone for both our teacher and "
    "student branches."
)

body(
    "Contrastive learning methods offer an alternative to MLM for representation learning. "
    "SimCSE (Gao et al., 2021) demonstrated that using the same sentence twice as a positive "
    "pair with different dropout masks — and in-batch negatives — substantially advances "
    "sentence embedding quality, while providing a theoretical connection to alignment and "
    "uniformity of the embedding space. DeCLUTR (Giorgi et al., 2021) extended this paradigm "
    "to long documents: positive pairs are formed from spans sampled within the same document, "
    "establishing that intra-document structural relationships can serve as self-supervised "
    "signal without labels. Both SimCSE and DeCLUTR are key baselines for our method: "
    "they demonstrate that view-based contrastive objectives improve representations, but "
    "neither applies an explicit IB compression constraint."
)

body(
    "The teacher-student architecture of our model is inspired by BYOL (Grill et al., 2020) "
    "and SimSiam (Chen & He, 2021). BYOL demonstrated that a two-network setup — an online "
    "network trained to predict an exponential-moving-average target network — learns powerful "
    "visual representations without negative pairs. SimSiam simplified this to a stop-gradient "
    "operation on the target branch, providing theoretical analysis showing that the "
    "predictor-plus-stop-gradient mechanism creates an implicit EM algorithm that prevents "
    "representational collapse. Our teacher branch directly applies this insight: "
    "torch.no_grad() enforces stop-gradient, preventing the trivial solution where both "
    "branches collapse to constant representations. Without stop-gradient, any alignment "
    "objective degenerates — the teacher would adapt to the student and both would converge "
    "to a degenerate fixed point."
)

body(
    "GlocalIB (Yang et al., 2025) is the direct methodological ancestor of our work. "
    "Published at NeurIPS 2025, it proposed a model-agnostic training wrapper for time "
    "series imputation combining a local reconstruction loss with a global alignment term: "
    "a teacher branch encodes the fully-observed time series; a student branch encodes the "
    "same series with values masked; the student is trained to align its representation with "
    "the teacher's via a tractable mutual information approximation. Yang et al. validated "
    "GlocalIB on nine imputation benchmarks, demonstrating consistent improvements over "
    "state-of-the-art imputation methods. We transfer this paradigm from continuous temporal "
    "data to discrete paragraph sequences, replacing the recurrent/graph encoder with "
    "RoBERTa-base and the temporal masking with paragraph dropping."
)

# 5.3
h2("5.3 Few-Shot and Low-Resource Text Classification")

body(
    "Few-shot text classification has been studied extensively in the meta-learning literature. "
    "Snell et al. (2017) showed that prototypical networks — which classify by computing "
    "distances to class prototype embeddings in a learned metric space — outperform more "
    "complex meta-learning approaches for N-way K-shot classification, and that representation "
    "quality is the dominant factor in few-shot generalization. This finding motivates our "
    "pre-training-centric experimental design: if we improve the encoder's representations, "
    "any downstream classification mechanism — including a simple linear probe — should benefit."
)

body(
    "For NLP specifically, Deng et al. (2020) proposed a meta-pretraining approach that "
    "trains a BERT model on a large unlabeled corpus before meta-learning on few-shot "
    "classification episodes. Their central finding — that unsupervised meta-pretraining "
    "quality determines few-shot downstream performance — is precisely the hypothesis we "
    "test in the legal domain: three pre-training objectives (MLM, GlocalIB β=0, GlocalIB "
    "full) produce three encoders whose quality difference is reflected in macro-F1 at "
    "N ∈ {10, 50, 100}."
)

body(
    "In the legal domain specifically, Sarkar et al. (2021) investigated few-shot and "
    "zero-shot approaches to legal text classification in the financial sector, documenting "
    "the prohibitive annotation cost that makes the few-shot regime the realistic operational "
    "setting for legal NLP systems. Their work is the closest applied prior work to ours; "
    "we extend it to the ECtHR multi-label scenario with a principled self-supervised "
    "pre-training approach rather than a meta-learning inference approach. Chalkidis et al. "
    "(2020) pre-trained LEGAL-BERT on 12GB of legal text, showing that domain-adaptive "
    "pre-training consistently improves ECtHR violation prediction — the same hypothesis "
    "our mlm baseline condition tests, with GlocalIB providing a stronger inductive bias "
    "on top of domain adaptation."
)

# 5.4
h2("5.4 Legal NLP Benchmarks")

body(
    "The ECtHR dataset was introduced by Chalkidis et al. (2019), who provided the first "
    "large-scale English-language corpus of European Court of Human Rights cases with "
    "multi-label article violation annotations. Each case is represented as a sequence of "
    "factual paragraphs (the 'facts' section of the judgment), and labels are the ECHR "
    "articles that were found to be violated. The paragraph-structured format directly "
    "motivates our chunk-and-pool encoder design, and the severe class imbalance "
    "(Article 3: 4,704 training cases; Article 5: 41 training cases) mandates macro-F1 "
    "as the primary evaluation metric."
)

body(
    "LexGLUE (Chalkidis et al., 2022) formalized ECtHR article prediction as one of seven "
    "legal NLU benchmark tasks, providing standardized train/validation/test splits, "
    "evaluation protocols, and baselines ranging from BiLSTM to LEGAL-BERT. The `ecthr_a` "
    "task within LexGLUE is our downstream evaluation task, accessed via HuggingFace "
    "Datasets. Reporting macro-F1 on the same test split makes our results directly "
    "comparable to all LexGLUE baselines, situating our pre-training contribution within "
    "the established legal NLP benchmark ecosystem."
)

# 5.5
h2("5.5 Positioning This Work")

body(
    "The table below maps existing methods to the four problems identified in Section 2:"
)

table = doc.add_table(rows=1, cols=5)
table.style = "Table Grid"
hdr = table.rows[0].cells
for i, text in enumerate(["Method", "SSL on legal text", "IB compression", "Long-doc encoding", "Few-shot focus"]):
    hdr[i].text = text
    hdr[i].paragraphs[0].runs[0].bold = True

rows_data = [
    ("MLM / RoBERTa",         "✓", "✗", "✗ (512-token limit)", "✗"),
    ("SimCSE",                 "✗", "✗", "✗ (sentence-level)",  "✗"),
    ("DeCLUTR",                "✗", "✗", "✓ (span-level)",      "✗"),
    ("LEGAL-BERT",             "✓", "✗", "✗ (512-token limit)", "✗"),
    ("Deep VIB",               "✗", "✓", "✗",                   "✗"),
    ("GlocalIB (Yang et al.)", "✗", "✓", "N/A (time series)",   "✗"),
    ("This work",              "✓", "✓", "✓ (chunk-and-pool)",  "✓"),
]
for row_data in rows_data:
    row = table.add_row().cells
    for i, text in enumerate(row_data):
        row[i].text = text
        if row_data[0] == "This work":
            row[i].paragraphs[0].runs[0].bold = True

doc.add_paragraph()
body(
    "No existing work combines legal-domain self-supervised pre-training, an explicit "
    "information bottleneck compression objective, paragraph-level long-document encoding, "
    "and systematic evaluation in the few-shot regime on the ECtHR benchmark. This "
    "combination is the specific contribution of the present work."
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
h1("6. Experimental Design")

body(
    "The experimental design follows a single controlled comparison principle: fix everything "
    "except the pre-training objective. Any difference in downstream few-shot performance "
    "is then attributable to the objective alone."
)

h2("6.1 Dataset")

body(
    "We use the ECtHR (ecthr_a) split from the LexGLUE benchmark "
    "(Chalkidis et al., 2022), accessed via HuggingFace Datasets "
    "(`coastalcph/lex_glue`). Each example is a list of factual paragraphs paired with "
    "multi-hot labels indicating which ECHR articles were violated. We apply a single "
    "filter: documents with fewer than 5 paragraphs are removed (they cannot be "
    "meaningfully masked). No other preprocessing is applied."
)

bullet("Train: ~9,000 cases (post-filter)")
bullet("Validation: ~1,000 cases")
bullet("Test: ~1,000 cases")
bullet("Classes: 10 ECHR articles (multi-label)")
bullet("Avg paragraphs per case: 23.1 (max: 558)")
bullet("Class imbalance: Article 3 dominates (4,704 cases); Article 5 is rarest (41 cases)")

h2("6.2 Experimental Conditions")

body("Three conditions, differing only in pre-training objective:")

table2 = doc.add_table(rows=1, cols=4)
table2.style = "Table Grid"
hdr2 = table2.rows[0].cells
for i, t in enumerate(["Condition", "Pre-training Objective", "IB Term", "Architecture"]):
    hdr2[i].text = t
    hdr2[i].paragraphs[0].runs[0].bold = True

for row_data in [
    ("mlm",           "Standard MLM (15% token masking)",    "None",              "RoBERTa-base (standard)"),
    ("glocal_beta0",  "GlocalIB alignment only (β=0)",       "Disabled",          "Teacher-student + probabilistic head"),
    ("glocal_ib",     "GlocalIB full (learnable β ≥ 0.01)",  "KL compression",    "Teacher-student + probabilistic head"),
]:
    row = table2.add_row().cells
    for i, t in enumerate(row_data):
        row[i].text = t

doc.add_paragraph()
body(
    "The `glocal_beta0` ablation is critical: it isolates the contribution of the IB "
    "compression term from the contribution of the teacher-student architecture itself. "
    "If glocal_ib outperforms glocal_beta0, the KL compression penalty is responsible "
    "for the improvement (H2 confirmed). If glocal_ib ≈ glocal_beta0, the benefit comes "
    "from the global-local structure alone, not from compression."
)

h2("6.3 Pre-training Protocol")

body(
    "All three conditions use the ECtHR training split as unlabeled pre-training data. "
    "Pre-training runs for 5 epochs with batch size 4 (32GB GPU) or 16 (Spark 128GB). "
    "The optimizer is AdamW with learning rate 1e-5. W&B logs per-step: loss, l_align, "
    "l_compress, beta, epoch. Checkpoints are saved after each epoch."
)

body(
    "For glocal_ib and glocal_beta0, each training step samples a batch of documents, "
    "creates the teacher input (all paragraphs) and student input (mask_paragraphs: "
    "20–40% randomly dropped, always keep ≥1), and runs the forward pass through "
    "GlocalIBModel. For mlm, the HuggingFace Trainer runs standard token-level "
    "15% masking via DataCollatorForLanguageModeling."
)

h2("6.4 Fine-tuning Protocol")

body(
    "After pre-training, we extract the RoBERTa encoder from each checkpoint and "
    "attach a DocumentClassifier: a chunk-and-pool encoder + Linear(768, 10) + sigmoid "
    "output for multi-label classification. Fine-tuning trains all weights (encoder + "
    "head) on N labeled examples sampled via sample_few_shot(), which ensures each of the "
    "10 classes has ≥N examples."
)

body("Fine-tuning runs for 10 epochs, AdamW lr=2e-5, BCELoss. "
     "N ∈ {10, 50, 100} × 5 random seeds = 15 runs per condition = 45 total runs.")

h2("6.5 Evaluation")

body(
    "Macro-F1 on the ECtHR test set, averaged over 5 seeds. We report mean ± std. "
    "The primary result is a performance-vs-N curve (log-scale x-axis) showing all three "
    "conditions with error bars. The ablation delta table (glocal_ib − glocal_beta0 at "
    "each N) quantifies the compression term's contribution."
)

body(
    "Beta trajectory diagnostic: the learned β from W&B logs for the glocal_ib condition "
    "is plotted over training steps. If β stays near 0.01 (lower bound), the IB term is "
    "inactive and H2 is threatened; if β varies adaptively (e.g., increasing, then settling), "
    "the model is genuinely learning the compression strength."
)

h2("6.6 Key Invariants and Risks")

bullet(
    "Stop-gradient is enforced. The teacher branch uses torch.no_grad() on every forward "
    "call. Without this, representations collapse. This is the single most critical "
    "implementation invariant."
)
bullet(
    "Beta lower bound (0.01). Prevents compression collapse where KL → 0 and the IB term "
    "disappears entirely. Removing the clamp would invalidate the glocal_ib condition."
)
bullet(
    "Macro-F1 required. Due to extreme class imbalance, micro-F1 would mask failures on "
    "rare classes. All results use macro-F1."
)
bullet(
    "Risk: IB does not survive fine-tuning. The compression learned during pre-training "
    "may be undone during full fine-tuning. Mitigation: measure representation geometry "
    "(cosine distance between class centroids) before and after fine-tuning."
)
bullet(
    "Risk: GlocalIB does not beat MLM. If H1 fails, the claim pivots to variance reduction "
    "(H3): even if mean macro-F1 is comparable, lower standard deviation across seeds "
    "is a meaningful practical contribution — more predictable performance in the "
    "few-shot regime."
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. REFERENCES
# ─────────────────────────────────────────────────────────────────────────────
h1("References")

ref_entry(
    "Tishby, N., Pereira, F. C., & Bialek, W.",
    "1999",
    "The Information Bottleneck Method",
    "Proceedings of the 37th Annual Allerton Conference on Communication, Control and Computing",
    "pp. 368–377. arXiv:physics/0004057."
)
ref_entry(
    "Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K.",
    "2017",
    "Deep Variational Information Bottleneck",
    "5th International Conference on Learning Representations (ICLR 2017)",
    "arXiv:1612.00410."
)
ref_entry(
    "Yang, J., Zhang, K., Zhang, G., Yu, P. S., & Ding, K.",
    "2025",
    "Glocal Information Bottleneck for Time Series Imputation",
    "Advances in Neural Information Processing Systems 38 (NeurIPS 2025)",
    "arXiv:2510.04910."
)
ref_entry(
    "Liu, Y., Ott, M., Goyal, N., et al.",
    "2019",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "arXiv preprint",
    "arXiv:1907.11692."
)
ref_entry(
    "Gao, T., Yao, X., & Chen, D.",
    "2021",
    "SimCSE: Simple Contrastive Learning of Sentence Embeddings",
    "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)",
    "pp. 6894–6910. arXiv:2104.08821."
)
ref_entry(
    "Giorgi, J., Nitski, O., Wang, B., & Bader, G.",
    "2021",
    "DeCLUTR: Deep Contrastive Learning for Unsupervised Textual Representations",
    "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP 2021)",
    "pp. 879–895. arXiv:2006.03659."
)
ref_entry(
    "Grill, J.-B., Strub, F., Altché, F., et al.",
    "2020",
    "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning",
    "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)",
    "arXiv:2006.07733."
)
ref_entry(
    "Chen, X., & He, K.",
    "2021",
    "Exploring Simple Siamese Representation Learning",
    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021)",
    "pp. 15750–15758. arXiv:2011.10566."
)
ref_entry(
    "Wang, H., Guo, X., Deng, Z., & Lu, Y.",
    "2022",
    "Rethinking Minimal Sufficient Representation in Contrastive Learning",
    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022 Oral)",
    "pp. 16085–16094."
)
ref_entry(
    "Chalkidis, I., Androutsopoulos, I., & Aletras, N.",
    "2019",
    "Neural Legal Judgment Prediction in English",
    "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)",
    "pp. 4317–4323. arXiv:1906.02059."
)
ref_entry(
    "Chalkidis, I., Jana, A., Hartung, D., et al.",
    "2022",
    "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English",
    "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)",
    "Vol. 1, pp. 4310–4330. arXiv:2110.00976."
)
ref_entry(
    "Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I.",
    "2020",
    "LEGAL-BERT: The Muppets Straight out of Law School",
    "Findings of the Association for Computational Linguistics: EMNLP 2020",
    "pp. 2898–2904."
)
ref_entry(
    "Snell, J., Swersky, K., & Zemel, R.",
    "2017",
    "Prototypical Networks for Few-Shot Learning",
    "Advances in Neural Information Processing Systems 30 (NeurIPS 2017)",
    "arXiv:1703.05175."
)
ref_entry(
    "Deng, S., Zhang, N., Sun, Z., Chen, J., & Chen, H.",
    "2020",
    "When Low Resource NLP Meets Unsupervised Language Model: Meta-Pretraining then Meta-Learning for Few-Shot Text Classification",
    "Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2020)",
    "pp. 13773–13774."
)
ref_entry(
    "Sarkar, R., Ojha, A. K., et al.",
    "2021",
    "Few-Shot and Zero-Shot Approaches to Legal Text Classification: A Case Study in the Financial Sector",
    "Proceedings of the Natural Legal Language Processing Workshop (NLLP 2021, EMNLP)",
    "pp. 102–106."
)
ref_entry(
    "Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.",
    "2019",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL 2019)",
    "pp. 4171–4186. arXiv:1810.04805."
)

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "GlocalIB_NLP_Report.docx")
doc.save(out_path)
print(f"Saved: {out_path}")
