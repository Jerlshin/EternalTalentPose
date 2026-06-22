

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from redstack.config.schema import LexiconSeedConfig
from redstack.domain.errors import ArtifactContractError
from redstack.pipelines.offline.registry import OfflineArtifactRegistry, OFFLINE_ARTIFACT_REGISTRY
from redstack.pipelines.offline.runner import StageReceipt, StageResult
from redstack.pipelines.offline.stages import OfflineStage
from redstack.ports._types import SourceMalformed, SourceOk

from redstack.pipelines.offline.context import OfflinePipelineContext

__all__: tuple[str, ...] = (
    "LexiconDiscoveryStage",
    "lexicon_discovery_stage",
    "tokenize",
)

#: Token pattern: alphanumerics + intra-word ``+`` / ``#`` / ``.`` (c++, c#, a/b).
_TOKEN_RE: Final = re.compile(r"[a-z0-9][a-z0-9+#.]*")

#: English stopwords excluded from terms/phrases (small, fixed, deterministic).
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
        "were", "with", "this", "these", "those", "we", "our", "i", "my", "they",
        "their", "you", "your", "he", "she", "his", "her", "but", "not", "have",
        "had", "will", "would", "can", "could", "should", "been", "being", "do",
        "did", "does", "using", "used", "use", "also", "into", "over", "per",
    }
)

#: Minimum PMI for a bigram/trigram to be admitted as a concept phrase.
_PMI_THRESHOLD: Final[float] = 2.0
#: Minimum co-occurrence count for a phrase to be considered (noise floor).
_PHRASE_MIN_COUNT: Final[int] = 3
#: Top-N terms retained per concept in the compiled lexicon (bounded artifact).
_TERMS_PER_CONCEPT: Final[int] = 50
#: Top-N co-occurrence edges retained per term in the term graph.
_EDGES_PER_TERM: Final[int] = 10


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text deterministically into lowercase content tokens.

    NFC-normalizes, lowercases, extracts ``_TOKEN_RE`` matches, drops stopwords
    and single characters. Pure and idempotent on already-normalized input. The
    same tokenizer is the one O5/feature competency matching reuses, so concept
    membership is consistent offline and online.
    """
    nfc = unicodedata.normalize("NFC", text).casefold()
    return [
        tok
        for tok in _TOKEN_RE.findall(nfc)
        if tok not in _STOPWORDS and len(tok) > 1
    ]


@dataclass(slots=True)
class _ConceptAccumulator:
    """Streaming c-TF-IDF accumulator for one concept (its seed-matched class).

    ``term_freq`` is the class term-frequency (summed over documents assigned to
    this concept); ``doc_count`` is how many documents matched the concept's
    seed. Bounded by vocabulary size, not by N.
    """

    seed_terms: frozenset[str]
    term_freq: Counter[str] = field(default_factory=Counter)
    doc_count: int = 0


@dataclass(frozen=True, slots=True)
class _Phrase:
    """A mined multi-word concept phrase with its count, PMI, and constituents."""

    count: int
    pmi: float
    terms: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Serialize for the concept dictionary / phrase graph artifacts."""
        return {"count": self.count, "pmi": round(self.pmi, 6), "terms": list(self.terms)}


class LexiconDiscoveryStage(OfflineStage):
    """O4 — stream descriptions, mine concept lexicons + phrase/term graphs."""

    stage_id = "O4"
    stage_version = "1.0"

    def __init__(
        self,
        seed: LexiconSeedConfig,
        registry: OfflineArtifactRegistry = OFFLINE_ARTIFACT_REGISTRY,
    ) -> None:
        """Construct with the injected human seed lexicon (authoring seed).

        Args:
            seed: ``lexicon/lexicon.seed.yaml`` parsed — concept → seed terms.
            registry: the artifact catalog (defaults to the frozen catalog).
        """
        super().__init__(registry)
        if not seed.concepts:
            msg = "lexicon seed must declare at least one concept"
            raise ArtifactContractError(msg)
        self._seed = seed

    def _run(
        self,
        ctx: OfflinePipelineContext,
        upstream: Mapping[str, StageReceipt],
    ) -> StageResult:
        accumulators = {
            concept: _ConceptAccumulator(
                seed_terms=frozenset(tokenize(" ".join(body.seed_terms)))
            )
            for concept, body in self._seed.concepts.items()
        }
        # Global document frequency (for IDF) + co-occurrence (term graph) +
        # adjacency counts (phrase PMI). All bounded by vocabulary, not by N.
        global_term_docs: Counter[str] = Counter()
        cooccurrence: dict[str, Counter[str]] = {}
        unigram_counts: Counter[str] = Counter()
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()
        total_unigrams = 0
        total_bigrams = 0
        doc_total = 0

        for record in ctx.candidate_source.stream():
            if isinstance(record, SourceMalformed):
                continue
            if not isinstance(record, SourceOk):
                continue
            for tokens in self._iter_description_tokens(record.raw):
                if not tokens:
                    continue
                doc_total += 1
                unique = set(tokens)
                global_term_docs.update(unique)
                self._assign_to_concepts(tokens, unique, accumulators)
                self._update_cooccurrence(unique, cooccurrence)
                total_unigrams += self._update_ngrams(
                    tokens, unigram_counts, bigram_counts, trigram_counts
                )
                total_bigrams += max(0, len(tokens) - 1)

        if doc_total == 0:
            msg = "lexicon discovery saw zero descriptions (empty pool)"
            raise ArtifactContractError(msg)

        lexicon = self._compile_lexicon(accumulators, global_term_docs, doc_total)
        phrases = self._mine_phrases(
            unigram_counts, bigram_counts, trigram_counts,
            total_unigrams, total_bigrams,
        )
        concepts_dict = self._build_concept_dictionary(lexicon, phrases, accumulators)
        term_graph = self._build_term_graph(cooccurrence)
        phrase_graph = self._build_phrase_graph(phrases)
        term_graph_node_count = len(cooccurrence)

        art_lexicon = self.emit_json(ctx, "lexicon_compiled", {"concepts": lexicon})
        art_concepts = self.emit_json(ctx, "concepts", {"concepts": concepts_dict})
        art_term = self.emit_json(ctx, "term_graph", term_graph)
        art_phrase = self.emit_json(ctx, "phrase_graph", phrase_graph)

        metrics: dict[str, object] = {
            "documents": doc_total,
            "concepts": len(lexicon),
            "phrases_mined": len(phrases),
            "term_graph_nodes": term_graph_node_count,
        }
        return StageResult(
            artifacts=(art_lexicon, art_concepts, art_term, art_phrase),
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Streaming token sources                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iter_description_tokens(raw: Mapping[str, object]) -> list[list[str]]:
        """Yield tokenized text from each description-bearing field of a record.

        Sources: profile summary + headline, and every career-history role
        description + title. Each becomes one "document" for c-TF-IDF/co-occurrence.
        """
        docs: list[list[str]] = []
        profile = raw.get("profile")
        if isinstance(profile, Mapping):
            for key in ("summary", "headline"):
                text = profile.get(key)
                if isinstance(text, str) and text.strip():
                    docs.append(tokenize(text))
        history = raw.get("career_history")
        if isinstance(history, (list, tuple)):
            for position in history:
                if isinstance(position, Mapping):
                    parts: list[str] = []
                    for key in ("title", "description"):
                        text = position.get(key)
                        if isinstance(text, str) and text.strip():
                            parts.append(text)
                    if parts:
                        docs.append(tokenize(" ".join(parts)))
        return docs

    @staticmethod
    def _assign_to_concepts(
        tokens: Sequence[str],
        unique: set[str],
        accumulators: Mapping[str, _ConceptAccumulator],
    ) -> None:
        """Assign a document to every concept whose seed it contains (c-TF-IDF class).

        A document may belong to multiple concepts (overlapping terminology is
        expected); each matched concept accrues the document's term frequencies.
        """
        for acc in accumulators.values():
            if unique & acc.seed_terms:
                acc.doc_count += 1
                acc.term_freq.update(tokens)

    @staticmethod
    def _update_cooccurrence(
        unique: set[str], cooccurrence: dict[str, Counter[str]]
    ) -> None:
        """Increment symmetric co-occurrence counts for terms sharing a document.

        Bounded: only distinct terms per document contribute, and the graph is
        pruned to top edges per term at emit time. Uses ``itertools.combinations``
        over the same sorted order the old nested-loop-with-slice used (avoids
        reallocating an ``ordered[i+1:]`` slice on every outer step — this is a
        100K-document hot loop) — verified to produce byte-identical counts.
        """
        ordered = sorted(unique)
        for a in ordered:
            cooccurrence.setdefault(a, Counter())
        for a, b in itertools.combinations(ordered, 2):
            cooccurrence[a][b] += 1
            cooccurrence[b][a] += 1

    @staticmethod
    def _update_ngrams(
        tokens: Sequence[str],
        unigram_counts: Counter[str],
        bigram_counts: Counter[tuple[str, str]],
        trigram_counts: Counter[tuple[str, str, str]],
    ) -> int:
        """Accumulate uni/bi/tri-gram counts for PMI phrase mining; return |tokens|."""
        unigram_counts.update(tokens)
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1
        for i in range(len(tokens) - 2):
            trigram_counts[(tokens[i], tokens[i + 1], tokens[i + 2])] += 1
        return len(tokens)

    # ------------------------------------------------------------------ #
    # c-TF-IDF compilation                                               #
    # ------------------------------------------------------------------ #
    def _compile_lexicon(
        self,
        accumulators: Mapping[str, _ConceptAccumulator],
        global_term_docs: Counter[str],
        doc_total: int,
    ) -> dict[str, object]:
        """Compute per-concept c-TF-IDF term weights → the compiled lexicon.

        For each concept class: ``weight(term) = tf_class(term) * log(1 + N / df(term))``
        where ``tf_class`` is the class term frequency and ``df`` the global
        document frequency. Seed terms are always retained (weight floored to a
        positive minimum) so the human seed is never lost. Returns the top
        ``_TERMS_PER_CONCEPT`` terms per concept, sorted by ``(-weight, term)``.
        """
        compiled: dict[str, object] = {}
        for concept, acc in accumulators.items():
            weights: dict[str, float] = {}
            for term, tf in acc.term_freq.items():
                df = global_term_docs.get(term, 1)
                idf = math.log1p(doc_total / df)
                weights[term] = float(tf) * idf
            # Guarantee seed coverage even when a seed term is sparse in-corpus.
            seed_floor = math.log1p(doc_total)
            for seed_term in acc.seed_terms:
                if weights.get(seed_term, 0.0) <= 0.0:
                    weights[seed_term] = seed_floor
            ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
            top = ranked[:_TERMS_PER_CONCEPT]
            # Normalize weights to [0,1] within the concept for stable downstream use.
            max_w = max((w for _, w in top), default=1.0) or 1.0
            compiled[concept] = {
                "terms": {term: round(w / max_w, 6) for term, w in top},
                "doc_count": acc.doc_count,
                "seed_terms": sorted(acc.seed_terms),
            }
        return compiled

    # ------------------------------------------------------------------ #
    # PMI phrase mining                                                  #
    # ------------------------------------------------------------------ #
    def _mine_phrases(
        self,
        unigram_counts: Counter[str],
        bigram_counts: Counter[tuple[str, str]],
        trigram_counts: Counter[tuple[str, str, str]],
        total_unigrams: int,
        total_bigrams: int,
    ) -> dict[str, _Phrase]:
        """Mine bigram/trigram phrases by pointwise mutual information.

        ``PMI(a,b) = log( P(a,b) / (P(a)·P(b)) )``. Bigrams above
        :data:`_PMI_THRESHOLD` with count ≥ :data:`_PHRASE_MIN_COUNT` are kept;
        trigrams are admitted when both constituent bigrams are themselves
        high-PMI (a deterministic, conservative gate). Returns phrase → ``_Phrase``
        sorted by phrase text at emit time.
        """
        if total_unigrams == 0:
            return {}
        phrases: dict[str, _Phrase] = {}
        kept_bigrams: set[tuple[str, str]] = set()
        for (a, b), count in bigram_counts.items():
            if count < _PHRASE_MIN_COUNT:
                continue
            p_ab = count / total_bigrams if total_bigrams else 0.0
            p_a = unigram_counts[a] / total_unigrams
            p_b = unigram_counts[b] / total_unigrams
            if p_ab <= 0.0 or p_a <= 0.0 or p_b <= 0.0:
                continue
            pmi = math.log(p_ab / (p_a * p_b))
            if pmi >= _PMI_THRESHOLD:
                kept_bigrams.add((a, b))
                phrases[f"{a} {b}"] = _Phrase(count=count, pmi=pmi, terms=(a, b))
        for (a, b, c), count in trigram_counts.items():
            if count < _PHRASE_MIN_COUNT:
                continue
            if (a, b) in kept_bigrams and (b, c) in kept_bigrams:
                tri_pmi = (phrases[f"{a} {b}"].pmi + phrases[f"{b} {c}"].pmi) / 2.0
                phrases[f"{a} {b} {c}"] = _Phrase(
                    count=count, pmi=tri_pmi, terms=(a, b, c)
                )
        return phrases

    # ------------------------------------------------------------------ #
    # Concept dictionary (anchor_text per concept — required by validator) #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_concept_dictionary(
        lexicon: Mapping[str, object],
        phrases: Mapping[str, _Phrase],
        accumulators: Mapping[str, _ConceptAccumulator],
    ) -> dict[str, object]:
        """Build the ConceptDictionary: per-concept vocab + composed ``anchor_text``.

        ``anchor_text`` is a deterministic, human-readable concatenation of the
        concept's seed terms followed by its top mined terms — the seed text O5
        embeds and O6 authors anchors from. Every concept gets a non-empty
        ``anchor_text`` (the ``_v_concepts`` invariant), since seed terms are
        always present.
        """
        dictionary: dict[str, object] = {}
        for concept, entry in lexicon.items():
            assert isinstance(entry, Mapping)
            terms_obj = entry.get("terms")
            top_terms = sorted(terms_obj) if isinstance(terms_obj, Mapping) else []
            seed_terms = accumulators[concept].seed_terms
            anchor_terms = sorted(seed_terms) + [
                t for t in top_terms if t not in seed_terms
            ]
            anchor_text = ", ".join(anchor_terms[:_TERMS_PER_CONCEPT])
            concept_phrases = sorted(
                phrase
                for phrase, body in phrases.items()
                if any(t in seed_terms for t in body.terms)
            )
            dictionary[concept] = {
                "vocab": top_terms,
                "phrases": concept_phrases,
                "anchor_text": anchor_text or concept,
                "expanded": False,
            }
        return dictionary

    # ------------------------------------------------------------------ #
    # Graph builders                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_term_graph(cooccurrence: Mapping[str, Counter[str]]) -> dict[str, object]:
        """Build the co-occurrence ``TermGraph`` (nodes + top weighted edges).

        Nodes are terms; edges are the top :data:`_EDGES_PER_TERM` co-occurring
        neighbours per term, emitted once per unordered pair in sorted order so
        the graph is acyclic-where-expected and byte-stable.
        """
        nodes = sorted(cooccurrence)
        seen: set[tuple[str, str]] = set()
        edges: list[dict[str, object]] = []
        for term in nodes:
            neighbours = cooccurrence[term].most_common(_EDGES_PER_TERM)
            for other, weight in neighbours:
                pair = (term, other) if term < other else (other, term)
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append({"source": pair[0], "target": pair[1], "weight": weight})
        edges.sort(key=lambda e: (str(e["source"]), str(e["target"])))
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _build_phrase_graph(phrases: Mapping[str, _Phrase]) -> dict[str, object]:
        """Build the ``PhraseGraph``: phrase nodes → constituent-term edges.

        A directed acyclic structure (phrase → term); nodes are phrases + their
        constituent terms; edges link each phrase to each term it contains.
        """
        phrase_nodes = sorted(phrases)
        term_nodes: set[str] = set()
        edges: list[dict[str, object]] = []
        for phrase in phrase_nodes:
            for term in phrases[phrase].terms:
                term_nodes.add(term)
                edges.append({"source": phrase, "target": term, "weight": 1})
        nodes = phrase_nodes + sorted(term_nodes - set(phrase_nodes))
        edges.sort(key=lambda e: (str(e["source"]), str(e["target"])))
        return {"nodes": nodes, "edges": edges}


def lexicon_discovery_stage(seed: LexiconSeedConfig) -> LexiconDiscoveryStage:
    """Factory: construct the O4 lexicon stage bound to the injected seed."""
    return LexiconDiscoveryStage(seed)