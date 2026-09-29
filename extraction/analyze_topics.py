import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "enriched_text.json"
)

TOPIC_OUTPUT = (
    PROCESSED_DIR
    / "topic_analysis.json"
)

TREND_OUTPUT = (
    PROCESSED_DIR
    / "trend_analysis.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Sentence-transformers model.
# Lightweight enough for the RTX 3050.
EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# Number of semantic topics.
#
# We calculate this automatically from dataset size,
# but keep it within reasonable bounds.
MIN_TOPICS = 5
MAX_TOPICS = 15

# Number of keywords to retain.
TOP_KEYWORDS = 50

# Number of hashtags to retain.
TOP_HASHTAGS = 50

# Number of representative texts per topic.
REPRESENTATIVE_TEXTS = 5

# Minimum word length for keyword extraction.
MIN_WORD_LENGTH = 3


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
Input file not found:

{INPUT_FILE}

Run analyze_text.py first.
"""
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(path, data):

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9_-]*\b",
        text
    )

    stop_words = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "was",
        "were",
        "are",
        "you",
        "your",
        "they",
        "their",
        "about",
        "there",
        "what",
        "when",
        "where",
        "which",
        "will",
        "would",
        "could",
        "should",
        "been",
        "being",
        "into",
        "than",
        "then",
        "them",
        "these",
        "those",
        "just",
        "very",
        "more",
        "some",
        "also",
        "only",
        "really",
        "like",
        "but",
        "not",
        "can",
        "its",
        "it's",
        "our",
        "out",
        "all",
        "too",
        "how",
        "why",
        "who",
        "get",
        "got",
        "had",
        "did",
        "does",
        "dont",
        "don't",
        "im",
        "i'm",
        "we",
        "our",
        "us",
        "he",
        "she",
        "his",
        "her",
        "it",
        "as",
        "on",
        "in",
        "of",
        "to",
        "a",
        "an",
        "is",
        "be",
        "or",
        "if",
        "at",
        "by",
        "up",
        "so",
        "no",
        "yes",
    }

    return [
        word
        for word in words
        if (
            len(word) >= MIN_WORD_LENGTH
            and word not in stop_words
        )
    ]


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(records):

    counter = Counter()

    for record in records:

        text = record.get(
            "text",
            ""
        )

        words = tokenize(text)

        counter.update(words)

    return [
        {
            "keyword": word,
            "mentions": count
        }

        for word, count
        in counter.most_common(
            TOP_KEYWORDS
        )
    ]


# ============================================================
# HASHTAG EXTRACTION
# ============================================================

def extract_hashtags(records):

    counter = Counter()

    for record in records:

        text = record.get(
            "text",
            ""
        )

        hashtags = re.findall(
            r"#([A-Za-z0-9_]+)",
            text
        )

        for hashtag in hashtags:

            counter.update(
                [hashtag.lower()]
            )

    return [
        {
            "hashtag": hashtag,
            "mentions": count
        }

        for hashtag, count
        in counter.most_common(
            TOP_HASHTAGS
        )
    ]


# ============================================================
# TF-IDF KEYWORDS
# ============================================================

def extract_tfidf_keywords(records):

    documents = [
        clean_text(
            record.get("text", "")
        )

        for record in records
    ]

    documents = [
        document
        for document in documents
        if document
    ]

    if not documents:

        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    scores = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[
        :TOP_KEYWORDS
    ]:

        results.append(
            {
                "term": terms[index],
                "tfidf_score": round(
                    float(scores[index]),
                    6
                )
            }
        )

    return results


# ============================================================
# DETERMINE NUMBER OF TOPICS
# ============================================================

def determine_topic_count(
    record_count
):

    if record_count < 20:

        return 3

    estimated = int(
        np.sqrt(record_count / 10)
    )

    return max(
        MIN_TOPICS,
        min(
            MAX_TOPICS,
            estimated
        )
    )


# ============================================================
# EMBEDDINGS
# ============================================================

def generate_embeddings(
    texts,
    model
):

    print(
        "\nGenerating semantic embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings
    )


# ============================================================
# TOPIC CLUSTERING
# ============================================================

def cluster_topics(
    records,
    embeddings,
    topic_count
):

    print(
        f"\nClustering into "
        f"{topic_count} semantic topics..."
    )

    model = MiniBatchKMeans(
        n_clusters=topic_count,
        random_state=42,
        batch_size=64,
        n_init=10
    )

    labels = model.fit_predict(
        embeddings
    )

    return labels


# ============================================================
# TOPIC KEYWORDS
# ============================================================

def get_topic_keywords(
    texts,
    labels,
    topic_id,
    count=10
):

    topic_texts = [
        texts[index]

        for index in range(
            len(texts)
        )

        if labels[index] == topic_id
    ]

    if not topic_texts:

        return []

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=1000
        )

        matrix = vectorizer.fit_transform(
            topic_texts
        )

        scores = np.asarray(
            matrix.mean(axis=0)
        ).ravel()

        terms = np.array(
            vectorizer.get_feature_names_out()
        )

        indices = scores.argsort()[::-1]

        return [
            terms[index]

            for index in indices[:count]
        ]

    except Exception:

        return []


# ============================================================
# TOPIC SENTIMENT
# ============================================================

def calculate_topic_sentiment(
    topic_records
):

    counts = Counter()

    for record in topic_records:

        sentiment = record.get(
            "sentiment",
            "unknown"
        )

        counts[sentiment] += 1

    total = len(topic_records)

    distribution = {}

    for sentiment, count in counts.items():

        distribution[sentiment] = {
            "count": count,
            "percentage": round(
                (
                    count / total * 100
                )
                if total
                else 0,
                2
            )
        }

    return distribution


# ============================================================
# TOPIC EMOTIONS
# ============================================================

def calculate_topic_emotions(
    topic_records
):

    counts = Counter()

    for record in topic_records:

        emotion = record.get(
            "emotion",
            "unknown"
        )

        counts[emotion] += 1

    return [
        {
            "emotion": emotion,
            "mentions": count
        }

        for emotion, count
        in counts.most_common(10)
    ]


# ============================================================
# REPRESENTATIVE TEXTS
# ============================================================

def get_representative_texts(
    topic_records
):

    sorted_records = sorted(
        topic_records,

        key=lambda record:
            float(
                record.get(
                    "sentiment_score",
                    0
                )
            ),

        reverse=True
    )

    return [
        {
            "source_type":
                record.get(
                    "source_type"
                ),

            "source_id":
                record.get(
                    "source_id"
                ),

            "text":
                record.get(
                    "text"
                )
        }

        for record
        in sorted_records[
            :REPRESENTATIVE_TEXTS
        ]
    ]


# ============================================================
# TOPIC ANALYSIS
# ============================================================

def build_topic_analysis(
    records,
    labels
):

    texts = [
        clean_text(
            record.get("text", "")
        )

        for record in records
    ]

    topic_groups = defaultdict(list)

    for index, topic_id in enumerate(
        labels
    ):

        topic_groups[
            int(topic_id)
        ].append(
            records[index]
        )

    topics = []

    for topic_id in sorted(
        topic_groups.keys()
    ):

        topic_records = (
            topic_groups[topic_id]
        )

        keywords = get_topic_keywords(
            texts,
            labels,
            topic_id
        )

        topic = {

            "topic_id":
                topic_id,

            "document_count":
                len(topic_records),

            "percentage":
                round(
                    (
                        len(topic_records)
                        / len(records)
                        * 100
                    ),
                    2
                ),

            "keywords":
                keywords,

            "sentiment":
                calculate_topic_sentiment(
                    topic_records
                ),

            "top_emotions":
                calculate_topic_emotions(
                    topic_records
                ),

            "representative_texts":
                get_representative_texts(
                    topic_records
                )
        }

        topics.append(
            topic
        )

    # Largest topics first
    topics.sort(
        key=lambda item:
            item["document_count"],
        reverse=True
    )

    # Human-friendly ranking
    for rank, topic in enumerate(
        topics,
        start=1
    ):

        topic["rank"] = rank

    return topics


# ============================================================
# TIME PARSING
# ============================================================

def normalize_timestamp(
    timestamp
):

    if timestamp is None:

        return None

    try:

        timestamp = float(
            timestamp
        )

        # JavaScript Date.now()
        # is milliseconds.

        if timestamp > 10_000_000_000:

            timestamp /= 1000

        return timestamp

    except (
        TypeError,
        ValueError
    ):

        return None


# ============================================================
# TREND ANALYSIS
# ============================================================

def calculate_trends(
    records,
    labels
):

    topic_data = defaultdict(
        lambda: {
            "timestamps": [],
            "sentiments": Counter(),
            "emotions": Counter()
        }
    )

    for index, record in enumerate(
        records
    ):

        topic_id = int(
            labels[index]
        )

        timestamp = normalize_timestamp(
            record.get(
                "created_at"
            )
        )

        if timestamp is not None:

            topic_data[
                topic_id
            ]["timestamps"].append(
                timestamp
            )

        topic_data[
            topic_id
        ]["sentiments"][
            record.get(
                "sentiment",
                "unknown"
            )
        ] += 1

        topic_data[
            topic_id
        ]["emotions"][
            record.get(
                "emotion",
                "unknown"
            )
        ] += 1

    trend_results = []

    for topic_id, data in topic_data.items():

        timestamps = sorted(
            data["timestamps"]
        )

        if not timestamps:

            continue

        min_time = min(
            timestamps
        )

        max_time = max(
            timestamps
        )

        duration = (
            max_time - min_time
        )

        # If there isn't enough time
        # variation, trend direction
        # cannot be meaningfully calculated.

        if duration < 3600:

            trend_direction = (
                "insufficient_data"
            )

            trend_score = 0.0

        else:

            midpoint = (
                min_time
                + duration / 2
            )

            early_count = sum(
                1

                for timestamp
                in timestamps

                if timestamp
                <= midpoint
            )

            late_count = sum(
                1

                for timestamp
                in timestamps

                if timestamp
                > midpoint
            )

            # Normalize by duration of
            # each half.

            first_rate = (
                early_count
                / (duration / 2)
            )

            second_rate = (
                late_count
                / (duration / 2)
            )

            if first_rate == 0:

                trend_score = 1.0

            else:

                trend_score = (
                    second_rate
                    - first_rate
                ) / first_rate

            trend_score = max(
                -1.0,
                min(
                    1.0,
                    trend_score
                )
            )

            if trend_score > 0.15:

                trend_direction = "rising"

            elif trend_score < -0.15:

                trend_direction = "declining"

            else:

                trend_direction = "stable"

        total = (
            data["sentiments"].total()
        )

        trend_results.append(
            {
                "topic_id":
                    topic_id,

                "mentions":
                    total,

                "trend_score":
                    round(
                        trend_score,
                        4
                    ),

                "trend_direction":
                    trend_direction,

                "first_timestamp":
                    min_time,

                "last_timestamp":
                    max_time,

                "sentiment_distribution":
                    dict(
                        data[
                            "sentiments"
                        ]
                    ),

                "top_emotions":
                    [
                        {
                            "emotion":
                                emotion,

                            "mentions":
                                count
                        }

                        for emotion, count

                        in data[
                            "emotions"
                        ].most_common(5)
                    ]
            }
        )

    trend_results.sort(
        key=lambda item:
            item["trend_score"],
        reverse=True
    )

    return trend_results


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "             SIH TOPIC & TREND ENGINE"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "\n[1/6] Loading enriched text..."
    )

    try:

        records = load_data()

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        return

    print(
        f"      Records loaded: "
        f"{len(records)}"
    )

    if not records:

        print(
            "\nNo records available."
        )

        return

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    print(
        "\n[2/6] Preparing text..."
    )

    valid_records = []

    for record in records:

        text = clean_text(
            record.get(
                "text",
                ""
            )
        )

        if text:

            record = {
                **record,
                "text": text
            }

            valid_records.append(
                record
            )

    records = valid_records

    texts = [
        record["text"]
        for record in records
    ]

    print(
        f"      Valid text records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    print(
        "\n[3/6] Extracting keywords..."
    )

    keywords = extract_keywords(
        records
    )

    hashtags = extract_hashtags(
        records
    )

    tfidf_keywords = (
        extract_tfidf_keywords(
            records
        )
    )

    print(
        f"      Keywords found: "
        f"{len(keywords)}"
    )

    print(
        f"      Hashtags found: "
        f"{len(hashtags)}"
    )

    print(
        f"      TF-IDF terms: "
        f"{len(tfidf_keywords)}"
    )

    # --------------------------------------------------------
    # LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print(
        "\n[4/6] Loading semantic model..."
    )

    print(
        f"      {EMBEDDING_MODEL}"
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL
        )
    )

    print(
        "      Semantic model loaded."
    )

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = generate_embeddings(
        texts,
        embedding_model
    )

    # --------------------------------------------------------
    # TOPIC CLUSTERING
    # --------------------------------------------------------

    topic_count = determine_topic_count(
        len(records)
    )

    labels = cluster_topics(
        records,
        embeddings,
        topic_count
    )

    # --------------------------------------------------------
    # TOPIC ANALYSIS
    # --------------------------------------------------------

    print(
        "\n[5/6] Building topic intelligence..."
    )

    topics = build_topic_analysis(
        records,
        labels
    )

    # --------------------------------------------------------
    # TREND ANALYSIS
    # --------------------------------------------------------

    print(
        "\n[6/6] Calculating trends..."
    )

    trends = calculate_trends(
        records,
        labels
    )

    # --------------------------------------------------------
    # TOPIC OUTPUT
    # --------------------------------------------------------

    topic_output = {

        "total_records":
            len(records),

        "topic_count":
            topic_count,

        "keywords":
            keywords,

        "hashtags":
            hashtags,

        "tfidf_keywords":
            tfidf_keywords,

        "topics":
            topics
    }

    # --------------------------------------------------------
    # TREND OUTPUT
    # --------------------------------------------------------

    trend_output = {

        "total_records":
            len(records),

        "topic_count":
            topic_count,

        "trends":
            trends
    }

    save_json(
        TOPIC_OUTPUT,
        topic_output
    )

    save_json(
        TREND_OUTPUT,
        trend_output
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "              TOPIC ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nRecords analyzed: "
        f"{len(records)}"
    )

    print(
        f"Semantic topics: "
        f"{topic_count}"
    )

    print(
        "\nTop keywords:"
    )

    for item in keywords[:15]:

        print(
            f"  {item['keyword']:<25}"
            f"{item['mentions']}"
        )

    print(
        "\nTop hashtags:"
    )

    if hashtags:

        for item in hashtags[:15]:

            print(
                f"  #{item['hashtag']:<24}"
                f"{item['mentions']}"
            )

    else:

        print(
            "  No hashtags detected."
        )

    print(
        "\nTopics:"
    )

    for topic in topics:

        print(
            f"\n  Topic {topic['topic_id']}"
        )

        print(
            f"    Documents: "
            f"{topic['document_count']}"
        )

        print(
            f"    Percentage: "
            f"{topic['percentage']}%"
        )

        print(
            "    Keywords: "
            + ", ".join(
                topic["keywords"][:8]
            )
        )

        sentiment = topic[
            "sentiment"
        ]

        if sentiment:

            dominant_sentiment = max(
                sentiment.items(),
                key=lambda item:
                    item[1]["count"]
            )

            print(
                f"    Dominant sentiment: "
                f"{dominant_sentiment[0]}"
            )

    print(
        "\nTrend directions:"
    )

    rising = sum(
        1

        for trend in trends

        if trend[
            "trend_direction"
        ] == "rising"
    )

    stable = sum(
        1

        for trend in trends

        if trend[
            "trend_direction"
        ] == "stable"
    )

    declining = sum(
        1

        for trend in trends

        if trend[
            "trend_direction"
        ] == "declining"
    )

    print(
        f"  Rising:    {rising}"
    )

    print(
        f"  Stable:    {stable}"
    )

    print(
        f"  Declining: {declining}"
    )

    print(
        f"\nTopic data:"
        f"\n  {TOPIC_OUTPUT}"
    )

    print(
        f"\nTrend data:"
        f"\n  {TREND_OUTPUT}"
    )

    print()
    

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()