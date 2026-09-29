import json
from pathlib import Path
from collections import Counter

import torch
from transformers import pipeline


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR = PROCESSED_DIR

POSTS_FILE = PROCESSED_DIR / "posts.json"
COMMENTS_FILE = PROCESSED_DIR / "comments.json"
ROOM_MESSAGES_FILE = PROCESSED_DIR / "room_messages.json"


# ============================================================
# MODELS
# ============================================================

SENTIMENT_MODEL = (
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

EMOTION_MODEL = (
    "SamLowe/roberta-base-go_emotions"
)


# ============================================================
# GPU CONFIGURATION
# ============================================================

if torch.cuda.is_available():

    DEVICE = 0
    DEVICE_NAME = torch.cuda.get_device_name(0)

else:

    DEVICE = -1
    DEVICE_NAME = "CPU"


# RTX 3050 6 GB
# Start conservatively with 16.
# If GPU memory is comfortable, you can later try 32.
BATCH_SIZE = 16


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:

        return ""

    return " ".join(
        str(text)
        .strip()
        .split()
    )


# ============================================================
# BUILD TEXT DATASET
# ============================================================

def build_text_dataset():

    posts = load_json(
        POSTS_FILE
    )

    comments = load_json(
        COMMENTS_FILE
    )

    room_messages = load_json(
        ROOM_MESSAGES_FILE
    )

    texts = []

    # --------------------------------------------------------
    # POSTS
    # --------------------------------------------------------

    for post in posts:

        text = clean_text(
            post.get("description")
        )

        if text:

            texts.append(
                {
                    "source_type": "post",

                    "source_id": post.get(
                        "post_id"
                    ),

                    "author_id": post.get(
                        "author_id"
                    ),

                    "text": text,

                    "created_at": post.get(
                        "created_at"
                    ),
                }
            )

    # --------------------------------------------------------
    # COMMENTS
    # --------------------------------------------------------

    for comment in comments:

        text = clean_text(
            comment.get("text")
        )

        if text:

            texts.append(
                {
                    "source_type": "comment",

                    "source_id": comment.get(
                        "comment_id"
                    ),

                    "author_id": comment.get(
                        "author_id"
                    ),

                    "text": text,

                    "created_at": comment.get(
                        "created_at"
                    ),
                }
            )

    # --------------------------------------------------------
    # ROOM MESSAGES
    # --------------------------------------------------------

    for message in room_messages:

        text = clean_text(
            message.get("text")
        )

        if text:

            texts.append(
                {
                    "source_type": "room_message",

                    "source_id": message.get(
                        "message_id"
                    ),

                    "author_id": message.get(
                        "author_id"
                    ),

                    "text": text,

                    "created_at": message.get(
                        "created_at"
                    ),
                }
            )

    return texts


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(
    texts,
    classifier,
    batch_size=BATCH_SIZE
):

    results = []

    total = len(texts)

    print(
        f"\nAnalyzing sentiment "
        f"using {DEVICE_NAME}..."
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        batch_texts = [
            item["text"]
            for item in batch
        ]

        try:

            predictions = classifier(
                batch_texts,
                truncation=True,
                max_length=512
            )

            for item, prediction in zip(
                batch,
                predictions
            ):

                results.append(
                    {
                        **item,

                        "sentiment":
                            prediction["label"],

                        "sentiment_score":
                            round(
                                float(
                                    prediction[
                                        "score"
                                    ]
                                ),
                                4
                            ),
                    }
                )

        except Exception as error:

            print(
                f"\nSentiment batch error: "
                f"{error}"
            )

            # Fall back to individual processing
            for item in batch:

                try:

                    prediction = classifier(
                        item["text"],
                        truncation=True,
                        max_length=512
                    )[0]

                    results.append(
                        {
                            **item,

                            "sentiment":
                                prediction[
                                    "label"
                                ],

                            "sentiment_score":
                                round(
                                    float(
                                        prediction[
                                            "score"
                                        ]
                                    ),
                                    4
                                ),
                        }
                    )

                except Exception as item_error:

                    print(
                        f"\nIndividual "
                        f"sentiment error: "
                        f"{item_error}"
                    )

                    results.append(
                        {
                            **item,

                            "sentiment":
                                "unknown",

                            "sentiment_score":
                                0.0,
                        }
                    )

        processed = min(
            start + batch_size,
            total
        )

        print(
            f"\r      Processed "
            f"{processed}/{total}",
            end=""
        )

    print()

    return results


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(
    texts,
    classifier,
    batch_size=BATCH_SIZE
):

    results = []

    total = len(texts)

    print(
        f"\nAnalyzing emotions "
        f"using {DEVICE_NAME}..."
    )

    for start in range(
        0,
        total,
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]

        batch_texts = [
            item["text"]
            for item in batch
        ]

        try:

            predictions = classifier(
                batch_texts,
                truncation=True,
                max_length=512,
                top_k=3
            )

            for item, prediction in zip(
                batch,
                predictions
            ):

                # Depending on the Transformers
                # version, batched output can
                # occasionally be nested.

                if (
                    prediction
                    and isinstance(
                        prediction[0],
                        list
                    )
                ):

                    prediction = prediction[0]

                emotions = []

                for emotion in prediction:

                    emotions.append(
                        {
                            "label":
                                emotion["label"],

                            "score":
                                round(
                                    float(
                                        emotion[
                                            "score"
                                        ]
                                    ),
                                    4
                                ),
                        }
                    )

                if emotions:

                    primary = emotions[0]

                else:

                    primary = {
                        "label": "unknown",
                        "score": 0.0,
                    }

                results.append(
                    {
                        **item,

                        "emotion":
                            primary["label"],

                        "emotion_score":
                            primary["score"],

                        "emotion_distribution":
                            emotions,
                    }
                )

        except Exception as error:

            print(
                f"\nEmotion batch error: "
                f"{error}"
            )

            # Fall back to individual processing
            for item in batch:

                try:

                    prediction = classifier(
                        item["text"],
                        truncation=True,
                        max_length=512,
                        top_k=3
                    )

                    if (
                        prediction
                        and isinstance(
                            prediction[0],
                            list
                        )
                    ):

                        prediction = prediction[0]

                    emotions = []

                    for emotion in prediction:

                        emotions.append(
                            {
                                "label":
                                    emotion[
                                        "label"
                                    ],

                                "score":
                                    round(
                                        float(
                                            emotion[
                                                "score"
                                            ]
                                        ),
                                        4
                                    ),
                            }
                        )

                    if emotions:

                        primary = emotions[0]

                    else:

                        primary = {
                            "label": "unknown",
                            "score": 0.0,
                        }

                    results.append(
                        {
                            **item,

                            "emotion":
                                primary["label"],

                            "emotion_score":
                                primary["score"],

                            "emotion_distribution":
                                emotions,
                        }
                    )

                except Exception as item_error:

                    print(
                        f"\nIndividual "
                        f"emotion error: "
                        f"{item_error}"
                    )

                    results.append(
                        {
                            **item,

                            "emotion":
                                "unknown",

                            "emotion_score":
                                0.0,

                            "emotion_distribution":
                                [],
                        }
                    )

        processed = min(
            start + batch_size,
            total
        )

        print(
            f"\r      Processed "
            f"{processed}/{total}",
            end=""
        )

    print()

    return results


# ============================================================
# COMBINE RESULTS
# ============================================================

def combine_results(
    sentiment_results,
    emotion_results
):

    # Use source_type + source_id as the
    # unique identifier.
    #
    # This is safer than source_id alone because
    # different source types could theoretically
    # contain the same ID.

    emotion_lookup = {
        (
            item["source_type"],
            item["source_id"]
        ): item

        for item in emotion_results
    }

    combined = []

    for sentiment in sentiment_results:

        key = (
            sentiment["source_type"],
            sentiment["source_id"]
        )

        emotion = emotion_lookup.get(
            key
        )

        record = {
            **sentiment,

            "emotion": (
                emotion["emotion"]
                if emotion
                else "unknown"
            ),

            "emotion_score": (
                emotion["emotion_score"]
                if emotion
                else 0.0
            ),

            "emotion_distribution": (
                emotion[
                    "emotion_distribution"
                ]
                if emotion
                else []
            ),
        }

        combined.append(
            record
        )

    return combined


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results):

    sentiment_counts = Counter(
        item["sentiment"]
        for item in results
    )

    emotion_counts = Counter(
        item["emotion"]
        for item in results
    )

    source_counts = Counter(
        item["source_type"]
        for item in results
    )

    return {

        "total_texts":
            len(results),

        "source_distribution":
            dict(source_counts),

        "sentiment_distribution":
            dict(sentiment_counts),

        "emotion_distribution":
            dict(emotion_counts),
    }


# ============================================================
# PRINT GPU INFORMATION
# ============================================================

def print_device_information():

    print()

    print(
        "Device information:"
    )

    print(
        f"  PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"  CUDA compiled: "
        f"{torch.version.cuda}"
    )

    print(
        f"  CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"  GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"  VRAM: "
            f"{round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)} GB"
        )

    else:

        print(
            "  GPU: Not available"
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 65)

    print(
        "           SIH AI TEXT INTELLIGENCE ENGINE"
    )

    print("=" * 65)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print_device_information()

    # --------------------------------------------------------
    # BUILD TEXT DATASET
    # --------------------------------------------------------

    print(
        "[1/5] Loading normalized datasets..."
    )

    try:

        texts = build_text_dataset()

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        return

    print(
        f"      Text records found: "
        f"{len(texts)}"
    )

    if not texts:

        print(
            "\nNo text records available."
        )

        return

    # --------------------------------------------------------
    # SENTIMENT MODEL
    # --------------------------------------------------------

    print(
        "\n[2/5] Loading sentiment model..."
    )

    print(
        f"      {SENTIMENT_MODEL}"
    )

    sentiment_classifier = pipeline(
        "sentiment-analysis",

        model=SENTIMENT_MODEL,

        tokenizer=SENTIMENT_MODEL,

        device=DEVICE
    )

    print(
        "      Sentiment model loaded."
    )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    print(
        "\n[3/5] Running sentiment analysis..."
    )

    sentiment_results = analyze_sentiment(
        texts,

        sentiment_classifier,

        batch_size=BATCH_SIZE
    )

    # --------------------------------------------------------
    # RELEASE UNUSED CUDA MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # EMOTION MODEL
    # --------------------------------------------------------

    print(
        "\n[4/5] Loading emotion model..."
    )

    print(
        f"      {EMOTION_MODEL}"
    )

    emotion_classifier = pipeline(
        "text-classification",

        model=EMOTION_MODEL,

        tokenizer=EMOTION_MODEL,

        top_k=3,

        device=DEVICE
    )

    print(
        "      Emotion model loaded."
    )

    # --------------------------------------------------------
    # EMOTION
    # --------------------------------------------------------

    emotion_results = analyze_emotion(
        texts,

        emotion_classifier,

        batch_size=BATCH_SIZE
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    print(
        "\n[5/5] Combining analysis results..."
    )

    results = combine_results(
        sentiment_results,

        emotion_results
    )

    summary = create_summary(
        results
    )

    # --------------------------------------------------------
    # SAVE ENRICHED DATA
    # --------------------------------------------------------

    enriched_file = (
        OUTPUT_DIR
        / "enriched_text.json"
    )

    with enriched_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,

            file,

            indent=2,

            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_file = (
        OUTPUT_DIR
        / "text_analysis_summary.json"
    )

    with summary_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,

            file,

            indent=2,

            ensure_ascii=False
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()

    print("=" * 65)

    print(
        "             TEXT ANALYSIS COMPLETE"
    )

    print("=" * 65)

    print(
        f"\nAnalyzed texts: "
        f"{summary['total_texts']}"
    )

    # --------------------------------------------------------
    # SOURCE DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nSource distribution:"
    )

    for label, count in sorted(
        summary[
            "source_distribution"
        ].items(),

        key=lambda item: item[1],

        reverse=True
    ):

        print(
            f"  {label:<20} {count}"
        )

    # --------------------------------------------------------
    # SENTIMENT
    # --------------------------------------------------------

    print(
        "\nSentiment:"
    )

    for label, count in sorted(
        summary[
            "sentiment_distribution"
        ].items(),

        key=lambda item: item[1],

        reverse=True
    ):

        print(
            f"  {label:<15} {count}"
        )

    # --------------------------------------------------------
    # EMOTION
    # --------------------------------------------------------

    print(
        "\nEmotion:"
    )

    for label, count in sorted(
        summary[
            "emotion_distribution"
        ].items(),

        key=lambda item: item[1],

        reverse=True
    ):

        print(
            f"  {label:<20} {count}"
        )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print(
        f"\nEnriched data:"
        f"\n  {enriched_file}"
    )

    print(
        f"\nSummary:"
        f"\n  {summary_file}"
    )

    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        print(
            "\nGPU memory:"
        )

        print(
            f"  Allocated: "
            f"{allocated:.2f} GB"
        )

        print(
            f"  Reserved:  "
            f"{reserved:.2f} GB"
        )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()