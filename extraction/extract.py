import json
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SERVICE_ACCOUNT = (
    PROJECT_ROOT
    / "credentials"
    / "firebase-service-account.json"
)

APP_NAME = "sih-mock-app-html"


# Collections used by the mock social-media application
COLLECTIONS = [
    "profiles",
    "posts",
    "rooms",
    "messages",
    "notifications",
    "activity",
    "roomMessages",
    "reports",
]


# ============================================================
# FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase():
    if not SERVICE_ACCOUNT.exists():
        raise FileNotFoundError(
            f"""
Firebase service account not found:

{SERVICE_ACCOUNT}

Make sure your Firebase Admin SDK JSON file is located at:

credentials/firebase-service-account.json
"""
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(SERVICE_ACCOUNT))
        firebase_admin.initialize_app(cred)

    return firestore.client()


# ============================================================
# FIRESTORE COLLECTION REFERENCE
# ============================================================

def get_collection_ref(db, collection_name):
    """
    Returns:

    artifacts/
        sih-mock-app-html/
            public/
                data/
                    <collection_name>
    """

    return (
        db
        .collection("artifacts")
        .document(APP_NAME)
        .collection("public")
        .document("data")
        .collection(collection_name)
    )


# ============================================================
# VALUE CONVERSION
# ============================================================

def convert_value(value):

    # Firestore Timestamp / datetime-like values
    if hasattr(value, "isoformat"):
        return value.isoformat()

    # Dictionary
    if isinstance(value, dict):
        return {
            key: convert_value(val)
            for key, val in value.items()
        }

    # List
    if isinstance(value, list):
        return [
            convert_value(item)
            for item in value
        ]

    return value


# ============================================================
# COLLECTION EXTRACTION
# ============================================================

def extract_collection(db, collection_name):

    collection_ref = get_collection_ref(
        db,
        collection_name
    )

    documents = []

    for document in collection_ref.stream():

        document_data = document.to_dict()

        record = {
            "id": document.id,
            **convert_value(document_data)
        }

        documents.append(record)

    return documents


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("       SIH SOCIAL MEDIA DATA EXTRACTION")
    print("=" * 65)

    # --------------------------------------------------------
    # STEP 1 — Firebase connection
    # --------------------------------------------------------

    print("\n[1/3] Connecting to Firebase...")

    try:

        db = initialize_firebase()

        print("      Firebase connection successful!")

    except Exception as error:

        print("\nERROR: Firebase connection failed.")
        print(error)

        return

    # --------------------------------------------------------
    # STEP 2 — Extract Firestore collections
    # --------------------------------------------------------

    print("\n[2/3] Extracting Firestore collections...\n")

    extracted_data = {}

    for collection_name in COLLECTIONS:

        try:

            print(
                f"      {collection_name:<20}",
                end=""
            )

            records = extract_collection(
                db,
                collection_name
            )

            extracted_data[collection_name] = records

            print(
                f"{len(records):>6} documents"
            )

        except Exception as error:

            print("FAILED")

            print(
                f"      Error: {error}"
            )

            extracted_data[collection_name] = []

    # --------------------------------------------------------
    # STEP 3 — Save raw data
    # --------------------------------------------------------

    print("\n[3/3] Saving extracted data...")

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / "raw_social_data.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            extracted_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 65)
    print("                 EXTRACTION COMPLETE")
    print("=" * 65)

    print("\nOutput file:")
    print(f"  {output_file}")

    print("\nExtracted documents:")

    total_documents = 0

    for collection_name, records in extracted_data.items():

        count = len(records)

        total_documents += count

        print(
            f"  {collection_name:<20} {count:>6}"
        )

    print(
        f"\n  {'TOTAL':<20} {total_documents:>6}"
    )

    print("\nRaw Firebase data successfully extracted.")
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()