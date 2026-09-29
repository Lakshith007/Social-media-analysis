import firebase_admin
from firebase_admin import credentials, firestore


SERVICE_ACCOUNT = (
    "credentials/firebase-service-account.json"
)


def get_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred)

    return firestore.client()