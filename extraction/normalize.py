import json
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "raw_social_data.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# HELPERS
# ============================================================

def load_raw_data():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found:\n{RAW_FILE}\n\n"
            "Run extraction first."
        )

    with RAW_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_json(filename, data):
    output_file = OUTPUT_DIR / filename

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# USERS
# ============================================================

def normalize_users(raw):
    users = []

    for profile in raw.get("profiles", []):

        user = {
            "user_id": profile.get("id"),
            "username": profile.get("username"),
            "bio": profile.get("bio"),
            "verified": profile.get("verified", False),

            "followers": profile.get(
                "followers",
                []
            ),

            "following": profile.get(
                "following",
                []
            ),

            "joined_at": profile.get(
                "joinedAt"
            ),

            "last_seen": profile.get(
                "lastSeen"
            ),

            "online": profile.get(
                "online",
                False
            )
        }

        users.append(user)

    return users


# ============================================================
# POSTS
# ============================================================

def normalize_posts(raw):

    posts = []

    for post in raw.get("posts", []):

        hashtags = post.get(
            "hashtags",
            []
        )

        # Some applications store hashtags as a string.
        if isinstance(hashtags, str):
            hashtags = [
                tag.strip()
                for tag in hashtags.split()
                if tag.strip()
            ]

        post_record = {
            "post_id": post.get("id"),

            "author_id": post.get(
                "authorId"
            ),

            "author_name": post.get(
                "authorName"
            ),

            "description": post.get(
                "description",
                ""
            ),

            "hashtags": hashtags,

            "created_at": post.get(
                "createdAt"
            ),

            "room_id": post.get(
                "roomId"
            ),

            "liked_by": post.get(
                "likedBy",
                []
            ),

            "reactions": post.get(
                "reactions",
                {}
            ),

            "comments": post.get(
                "comments",
                []
            ),

            "poll": post.get(
                "poll"
            ),

            "media_count": len(
                post.get(
                    "imageBase64s",
                    []
                )
            )
        }

        posts.append(post_record)

    return posts


# ============================================================
# COMMENTS
# ============================================================

def normalize_comments(raw):

    comments = []

    for post in raw.get("posts", []):

        post_id = post.get("id")

        post_comments = post.get(
            "comments",
            []
        )

        for comment in post_comments:

            comment_record = {
                "comment_id": comment.get(
                    "id"
                ),

                "post_id": post_id,

                "author_id": comment.get(
                    "authorId"
                ),

                "author_name": comment.get(
                    "authorName"
                ),

                "text": comment.get(
                    "text",
                    ""
                ),

                "liked_by": comment.get(
                    "likedBy",
                    []
                ),

                "replies": comment.get(
                    "replies",
                    []
                ),

                "created_at": comment.get(
                    "createdAt"
                )
            }

            comments.append(comment_record)

    return comments


# ============================================================
# FOLLOW RELATIONSHIPS
# ============================================================

def normalize_follow_relationships(raw):

    relationships = []

    for profile in raw.get(
        "profiles",
        []
    ):

        user_id = profile.get("id")

        following = profile.get(
            "following",
            []
        )

        for target_id in following:

            relationships.append(
                {
                    "source": user_id,
                    "target": target_id,
                    "relationship": "follows"
                }
            )

    return relationships


# ============================================================
# LIKES / REACTIONS
# ============================================================

def normalize_post_interactions(raw):

    interactions = []

    for post in raw.get(
        "posts",
        []
    ):

        post_id = post.get("id")

        # ----------------------------
        # Likes
        # ----------------------------

        for user_id in post.get(
            "likedBy",
            []
        ):

            interactions.append(
                {
                    "user_id": user_id,
                    "target_id": post_id,
                    "interaction": "like",
                    "created_at": post.get(
                        "createdAt"
                    )
                }
            )

        # ----------------------------
        # Reactions
        # ----------------------------

        reactions = post.get(
            "reactions",
            {}
        )

        if isinstance(
            reactions,
            dict
        ):

            for reaction_type, users in reactions.items():

                if isinstance(
                    users,
                    list
                ):

                    for user_id in users:

                        interactions.append(
                            {
                                "user_id": user_id,
                                "target_id": post_id,
                                "interaction": reaction_type,
                                "created_at": post.get(
                                    "createdAt"
                                )
                            }
                        )

    return interactions


# ============================================================
# ROOMS
# ============================================================

def normalize_rooms(raw):

    rooms = []

    for room in raw.get(
        "rooms",
        []
    ):

        rooms.append(
            {
                "room_id": room.get(
                    "id"
                ),

                "name": room.get(
                    "name"
                ),

                "description": room.get(
                    "description"
                ),

                "created_by": room.get(
                    "createdBy"
                ),

                "members": room.get(
                    "members",
                    []
                ),

                "moderators": room.get(
                    "moderators",
                    []
                ),

                "private": room.get(
                    "private",
                    False
                ),

                "pinned_posts": room.get(
                    "pinnedPosts",
                    []
                ),

                "created_at": room.get(
                    "createdAt"
                )
            }
        )

    return rooms


# ============================================================
# ROOM MESSAGES
# ============================================================

def normalize_room_messages(raw):

    messages = []

    for message in raw.get(
        "roomMessages",
        []
    ):

        messages.append(
            {
                "message_id": message.get(
                    "id"
                ),

                "room_id": message.get(
                    "roomId"
                ),

                "author_id": message.get(
                    "authorId"
                ),

                "text": message.get(
                    "text",
                    ""
                ),

                "created_at": message.get(
                    "createdAt"
                )
            }
        )

    return messages


# ============================================================
# ACTIVITY
# ============================================================

def normalize_activity(raw):

    activities = []

    for activity in raw.get(
        "activity",
        []
    ):

        activities.append(
            {
                "activity_id": activity.get(
                    "id"
                ),

                "type": activity.get(
                    "type"
                ),

                "text": activity.get(
                    "text"
                ),

                "actor_id": activity.get(
                    "actorId"
                ),

                "created_at": activity.get(
                    "createdAt"
                )
            }
        )

    return activities


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 65)
    print("          SIH SOCIAL DATA NORMALIZATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    print("\n[1/3] Loading raw Firebase data...")

    try:
        raw = load_raw_data()
    except Exception as error:
        print(f"\nERROR: {error}")
        return

    print("      Raw data loaded successfully.")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("\n[2/3] Normalizing datasets...\n")

    datasets = {

        "users.json":
            normalize_users(raw),

        "posts.json":
            normalize_posts(raw),

        "comments.json":
            normalize_comments(raw),

        "follow_relationships.json":
            normalize_follow_relationships(raw),

        "post_interactions.json":
            normalize_post_interactions(raw),

        "rooms.json":
            normalize_rooms(raw),

        "room_messages.json":
            normalize_room_messages(raw),

        "activity.json":
            normalize_activity(raw),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("[3/3] Saving processed datasets...\n")

    total_records = 0

    for filename, data in datasets.items():

        output_file = save_json(
            filename,
            data
        )

        count = len(data)

        total_records += count

        print(
            f"      {filename:<32}"
            f"{count:>6} records"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("              NORMALIZATION COMPLETE")
    print("=" * 65)

    print(
        f"\nTotal processed records: "
        f"{total_records}"
    )

    print(
        f"\nOutput directory:\n"
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nRaw Firebase data has been preserved."
    )

    print(
        "Processed datasets are ready for AI analytics."
    )

    print()


if __name__ == "__main__":
    main()