"""
Pinecone namespace naming, shared by the ingestion side (which writes a
profile's embeddings) and the RAG side (which reads them back). Keeping
this in one place guarantees both sides always agree on the name.

One Pinecone index is shared across all profiles; each profile gets its
own namespace within that index, which is Pinecone's built-in mechanism
for keeping different profiles' vectors from ever mixing in a query.
"""


def namespace_for_profile(profile_id: str) -> str:
    """
    Turn a profile_id into a Pinecone-safe namespace.

    Pinecone namespaces are plain strings with no strict charset
    requirement, but we still sanitize so arbitrary CMS profile_ids can't
    produce an empty or purely-whitespace namespace.
    """
    safe = "".join(c for c in profile_id if c.isalnum() or c in ("-", "_"))
    safe = safe or "default"
    return f"profile_{safe}"


# Backwards-compatible alias (previously this returned a Chroma collection
# name). Kept so any other callers don't need to change.
collection_name_for_profile = namespace_for_profile