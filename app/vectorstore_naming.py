"""
Chroma collection naming, shared by the ingestion side (which writes a
profile's embeddings) and the RAG side (which reads them back). Keeping
this in one place guarantees both sides always agree on the name.
"""


def collection_name_for_profile(profile_id: str) -> str:
    """
    Turn a profile_id into a Chroma-safe collection name.

    Chroma collection names must be 3-63 chars, alphanumeric/underscore/hyphen.
    We sanitize and prefix so arbitrary CMS profile_ids are always valid.
    """
    safe = "".join(c for c in profile_id if c.isalnum() or c in ("-", "_"))
    safe = safe or "default"
    name = f"profile_{safe}"
    return name[:63]