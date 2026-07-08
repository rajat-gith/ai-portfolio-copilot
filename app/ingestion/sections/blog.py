from typing import List

from langchain_core.documents import Document

from ...clients.cms_client import CMSClient
from ..normalize import normalize_list_response, safe_get, safe_list, truncate_text

ENDPOINT = "public/blog"
SECTION = "blog"
CONTENT_TRUNCATE_LIMIT = 1200


def build_documents(cms_client: CMSClient, profile_id: str) -> List[Document]:
    """Build one Document per blog entry.

    Expected raw shape:
    {
      "_id", "title", "content", "tags",
      "author": { "name", "userId" },
      "coverImage", "isPublished",
      "publishedAt", "createdAt", "updatedAt"
    }
    """
    raw = cms_client.fetch(ENDPOINT)
    entries = normalize_list_response(raw)

    docs: List[Document] = []
    for blog in entries:
        tags_str = safe_list(blog, "tags")
        is_published = bool(blog.get("isPublished", False))

        author = blog.get("author", {}) or {}
        author_name = safe_get(author, "name")
        author_user_id = safe_get(author, "userId")

        content = truncate_text(safe_get(blog, "content"), CONTENT_TRUNCATE_LIMIT)

        text = (
            f"Blog Title: {safe_get(blog, 'title')}\n"
            f"Author: {author_name} (User ID: {author_user_id})\n"
            f"Published: {'Yes' if is_published else 'No'}\n"
            f"Published On: {safe_get(blog, 'publishedAt')}\n"
            f"Tags: {tags_str}\n"
            f"Content: {content}\n"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "section": SECTION,
                    "profile_id": profile_id,
                    "blog_id": str(blog.get("_id", "")),
                    "is_published": is_published,
                },
            )
        )

    return docs
