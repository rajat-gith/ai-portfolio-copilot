from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about a single user's profile.
    You will be given:
    - a QUESTION (what the user asked)
    - a CONTEXT (one or more profile excerpts from the user's data)

    IMPORTANT RULES — follow exactly:
    1. Treat the CONTEXT as authoritative facts about the user. Do NOT invent new facts.
    2. Answer the question directly (one or two short sentences) if the answer is present in CONTEXT.
    3. After your answer, add a small "SOURCES:" section listing the section types and ids used (e.g. "education: 4172dff...").
    4. If the required fact is not present, say "I don't know from the provided profile data."

    QUESTION:
    {question}

    CONTEXT:
    {context}

    Answer now, then list SOURCES (section: id). Be concise.
    """
)
