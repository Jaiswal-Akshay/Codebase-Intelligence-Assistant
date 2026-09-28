"""Lesson 2: improve retrieval by scoring meaningful word matches."""

import re

documents = [
    {
        "name": "authentication.py",
        "text": "The login function checks a user's email and password."
    },
    {
        "name": "orders.py",
        "text": "The order service creates orders and calculates the total price."
    },
    {
        "name": "database.py",
        "text": "The database module connects the application to PostgreSQL."
    },
]

STOP_WORDS = {
    "a", "an", "and", "are", "at", "be", "but", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "of", "on", "or",
    "the", "this", "to", "us", "what", "where", "which", "with",
}


def words(text):
    all_words = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return all_words - STOP_WORDS


def retrieve(question, documents):
    question_words = words(question)
    results = []

    for document in documents:
        document_words = words(document["text"])
        matching_words = question_words.intersection(document_words)

        if matching_words:
            results.append({
                **document,
                "matching_words": ", ".join(sorted(matching_words)),
                "score": len(matching_words),
            })

    return sorted(results, key=lambda result: result["score"], reverse=True)


question = input("Ask a question about the project: ")
matches = retrieve(question, documents)

if matches:
    print("\nRelevant project information:\n")

    for match in matches:
        print(f"- {match['name']}: {match['text']}")
        print(f"  Score: {match['score']}")
        print(f"  Matching words: {match['matching_words']}")
else:
    print("\nI could not find relevant information.")