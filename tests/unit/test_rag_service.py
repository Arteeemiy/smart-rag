from smart_rag.services.rag_service import RAGService


class DummyRetrievalService:
    def retrieve_context(self, query: str, top_k: int | None = None):
        return "ctx", [{"section": "A", "distance": 1.0}], "retrieved"


class DummyLLMService:
    def generate_response(self, context: str, question: str):
        return "answer"


def test_rag_service_returns_answer():
    service = RAGService(DummyRetrievalService(), DummyLLMService())
    result = service.ask("test")
    assert result["answer"] == "answer"
    assert result["has_context"] is True
