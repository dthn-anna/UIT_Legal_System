from app.generation.prompt import build_user_prompt
from app.models.schemas import SourceResult


def test_prompt_contains_citation_metadata():
    source = SourceResult(
        passage_id="p1",
        document="Quy chế đào tạo",
        article="Điều 10",
        content="Nội dung quy định.",
        score=0.03,
    )

    prompt = build_user_prompt("Câu hỏi?", [source])

    assert "Quy chế đào tạo" in prompt
    assert "Điều 10" in prompt
    assert "Nội dung quy định." in prompt
    assert "Câu hỏi?" in prompt
