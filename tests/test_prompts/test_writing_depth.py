from autoreview.llm.prompts.writing import build_section_writing_prompt


def test_section_writing_prompt_includes_depth_instructions():
    prompt = build_section_writing_prompt(
        section_id="s1",
        section_title="Deep Learning",
        section_description="Overview of DL methods",
        outline_context="Full outline here",
        relevant_extractions="Extractions here",
        target_word_count=1200,
        depth_instructions="Exhaustively trace evidence chains. Target approximately 1200 words.",
    )
    assert "1200" in prompt
    assert "Exhaustively trace" in prompt
    assert "DEPTH AND LENGTH GUIDANCE" in prompt


def test_section_writing_prompt_without_depth_is_unchanged():
    prompt = build_section_writing_prompt(
        section_id="s1",
        section_title="Deep Learning",
        section_description="Overview",
        outline_context="Outline",
        relevant_extractions="Extractions",
    )
    assert "DEPTH AND LENGTH GUIDANCE" not in prompt
