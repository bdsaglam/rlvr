def test_environment_loading():
    """Test if the MuSiQue environment can be loaded."""
    from vf_musique import load_environment

    env = load_environment(
        datasets_str="bdsaglam/musique,answerable,train[:100]",
        retriever="golden",  # Simplest retriever
    )

    tool_names = [tool.__name__ for tool in env.tools]
    assert len(tool_names) == 2, "Expected 2 tools"
    assert "retrieve_documents" in tool_names, "Expected retrieve_documents tool"
    assert "get_document" in tool_names, "Expected get_document tool"

    return True
