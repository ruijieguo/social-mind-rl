from scripts.data.merge_and_dedupe import jaccard_4gram, build_minhash_index


def test_jaccard_identical():
    assert jaccard_4gram("hello world", "hello world") == 1.0


def test_jaccard_disjoint():
    assert jaccard_4gram("hello world", "zzzzzzzz") < 0.1


def test_jaccard_partial_overlap():
    s = jaccard_4gram("the quick brown fox", "the quick brown dog")
    assert 0.2 < s < 0.9


def test_minhash_index_finds_near_duplicates():
    corpus = [
        ("a", "the cat sat on the mat"),
        ("b", "a different sentence entirely about dogs"),
        ("c", "the cat sat on the rug"),  # near-dup of a
    ]
    index = build_minhash_index(corpus)
    # Querying 'a' should return 'c' as candidate
    candidates = index.query("a", "the cat sat on the mat")
    assert "c" in candidates
    assert "b" not in candidates
