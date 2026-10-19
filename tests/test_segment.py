from app import data
from app.segment import segment


def test_gold_spans_are_dev_segments():
    answers = {a["answer_id"]: a["ai_answer"] for a in data.load_answers()}
    gold = data.read_csv("development_gold.csv")  # gold loaded only here, never by app code under test
    spans = {(aid, s, e) for aid, text in answers.items() for _, s, e, _ in segment(text)}
    for g in gold:
        s, e = int(g["start_char"]), int(g["end_char"])
        assert (g["answer_id"], s, e) in spans, g["claim_id"]
        assert answers[g["answer_id"]][s:e] == g["quote"]
    assert len(gold) == 376


def test_segment_counts():
    by_split = {"development": 0, "check": 0}
    for a in data.load_answers():
        by_split[a["split"]] += len(segment(a["ai_answer"]))
    assert by_split == {"development": 659, "check": 166}


def test_no_leftover_text():
    for a in data.load_answers():
        text = a["ai_answer"]
        chars = list(text)
        for _, s, e, sent in segment(text):
            assert text[s:e] == sent
            chars[s:e] = [" "] * (e - s)
        assert "".join(chars).strip() == "", a["answer_id"]


def test_indices_sequential():
    segs = segment("One. Two!\nThree?")
    assert segs == [(0, 0, 4, "One."), (1, 5, 9, "Two!"), (2, 10, 16, "Three?")]
