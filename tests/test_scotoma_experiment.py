from collections import Counter
from itertools import combinations

import pytest

from vlm_perception.models import Colour
from vlm_perception.scotoma.experiment import (
    BLUR_FRACTIONS,
    ENGLISH_POOL,
    MIN_POOL_HAMMING,
    POOLS,
    PSEUDO_POOL,
    STRING_LENGTH,
    generate_stimuli,
    hamming,
    legibility_conditions,
    ordered_pairs,
    scotoma_conditions,
)
from vlm_perception.scotoma.scoring import (
    bias_index,
    normalise,
    normalised_levenshtein,
    parse_transcriptions,
    positional_hamming,
    score_transcription,
)

# --- string pools ---


@pytest.mark.parametrize("pool", [ENGLISH_POOL, PSEUDO_POOL])
def test_pool_strings_length_matched_uppercase_spacefree(pool):
    assert len(pool) == 8
    for s in pool:
        assert len(s) == STRING_LENGTH
        assert s == s.upper()
        assert s.isalpha()


@pytest.mark.parametrize("pool", [ENGLISH_POOL, PSEUDO_POOL])
def test_pool_min_pairwise_hamming(pool):
    for a, b in combinations(pool, 2):
        assert hamming(a, b) >= MIN_POOL_HAMMING


# --- pairing and counterbalancing ---


def test_ordered_pairs_every_string_once_per_role():
    pairs = ordered_pairs(ENGLISH_POOL)
    assert len(pairs) == 8
    assert Counter(p[0] for p in pairs) == Counter(ENGLISH_POOL)
    assert Counter(p[1] for p in pairs) == Counter(ENGLISH_POOL)


def test_ordered_pairs_colour_counterbalanced():
    pairs = ordered_pairs(ENGLISH_POOL)
    colours = Counter(p[2] for p in pairs)
    assert colours[Colour.red] == 4
    assert colours[Colour.cyan] == 4


def test_ordered_pairs_reciprocal():
    pairs = ordered_pairs(ENGLISH_POOL)
    keys = {(a, b) for a, b, _ in pairs}
    assert all((b, a) in keys for a, b in keys)


# --- condition generation ---


def test_condition_count_192():
    assert len(scotoma_conditions()) == 192


def test_conditions_cover_factors():
    conditions = scotoma_conditions()
    assert {c.blur_fraction for c in conditions} == set(BLUR_FRACTIONS)
    assert {c.blurred_on_top for c in conditions} == {True, False}
    assert {c.pool for c in conditions} == set(POOLS)
    assert all(c.offset_fraction == 0.38 for c in conditions)


def test_condition_filenames_unique():
    conditions = scotoma_conditions() + legibility_conditions()
    names = [c.image_filename for c in conditions]
    assert len(names) == len(set(names))


def test_condition_colour_roles_complementary():
    for c in scotoma_conditions():
        assert {c.colour_real, c.colour_robot} == {Colour.red, Colour.cyan}


def test_d_pair_recorded():
    for c in scotoma_conditions():
        assert c.d_pair is not None
        assert c.d_pair >= MIN_POOL_HAMMING


def test_legibility_conditions_solo():
    conditions = legibility_conditions()
    assert len(conditions) == 16
    assert all(c.is_solo for c in conditions)
    assert all(c.blur_fraction == 0.0 for c in conditions)
    colours = Counter(c.colour_real for c in conditions)
    assert colours[Colour.red] == 8
    assert colours[Colour.cyan] == 8


def test_generate_stimuli_renders_files(tmp_path):
    conditions = scotoma_conditions()[:2] + legibility_conditions()[:1]
    paths = generate_stimuli(tmp_path, conditions)
    assert len(paths) == 3
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# --- scoring ---


def test_normalise_strips_nonletters():
    assert normalise("The text says: 'ABSOLUTELY!'") == "THETEXTSAYSABSOLUTELY"


def test_levenshtein_known_values():
    assert normalised_levenshtein("ABC", "ABC") == 0.0
    assert normalised_levenshtein("ABC", "ABD") == pytest.approx(1 / 3)
    assert normalised_levenshtein("ABC", "") == 1.0
    assert normalised_levenshtein("ABCD", "BCD") == pytest.approx(1 / 4)


def test_positional_hamming_known_values():
    assert positional_hamming("ABC", "ABC") == 0.0
    assert positional_hamming("ABC", "ABD") == pytest.approx(1 / 3)
    # deletion shifts every position: high hamming, low levenshtein
    assert positional_hamming("ABCD", "BCD") == 1.0


def test_bias_index_sign_convention():
    # transcription matches robot exactly: d_real > 0, d_robot = 0 -> +1
    assert bias_index(0.8, 0.0) == 1.0
    assert bias_index(0.0, 0.8) == -1.0
    assert bias_index(0.0, 0.0) == 0.0
    assert bias_index(0.5, 0.5) == 0.0


def test_parse_transcriptions_json_text():
    assert parse_transcriptions('{"text": "everything"}') == ["EVERYTHING"]


def test_parse_transcriptions_json_messages():
    parsed = parse_transcriptions('{"messages": ["ABSOLUTELY", "background"]}')
    assert parsed == ["ABSOLUTELY", "BACKGROUND"]


def test_parse_transcriptions_json_after_reasoning():
    raw = 'Let me look closely.\nThe letters are B, A...\n{"text": "BACKGROUND"}'
    assert parse_transcriptions(raw) == ["BACKGROUND"]


def test_parse_transcriptions_bare_word_fallback():
    assert parse_transcriptions("EVERYTHING") == ["EVERYTHING"]


def test_parse_transcriptions_empty():
    assert parse_transcriptions("") == []


def test_score_transcription_reads_robot_stream():
    score = score_transcription('{"text": "BACKGROUND"}', "ABSOLUTELY", "BACKGROUND")
    assert score.dist_robot_lev == 0.0
    assert score.dist_real_lev > 0
    assert score.bias_index_lev == 1.0
    assert score.bias_index_ham == 1.0


def test_score_transcription_reads_real_stream():
    score = score_transcription('{"text": "ABSOLUTELY"}', "ABSOLUTELY", "BACKGROUND")
    assert score.bias_index_lev == -1.0


def test_score_transcription_both_streams_recovered():
    score = score_transcription(
        '{"messages": ["ABSOLUTELY", "BACKGROUND"]}', "ABSOLUTELY", "BACKGROUND"
    )
    assert score.dist_real_lev == 0.0
    assert score.dist_robot_lev == 0.0
    assert score.bias_index_lev == 0.0


def test_score_transcription_solo():
    score = score_transcription('{"text": "ABSOLUTELY"}', "ABSOLUTELY", "")
    assert score.dist_real_lev == 0.0
    assert score.dist_robot_lev is None
    assert score.bias_index_lev is None


def test_score_transcription_unparseable():
    score = score_transcription("", "ABSOLUTELY", "BACKGROUND")
    assert score.raw_transcription is None
    assert score.bias_index_lev is None
