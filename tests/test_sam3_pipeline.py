import numpy as np
import pytest

from models.sam3_pipeline import decode_inputs, postprocess_proposals


def test_every_proposal_retains_its_mask_and_presence_adjusted_score():
    masks = np.array([[[[-1, 0, 1]], [[2, -2, 0]], [[-1, -2, -3]]]], dtype=np.float32)
    matching = np.log(np.array([[[[1 / 3], [1], [3]]]], dtype=np.float32))
    presence = np.log(np.array([[[3]]], dtype=np.float32))
    result = postprocess_proposals(masks, matching, presence)
    np.testing.assert_array_equal(result["masks"], [[[0, 0, 1]], [[1, 0, 0]], [[0, 0, 0]]])
    np.testing.assert_allclose(result["scores"], [0.1875, 0.375, 0.5625], atol=1e-7)
    assert result["masks"].dtype == np.uint8
    assert result["scores"].dtype == np.float32


def test_only_final_decoder_layer_is_scored_without_sorting():
    masks = np.ones((1, 2, 1, 1), dtype=np.float32)
    matching = np.array([[[[100], [-100]]], [[[-100], [100]]]], dtype=np.float32)
    presence = np.array([[[100]], [[0]]], dtype=np.float32)
    result = postprocess_proposals(masks, matching, presence)
    np.testing.assert_allclose(result["scores"], [0, 0.5], atol=1e-7)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_output_is_not_returned(bad):
    with pytest.raises(ValueError):
        postprocess_proposals(np.array([[[[bad]]]]), np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1)))


def test_mismatched_proposal_count_fails_instead_of_pairing_wrong_scores():
    with pytest.raises(ValueError):
        postprocess_proposals(np.zeros((1, 2, 1, 1)), np.zeros((1, 1, 3, 1)), np.zeros((1, 1, 1)))


def test_utf8_prompt_roundtrips_without_character_loss():
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    decoded_image, prompt = decode_inputs({"image": image, "prompt": np.frombuffer("café bag".encode(), dtype=np.uint8)})
    assert prompt == "café bag"
    np.testing.assert_array_equal(decoded_image, image)


@pytest.mark.parametrize("prompt", [b"", b"\xff", b"   "])
def test_invalid_prompt_bytes_are_rejected(prompt):
    with pytest.raises(ValueError):
        decode_inputs({"image": np.zeros((3, 5, 3), dtype=np.uint8), "prompt": np.frombuffer(prompt, dtype=np.uint8)})


@pytest.mark.parametrize("image", [np.zeros((3, 5), dtype=np.uint8), np.zeros((3, 5, 4), dtype=np.uint8),
                                    np.zeros((0, 5, 3), dtype=np.uint8), np.zeros((3, 5, 3), dtype=np.float32)])
def test_invalid_image_contract_is_rejected(image):
    with pytest.raises(ValueError):
        decode_inputs({"image": image, "prompt": np.frombuffer(b"bag", dtype=np.uint8)})
