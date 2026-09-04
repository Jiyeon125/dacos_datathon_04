from src.data_loader import load_bundle
from src.validation import assert_valid_bundle


def test_committed_deployment_data_matches_contract():
    bundle = load_bundle()
    report = assert_valid_bundle(bundle)
    assert report["통과여부"].all()

