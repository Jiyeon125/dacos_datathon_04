from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.candidate_generator import PAIR_A_CODE
from src.data_loader import load_bundle


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_default_scenario_renders_without_exception():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "학교 통합, 교육여건은 어떻게 달라질까?"
    assert len(app.selectbox) == 2
    assert len(app.selectbox[0].options) == 92
    assert app.selectbox[0].value == "213021106"
    assert app.selectbox[1].value is None
    app.selectbox[1].set_value("213021124").run(timeout=30)
    assert not app.exception
    assert "가남초등학교 → 가야초등학교" in app.subheader[0].value
    assert any("선택한 B학교의 교육자원 여유" in item.value for item in app.markdown)


def test_streamlit_no_candidate_school_has_explicit_empty_state():
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    counts = load_bundle().candidate_counts
    no_candidate_code = counts.loc[counts["소규모학교명"].eq("녹명초등학교"), PAIR_A_CODE].iloc[0]
    app.selectbox[0].set_value(no_candidate_code).run(timeout=30)
    assert not app.exception
    assert any("3km 이내에 선택 가능한 수용학교가 없습니다" in item.value for item in app.info)


def test_streamlit_keeps_all_small_schools_and_explains_gis_exclusion():
    bundle = load_bundle()
    excluded_code = bundle.excluded_schools["학교코드(KEDI)"].iloc[0]
    app = AppTest.from_file(APP_PATH).run(timeout=30)
    app.selectbox[0].set_value(excluded_code).run(timeout=30)
    assert not app.exception
    assert any("현재 GIS 분석에서는 제외됩니다" in item.value for item in app.warning)
