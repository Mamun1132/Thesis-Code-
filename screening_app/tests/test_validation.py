import math

import pytest

from screening.predictor import BINARY, NUMERIC_RANGES, ValidationError, validate


def test_valid_input_passes(valid):
    clean, warnings = validate(valid)
    assert clean["Age"] == 27 and warnings == []


@pytest.mark.parametrize("field", [f for f in NUMERIC_RANGES if f != "BMI"] + BINARY)
def test_required_fields(valid, field):
    valid[field] = ""
    with pytest.raises(ValidationError) as e:
        validate(valid)
    assert field in e.value.errors


def test_bmi_optional_is_imputed_with_warning(valid):
    valid["BMI"] = ""
    clean, warnings = validate(valid)
    assert math.isnan(clean["BMI"]) and len(warnings) == 1


@pytest.mark.parametrize("field", list(NUMERIC_RANGES))
def test_range_bounds(valid, field):
    lo, hi, _ = NUMERIC_RANGES[field]
    for ok in (lo, hi):
        v = dict(valid, **{field: ok})
        if field in ("Systolic BP", "Diastolic"):
            v.update({"Systolic BP": 200, "Diastolic": 40} if field == "Diastolic" else {"Diastolic": 40})
            v[field] = ok
        validate(v)
    for bad in (lo - 0.1, hi + 0.1):
        with pytest.raises(ValidationError) as e:
            validate(dict(valid, **{field: bad}))
        assert field in e.value.errors


@pytest.mark.parametrize("value", ["abc", "12a", None, "nan", "inf"])
def test_non_numeric_rejected(valid, value):
    valid["Heart Rate"] = value
    with pytest.raises(ValidationError):
        validate(valid)


@pytest.mark.parametrize("value", [2, -1, 0.5])
def test_binary_must_be_0_or_1(valid, value):
    valid["Gestational Diabetes"] = value
    with pytest.raises(ValidationError) as e:
        validate(valid)
    assert "Gestational Diabetes" in e.value.errors


def test_diastolic_must_be_below_systolic(valid):
    valid.update({"Systolic BP": 100, "Diastolic": 100})
    with pytest.raises(ValidationError) as e:
        validate(valid)
    assert "Diastolic" in e.value.errors


def test_string_numbers_from_form_accepted(valid):
    clean, _ = validate({k: str(v) for k, v in valid.items()})
    assert clean["BS"] == 11.0


def test_impossible_age_325_rejected(valid):
    with pytest.raises(ValidationError):
        validate(dict(valid, Age=325))
