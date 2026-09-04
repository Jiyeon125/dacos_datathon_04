from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherReferenceModel:
    """2025 부산 초등학교의 관측 교원 배치 패턴을 요약한 참고모형."""

    intercept: float = 4.260031589573636
    class_coefficient: float = 1.3276399234497231
    residual_q10: float = -2.7959178500211834
    residual_q90: float = 2.8556253305917387
    training_school_count: int = 296
    validation_mae: float = 1.7846935221700462
    validation_r2: float = 0.9802636030978894
    reference_date: str = "2025-10-01"
    usage_label: str = "현재 학교 배치 패턴 참고값"
    version: str = "busan-elementary-class-linear-2025-v1"

    def predict(self, total_classes: float) -> dict[str, float | int | str]:
        if total_classes < 0:
            raise ValueError("학급 수는 0 이상이어야 합니다.")
        estimate = self.intercept + self.class_coefficient * float(total_classes)
        return {
            "input_total_classes": int(total_classes),
            "estimate": estimate,
            "range_low": max(0.0, estimate + self.residual_q10),
            "range_high": estimate + self.residual_q90,
            "validation_mae": self.validation_mae,
            "validation_r2": self.validation_r2,
            "training_school_count": self.training_school_count,
            "reference_date": self.reference_date,
            "usage_label": self.usage_label,
        }


TEACHER_REFERENCE_MODEL = TeacherReferenceModel()


def predict_teacher_reference(total_classes: float) -> dict[str, float | int | str]:
    return TEACHER_REFERENCE_MODEL.predict(total_classes)
