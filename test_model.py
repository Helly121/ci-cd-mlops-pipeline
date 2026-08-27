from app import predict_result


def test_placed_student():
    result = predict_result(
        8.5,   # CGPA
        92,    # Attendance
        85,    # CodingScore
        3,     # Projects
        1      # Internship
    )

    assert result == 1


def test_not_placed_student():
    result = predict_result(
        5.9,   # CGPA
        60,    # Attendance
        42,    # CodingScore
        1,     # Projects
        0      # Internship
    )

    assert result == 0