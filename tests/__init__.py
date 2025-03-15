import pytest
import pandas as pd
from utils.processing import flatten, merge_entries, fill_gaps, process_dataframe, EXP_LEVELS

def test_flatten():
    entry = {
        "name": "Job 1",
        "salary": {"from": 50000, "to": 70000},
        "location": {"city": "Moscow", "country": "Russia"},
    }

    expected = {
        "name": "Job 1",
        "salary_from": 50000,
        "salary_to": 70000,
        "location_city": "Moscow",
        "location_country": "Russia",
    }

    result = flatten(entry)

    assert result == expected


def test_merge_entries():
    entries = [
        {"name": "Job 1", "salary": 50000},
        {"name": "Job 2", "salary": 60000, "location": "Moscow"},
    ]

    expected = {
        "name": ["Job 1", "Job 2"],
        "salary": [50000, 60000],
        "location": [pd.NA, "Moscow"],
    }

    result = merge_entries(entries)

    assert result == expected


def test_fill_gaps():
    data = {
        "salary_from": [1000, None, 2000],
        "salary_to": [None, 3000, None],
        "salary_currency": ["RUR", "USD", "RUR"],
        "address_lat": [55.0, None, 54.0],
        "address_lng": [37.0, None, 38.0],
    }

    df = pd.DataFrame(data)

    expected = {
        "salary": [1000, 3000 * 100, 2000],
        "address_lat": [55.0, 54.0],
        "address_lng": [37.0, 38.0],
    }

    result = fill_gaps(df)

    assert result[["salary", "address_lat", "address_lng"]].values.tolist() == expected


# Тест для функции process_dataframe
def test_process_dataframe():
    data = {
        "premium": ["True", "False", "True"],
        "name": ["Job 1", "Job 2", "Job 3"],
        "has_test": ["True", "False", "True"],
        "response_letter_required": ["True", "True", "False"],
        "area_name": ["Area 1", "Area 2", "Area 3"],
        "published_at": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "created_at": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "snippet_requirement": ["Requirement 1", "Requirement 2", "Requirement 3"],
        "snippet_responsibility": ["Responsibility 1", "Responsibility 2", "Responsibility 3"],
        "schedule_name": ["Full-time", "Part-time", "Freelance"],
        "professional_roles_name": ["Developer", "Manager", "Tester"],
        "experience_name": ["От 1 года до 3 лет", "Более 6 лет", "От 3 до 6 лет"],
        "employment_name": ["Full-time", "Part-time", "Freelance"],
        "salary_from": [1000, 2000, 3000],
        "salary_to": [1500, 2500, 3500],
        "salary_currency": ["RUR", "USD", "RUR"],
        "address_lat": [55.0, 56.0, 57.0],
        "address_lng": [37.0, 38.0, 39.0],
    }

    df = pd.DataFrame(data)

    expected = {
        "premium": [1, 0, 1],
        "name": ["Job 1", "Job 2", "Job 3"],
        "has_test": [1, 0, 1],
        "response_letter_required": [1, 1, 0],
        "area_name": ["Area 1", "Area 2", "Area 3"],
        "published_at": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "created_at": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "snippet_requirement": ["Requirement 1", "Requirement 2", "Requirement 3"],
        "snippet_responsibility": ["Responsibility 1", "Responsibility 2", "Responsibility 3"],
        "schedule_name": ["Full-time", "Part-time", "Freelance"],
        "professional_roles_name": ["Developer", "Manager", "Tester"],
        "experience_name": ["От 1 года до 3 лет", "Более 6 лет", "От 3 до 6 лет"],
        "employment_name": ["Full-time", "Part-time", "Freelance"],
        "salary": [1000, 250000, 3000],
        "address_lat": [55.0, 56.0, 57.0],
        "address_lng": [37.0, 38.0, 39.0],
        "is_salary_set": ["Указана", "Указана", "Указана"],
    }

    result = process_dataframe(df)

    assert result[["premium", "has_test", "is_salary_set", "salary"]].values.tolist() == [
        [1, 1, "Указана", 1000],
        [0, 0, "Указана", 250000],
        [1, 1, "Указана", 3000],
    ]

if __name__ == "__main__":
    pytest.main()