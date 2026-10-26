# Data

`student_performance.csv` – Student Performance Dataset (Kaggle), 1,000,000 rows, no missing values.

| Column | Description |
|---|---|
| student_id | Unique identifier (not used as a feature) |
| weekly_self_study_hours | Self-study hours per week (0–40) |
| attendance_percentage | Class attendance (50–100 %) |
| class_participation | Participation score (0–10) |
| total_score | Overall score (0–100) – **not used as a feature** (leakage) |
| grade | Letter grade A–F derived from total_score (A ≥ 85, B ≥ 70, C ≥ 55, D ≥ 40, F < 40) |

Derived target (`fail`): 1 if grade is D or F, otherwise 0.
Source: Qureshi, N. (n.d.). *Student performance dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-da
