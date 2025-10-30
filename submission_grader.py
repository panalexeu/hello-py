import pandas as pd
from sklearn.metrics import f1_score


def grade_submission(
        submission_path: str = './submission.csv',
        golden_path: str = './data/test_with_labels.csv'
) -> float:
    # Load files
    sub_df = pd.read_csv(submission_path)
    gold_df = pd.read_csv(golden_path)

    # Merge on PassengerId
    merged = pd.merge(
        gold_df[['PassengerId', 'Transported']],
        sub_df[['PassengerId', 'Transported']],
        on='PassengerId',
        suffixes=('_true', '_pred')
    )

    # Calculate F1 score
    f1 = f1_score(merged['Transported_true'], merged['Transported_pred'])

    return f1


if __name__ == '__main__':
    f1_score_result = grade_submission()
    print(f"F1 Score: {f1_score_result:.4f}")
