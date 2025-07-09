import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(csv_path='cxr-pneumonia-dataset.csv'):
    """
    Loads a dataset from a CSV file and splits it into training, validation, and test sets
    with stratification based on 'set_name' and 'class' columns.

    Args:
        csv_path (str): The path to the input CSV file.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
               (train_df, val_df, test_df)
    """
    # 1. 데이터 로드
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return None, None, None

    print(f"Successfully loaded dataset with {len(df)} records.")

    # 2. 계층화를 위한 새로운 컬럼 생성 ('set_name'과 'class' 조합)
    # 이 컬럼은 분할 시에만 사용됩니다.
    df['stratify_col'] = df['set_name'] + '_' + df['class'].astype(str)

    # 3. Train (80%) / Temp (20%) 분할
    # stratify 옵션을 통해 'stratify_col'의 분포를 유지하며 분할합니다.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['stratify_col'],
        random_state=42
    )

    # 4. Temp (20%) -> Validation (10%) / Test (10%) 분할
    # test_size=0.5는 temp_df의 50%를 의미하며, 이는 원본 데이터의 10%에 해당합니다.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['stratify_col'],
        random_state=42
    )

    # 분할에 사용된 'stratify_col' 컬럼 제거
    for d in [train_df, val_df, test_df]:
        d.drop(columns='stratify_col', inplace=True)

    # 5. 분할 결과 확인
    print(f"\n--- Split Summary ---")
    print(f"Total data:      {len(df)}")
    print(f"Train data:      {len(train_df)} ({len(train_df)/len(df):.2%})")
    print(f"Validation data: {len(val_df)} ({len(val_df)/len(df):.2%})")
    print(f"Test data:       {len(test_df)} ({len(test_df)/len(df):.2%})")

    print('\n--- Train Set Distribution ---')
    print((train_df.groupby(['set_name', 'class']).size() / len(train_df)).round(4))

    print('\n--- Validation Set Distribution ---')
    print((val_df.groupby(['set_name', 'class']).size() / len(val_df)).round(4))

    print('\n--- Test Set Distribution ---')
    print((test_df.groupby(['set_name', 'class']).size() / len(test_df)).round(4))
    
    return train_df, val_df, test_df

if __name__ == '__main__':
    # 스크립트를 직접 실행할 때 함수를 호출합니다.
    train_dataset, validation_dataset, test_dataset = split_dataset()

    if train_dataset is not None:
        # 생성된 DataFrame을 CSV로 저장 (필요 시 주석 해제)
        train_dataset.to_csv('train_dataset.csv', index=False)
        validation_dataset.to_csv('validation_dataset.csv', index=False)
        test_dataset.to_csv('test_dataset.csv', index=False)
        print("\nDataFrames are ready. Uncomment the lines at the end of the script to save them to CSV files.")