import numpy as np
import pandas as pd


if __name__ == '__main__':
    postings = pd.read_csv('data/datasets/postings.csv')
    columns_to_keep = ["company_name", "title", "description", "location", "normalized_salary"]
    postings = postings[columns_to_keep]

    # remove postings without salary
    postings = postings.dropna(subset=['normalized_salary', 'description'])

    # remove postings with extreme salaries
    postings = postings.sort_values(by='normalized_salary', ascending=False)
    postings = postings[(postings.normalized_salary < 500000) & (postings.normalized_salary > 10000)]

    # transform target of salary as log because distribution is skewed 
    postings['target_salary'] = np.log1p(postings['normalized_salary'])
    postings = postings.drop(columns=['normalized_salary'])

    postings = postings.reset_index(drop=True)

    print(postings.info())

    postings.to_csv('data/datasets/postings_cleaned.csv', index=True)
