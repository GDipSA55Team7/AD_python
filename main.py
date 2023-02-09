# %%
import pandas as pd
import numpy as np
import json
import requests
from pandas import json_normalize
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity


def main():
    # get JSON for all open
    url = "https://8aa1-103-252-200-84.ap.ngrok.io/api/jobs/allopen"

    response = requests.get(f"{url}")

    data_json_open = json.loads(response.text)

    print(data_json_open)

    # get JSON for all jobs tagged with a locum
    url = "https://8aa1-103-252-200-84.ap.ngrok.io/api/jobs/allapplied"

    response = requests.get(f"{url}")

    data_json_applied = json.loads(response.text)

    print(data_json_applied)

    df_open = json_normalize(data_json_open)

    df_applied = json_normalize(data_json_applied)

    def categorize_hours(hours):
        if hours <= 4:
            return 'short'
        elif hours > 4 and hours <= 8:
            return 'medium'
        else:
            return 'long'

    def process_postal_code(postal_code):
        first_two_digits = str(postal_code[:2])

        if first_two_digits in ['31', '32', '33', '34', '35', '36', '37', '56', '57']:
            return 'central'
        elif first_two_digits in ['53', '54', '55', '82', '72', '73', '77', '78', '75', '76', '79', '80']:
            return 'north'
        elif first_two_digits in ['38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '81',
                                  '51', '52']:
            return 'east'
        elif first_two_digits in ['58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71']:
            return 'west'
        elif first_two_digits in ['01', '02', '03', '04', '05', '06', '07', '08', '14', '15', '16', '09', '10', '11',
                                  '12', '13', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28',
                                  '29', '30']:
            return 'south'
        else:
            return 'Invalid Postal Code'

    def process_dataframe(df):
        # select the columns to keep
        if 'freelancer.id' in df:
            df = df[["id", "startDateTime", "endDateTime", "clinic.postalCode", "freelancer.id"]]
            df["user_id"] = df["freelancer.id"]
            df = df.drop(columns=['freelancer.id'])
        else:
            df = df[["id", "startDateTime", "endDateTime", "clinic.postalCode"]]

        # rename id column to job id
        df["job_id"] = df["id"]

        # calculate the duration of the job in hours
        df["duration"] = (pd.to_datetime(df["endDateTime"]) - pd.to_datetime(df["startDateTime"])) / pd.Timedelta(
            hours=1)

        # categorize hours
        df['duration'] = df['duration'].apply(categorize_hours)

        # Convert the start time column to a pandas datetime object
        df['startDateTime'] = pd.to_datetime(df['startDateTime'])

        # Extract the day of the week from the start time and store it in a new column 'day'
        df['day'] = df['startDateTime'].dt.date.apply(lambda x: x.strftime('%A'))

        # Extract the hour of the day from the start time
        df['start_hour'] = df['startDateTime'].dt.hour

        # Process postal code as regions
        df['region'] = df['clinic.postalCode'].apply(process_postal_code)

        # create a new column 'start_period' based on the hour of the day
        df['start_period'] = np.where((df['start_hour'] >= 6) & (df['start_hour'] < 12), 'morning',
                                      np.where((df['start_hour'] >= 12) & (df['start_hour'] < 18), 'afternoon',
                                               'night'))

        # drop the end_time column
        df.drop(columns=["endDateTime", "start_hour", "startDateTime", "clinic.postalCode", "id"], inplace=True)

        return df

    df_open_processed = process_dataframe(df_open)

    df_applied_processed = process_dataframe(df_applied)

    # group the dataframe by user ID and other columns and find top 3 repeats for each user id
    grouped = df_applied_processed.groupby(['user_id', 'duration', 'day', 'region', 'start_period']).size().reset_index(
        name='counts')
    grouped = grouped.sort_values(by=['user_id', 'counts'], ascending=[True, False])
    df_applied_processed = grouped.groupby('user_id').head(3).reset_index(drop=True)

    # convert categorical variables to numerical using one-hot encoding
    df_applied_processed = pd.get_dummies(df_applied_processed, columns=['day', 'region', 'duration', 'start_period'])
    df_open_processed = pd.get_dummies(df_open_processed, columns=['day', 'region', 'duration', 'start_period'])

    def one_hot_encode(df_input):
        # create a list of column names
        column_names = ['day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday', 'day_Friday', 'day_Saturday',
                        'day_Sunday', 'region_Central', 'region_East', 'region_North', 'region_West', 'region_South',
                        'duration_long', 'duration_medium', 'duration_short', 'start_period_morning',
                        'start_period_afternoon', 'start_period_night']

        # create an empty dataframe with only columns
        df = pd.DataFrame(columns=column_names)

        df_input = df_input.reindex(columns=sorted(df_input.columns.union(df.columns)), fill_value=0)

        # concatenate the dataframes
        df_concat = pd.concat([df, df_input])

        return df_concat

    df_applied_encoded = one_hot_encode(df_applied_processed)
    df_open_encoded = one_hot_encode(df_open_processed)

    # calculate cosine similarity between each row in historical data and open jobs
    similarity_matrix = cosine_similarity(df_applied_encoded.drop(["user_id", "counts"], axis=1),
                                          df_open_encoded.drop("job_id", axis=1))

    # create dataframe with user_id, job_id, row index in historical data, row index in open jobs, and similarity score
    similarity_scores = []
    for i, row in df_applied_encoded.iterrows():
        user_id = row["user_id"]
        for j, job in df_open_encoded.iterrows():
            similarity_score = cosine_similarity(row.drop(["user_id", "counts"], axis=0).values.reshape(1, -1),
                                                 job.drop("job_id", axis=0).values.reshape(1, -1))[0][0]
            if similarity_score >= 0.75:
                similarity_scores.append([user_id, i, job["job_id"], j, similarity_score])

    similarity_scores = pd.DataFrame(similarity_scores,
                                     columns=["user_id", "applied_data_index", "open_job_id", "open_jobs_index",
                                              "similarity_score"])
    similarity_scores = similarity_scores.sort_values(by="user_id")

    recommended_for_sql = pd.DataFrame(similarity_scores, columns=["user_id", "open_job_id", "similarity_score"])

    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                           .format(host="sql_url",
                                   db="ad_locum", user="admin", pw="pw"))

    recommended_for_sql.to_sql('recommended_job', engine, index=True, index_label="id", if_exists='replace')


if __name__ == '__main__':
    main()
