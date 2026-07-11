import pandas as pd
from datetime import datetime

def calculate_risk_score(person_cases_df, reference_date=None):
    """
    person_cases_df: all FIR records linked to one person
    Expected to have at least 'FIR_YEAR', 'FIR_MONTH', 'CrimeHead_Name'
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Severity weights based on Karnataka FIR categories
    SEVERITY_WEIGHTS = {
        'MURDER': 10, 'ATTEMPT TO MURDER': 9, 'DACOITY': 9, 
        'ROBBERY': 8, 'RAPE': 10, 'POCSO': 10,
        'BURGLARY - NIGHT': 7, 'BURGLARY - DAY': 6,
        'THEFT': 5, 'MOTOR VEHICLE ACCIDENTS FATAL': 8,
        'CASES OF HURT': 6, 'CHEATING': 4, 'DEFAULT': 3
    }

    frequency = len(person_cases_df)
    frequency_score = min(frequency / 10, 1.0) * 100  # Cap at 10 cases

    # Estimate recentness using FIR_YEAR and FIR_MONTH (approximate to first day of month)
    latest_year = person_cases_df['FIR_YEAR'].max()
    latest_month_in_year = person_cases_df[person_cases_df['FIR_YEAR'] == latest_year]['FIR_MONTH'].max()
    
    # Handle NaNs or missing
    if pd.isna(latest_year) or pd.isna(latest_month_in_year):
        latest_date = datetime(2016, 1, 1) # fallback
    else:
        latest_date = datetime(int(latest_year), int(latest_month_in_year), 1)

    days_since_last = (reference_date - latest_date).days
    recency_score = max(0, 100 - (days_since_last / 3.65)) # Decays over a few years

    # Severity score
    def get_severity(crime):
        if not isinstance(crime, str):
            return SEVERITY_WEIGHTS['DEFAULT']
        return SEVERITY_WEIGHTS.get(crime.upper(), SEVERITY_WEIGHTS['DEFAULT'])

    severity_values = person_cases_df['CrimeHead_Name'].apply(get_severity)
    severity_score = (severity_values.mean() / 10) * 100

    # Trend multiplier
    # Calculate cases in the last 2 years vs before
    recent_cases = person_cases_df[person_cases_df['FIR_YEAR'] >= reference_date.year - 2]
    old_cases = person_cases_df[person_cases_df['FIR_YEAR'] < reference_date.year - 2]
    trend_multiplier = 1.2 if len(recent_cases) > len(old_cases) else 1.0

    final_score = (
        0.35 * frequency_score +
        0.30 * recency_score +
        0.35 * severity_score
    ) * trend_multiplier

    return {
        'risk_score': round(min(final_score, 100), 1),
        'frequency': frequency,
        'avg_severity': round(severity_values.mean(), 1),
        'trending_up': trend_multiplier > 1.0
    }

def get_top_risk_offenders(df, top_n=10):
    """
    Groups by Victim_Names/Accused (using Victim_Names as a mock identifier here since it contains lists of names)
    and calculates risk scores.
    """
    # For the datathon, we assume 'Victim_Names' JSON strings might contain repeated names to simulate offenders.
    # In reality, this should be done on a proper Accused/Offender identifier column.
    
    # Flatten names
    import json
    records = []
    
    # Sample a chunk to avoid processing 1.6M rows for this function
    sample_df = df.head(10000).copy()
    
    for idx, row in sample_df.iterrows():
        try:
            names = json.loads(row['Victim_Names'])
            for name in names:
                records.append({
                    'Name': name,
                    'FIR_YEAR': row['FIR_YEAR'],
                    'FIR_MONTH': row['FIR_MONTH'],
                    'CrimeHead_Name': row['CrimeHead_Name']
                })
        except:
            pass
            
    if not records:
        return pd.DataFrame()
        
    flat_df = pd.DataFrame(records)
    grouped = flat_df.groupby('Name')
    
    scores = []
    for name, group in grouped:
        if len(group) > 1: # Only repeat offenders
            score_data = calculate_risk_score(group)
            score_data['Name'] = name
            scores.append(score_data)
            
    res_df = pd.DataFrame(scores)
    if not res_df.empty:
        return res_df.sort_values('risk_score', ascending=False).head(top_n)
    return res_df
