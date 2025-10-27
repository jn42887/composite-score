import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import base64
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import unicodedata
import re
from sklearn.preprocessing import StandardScaler
import time
import random

# Read credentials from environment variables (GitHub Actions will provide these)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
EPM_API_KEY = os.getenv('EPM_API_KEY')

# GitHub settings
GITHUB_USERNAME = 'jn42887'
REPO_NAME = 'composite-score'
BRANCH = 'main'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_name(name):
    """Normalize player names for better matching"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ASCII', 'ignore').decode('ASCII')
    name = name.lower()
    
    # Handle common variations
    name = name.replace('.', '').replace('-', ' ').replace("'", "")
    name = name.replace(' jr', '').replace(' sr', '').replace(' iii', '').replace(' iv', '').replace(' ii', '').replace(' i', '')
    
    # Handle common nicknames and variations
    nickname_map = {
        'anthony': 'tony',
        'christopher': 'chris',
        'michael': 'mike',
        'matthew': 'matt',
        'joseph': 'joe',
        'robert': 'bob',
        'richard': 'rick',
        'william': 'bill',
        'james': 'jim',
        'daniel': 'dan',
        'benjamin': 'ben',
        'alexander': 'alex',
        'nicholas': 'nick',
        'jonathan': 'jon',
        'timothy': 'tim',
        'patrick': 'pat',
        'gregory': 'greg',
        'jeffrey': 'jeff',
        'kenneth': 'ken',
        'stephen': 'steve',
        'thomas': 'tom',
        'charles': 'chuck',
        'edward': 'ed',
        'ronald': 'ron',
        'lawrence': 'larry',
        'kevin': 'kev',
        'brian': 'bri',
        'george': 'geo',
        'mark': 'marc',
        'paul': 'pau',
        'steven': 'steve',
        'andrew': 'andy',
        'joshua': 'josh',
        'kenneth': 'kenny',
        'ryan': 'ry',
        'jacob': 'jake',
        'gary': 'gar',
        'nicholas': 'nico',
        'eric': 'erik',
        'jonathan': 'johnny',
        'stephen': 'steph',
        'christopher': 'chris',
        'matthew': 'matty',
        'joseph': 'joey',
        'robert': 'robby',
        'richard': 'rich',
        'william': 'will',
        'james': 'jimmy',
        'daniel': 'danny',
        'benjamin': 'benny',
        'alexander': 'alex',
        'timothy': 'timmy',
        'patrick': 'patty',
        'gregory': 'greg',
        'jeffrey': 'jeffy',
        'kenneth': 'kenny',
        'stephen': 'stevie',
        'thomas': 'tommy',
        'charles': 'charlie',
        'edward': 'eddie',
        'ronald': 'ronnie',
        'lawrence': 'larry',
        'kevin': 'kev',
        'brian': 'bri',
        'george': 'georgie',
        'mark': 'marky',
        'paul': 'pauly',
        'steven': 'stevie',
        'andrew': 'andy',
        'joshua': 'josh',
        'ryan': 'ry',
        'jacob': 'jake',
        'gary': 'gar',
        'nicholas': 'nico',
        'eric': 'erik',
        'jonathan': 'johnny',
        'stephen': 'steph'
    }
    
    # Apply nickname mapping
    words = name.split()
    normalized_words = []
    for word in words:
        if word in nickname_map:
            normalized_words.append(nickname_map[word])
        else:
            normalized_words.append(word)
    
    name = ' '.join(normalized_words)
    name = ' '.join(name.split())  # Clean up extra spaces
    return name

def normalize_team(team):
    """Normalize team abbreviations"""
    if pd.isna(team):
        return ""
    team = str(team).strip().upper()
    if '/' in team:
        return team
    if team in ['TOT', '2TM', '3TM']:
        return 'MULTI'
    team_mapping = {
        'BRK': 'BKN', 'BKN': 'BKN',
        'CHO': 'CHA', 'CHA': 'CHA',
        'PHO': 'PHX', 'PHX': 'PHX',
        'NOP': 'NOP', 'NOH': 'NOP',
    }
    return team_mapping.get(team, team)

def push_to_github(df, file_path, commit_message=None):
    """Push a dataframe as CSV to GitHub"""
    if commit_message is None:
        commit_message = f"Auto-update {file_path} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    csv_content = df.to_csv(index=False)
    content_encoded = base64.b64encode(csv_content.encode()).decode()
    
    api_url = f'https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents/{file_path}'
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }
    
    response = requests.get(api_url, headers=headers)
    sha = None
    if response.status_code == 200:
        sha = response.json()['sha']
    
    data = {
        'message': commit_message,
        'content': content_encoded,
        'branch': BRANCH,
    }
    if sha:
        data['sha'] = sha
    
    response = requests.put(api_url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        print(f"✓ Successfully pushed {file_path} to GitHub!")
        return True
    else:
        print(f"✗ Failed to push {file_path}")
        print(f"  Status: {response.status_code}")
        return False

def scale_to_target(series, target_mean, target_std):
    """Scale a series to have target mean and std, handling NaN values"""
    valid_data = series.dropna()
    if len(valid_data) == 0:
        return series
    current_mean = valid_data.mean()
    current_std = valid_data.std()
    if current_std == 0:
        return series
    scaled = (series - current_mean) / current_std * target_std + target_mean
    return scaled

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_epm_data():
    """Fetch EPM data from API"""
    print("Fetching EPM data from API...")
    
    headers = {
        "Authorization": EPM_API_KEY,
        "Accept": "application/json"
    }
    
    epm_url = "https://dunksandthrees.com/api/v1/epm"
    lookback_days = 210
    all_data = []
    
    for i in range(lookback_days):
        date_str = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        params = {"date": date_str, "game_optimized": 0}
        try:
            response = requests.get(epm_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            daily_data = response.json()
            if daily_data:
                all_data.extend(daily_data)
        except:
            continue
    
    df_epm = pd.DataFrame(all_data)
    df_epm.sort_values(by=["player_id", "game_dt"], ascending=[True, False], inplace=True)
    df_epm = df_epm.drop_duplicates(subset="player_id", keep="first")
    
    print(f"  Loaded {len(df_epm)} EPM records")
    return df_epm

def fetch_lebron_data():
    """Fetch Lebron data from CSV"""
    print("Fetching Lebron data...")
    url = "https://r2-bbi.fanspo.com/all_impact_metrics_all_seasons.csv"
    lebron = pd.read_csv(url)
    lebron = lebron[lebron['Year'] == 2025]
    print(f"  Loaded {len(lebron)} Lebron records")
    return lebron

def fetch_darko_data():
    """Fetch DARKO data from DPM parquet file"""
    print("Fetching DARKO data...")
    url = "https://www.dropbox.com/scl/fi/yxpvvv2ttm2udevahiufs/darko_career_dpm_talent.parq?rlkey=isq4ols3uhxlod2fkisbn33d7&dl=1"
    
    try:
        # Try direct parquet reading first
        darko = pd.read_parquet(url)
    except (ImportError, Exception) as e:
        print(f"  Direct parquet reading failed: {e}")
        print("  Downloading file and reading locally...")
        
        # Download the file first
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save temporarily and read
        temp_file = "temp_darko.parquet"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        try:
            # Try reading the downloaded parquet file
            darko = pd.read_parquet(temp_file)
            print("  Successfully read parquet file locally")
        except Exception as parquet_error:
            print(f"  Local parquet reading failed: {parquet_error}")
            print("  This suggests pyarrow/fastparquet is not installed")
            print("  Please install pyarrow: pip install pyarrow")
            raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    print(f"  Loaded {len(darko)} DARKO records")
    return darko

def fetch_xrapm_data():
    """Fetch xRAPM data from website"""
    print("Fetching xRAPM data...")
    url = "https://xrapm.com/"
    response = requests.get(url)
    response.encoding = 'utf-8'
    
    soup = BeautifulSoup(response.text, 'html.parser', from_encoding='utf-8')
    table = soup.find('table', {'id': 'sortableTable'})
    
    headers = []
    thead = table.find('thead')
    if thead:
        for th in thead.find_all('th'):
            headers.append(th.text.strip())
    
    data = []
    tbody = table.find('tbody')
    if tbody:
        all_tds = tbody.find_all('td')
        for i in range(0, len(all_tds), 5):
            if i + 4 < len(all_tds):
                row_data = []
                player_link = all_tds[i].find('a')
                row_data.append(player_link.text.strip() if player_link else all_tds[i].text.strip())
                row_data.append(all_tds[i+1].text.strip())
                row_data.append(all_tds[i+2].text.strip())
                row_data.append(all_tds[i+3].text.strip())
                row_data.append(all_tds[i+4].text.strip())
                data.append(row_data)
    
    xrapm = pd.DataFrame(data, columns=headers)
    
    def clean_numeric(value):
        if '(' in value:
            return value.split('(')[0].strip()
        return value
    
    xrapm['Offense'] = xrapm['Offense'].apply(clean_numeric).astype(float)
    xrapm['Defense(*)'] = xrapm['Defense(*)'].apply(clean_numeric).astype(float)
    xrapm['Total'] = xrapm['Total'].apply(clean_numeric).astype(float)
    
    print(f"  Loaded {len(xrapm)} xRAPM records")
    return xrapm

def fetch_skills_data():
    """Fetch EPM skills data from API"""
    print("Fetching EPM skills data from API...")
    
    headers = {
        "Authorization": EPM_API_KEY,
        "Accept": "application/json"
    }
    
    API_URL = 'https://dunksandthrees.com/api/v1/epm'
    all_data = []
    lookback_days = 210
    
    for i in range(lookback_days):
        date_str = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
        params = {"date": date_str, "game_optimized": "0"}
        
        try:
            response = requests.get(API_URL, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            daily_data = response.json()
            if daily_data:
                all_data.extend(daily_data)
        except:
            continue
    
    if not all_data:
        print("  Warning: No skills data retrieved")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df.sort_values(by=["player_id", "game_dt"], ascending=[True, False], inplace=True)
    df_latest = df.drop_duplicates(subset="player_id", keep="first")
    
    print(f"  Processed {len(df_latest)} unique players")
    return df_latest

def get_current_teams_nba_com():
    """Fetch current rosters from NBA API"""
    print("Fetching current NBA rosters from NBA API...")
    
    # Determine current season
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # NBA season typically starts in October, so if we're before October, we're still in the previous season
    if current_month < 10:
        season_year = current_year - 1
    else:
        season_year = current_year
    
    season = f'{season_year}-{str(season_year + 1)[2:]}'
    print(f"  Using season: {season}")
    
    player_to_team = {}
    
    nba_teams = {
        1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN', 1610612766: 'CHA',
        1610612741: 'CHI', 1610612739: 'CLE', 1610612742: 'DAL', 1610612743: 'DEN',
        1610612765: 'DET', 1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
        1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM', 1610612748: 'MIA',
        1610612749: 'MIL', 1610612750: 'MIN', 1610612740: 'NOP', 1610612752: 'NYK',
        1610612760: 'OKC', 1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
        1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
        1610612762: 'UTA', 1610612764: 'WAS',
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com'
    }
    
    for team_id, team_abbr in nba_teams.items():
        try:
            api_url = f'https://stats.nba.com/stats/commonteamroster?LeagueID=00&Season={season}&TeamID={team_id}'
            time.sleep(random.uniform(0.5, 1.0))  # Rate limiting
            
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
            
            data = response.json()
            if 'resultSets' in data and len(data['resultSets']) > 0:
                players = data['resultSets'][0]['rowSet']
                for player in players:
                    if len(player) > 3:
                        first_name = player[3]
                        last_name = player[2]
                        full_name = f"{first_name} {last_name}"
                        normalized = normalize_name(full_name)
                        player_to_team[normalized] = team_abbr
        except Exception as e:
            print(f"  Error fetching roster for {team_abbr}: {e}")
            continue
    
    print(f"  Found current teams for {len(player_to_team)} players")
    return player_to_team

# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def find_match_multi_strategy(lebron_key, lebron_name, candidate_df):
    """Try multiple matching strategies"""
    exact_match = candidate_df[candidate_df['match_key'] == lebron_key]
    if not exact_match.empty:
        return exact_match.iloc[0]['match_key'], 100
    
    name_match = candidate_df[candidate_df['name_only'] == lebron_name]
    if not name_match.empty:
        return name_match.iloc[0]['match_key'], 95
    
    candidate_keys = candidate_df['match_key'].tolist()
    if candidate_keys:
        match, score = process.extractOne(lebron_key, candidate_keys, scorer=fuzz.ratio)
        if score >= 85:
            return match, score
    
    candidate_names = candidate_df['name_only'].tolist()
    if candidate_names:
        match_name, score = process.extractOne(lebron_name, candidate_names, scorer=fuzz.ratio)
        if score >= 90:
            matched_row = candidate_df[candidate_df['name_only'] == match_name].iloc[0]
            return matched_row['match_key'], score
    
    return None, 0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("=" * 60)
    print("Starting NBA Data Update")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 1. Fetch all datasets
        epm = fetch_epm_data()
        lebron = fetch_lebron_data()
        darko = fetch_darko_data()
        xrapm = fetch_xrapm_data()
        skills = fetch_skills_data()
        
        # 2. Use Lebron as base dataset (reliable approach)
        print("\nUsing Lebron dataset as base...")
        lebron_df = lebron.copy()
        lebron_df['normalized_name'] = lebron_df['player_name'].apply(normalize_name)
        lebron_df['normalized_team'] = lebron_df['Tm'].apply(normalize_team)
        lebron_df['match_key'] = lebron_df['normalized_name'] + '|' + lebron_df['normalized_team']
        lebron_df['name_only'] = lebron_df['normalized_name']
        
        base_df = lebron_df[['player_name', 'Tm', 'normalized_name', 'match_key', 'name_only']].copy()
        base_df.columns = ['player_name', 'team', 'normalized_name', 'match_key', 'name_only']
        print(f"Using Lebron dataset as base with {len(base_df)} players")
        
        # 3. Get current NBA rosters for team updates
        print("\nGetting current NBA rosters for team updates...")
        current_teams = get_current_teams_nba_com()
        
        # 3. Prepare datasets for merging
        print("\nPreparing datasets for merge...")
        
        # EPM
        epm_df = epm.copy()
        epm_df['normalized_name'] = epm_df['player_name'].apply(normalize_name)
        epm_df['normalized_team'] = epm_df['team_id'].apply(lambda x: normalize_team(str(x)[-3:]) if pd.notna(x) else "")
        epm_df['match_key'] = epm_df['normalized_name'] + '|' + epm_df['normalized_team']
        epm_df['name_only'] = epm_df['normalized_name']
        
        # Aggregate EPM
        epm_numeric_cols = epm_df.select_dtypes(include=[np.number]).columns.tolist()
        epm_non_numeric_cols = epm_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        epm_agg_dict = {}
        for col in epm_numeric_cols:
            if col not in ['normalized_name', 'normalized_team']:
                epm_agg_dict[col] = 'mean'
        for col in epm_non_numeric_cols:
            if col not in ['match_key', 'normalized_name', 'normalized_team', 'name_only']:
                epm_agg_dict[col] = 'first'
        
        epm_agg_dict['player_name'] = 'first'
        epm_agg_dict['player_id'] = 'first'
        epm_agg_dict['name_only'] = 'first'
        
        epm_agg = epm_df.groupby('match_key').agg(epm_agg_dict).reset_index()
        epm_agg = epm_agg.add_prefix('epm_')
        epm_agg = epm_agg.rename(columns={'epm_match_key': 'match_key', 'epm_name_only': 'name_only'})
        
        # Lebron
        lebron_df = lebron.copy()
        lebron_df['normalized_name'] = lebron_df['player_name'].apply(normalize_name)
        lebron_df['normalized_team'] = lebron_df['Tm'].apply(normalize_team)
        lebron_df['match_key'] = lebron_df['normalized_name'] + '|' + lebron_df['normalized_team']
        lebron_df['name_only'] = lebron_df['normalized_name']
        
        # DARKO
        darko_df = darko.copy()
        # Filter for current season (2025) and get latest data per player
        darko_df = darko_df[darko_df['season'] == 2025]
        darko_df = darko_df.sort_values(['player_name', 'season'], ascending=[True, False])
        darko_df = darko_df.drop_duplicates(subset='player_name', keep='first')
        
        darko_df['normalized_name'] = darko_df['player_name'].apply(normalize_name)
        darko_df['normalized_team'] = darko_df['team_name'].apply(normalize_team)
        darko_df['match_key'] = darko_df['normalized_name'] + '|' + darko_df['normalized_team']
        darko_df['name_only'] = darko_df['normalized_name']
        
        # xRAPM
        xrapm_df = xrapm.copy()
        xrapm_df['normalized_name'] = xrapm_df['Player'].apply(normalize_name)
        xrapm_df['normalized_team'] = xrapm_df['Team'].apply(normalize_team)
        xrapm_df['match_key'] = xrapm_df['normalized_name'] + '|' + xrapm_df['normalized_team']
        xrapm_df['name_only'] = xrapm_df['normalized_name']
        
        # Add prefixes
        darko_df_merged = darko_df.add_prefix('darko_')
        darko_df_merged = darko_df_merged.rename(columns={'darko_match_key': 'match_key', 'darko_name_only': 'name_only'})
        
        xrapm_df_merged = xrapm_df.add_prefix('xrapm_')
        xrapm_df_merged = xrapm_df_merged.rename(columns={'xrapm_match_key': 'match_key', 'xrapm_name_only': 'name_only'})
        
        lebron_df_merged = lebron_df.add_prefix('lebron_')
        lebron_df_merged = lebron_df_merged.rename(columns={'lebron_match_key': 'match_key', 'lebron_name_only': 'name_only'})
        
        # Start merge with roster base
        print("Merging datasets with current roster as base...")
        combined = base_df.copy()
        
        # Match DARKO
        darko_match_map = {}
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], darko_df_merged)
            if match_key:
                darko_match_map[idx] = match_key
        
        # Match xRAPM
        xrapm_match_map = {}
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], xrapm_df_merged)
            if match_key:
                xrapm_match_map[idx] = match_key
        
        # Match EPM
        epm_match_map = {}
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], epm_agg)
            if match_key:
                epm_match_map[idx] = match_key
        
        # Merge data
        xrapm_dict = xrapm_df_merged.set_index('match_key').to_dict('index')
        for idx, match_key in xrapm_match_map.items():
            if match_key in xrapm_dict:
                for col, val in xrapm_dict[match_key].items():
                    if col not in ['match_key', 'name_only']:
                        combined.at[idx, col] = val
        
        darko_dict = darko_df_merged.set_index('match_key').to_dict('index')
        for idx, match_key in darko_match_map.items():
            if match_key in darko_dict:
                for col, val in darko_dict[match_key].items():
                    if col not in ['match_key', 'name_only']:
                        combined.at[idx, col] = val
        
        epm_dict = epm_agg.set_index('match_key').to_dict('index')
        for idx, match_key in epm_match_map.items():
            if match_key in epm_dict:
                for col, val in epm_dict[match_key].items():
                    if col not in ['match_key', 'name_only']:
                        combined.at[idx, col] = val
        
        # Add final columns
        combined['final_player_name'] = combined['player_name']
        combined['final_team'] = combined['team']
        
        # Update teams using current roster data
        combined['current_team'] = combined['team']  # Start with original team
        roster_matches = 0
        
        for idx, row in combined.iterrows():
            normalized = normalize_name(row['final_player_name'])
            original_team = normalize_team(row['final_team'])
            
            if normalized in current_teams:
                combined.at[idx, 'current_team'] = current_teams[normalized]
                roster_matches += 1
                if current_teams[normalized] != original_team:
                    print(f"  Updated {row['final_player_name']}: {original_team} → {current_teams[normalized]}")
            else:
                # Try fuzzy matching for players that didn't match exactly
                fuzzy_match = None
                best_score = 0
                
                for roster_name in current_teams.keys():
                    score = fuzz.ratio(normalized, roster_name)
                    if score > best_score and score >= 85:  # 85% similarity threshold
                        best_score = score
                        fuzzy_match = roster_name
                
                if fuzzy_match:
                    combined.at[idx, 'current_team'] = current_teams[fuzzy_match]
                    roster_matches += 1
                    print(f"  Fuzzy matched {row['final_player_name']} → {fuzzy_match} (score: {best_score})")
                    if current_teams[fuzzy_match] != original_team:
                        print(f"    Updated team: {original_team} → {current_teams[fuzzy_match]}")
        
        print(f"\nRoster assignment results:")
        print(f"  - Players matched with current rosters: {roster_matches}")
        print(f"  - Players using original team data: {len(combined) - roster_matches}")
        
        # Debug: Show some roster data
        cavs_players = combined[combined['current_team'] == 'CLE']
        print(f"\nDebug: Found {len(cavs_players)} Cavs players in final dataset:")
        for idx, row in cavs_players.head().iterrows():
            print(f"  - {row['final_player_name']}")
        
        # 3. Create composite scores
        print("\nCreating composite scores...")
        
        cs = combined[['lebron_Year', 'lebron_Age', 'lebron_player', 'current_team', 'lebron_Position']].copy()
        cs.columns = ['Season', 'Age', 'Player', 'Team', 'Pos']
        cs['MP65'] = combined['lebron_MIN']/combined['lebron_G']*65
        # Handle xRAPM defense (invert if available)
        if 'xrapm_Defense(*)' in combined.columns:
            combined['xrapm_Defense(*)'] = -1*combined['xrapm_Defense(*)']
        
        # Assign metrics with fallbacks for missing data
        cs['lebron_off'] = combined.get('lebron_predOLEBRON', 0)
        cs['lebron_def'] = combined.get('lebron_predDLEBRON', 0)
        cs['xrapm_off'] = combined.get('xrapm_Offense', 0)
        cs['xrapm_def'] = combined.get('xrapm_Defense(*)', 0)
        cs['darko_off'] = combined.get('darko_o_dpm', 0)
        cs['darko_def'] = combined.get('darko_d_dpm', 0)
        cs['epm_off'] = combined.get('epm_oepm', 0)
        cs['epm_def'] = combined.get('epm_depm', 0)
        
        # Scale metrics
        epm_off_mean = cs['epm_off'].mean()
        epm_off_std = cs['epm_off'].std()
        epm_def_mean = cs['epm_def'].mean()
        epm_def_std = cs['epm_def'].std()
        
        cs['lebron_off_scaled'] = scale_to_target(cs['lebron_off'], epm_off_mean, epm_off_std)
        cs['xrapm_off_scaled'] = scale_to_target(cs['xrapm_off'], epm_off_mean, epm_off_std)
        cs['darko_off_scaled'] = scale_to_target(cs['darko_off'], epm_off_mean, epm_off_std)
        cs['epm_off_scaled'] = cs['epm_off']
        
        cs['lebron_def_scaled'] = scale_to_target(cs['lebron_def'], epm_def_mean, epm_def_std)
        cs['xrapm_def_scaled'] = scale_to_target(cs['xrapm_def'], epm_def_mean, epm_def_std)
        cs['darko_def_scaled'] = scale_to_target(cs['darko_def'], epm_def_mean, epm_def_std)
        cs['epm_def_scaled'] = cs['epm_def']
        
        cs['combined_off'] = cs[['lebron_off_scaled', 'xrapm_off_scaled', 'darko_off_scaled', 'epm_off_scaled']].mean(axis=1, skipna=True)
        cs['combined_def'] = cs[['lebron_def_scaled', 'xrapm_def_scaled', 'darko_def_scaled', 'epm_def_scaled']].mean(axis=1, skipna=True)
        cs['combined_tot'] = cs['combined_off'] + cs['combined_def']
        
        # Calculate projections
        cs['Multi-Year WAR'] = (0.1141*cs['combined_tot']*cs['combined_tot']+1.3037*cs['combined_tot']+2.8285)*1.05
        cs['Multi-Year PV'] = cs['Multi-Year WAR'] * 6000000
        
        cs['Y1_off'] = cs['combined_off']+1.8531-0.0675*(cs['Age']+1)
        cs['Y1_def'] = cs['combined_def']+0.7272-0.0261*(cs['Age']+1)
        cs['Y1_tot'] = cs['Y1_off'] + cs['Y1_def']
        cs['Y1_war'] = (0.1141*cs['Y1_tot']*cs['Y1_tot']+1.3037*cs['Y1_tot']+2.8285)*1.05
        cs['Y1_PV'] = cs['Y1_war']*6000000*1.07
        
        cs['Y2_off'] = cs['Y1_off']+1.8531-0.0675*(cs['Age']+2)
        cs['Y2_def'] = cs['Y1_def']+0.7272-0.0261*(cs['Age']+2)
        cs['Y2_tot'] = cs['Y2_off'] + cs['Y2_def']
        cs['Y2_war'] = (0.1141*cs['Y2_tot']*cs['Y2_tot']+1.3037*cs['Y2_tot']+2.8285)*1.05
        cs['Y2_PV'] = cs['Y2_war']*6000000*1.07*1.1
        
        cs['Y3_off'] = cs['Y2_off']+1.8531-0.0675*(cs['Age']+3)
        cs['Y3_def'] = cs['Y2_def']+0.7272-0.0261*(cs['Age']+3)
        cs['Y3_tot'] = cs['Y3_off'] + cs['Y3_def']
        cs['Y3_war'] = (0.1141*cs['Y3_tot']*cs['Y3_tot']+1.3037*cs['Y3_tot']+2.8285)*1.05
        cs['Y3_PV'] = cs['Y3_war']*6000000*1.07*1.1*1.1
        
        cs['Y4_off'] = cs['Y3_off']+1.8531-0.0675*(cs['Age']+4)
        cs['Y4_def'] = cs['Y3_def']+0.7272-0.0261*(cs['Age']+4)
        cs['Y4_tot'] = cs['Y4_off'] + cs['Y4_def']
        cs['Y4_war'] = (0.1141*cs['Y4_tot']*cs['Y4_tot']+1.3037*cs['Y4_tot']+2.8285)*1.05
        cs['Y4_PV'] = cs['Y4_war']*6000000*1.07*1.1*1.1*1.1
        
        cs['Y5_off'] = cs['Y4_off']+1.8531-0.0675*(cs['Age']+5)
        cs['Y5_def'] = cs['Y4_def']+0.7272-0.0261*(cs['Age']+5)
        cs['Y5_tot'] = cs['Y5_off'] + cs['Y5_def']
        cs['Y5_war'] = (0.1141*cs['Y5_tot']*cs['Y5_tot']+1.3037*cs['Y5_tot']+2.8285)*1.05
        cs['Y5_PV'] = cs['Y5_war']*6000000*1.07*1.1*1.1*1.1*1.1
        
        # 4. Push to GitHub
        print("\nPushing to GitHub...")
        
        # Push composite scores
        push_to_github(cs, 'Composite Projections copy.csv')
        
        # Push skills data
        if not skills.empty:
            push_to_github(skills, 'skills_data_full_response.csv')
        
        print("\n" + "=" * 60)
        print("✓ Update Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
