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

def capitalize_name(name):
    """Properly capitalize player names"""
    if pd.isna(name) or name == '':
        return name
    
    # Handle special cases
    special_cases = {
        'lebron james': 'LeBron James',
        'michael jordan': 'Michael Jordan',
        'kobe bryant': 'Kobe Bryant',
        'shaquille oneal': 'Shaquille O\'Neal',
        'dwyane wade': 'Dwyane Wade',
        'tim duncan': 'Tim Duncan',
        'kevin garnett': 'Kevin Garnett',
        'paul pierce': 'Paul Pierce',
        'ray allen': 'Ray Allen',
        'allen iverson': 'Allen Iverson',
        'tracy mcgrady': 'Tracy McGrady',
        'vince carter': 'Vince Carter',
        'grant hill': 'Grant Hill',
        'penny hardaway': 'Penny Hardaway',
        'chris webber': 'Chris Webber',
        'jason kidd': 'Jason Kidd',
        'steve nash': 'Steve Nash',
        'dirk nowitzki': 'Dirk Nowitzki',
        'kevin durant': 'Kevin Durant',
        'russell westbrook': 'Russell Westbrook',
        'james harden': 'James Harden',
        'stephen curry': 'Stephen Curry',
        'klay thompson': 'Klay Thompson',
        'draymond green': 'Draymond Green',
        'kawhi leonard': 'Kawhi Leonard',
        'paul george': 'Paul George',
        'jimmy butler': 'Jimmy Butler',
        'damian lillard': 'Damian Lillard',
        'kyrie irving': 'Kyrie Irving',
        'anthony davis': 'Anthony Davis',
        'joel embiid': 'Joel Embiid',
        'giannis antetokounmpo': 'Giannis Antetokounmpo',
        'luka doncic': 'Luka Doncic',
        'jayson tatum': 'Jayson Tatum',
        'donovan mitchell': 'Donovan Mitchell',
        'darius garland': 'Darius Garland',
        'evan mobley': 'Evan Mobley',
        'jarrett allen': 'Jarrett Allen',
        'max strus': 'Max Strus',
        'dean wade': 'Dean Wade',
        'sam merrill': 'Sam Merrill',
        'lonzo ball': 'Lonzo Ball',
        'thomas bryant': 'Thomas Bryant',
        'deandre hunter': 'De\'Andre Hunter',
        'larry nance': 'Larry Nance Jr.',
        'craig porter': 'Craig Porter Jr.',
        'jaylon tyson': 'Jaylon Tyson',
        'tyrese proctor': 'Tyrese Proctor',
        'naeqwan tomlin': 'Nae\'Qwan Tomlin',
        'luke travers': 'Luke Travers'
    }
    
    name_lower = str(name).lower().strip()
    if name_lower in special_cases:
        return special_cases[name_lower]
    
    # Default capitalization for other names
    words = str(name).split()
    capitalized_words = []
    for word in words:
        if word.lower() in ['jr', 'sr', 'ii', 'iii', 'iv']:
            capitalized_words.append(word.upper())
        else:
            capitalized_words.append(word.capitalize())
    
    return ' '.join(capitalized_words)

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
        'thomas': 'thomas',  # Don't convert Thomas to Tom
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
    """Scrape current rosters from NBA.com"""
    print("Fetching current NBA rosters from NBA.com...")
    
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
    
    team_names = {
        'ATL': 'hawks', 'BOS': 'celtics', 'BKN': 'nets', 'CHA': 'hornets',
        'CHI': 'bulls', 'CLE': 'cavaliers', 'DAL': 'mavericks', 'DEN': 'nuggets',
        'DET': 'pistons', 'GSW': 'warriors', 'HOU': 'rockets', 'IND': 'pacers',
        'LAC': 'clippers', 'LAL': 'lakers', 'MEM': 'grizzlies', 'MIA': 'heat',
        'MIL': 'bucks', 'MIN': 'timberwolves', 'NOP': 'pelicans', 'NYK': 'knicks',
        'OKC': 'thunder', 'ORL': 'magic', 'PHI': '76ers', 'PHX': 'suns',
        'POR': 'blazers', 'SAC': 'kings', 'SAS': 'spurs', 'TOR': 'raptors',
        'UTA': 'jazz', 'WAS': 'wizards'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    for i, (team_id, team_abbr) in enumerate(nba_teams.items()):
        try:
            team_name = team_names[team_abbr]
            # Prefer dedicated roster page to avoid non-roster mentions
            url = f"https://www.nba.com/team/{team_id}/{team_name}/roster"
            time.sleep(random.uniform(1.0, 2.0))
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            # Restrict scraping to the roster table only: the table whose header contains a 'PLAYER' column
            player_links = []
            roster_tables = []
            for table in soup.find_all('table'):
                thead = table.find('thead')
                if not thead:
                    continue
                headers = [th.get_text(strip=True).upper() for th in thead.find_all('th')]
                if any(h in ('PLAYER', 'NAME') for h in headers):
                    roster_tables.append(table)
            tables_to_scan = roster_tables if roster_tables else soup.find_all('table')
            for table in tables_to_scan:
                tbody = table.find('tbody') or table
                player_links.extend(tbody.find_all('a', href=lambda x: x and '/player/' in str(x)))
            
            seen_players = set()
            for link in player_links:
                player_name = link.text.strip()
                if not player_name or len(player_name) < 3:
                    continue
                # Heuristic: skip items that don't look like full names
                if ' ' not in player_name:
                    continue
                normalized = normalize_name(player_name)
                if normalized not in seen_players:
                    seen_players.add(normalized)
                    # Only set the team the first time we see a player; this avoids
                    # overwriting with non-roster references (e.g., history/honors) from other pages
                    if normalized not in player_to_team:
                        player_to_team[normalized] = team_abbr
        except:
            continue
    
    print(f"  Found current teams for {len(player_to_team)} players")
    return player_to_team

def get_current_teams_from_epm():
    """Fetch current teams from EPM API season-epm endpoint and return mapping normalized_name -> team abbr.
    
    Uses /api/v1/season-epm which has team_id and team_alias (team abbreviation) fields.
    """
    print("Fetching current NBA rosters from EPM API (season-epm)...")

    if not EPM_API_KEY:
        print("  ERROR: EPM_API_KEY is not set. Cannot fetch current teams.")
        return {}

    # Get current season (2026)
    current_season = 2026
    url = "https://dunksandthrees.com/api/v1/season-epm"
    
    headers = {
        "Authorization": EPM_API_KEY,
        "Accept": "application/json"
    }
    
    params = {
        "season": current_season,
        "seasontype": 2  # Regular season
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data or not isinstance(data, list):
            print("  ERROR: EPM API returned no data or invalid format")
            return {}
        
        # Build mapping from player_name to team_alias
        player_to_team = {}
        for record in data:
            player_name = record.get('player_name')
            team_alias = record.get('team_alias')
            
            if not player_name or not team_alias:
                continue
            
            normalized = normalize_name(str(player_name).strip())
            team_abbr = normalize_team(str(team_alias).strip())
            player_to_team[normalized] = team_abbr
        
        # Debug summary
        try:
            team_counts = pd.Series(list(player_to_team.values())).value_counts().to_dict()
            top_counts = dict(list(team_counts.items())[:10])
            print(f"  Found current teams for {len(player_to_team)} players (EPM API)")
            print(f"  Team counts (top): {top_counts}")
        except Exception:
            print(f"  Found current teams for {len(player_to_team)} players (EPM API)")
        
        return player_to_team
        
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: EPM API request failed: {e}")
        return {}
    except Exception as e:
        print(f"  ERROR: Failed to process EPM API response: {e}")
        return {}

# ============================================================================
# MERGE FUNCTIONS
# ============================================================================

def find_match_multi_strategy(lebron_key, lebron_name, candidate_df):
    """Try multiple matching strategies"""
    exact_match = candidate_df[candidate_df['match_key'] == lebron_key]
    if not exact_match.empty:
        return exact_match.iloc[0]['match_key'], 100
    
    # Try exact name match first
    name_match = candidate_df[candidate_df['name_only'] == lebron_name]
    if not name_match.empty:
        return name_match.iloc[0]['match_key'], 95
    
    # Try name match with MULTI team (for players like Thomas Bryant with TOT team)
    multi_match = candidate_df[(candidate_df['name_only'] == lebron_name) & 
                              (candidate_df['match_key'].str.contains('|MULTI'))]
    if not multi_match.empty:
        return multi_match.iloc[0]['match_key'], 90
    
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
        
        # 2. Get NBA rosters from EPM API as base dataset
        print("\nGetting NBA rosters from EPM API as base dataset...")
        current_teams = get_current_teams_from_epm()
        
        # Create base dataset from scraped rosters
        roster_data = []
        seen_base_players = set()  # Track players we've already added
        for normalized_name, team_abbr in current_teams.items():
            # Skip if we've already added this player
            if normalized_name in seen_base_players:
                print(f"  WARNING: Skipping duplicate player in roster: {normalized_name}")
                continue
            seen_base_players.add(normalized_name)
            
            roster_data.append({
                'player_name': normalized_name,  # We'll improve this with actual names
                'team': team_abbr,
                'normalized_name': normalized_name,
                'match_key': normalized_name + '|' + team_abbr,
                'name_only': normalized_name
            })
        
        base_df = pd.DataFrame(roster_data)
        
        # Deduplicate base_df - ensure each player appears only once
        # If same player appears multiple times, keep the first occurrence
        initial_base_count = len(base_df)
        base_df = base_df.drop_duplicates(subset=['normalized_name'], keep='first')
        if len(base_df) < initial_base_count:
            print(f"  Removed {initial_base_count - len(base_df)} duplicate entries from roster base")
        
        print(f"Created base dataset with {len(base_df)} unique players from EPM API")
        
        if len(base_df) == 0:
            print("  ERROR: No players found in base dataset. Cannot proceed.")
            print("  This likely means the EPM API failed or returned no data.")
            return
        
        # 3. Prepare datasets for merging
        print("\nPreparing datasets for merge...")
        
        # EPM
        epm_df = epm.copy()
        epm_df['normalized_name'] = epm_df['player_name'].apply(normalize_name)
        epm_df['normalized_team'] = epm_df['team_id'].apply(lambda x: normalize_team(str(x)[-3:]) if pd.notna(x) else "")
        epm_df['match_key'] = epm_df['normalized_name'] + '|' + epm_df['normalized_team']
        epm_df['name_only'] = epm_df['normalized_name']
        
        # Deduplicate EPM by name_only first - prefer MULTI (aggregate stats) over individual team entries
        # This prevents averaging incorrect values from different team entries
        epm_df = epm_df.sort_values(['name_only', 'normalized_team'], ascending=[True, True])
        epm_df['team_priority'] = epm_df['normalized_team'].apply(lambda x: 1 if x == 'MULTI' else 0)
        epm_df = epm_df.sort_values(['name_only', 'team_priority'], ascending=[True, False])
        epm_df = epm_df.drop_duplicates(subset=['name_only'], keep='first')
        epm_df = epm_df.drop('team_priority', axis=1)
        
        # Now recalculate match_key after deduplication
        epm_df['match_key'] = epm_df['normalized_name'] + '|' + epm_df['normalized_team']
        
        # Aggregate EPM - should only be needed if same player+team combo appears multiple times
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
        
        print(f"  Deduplicated EPM data: {len(epm)} -> {len(epm_df)} -> {len(epm_agg)} unique entries")
        
        # Lebron
        lebron_df = lebron.copy()
        # Deduplicate LEBRON data - if same player appears multiple times, keep the one with most recent/complete data
        lebron_df['normalized_name'] = lebron_df['player_name'].apply(normalize_name)
        lebron_df['normalized_team'] = lebron_df['Tm'].apply(normalize_team)
        lebron_df['match_key'] = lebron_df['normalized_name'] + '|' + lebron_df['normalized_team']
        lebron_df['name_only'] = lebron_df['normalized_name']
        
        # Sort by name and team, then drop duplicates - this ensures we keep one entry per player+team combo
        # If same player appears with same team, keep first (could add additional sorting by Year/other fields if needed)
        lebron_df = lebron_df.sort_values(['normalized_name', 'normalized_team', 'Year'], ascending=[True, True, False])
        lebron_df = lebron_df.drop_duplicates(subset=['match_key'], keep='first')
        
        # Also deduplicate by name_only - if same player name appears multiple times (different teams), keep the one with MULTI team
        # MULTI/TOT entries are aggregate stats across teams, which are more accurate
        lebron_df = lebron_df.sort_values(['name_only', 'normalized_team'], ascending=[True, True])
        # Prefer MULTI teams (they represent aggregate stats)
        lebron_df['team_priority'] = lebron_df['normalized_team'].apply(lambda x: 1 if x == 'MULTI' else 0)
        lebron_df = lebron_df.sort_values(['name_only', 'team_priority'], ascending=[True, False])
        lebron_df = lebron_df.drop_duplicates(subset=['name_only'], keep='first')
        lebron_df = lebron_df.drop('team_priority', axis=1)
        
        print(f"  Deduplicated LEBRON data: {len(lebron)} -> {len(lebron_df)} unique players")
        
        # DARKO
        darko_df = darko.copy()
        # Filter for current season (2025) and get most recent daily record per player
        if 'season' in darko_df.columns:
            try:
                # Use the latest available season (handles float/int like 2026.0)
                season_series = pd.to_numeric(darko_df['season'], errors='coerce')
                if season_series.notna().any():
                    target_season = int(season_series.max())
                    print(f"  Using DARKO season: {target_season}")
                    darko_df = darko_df[season_series == target_season]
                else:
                    print("  WARNING: DARKO 'season' not parseable; using all rows")
            except Exception:
                print("  WARNING: Failed to parse DARKO 'season'; using all rows")
        
        # Choose latest record per player using best available date-like column
        date_candidates = ['date','dt','as_of','calc_date','run_date','game_date','game_dt','day','day_dt','updated_at','timestamp']
        date_col = None
        for cand in date_candidates:
            if cand in darko_df.columns:
                parsed = pd.to_datetime(darko_df[cand], errors='coerce')
                if parsed.notna().any():
                    darko_df['_parsed_date'] = parsed
                    date_col = cand
                    break
        if date_col is not None:
            # Keep the last (most recent) record per player
            darko_df = darko_df.sort_values(['player_name', '_parsed_date'])
            darko_df = darko_df.drop_duplicates(subset='player_name', keep='last')
            darko_df = darko_df.drop(columns=['_parsed_date'])
        else:
            # Fallback: keep last occurrence per player based on file order
            darko_df = darko_df.reset_index()
            darko_df = darko_df.sort_values(['player_name', 'index'])
            darko_df = darko_df.groupby('player_name', as_index=False).tail(1)
            darko_df = darko_df.drop(columns=['index'])
        
        darko_df['normalized_name'] = darko_df['player_name'].astype(str).apply(normalize_name)
        darko_df['normalized_team'] = darko_df['team_name'].astype(str).apply(normalize_team)
        darko_df['match_key'] = darko_df['normalized_name'].astype(str) + '|' + darko_df['normalized_team'].astype(str)
        darko_df['name_only'] = darko_df['normalized_name']
        
        # xRAPM
        xrapm_df = xrapm.copy()
        xrapm_df['normalized_name'] = xrapm_df['Player'].apply(normalize_name)
        xrapm_df['normalized_team'] = xrapm_df['Team'].apply(normalize_team)
        xrapm_df['match_key'] = xrapm_df['normalized_name'] + '|' + xrapm_df['normalized_team']
        xrapm_df['name_only'] = xrapm_df['normalized_name']
        
        # Deduplicate xRAPM - ensure each player+team combo appears only once
        initial_xrapm_count = len(xrapm_df)
        xrapm_df = xrapm_df.sort_values(['normalized_name', 'normalized_team'])
        xrapm_df = xrapm_df.drop_duplicates(subset=['match_key'], keep='first')
        
        # Also deduplicate by name_only - prefer MULTI teams (aggregate stats)
        xrapm_df['team_priority'] = xrapm_df['normalized_team'].apply(lambda x: 1 if x == 'MULTI' else 0)
        xrapm_df = xrapm_df.sort_values(['name_only', 'team_priority'], ascending=[True, False])
        xrapm_df = xrapm_df.drop_duplicates(subset=['name_only'], keep='first')
        xrapm_df = xrapm_df.drop('team_priority', axis=1)
        
        if len(xrapm_df) < initial_xrapm_count:
            print(f"  Deduplicated xRAPM data: {initial_xrapm_count} -> {len(xrapm_df)} unique players")
        
        # Add prefixes
        darko_df_merged = darko_df.add_prefix('darko_')
        darko_df_merged = darko_df_merged.rename(columns={'darko_match_key': 'match_key', 'darko_name_only': 'name_only'})
        
        xrapm_df_merged = xrapm_df.add_prefix('xrapm_')
        xrapm_df_merged = xrapm_df_merged.rename(columns={'xrapm_match_key': 'match_key', 'xrapm_name_only': 'name_only'})
        
        lebron_df_merged = lebron_df.add_prefix('lebron_')
        lebron_df_merged = lebron_df_merged.rename(columns={'lebron_match_key': 'match_key', 'lebron_name_only': 'name_only'})
        
        # Start merge with roster base
        print("Merging datasets with roster as base...")
        combined = base_df.copy()
        
        # Match DARKO
        darko_match_map = {}
        matched_darko_keys = set()
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], darko_df_merged)
            if match_key:
                if match_key in matched_darko_keys:
                    print(f"  WARNING: DARKO entry {match_key} already matched, skipping duplicate for {row['normalized_name']}")
                    continue
                matched_darko_keys.add(match_key)
                darko_match_map[idx] = match_key
        
        # Match xRAPM
        xrapm_match_map = {}
        matched_xrapm_keys = set()
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], xrapm_df_merged)
            if match_key:
                if match_key in matched_xrapm_keys:
                    print(f"  WARNING: xRAPM entry {match_key} already matched, skipping duplicate for {row['normalized_name']}")
                    continue
                matched_xrapm_keys.add(match_key)
                xrapm_match_map[idx] = match_key
        
        # Match EPM
        epm_match_map = {}
        matched_epm_keys = set()
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], epm_agg)
            if match_key:
                if match_key in matched_epm_keys:
                    print(f"  WARNING: EPM entry {match_key} already matched, skipping duplicate for {row['normalized_name']}")
                    continue
                matched_epm_keys.add(match_key)
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
        
        # Match LeBRON
        lebron_match_map = {}
        # Track which LEBRON entries have been matched to prevent duplicates
        matched_lebron_keys = set()
        for idx, row in combined.iterrows():
            match_key, score = find_match_multi_strategy(row['match_key'], row['name_only'], lebron_df_merged)
            if match_key:
                # Check if this LEBRON entry was already matched to another roster entry
                if match_key in matched_lebron_keys:
                    print(f"  WARNING: LEBRON entry {match_key} already matched, skipping duplicate match for roster entry {row['normalized_name']}")
                    continue
                matched_lebron_keys.add(match_key)
                lebron_match_map[idx] = match_key
        
        lebron_dict = lebron_df_merged.set_index('match_key').to_dict('index')
        for idx, match_key in lebron_match_map.items():
            if match_key in lebron_dict:
                for col, val in lebron_dict[match_key].items():
                    if col not in ['match_key', 'name_only']:
                        combined.at[idx, col] = val
        
        # Add final columns with proper capitalization
        # Use LeBRON player name if available, otherwise use roster name
        def get_final_name(row):
            if pd.notna(row.get('lebron_player_name')):
                return capitalize_name(row['lebron_player_name'])
            else:
                return capitalize_name(row['player_name'])
        
        combined['final_player_name'] = combined.apply(get_final_name, axis=1)
        combined['final_team'] = combined['team']
        
        # Remove duplicates BEFORE filtering - if same player appears multiple times, keep the one with most data
        print("\nRemoving duplicate player entries from combined data...")
        initial_count = len(combined)
        
        # Count how many data sources each player has
        def count_data_sources(row):
            count = 0
            if any(col.startswith('lebron_') and pd.notna(row.get(col)) and row.get(col) != 0 
                   for col in combined.columns if col.startswith('lebron_')):
                count += 1
            if any(col.startswith('epm_') and pd.notna(row.get(col)) and row.get(col) != 0 
                   for col in combined.columns if col.startswith('epm_')):
                count += 1
            if any(col.startswith('darko_') and pd.notna(row.get(col)) and row.get(col) != 0 
                   for col in combined.columns if col.startswith('darko_')):
                count += 1
            if any(col.startswith('xrapm_') and pd.notna(row.get(col)) and row.get(col) != 0 
                   for col in combined.columns if col.startswith('xrapm_')):
                count += 1
            return count
        
        combined['data_source_count'] = combined.apply(count_data_sources, axis=1)
        
        # Sort by data source count (descending) so we keep the entry with most data
        combined = combined.sort_values('data_source_count', ascending=False)
        
        # Drop duplicates, keeping the first (which has most data sources)
        combined = combined.drop_duplicates(subset=['final_player_name'], keep='first')
        combined = combined.drop('data_source_count', axis=1)
        
        duplicate_count = initial_count - len(combined)
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate player entries before filtering")
        
        # Filter out players with no data in any metrics source
        print("\nFiltering players with no metrics data...")
        initial_count = len(combined)
        
        # Check if player has data in any of the metrics sources
        has_data = []
        for idx, row in combined.iterrows():
            has_lebron = any(col.startswith('lebron_') and pd.notna(row.get(col)) and row.get(col) != 0 
                           for col in combined.columns if col.startswith('lebron_'))
            has_epm = any(col.startswith('epm_') and pd.notna(row.get(col)) and row.get(col) != 0 
                        for col in combined.columns if col.startswith('epm_'))
            has_darko = any(col.startswith('darko_') and pd.notna(row.get(col)) and row.get(col) != 0 
                          for col in combined.columns if col.startswith('darko_'))
            has_xrapm = any(col.startswith('xrapm_') and pd.notna(row.get(col)) and row.get(col) != 0 
                          for col in combined.columns if col.startswith('xrapm_'))
            
            has_data.append(has_lebron or has_epm or has_darko or has_xrapm)
        
        combined['has_data'] = has_data
        combined = combined[combined['has_data']].drop('has_data', axis=1)
        
        filtered_count = len(combined)
        print(f"  Filtered from {initial_count} to {filtered_count} players with metrics data")
        print(f"  Removed {initial_count - filtered_count} players with no data")
        
        # Current team is already set from roster data
        combined['current_team'] = combined['team']
        
        # 3. Create composite scores
        print("\nCreating composite scores...")
        
        cs = combined[['final_player_name', 'team']].copy()
        cs.columns = ['Player', 'Team']
        
        # Add basic info - use defaults for missing data
        cs['Season'] = 2025  # Current season
        
        # Use LeBRON age when available, don't default to 25 if missing
        if 'lebron_Age' in combined.columns:
            cs['Age'] = combined['lebron_Age']  # Keep NaN if missing
        else:
            cs['Age'] = None  # Explicitly None if column doesn't exist
        
        # Use LeBRON position when available, otherwise default to 'G'
        if 'lebron_Position' in combined.columns:
            cs['Pos'] = combined['lebron_Position'].fillna('G')
        else:
            cs['Pos'] = 'G'
        
        # Try to get Lebron data where available
        if 'lebron_Year' in combined.columns:
            cs['lebron_Year'] = combined['lebron_Year'].fillna(2025)
        else:
            cs['lebron_Year'] = 2025
            
        if 'lebron_Age' in combined.columns:
            cs['lebron_Age'] = combined['lebron_Age']  # Keep NaN if missing
        else:
            cs['lebron_Age'] = None
            
        if 'lebron_Position' in combined.columns:
            cs['lebron_Position'] = combined['lebron_Position'].fillna('G')
        else:
            cs['lebron_Position'] = 'G'
        
        # Calculate MP65 if Lebron data available
        if 'lebron_MIN' in combined.columns and 'lebron_G' in combined.columns:
            cs['MP65'] = (combined['lebron_MIN'] / combined['lebron_G'] * 65).fillna(30)
        else:
            cs['MP65'] = 30  # Default minutes
        # Handle xRAPM defense (invert if available)
        if 'xrapm_Defense(*)' in combined.columns:
            combined['xrapm_Defense(*)'] = -1*combined['xrapm_Defense(*)']
        
        # Assign metrics - keep NaN for missing data (don't fill with 0, as 0 is an actual value)
        # Prefer multi-year LEBRON fields when available
        if 'lebron_multiOLEBRON' in combined.columns:
            cs['lebron_off'] = combined['lebron_multiOLEBRON']
        elif 'lebron_predOLEBRON' in combined.columns:
            cs['lebron_off'] = combined['lebron_predOLEBRON']
        else:
            cs['lebron_off'] = None
            
        if 'lebron_multiDLEBRON' in combined.columns:
            cs['lebron_def'] = combined['lebron_multiDLEBRON']
        elif 'lebron_predDLEBRON' in combined.columns:
            cs['lebron_def'] = combined['lebron_predDLEBRON']
        else:
            cs['lebron_def'] = None
            
        if 'xrapm_Offense' in combined.columns:
            cs['xrapm_off'] = combined['xrapm_Offense']
        else:
            cs['xrapm_off'] = None
            
        if 'xrapm_Defense(*)' in combined.columns:
            cs['xrapm_def'] = combined['xrapm_Defense(*)']
        else:
            cs['xrapm_def'] = None
            
        if 'darko_o_dpm' in combined.columns:
            cs['darko_off'] = combined['darko_o_dpm']
        else:
            cs['darko_off'] = None
            
        if 'darko_d_dpm' in combined.columns:
            cs['darko_def'] = combined['darko_d_dpm']
        else:
            cs['darko_def'] = None
            
        if 'epm_oepm' in combined.columns:
            cs['epm_off'] = combined['epm_oepm']
        else:
            cs['epm_off'] = None
            
        if 'epm_depm' in combined.columns:
            cs['epm_def'] = combined['epm_depm']
        else:
            cs['epm_def'] = None
        
        # Scale metrics - use only non-NaN values for mean/std calculations
        # These will be NaN if no EPM data exists
        epm_off_mean = cs['epm_off'].mean(skipna=True)
        epm_off_std = cs['epm_off'].std(skipna=True) if pd.notna(epm_off_mean) else 0
        epm_def_mean = cs['epm_def'].mean(skipna=True)
        epm_def_std = cs['epm_def'].std(skipna=True) if pd.notna(epm_def_mean) else 0
        
        # LEBRON data removed from composite - using EPM, xRAPM, and DARKO only
        cs['xrapm_off_scaled'] = scale_to_target(cs['xrapm_off'], epm_off_mean, epm_off_std)
        cs['darko_off_scaled'] = scale_to_target(cs['darko_off'], epm_off_mean, epm_off_std)
        cs['epm_off_scaled'] = cs['epm_off']
        
        cs['xrapm_def_scaled'] = scale_to_target(cs['xrapm_def'], epm_def_mean, epm_def_std)
        cs['darko_def_scaled'] = scale_to_target(cs['darko_def'], epm_def_mean, epm_def_std)
        cs['epm_def_scaled'] = cs['epm_def']
        
        # Calculate combined metrics - only average available (non-NaN) metrics
        # Using EPM, xRAPM, and DARKO (LEBRON removed due to data issues)
        cs['combined_off'] = cs[['xrapm_off_scaled', 'darko_off_scaled', 'epm_off_scaled']].mean(axis=1, skipna=True)
        cs['combined_def'] = cs[['xrapm_def_scaled', 'darko_def_scaled', 'epm_def_scaled']].mean(axis=1, skipna=True)
        # combined_tot will be NaN if either combined_off or combined_def is NaN
        cs['combined_tot'] = cs['combined_off'] + cs['combined_def']
        
        # Calculate projections
        cs['Multi-Year WAR'] = (0.1141*cs['combined_tot']*cs['combined_tot']+1.3037*cs['combined_tot']+2.8285)*1.05
        cs['Multi-Year PV'] = cs['Multi-Year WAR'] * 6000000
        
        cs['Y1_off'] = cs['combined_off']+1.8531-0.0675*(cs['Age'].fillna(0)+1)
        cs['Y1_def'] = cs['combined_def']+0.7272-0.0261*(cs['Age'].fillna(0)+1)
        cs['Y1_tot'] = cs['Y1_off'] + cs['Y1_def']
        cs['Y1_war'] = (0.1141*cs['Y1_tot']*cs['Y1_tot']+1.3037*cs['Y1_tot']+2.8285)*1.05
        cs['Y1_PV'] = cs['Y1_war']*6000000*1.07
        
        cs['Y2_off'] = cs['Y1_off']+1.8531-0.0675*(cs['Age'].fillna(0)+2)
        cs['Y2_def'] = cs['Y1_def']+0.7272-0.0261*(cs['Age'].fillna(0)+2)
        cs['Y2_tot'] = cs['Y2_off'] + cs['Y2_def']
        cs['Y2_war'] = (0.1141*cs['Y2_tot']*cs['Y2_tot']+1.3037*cs['Y2_tot']+2.8285)*1.05
        cs['Y2_PV'] = cs['Y2_war']*6000000*1.07*1.1
        
        cs['Y3_off'] = cs['Y2_off']+1.8531-0.0675*(cs['Age'].fillna(0)+3)
        cs['Y3_def'] = cs['Y2_def']+0.7272-0.0261*(cs['Age'].fillna(0)+3)
        cs['Y3_tot'] = cs['Y3_off'] + cs['Y3_def']
        cs['Y3_war'] = (0.1141*cs['Y3_tot']*cs['Y3_tot']+1.3037*cs['Y3_tot']+2.8285)*1.05
        cs['Y3_PV'] = cs['Y3_war']*6000000*1.07*1.1*1.1
        
        cs['Y4_off'] = cs['Y3_off']+1.8531-0.0675*(cs['Age'].fillna(0)+4)
        cs['Y4_def'] = cs['Y3_def']+0.7272-0.0261*(cs['Age'].fillna(0)+4)
        cs['Y4_tot'] = cs['Y4_off'] + cs['Y4_def']
        cs['Y4_war'] = (0.1141*cs['Y4_tot']*cs['Y4_tot']+1.3037*cs['Y4_tot']+2.8285)*1.05
        cs['Y4_PV'] = cs['Y4_war']*6000000*1.07*1.1*1.1*1.1
        
        cs['Y5_off'] = cs['Y4_off']+1.8531-0.0675*(cs['Age'].fillna(0)+5)
        cs['Y5_def'] = cs['Y4_def']+0.7272-0.0261*(cs['Age'].fillna(0)+5)
        cs['Y5_tot'] = cs['Y5_off'] + cs['Y5_def']
        cs['Y5_war'] = (0.1141*cs['Y5_tot']*cs['Y5_tot']+1.3037*cs['Y5_tot']+2.8285)*1.05
        cs['Y5_PV'] = cs['Y5_war']*6000000*1.07*1.1*1.1*1.1*1.1
        
        # Remove duplicate players - keep first occurrence (typically the one with more complete data)
        print("\nRemoving duplicate players...")
        initial_count = len(cs)
        cs = cs.drop_duplicates(subset=['Player'], keep='first')
        duplicate_count = initial_count - len(cs)
        if duplicate_count > 0:
            print(f"  Removed {duplicate_count} duplicate player entries")
        
        # 4. Push to GitHub
        print("\nPushing to GitHub...")
        
        # Push composite scores
        cs.to_csv('Composite Projections copy.csv', index=False)
        print("✓ Saved Composite Projections copy.csv")
        
        # Push skills data
        if not skills.empty:
            skills.to_csv('skills_data_full_response.csv', index=False)
            print("✓ Saved skills_data_full_response.csv")
        
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
