"""

Script to create custom features for Team IAA submissions.

"""

__author__ = 'dickeym'

import pandas as pd 
import numpy as np 

def get_conf_win_pcts(league):
    """
    Calculate regular season standings (conf. win pct) within each conference 
     and season.

    Parameters
    ----------
    league : str
        Either 'women' or 'men'.

    Returns
    -------
    conf_win_pcts_df : DataFrame
        Conference win %s and rankings/standings by team/season.
    """
    
    ## Read and merge datasets
    if league == 'women':
        prefix = 'W'
    else:
        prefix = 'M'
    teams_df = pd.read_csv(f"{prefix}Teams.csv")
    conferences_df = pd.read_csv(f"{prefix}TeamConferences.csv")
    reg_season_results_df = pd.read_csv(f"{prefix}RegularSeasonCompactResults.csv")
    teams_df = teams_df.merge(conferences_df)
    
    teams_df.columns = [f'W{c}' if c != 'Season' else c for c in teams_df.columns]
    reg_season_results_df = reg_season_results_df.merge(teams_df, on = ['WTeamID', 'Season'])
    teams_df.columns = [f"L{c[1:]}" if c != 'Season' else c for c in teams_df.columns]
    reg_season_results_df = reg_season_results_df.merge(teams_df, on = ['LTeamID', 'Season'])    
    
    ## Limit to conference games
    conf_games = reg_season_results_df[reg_season_results_df['WConfAbbrev'] == 
                                   reg_season_results_df['LConfAbbrev']].copy()
    
    ## Gather number of wins and losses by team, conference, and season
    seasons, confs, teams, n_wins, n_losses  = [], [], [], [], []
    for i, row in conferences_df.iterrows():
        seasons.append(row['Season'])
        confs.append(row['ConfAbbrev'])
        teams.append(row['TeamID'])
        n_wins.append(len(conf_games[(conf_games['Season'] == row['Season']) &
                                     (conf_games['WTeamID'] == row['TeamID'])]))
        n_losses.append(len(conf_games[(conf_games['Season'] == row['Season']) &
                                     (conf_games['LTeamID'] == row['TeamID'])]))
    conf_win_pcts_df = pd.DataFrame({'Season': seasons,
                                     'ConfAbbrev': confs,
                                     'TeamID': teams,
                                     'n_wins': n_wins,
                                     'n_losses': n_losses})
    conf_win_pcts_df['conf_win_pct'] = conf_win_pcts_df['n_wins']/(conf_win_pcts_df['n_wins'] + conf_win_pcts_df['n_losses'])
    
    ## Rank within conference and season
    conf_win_pcts_df['rank_in_conf'] = conf_win_pcts_df.groupby(['Season', 'ConfAbbrev'])['conf_win_pct'].rank(ascending=False, method = 'max')
    
    return conf_win_pcts_df


def exhaust_possible_seeds(tourney_slots_df, parent_seeds):
    """
    Recursive function to get all of the possible seeds for each slot.
     Can be used to add a "possible_teams" column to the tournament slots data

    Parameters
    ----------
    tourney_slots_df : DataFrame
        The provided tournament slots data.
    parent_seeds : list
        A list of seeds from rounds > 1 (which are really the names of rounds).

    Returns
    -------
    list
        A list of all possible seeds that could reach that round.

    """
    parent_rows = tourney_slots_df[
        tourney_slots_df['Slot'].isin(parent_seeds)].copy()
    if np.min(parent_rows['round']) != 1:
        parent_seeds.extend(list(parent_rows['StrongSeed']))
        parent_seeds.extend(list(parent_rows['WeakSeed']))
        return exhaust_possible_seeds(tourney_slots_df, parent_seeds)
    else:
        rd1_seeds = parent_rows[parent_rows['round'] == 1]
        possible_seeds = list(rd1_seeds['StrongSeed'])
        possible_seeds.extend(list(rd1_seeds['WeakSeed']))
        return possible_seeds


def get_round_met(tourney_slots_df, seed_1, seed_2):
    """
    Get the round that two seeds meet in the slots dataframe

    Parameters
    ----------
    tourney_slots_df : DataFrame
        The provided tournament slots data.
    seed_1 : str
        Seed of team 1.
    seed_2 : str
        Seed of team 2.

    Returns
    -------
    rd : int
        Minimum round number that two teams can feasibly meet.
    """
    
    teams_in_slot = tourney_slots_df.possible_teams.apply(lambda x: ((seed_1 in x) and 
                                                                    (seed_2 in x)))
    slots_with_teams = tourney_slots_df[teams_in_slot]
    
    rd = np.min(slots_with_teams['round'])
    
    return rd


def get_rolling_avg_round_reached(league, by = 'conf_standing', start_season = 2000,
                                 end_season = 2020, n_year_avg = 5):
    """
    Calculate avg. round reached in NCAA tourney the last X years for teams 
     that are in each standing/position in each conference.
     
    Parameters
    ----------
    league : str
        Either 'women' or 'men'.
    by : str
        Either 'conf_standing' or 'coach/team'.  Note: no coach data available for women's yet,
         therefore just group by 'TeamID' instead.
    current_season : int
        Year of current season to calculate the last X years for.
    n_year_avg : int
        How many years back to look at tournament results for.

    Returns
    -------
    full_avg_max_rd_df : DataFrame
        Avg. round reached in tourney by conference standing or coach since X year.
    """
    ## Read and merge datasets
    if league == 'women':
        prefix = 'W'
    else:
        prefix = 'M'
    ## Read in tourney results
    tourney_results_df = pd.read_csv(f"{prefix}NCAATourneyCompactResults.csv")
    tourney_seeds_df = pd.read_csv(f"{prefix}NCAATourneySeeds.csv")
    tourney_slots_df = pd.read_csv(f"{prefix}NCAATourneySlots.csv")
    
    ## Merge in seeds to tourney results
    tourney_seeds_df = tourney_seeds_df.rename(columns = {'TeamID': 'WTeamID', 'Seed': 'WSeed'})
    tourney_results_df = tourney_results_df.merge(tourney_seeds_df, on = ['WTeamID', 'Season'])
    tourney_seeds_df = tourney_seeds_df.rename(columns = {'WTeamID': 'LTeamID', 'WSeed': 'LSeed'})
    tourney_results_df = tourney_results_df.merge(tourney_seeds_df, on = ['LTeamID', 'Season'])
    
    ## Add a field to slots with the possible teams that could reach that round
    tourney_slots_df['round'] = tourney_slots_df['Slot'].apply(lambda x: int(x[1]))
    possible_teams = []
    for i, row in tourney_slots_df.iterrows():
        if row['round'] != 1:
            possible_seeds = exhaust_possible_seeds(
            tourney_slots_df, [row['StrongSeed'], row['WeakSeed']])
            possible_teams.append(possible_seeds)
        else:
            possible_teams.append([row['StrongSeed'], row['WeakSeed']])
    tourney_slots_df['possible_teams'] = possible_teams
    
    ## Add a column to tourney results with the round that the 2 teams met
    tourney_results_df['round'] = tourney_results_df.apply(lambda x: get_round_met(tourney_slots_df, x['WSeed'], x['LSeed']),
                                                  axis = 1)
    
    ## Double the tourney results to have one record per team playing
    tourney_results_df['TeamID'] = tourney_results_df.apply(lambda x: [x['WTeamID'], x['LTeamID']], axis = 1)
    tourney_results_df = tourney_results_df.explode('TeamID')
    
    ## Get conference winnning pcts and ranks by year/team to merge with tourney results
    if by == 'conf_standing':
        ## Merge conf standings into tourney results
        conf_win_pcts_df = get_conf_win_pcts(league).drop(columns = ['n_wins', 'n_losses'])
        tourney_results_df = tourney_results_df.merge(conf_win_pcts_df, on = ['TeamID', 'Season'])
        group_by_1_cols = ['Season', 'ConfAbbrev', 'rank_in_conf']
        group_by_2_cols = ['ConfAbbrev', 'rank_in_conf']
        colname_prefix = 'conf'
    elif (by == 'coach/team') and (league == 'men'):
        coach_df = pd.read_csv(f"{prefix}TeamCoaches.csv") ## right now prefix can only be M
        tourney_results_df = tourney_results_df.merge(coach_df, on = ['TeamID', 'Season'])
        group_by_1_cols = ['Season', 'CoachName']
        group_by_2_cols = 'CoachName'
        colname_prefix = 'coach'
    else:
        ## Instead of coach for women, we'll use teamID
        group_by_1_cols = ['Season', 'TeamID']
        group_by_2_cols = 'TeamID'
        colname_prefix = 'team'
        
    ## For teams that win the championship, add another 1 to their round to give credit for "advancing"
    tourney_results_df['round'] = tourney_results_df.apply(lambda x: x['round'] + 1 if 
                                                             ((x['round'] == 6) and (x['WTeamID']==x['TeamID']))
                                                            else x['round'], axis = 1)
    
    ## Roll across required seasons and calculate new avg. rounds by conf/standing
    avg_max_rd_dfs = []
    for season in range(start_season, end_season):
        ## Limit the tourney results to an n_year_avg range
        yr_results_df = tourney_results_df[(tourney_results_df['Season'] >=  season-n_year_avg) &
                                           (tourney_results_df['Season'] < season)]
        
        ## Get the maximum round reached for each season
        max_rounds = (yr_results_df.groupby(group_by_1_cols)
                      .agg({'round': max}).reset_index().rename(columns = {'round': 'max_round'}))
        
        ## Sum the maximum rounds by team to be later used in an average
        ### (avg. needs to include years where they were not in the tourney, counted as 0)
        avg_max_rd_df = (max_rounds.groupby(group_by_2_cols)
                                   .agg({'max_round': np.sum, 'Season': [len, max, min]})
                                   .reset_index().rename(columns = {'max_round': 'total_rounds'}))
        
        ## Get rid of the multi-level columns
        avg_max_rd_df.columns = ['_'.join(col) if '' not in col else ''.join(col) for col in
                                 avg_max_rd_df.columns.values]
        
        ## Calculate number of seasons and list the current season iteration as the "valid" szn.
        min_season = np.min(avg_max_rd_df['Season_min'])
        max_season = np.max(avg_max_rd_df['Season_max'])
        avg_max_rd_df['Season'] = season
        avg_max_rd_df['avg_rd_season_range'] = f"{min_season}-{max_season}"

        ## Calculate the average round reached
        avg_max_rd_df[f'{colname_prefix}_avg_round'] = avg_max_rd_df['total_rounds_sum']/(max_season-min_season+1)
        avg_max_rd_dfs.append(avg_max_rd_df)
    
    ## Add all the seasons together and format column names
    full_avg_max_rd_df = pd.concat(avg_max_rd_dfs).drop_duplicates()
    
    return full_avg_max_rd_df