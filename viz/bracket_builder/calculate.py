import os
import numpy as np
import pandas as pd


def exhaust_possible_seeds(tourney_slots_df, seeds, max_depth_seeds = None):
    """
    Recursive function to get all of the possible seeds for each slot.
     Can be used to add a "possible_teams" column to the tournament slots data

    Parameters
    ----------
    tourney_slots_df : DataFrame
        The provided tournament slots data.
    parent_seeds : list
        A list of seeds from each round. (When rd > 1 they are really the names of rounds).

    Returns
    -------
    list
        A list of all possible seeds that could reach that round.

    """
    
    ## Get slots that seeds came from (earlier in the tourney)
    earlier_rd_rows = tourney_slots_df[tourney_slots_df['Slot'].isin(seeds)].copy()
    
    ## Get the round that these are still "seeds"
    seed_rd_rows = tourney_slots_df[(tourney_slots_df['StrongSeed'].isin(seeds)) |
                                    (tourney_slots_df['WeakSeed'].isin(seeds))].copy()
    ## Find the earliest round that the seed is in
    min_round_seeds = np.min(seed_rd_rows['round'])
    
    ## Find the ones in this dataframe that are new
    new_seeds = list((set(earlier_rd_rows['StrongSeed']) | set(earlier_rd_rows['WeakSeed'])) - set(seeds))
    if min_round_seeds == 1:
        seeds_reached_max_depth = list(set(seeds) - set(new_seeds))
    else:
        seeds_reached_max_depth = None
    
    if len(new_seeds) > 0:
        ## If there's another level deeper, add new seeds we got this time, and run again
        return exhaust_possible_seeds(tourney_slots_df, new_seeds, seeds_reached_max_depth)
    else:
        ## If this is the deepest level, don't run it again, just return the list
        if min_round_seeds == 0:
            possible_seeds = seeds
            if max_depth_seeds is not None:
                possible_seeds.extend(max_depth_seeds)
        else:
            possible_seeds = seeds
        
        possible_seeds = list(set([s for s in possible_seeds if s not in tourney_slots_df['Slot']]))  ## dedup
        possible_seeds.sort()
        
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


def find_round_prob(sub_df, probs_df, team_id, rnd):
    """
    Get the probability that a given team reaches a round
    
    Parameters
    ----------
    sub_df : DataFrame
        Submission dataframe with round added.
    team_name : str
        Name of team.
    rnd : int
        Round of interest to get probability for.
    
    Returns
    -------
    rnd_prob : float
        Probability that a team reaches the round.
    """
    ## Get all possible matchups for the team in prior round
    team_round_preds = sub_df[((sub_df['TeamID_1'] == team_id) | 
                               (sub_df['TeamID_2'] == team_id)) &
                               (sub_df['round'] == rnd-1)].copy()
    
    ## Defining a dict of column names corresponding to the round before
    rd_cols = {0: 'Round0',
               1: 'Round1',
               2: 'Round2',
               3: 'Sweet16',
               4: 'Elite8',
               5: 'Final4',
               6: 'Final',
               7: 'Champ'}
    
    if len(team_round_preds) == 1:
        ## only 1 matchup to worry about, just get the corresponding prob for that team
        if team_id in list(team_round_preds['TeamID_1']):
            rnd_prob = float(team_round_preds['Pred'])
        else:
            rnd_prob = 1-float(team_round_preds['Pred'])
        if rnd > 0:
            team_prob_reaching = float(probs_df[probs_df['TeamID'] == team_id][rd_cols[rnd-1]])
            rnd_prob = rnd_prob*team_prob_reaching
    elif len(team_round_preds) == 0:
        ## If they didn't have any games in the prior round, 
        ### probability is 1 (applies to non-play-in teams making round 1)
        rnd_prob = 1
    else:
        ## Using a json-like structure containing probabilities that an opposing team would make that round
        ### and probability that the team of interest would beat that team
        conditional_team_probs = {}
        
        ## Keys for the dict: possible teams
        possible_teams = list((set(team_round_preds['TeamID_1']) | 
                               set(team_round_preds['TeamID_2'])) - set([team_id]))
        
        ## Used for lookup in value 1 below
        prob_reaching_df = (probs_df[probs_df['TeamID'].isin(possible_teams)]
                                     [['TeamID', rd_cols[rnd-1]]]
                                     .rename(columns = {rd_cols[rnd-1]: 'prob_reaching_rd'}))
        
        for t in possible_teams:
            within_dict = {}
            ## Value 1 (conditional part): chances that the opposing teams make the round
            within_dict['prob_reaching'] = float(prob_reaching_df[prob_reaching_df['TeamID'] == t]['prob_reaching_rd'])            
            
            ## Value 2: win probability for the team of interest over the possible opposing team
            if team_id < t:
                win_prob = float(team_round_preds[team_round_preds['TeamID_2'] == t]['Pred'])
            else:
                win_prob = 1-float(team_round_preds[team_round_preds['TeamID_1'] == t]['Pred'])
            within_dict['win_prob'] = win_prob
            conditional_team_probs[t] = within_dict
        
        ### Get probability for the team for the prior round
        team_prob_reaching = float(probs_df[probs_df['TeamID'] == team_id][rd_cols[rnd-1]])
        
        ### Calculate probability of making to the round of interest
        rnd_games_probs = (np.sum([(t['prob_reaching']*t['win_prob'])
                             for t in conditional_team_probs.values()]))
        
        rnd_prob = rnd_games_probs*team_prob_reaching
    
    return rnd_prob


def compute_conditional_probs(sub_filepath, league = 'men'):
    """
    Function to take the submission file and calculate conditional probabilities for each team/round.

    :param sub_filepath (str): location of Kaggle data submission
    :param league (str): either 'men' or 'women'
    :return: DataFrame containing probabilities for each team to make each round

    """
    
    ## Prefix according to league
    if league == 'men':
        prefix = 'M'
    else:
        prefix = 'W'
    
    ## Get the sample submission and break out the ID
    sub_df = pd.read_csv(sub_filepath)
    sub_df['Season']    = sub_df['ID'].apply(lambda x: x.split('_')[0]).astype(int)
    sub_df['TeamID_1']  = sub_df['ID'].apply(lambda x: x.split('_')[1]).astype(int)
    sub_df['TeamID_2']  = sub_df['ID'].apply(lambda x: x.split('_')[2]).astype(int)
    
    ## Get the seeds and slots
    tourney_seeds_df = pd.read_csv(f"{prefix}NCAATourneySeeds.csv")
    tourney_seeds_df = tourney_seeds_df[tourney_seeds_df['Season'] == 2021].copy()
    tourney_slots_df = pd.read_csv(f"{prefix}NCAATourneySlots.csv")
    if league == 'men':
        tourney_slots_df = tourney_slots_df[tourney_slots_df['Season'] == 2021].copy()
    
    ## Merge in seeds to submission
    tourney_seeds_df = tourney_seeds_df.rename(columns = {'TeamID': 'TeamID_1', 'Seed': 'Seed_1'})
    sub_df = sub_df.merge(tourney_seeds_df, on = ['TeamID_1', 'Season'])
    tourney_seeds_df = tourney_seeds_df.rename(columns = {'TeamID_1': 'TeamID_2', 'Seed_1': 'Seed_2'})
    sub_df = sub_df.merge(tourney_seeds_df, on = ['TeamID_2', 'Season'])
    
    ## Add a field to slots with the possible teams that could reach that round
    tourney_slots_df['round'] = tourney_slots_df['Slot'].apply(lambda x: int(x[1]) if x.startswith('R')
                                                               else 0)
    tourney_slots_df['possible_teams'] = tourney_slots_df.apply(lambda x: 
                                        exhaust_possible_seeds(tourney_slots_df,
                                                               [x['StrongSeed'],
                                                                x['WeakSeed']]), axis = 1)
    
    ## Add a column to tourney results with the round that the 2 teams meet
    sub_df['round'] = sub_df.apply(lambda x: get_round_met(tourney_slots_df, x['Seed_1'], x['Seed_2']),
                                             axis = 1)
        
    ## Get the team names to merge in
    team_names_df = pd.read_csv(f"{prefix}Teams.csv")
    
    ## Merge team 1 name
    team_names_df = (team_names_df[['TeamID', 'TeamName']]
                    .rename(columns={'TeamName': 'TeamName_1',
                                     'TeamID': 'TeamID_1'}))
    sub_df = sub_df.merge(team_names_df, how='left', on='TeamID_1')

    ## Merge team 2 name
    team_names_df = team_names_df.rename(columns={'TeamName_1': 'TeamName_2',
                                            'TeamID_1': 'TeamID_2'})
    sub_df = sub_df.merge(team_names_df, how='left', on='TeamID_2')
        
    ## For each team, go through each round and calculate the prob. that they'll be in the next round
    ##  based on the submission.
    team_names = list(set(sub_df['TeamName_1']) | set(sub_df['TeamName_2']))
    team_names.sort()
    probs_df = pd.DataFrame({'TeamName': team_names})
    team_names_df = team_names_df.rename(columns={'TeamName_2': 'TeamName',
                                            'TeamID_2': 'TeamID'})
    probs_df = probs_df.merge(team_names_df, on = 'TeamName')
    probs_df['Round0'] = 1
    
    ## Fill in probabilities, round by round
    probs_df['Round1']  = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 1))
    probs_df['Round2']  = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 2))
    probs_df['Sweet16'] = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 3))
    probs_df['Elite8']  = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 4))
    probs_df['Final4']  = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 5))
    probs_df['Final']   = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 6))
    probs_df['Champ']   = probs_df['TeamID'].apply(lambda x: find_round_prob(sub_df, probs_df, x, 7))
    
    probs_df = probs_df.drop(columns = ['Round0'])
    
    return probs_df