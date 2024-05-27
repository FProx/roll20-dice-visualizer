import argparse
import re, mmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(prog='Roll20 Dice Analyzer',
                                 description='This script takes in a chat archive from roll20.net and generates plots with dice distributions for each player.',
                                 epilog='For inquiries, please contact @FProx on GitHub https://github.com/FProx')
parser.add_argument('file_path', help='relative path to chat archive. Its format should be .htm')
parser.add_argument('-d', '--dice_size', type=int, default=20)
parser.add_argument('--pseudonymized', action='store_true', default=False)

player_pattern = br'data-playerid="([^"]+)">(?:(?!<div class="message).)+'
dice_pattern = r'diceroll d(\d+).+?didroll">(\d+)'
name_pattern = r'<span class="by">([^:]+):</span>'

def plot_dice_rolls_of_size(df, dice_size=20):
    dsize_df = df.loc[df['dice_size'] == dice_size]
    unique_result_counts = dsize_df.groupby(df.columns.tolist(), as_index=False).size()
    total_rolls_per_player = unique_result_counts[['playerid', 'size']].groupby(['playerid']).sum()
    unique_result_counts['player_percentage'] = unique_result_counts.apply(lambda row: row['size'] / total_rolls_per_player.at[row['playerid'], 'size'], axis=1)
    
    for player in unique_result_counts['playerid'].unique():
        player_rolls = dsize_df.loc[dsize_df['playerid'] == player]
        player_result_counts = unique_result_counts.loc[unique_result_counts['playerid'] == player].copy()
        
        player_name = player_rolls['player_name'].iat[0] if 'player_name' in player_rolls.columns.tolist() else player
        
        expected_percentage = 1 / dice_size
        actual_percentages = player_result_counts['player_percentage'].tolist()
        player_mae = np.sum([abs(x - expected_percentage) for x in actual_percentages]) / len(actual_percentages)
        player_rmse = np.sqrt(np.sum([(x - expected_percentage)**2 for x in actual_percentages]))
        player_rolls_total = total_rolls_per_player.at[player, 'size']
        
        roll_description = player_rolls['dice_roll'].describe()
        percentiles = np.array([roll_description['25%'], roll_description['50%'], roll_description['75%'], roll_description['max']])
        player_result_counts['nearest_distance'] = player_result_counts.apply(lambda row: min(row['dice_roll'], abs(row['dice_roll'] - percentiles).min()), axis=1)
        
        palette = 'flare_r'
        sns.set_style('whitegrid')
        ax = sns.barplot(data=player_result_counts, x='dice_roll', y='player_percentage', order=np.arange(dice_size)+1, hue='nearest_distance', palette=palette, legend=False)
        ax.axhline(expected_percentage, c=sns.color_palette(palette)[0], ls='--')
        plt.xlabel(f'd{dice_size} result')
        plt.yticks(np.arange(0, player_result_counts['player_percentage'].max(), .01))
        plt.ylabel('percentage')
        title = f'd{dice_size} distribution for {player_name} ({player_rolls_total} rolls)\nRMSE={player_rmse:.4f}; MAE={player_mae:.4f};'
        plt.title(title)
        plt.show()

def create_dataframe_from_roll20_chat(raw_text_path, is_pseudonomized=False):
    roll_data = {'playerid': [], 'dice_size': [], 'dice_roll': []}
    player_name_map = {}
    
    with open(raw_text_path, 'r+') as file:
        file_data = mmap.mmap(file.fileno(), 0)
        message_matches = re.finditer(player_pattern, file_data)
        for message_match in message_matches:
            message = message_match.group(0).decode('utf-8')
            playerid = message_match.group(1).decode('utf-8')
            if not (is_pseudonomized or playerid in player_name_map.keys()):
                name_match = re.search(name_pattern, message)
                if name_match:
                    player_name_map[playerid] = name_match.group(1)
            diceroll_matches = re.finditer(dice_pattern, message)
            for diceroll_match in diceroll_matches:
                size, roll = diceroll_match.groups()
                roll_data['playerid'].append(playerid)
                roll_data['dice_size'].append(int(size))
                roll_data['dice_roll'].append(int(roll))
    
    roll_df = pd.DataFrame.from_dict(roll_data, orient='columns')
    if not is_pseudonomized:
        roll_df['player_name'] = roll_df.apply(lambda row: player_name_map.get(row['playerid'], None), axis=1)
    return roll_df

def main(args):
    roll_df = create_dataframe_from_roll20_chat(args.file_path, args.pseudonymized)
    plot_dice_rolls_of_size(roll_df, args.dice_size)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)