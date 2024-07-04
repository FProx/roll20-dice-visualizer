import argparse
import re, mmap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(prog='Roll20 Dice Analyzer',
                                 description='This script takes in a chat archive from roll20.net and generates plots with dice distributions for each player.',
                                 epilog='For inquiries, please contact @FProx on GitHub: https://github.com/FProx')
parser.add_argument('file_path', help='relative path to chat archive. Its format should be .htm')
parser.add_argument('-d', '--dice_size', type=int)
names_group = parser.add_mutually_exclusive_group()
names_group.add_argument('-p', '--players', nargs='+')
names_group.add_argument('-x', '--exclude', nargs='+')
parser.add_argument('--absolute', action='store_true', default=False)
parser.add_argument('--pseudonymized', action='store_true', default=False)
parser.add_argument('-s', '--save_figure', action='store', default=False)

player_pattern = br'data-playerid="([^"]+)">(?:(?!<div class="message).)+'
dice_pattern = r'diceroll d(\d+).+?didroll">(\d+)'
name_pattern = r'<span class="by">([^:]+):</span>'

def create_dataframe_from_roll20_chat(raw_text_path, only_include_players=None, exclude_players=None, is_pseudonomized=False):
    roll_data = {'playerid': [], 'dice_size': [], 'dice_roll': []}
    player_name_map = {} # maps playerid to the player name as it would be displayed in the chat log.
    
    with open(raw_text_path, 'r+') as file: # fill roll_data dict
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
    
    roll_df = pd.DataFrame.from_dict(roll_data, orient='columns') # create df from dict and filter it
    roll_df['display_name'] = roll_df['playerid'] if is_pseudonomized else roll_df.apply(lambda row: player_name_map.get(row['playerid'], None), axis=1)
    if only_include_players:
        roll_df = roll_df.loc[roll_df['display_name'].isin(only_include_players)]
    elif exclude_players:
        roll_df = roll_df.drop(roll_df.loc[roll_df['display_name'].isin(exclude_players)].index)
    
    return roll_df[['display_name', 'dice_size', 'dice_roll']]

def normalize_dice_df(df):
    norm_df = df.copy()
    norm_df['dice_roll'] = (norm_df['dice_roll']-1)/(norm_df['dice_size']-1)
    return norm_df

def filter_dice_size(df, dice_size):
    filtered_df = df.loc[df['dice_size'] == dice_size]
    return filtered_df

def generate_plot_args(df, dice_size=None, show_count=False):
    total_rolls = len(df)
    player_total_rolls = df.groupby(['display_name']).count()['dice_size']
    df['display_name_and_rolls'] = df['display_name'].apply(lambda name: f'{name} ({player_total_rolls[name]} rolls)')
    
    dice_result_dist_string = f'd{dice_size}' if dice_size else 'Dice'
    dice_result_dist_string += ' Distribution'
    title = f'Absolute {dice_result_dist_string}' if show_count else f'Relative {dice_result_dist_string}'
    is_single_player = len(df['display_name'].unique()) == 1
    if is_single_player:
        title += f' for Player {df['display_name'].iloc[0]} ({player_total_rolls[df['display_name'].iloc[0]]} rolls)'
    else:
        title += f' ({total_rolls} rolls)'
    
    figure_args = {
        'df': df,
        'title': title,
        'xlabel': 'Dice Result' if dice_size else 'Dice Result Percentile',
        'xticks': np.arange(1, dice_size + 1, 1) if dice_size else np.arange(0, 1.1, .1)
    }
    histplot_args = {
        'x': 'dice_roll',
        'hue': None if is_single_player else 'display_name_and_rolls',
        'stat': 'count' if show_count else 'probability',
        'bins': dice_size if dice_size else 6,
        'discrete': True if dice_size else False,
        'common_norm': True if is_single_player else False,
        'element': 'bars' if is_single_player else 'step', # for visibility
        'fill': True if is_single_player else False
    }
    return figure_args, histplot_args

def plot_dice_results(df, title, xlabel, xticks, save_figure=False, **histplot_args):
    sns.set_style('whitegrid')
    
    ax = sns.histplot(df, **histplot_args)
    plt.title(title)
    if histplot_args['discrete']:
        plt.xlim(df[histplot_args['x']].min() - .5, df[histplot_args['x']].max() + .5)
    else:
        plt.xlim(df[histplot_args['x']].min(), df[histplot_args['x']].max())
    plt.xticks(xticks)
    plt.xlabel(xlabel)
    
    if ax.get_legend():
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title='Player Name (# Rolls)')
        plt.tight_layout()
    if save_figure:
        plt.savefig(f'{save_figure}/{title}.svg')
    else:
        plt.show()

def main(args):
    roll_df = create_dataframe_from_roll20_chat(args.file_path, args.players, args.exclude, args.pseudonymized)
    if not args.dice_size:
        roll_df = normalize_dice_df(roll_df)
    else:
        roll_df = filter_dice_size(roll_df, args.dice_size)
    figure_args, histplot_args = generate_plot_args(roll_df, args.dice_size, args.absolute)
    plot_dice_results(**figure_args, **histplot_args, save_figure=args.save_figure)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
