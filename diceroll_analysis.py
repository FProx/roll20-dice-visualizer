import argparse
import re, mmap
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def iso_datetime_from_string(input):
    try:
        return datetime.fromisoformat(input)
    except ValueError:
        raise argparse.ArgumentTypeError(f'not a valid date: {input!r}\nPlease specify the date using ISO format.')

parser = argparse.ArgumentParser(prog='Roll20 Dice Analyzer',
                                 description='This script takes in a chat archive from roll20.net and generates plots with dice distributions for each player.',
                                 epilog='For inquiries, please contact @FProx on GitHub: https://github.com/FProx')
# required arguments
parser.add_argument('file_path', help='relative path to chat archive. Its format should be .htm')

# filter arguments
filter_group = parser.add_argument_group(title='filter arguments', description='All of these are optional, use them to narrow down the dataset to the parts that are of interest.')
filter_group.add_argument('-d', '--dice_type', type=int, help='use this if results of a specific dice type should be explored. Please pass the side count as an integer.')
names_group = filter_group.add_mutually_exclusive_group()
names_group_help_appendix = 'Separate multiple players by leaving a space. If a player name contains a space, please enclose it with ". Only exact name matches work.'
names_group.add_argument('-p', '--players', nargs='+', help=f'display only results of one or multiple players. {names_group_help_appendix}')
names_group.add_argument('-x', '--exclude', nargs='+', help=f'exclude one or multiple players. {names_group_help_appendix}')
date_group = parser.add_argument_group(title='date arguments', description='Narrow down dataset to specific date interval.')
date_group.add_argument('-b', '--before', type=iso_datetime_from_string, help='only process dice rolls before (and including) this date. Please specify the date using ISO format.')
date_group.add_argument('-a', '--after', type=iso_datetime_from_string, help='only process dice rolls after (and including) this date. Please specify the date using ISO format.')
date_group.add_argument('-e', '--exact', type=iso_datetime_from_string, help='only process dice rolls that happened on this specific day. Please specify the date using ISO format.')

# display arguments
display_group = parser.add_argument_group(title='display arguments', description='Change how the results should be displayed.')
display_group.add_argument('--absolute', action='store_true', default=False, help='if set, displays absolute result counts instead of probabilities (the latter is the default).')
display_group.add_argument('--pseudonymized', action='store_true', default=False, help='if set, shows pseudonomized player ids instead of the names shown in the chat window (the latter is the default).')
display_group.add_argument('-s', '--save_path', action='store', help='saves the generated figure at the specified path and does not show it to the user. By default, the generated figure will be shown to the user but not saved.')

player_pattern = br'data-playerid="([^"]+)">(?:(?!<div class="message).)+'
dice_pattern = r'diceroll d(\d+).+?didroll">(\d+)'
name_pattern = r'<span class="by">([^:]+)'
date_pattern = r'<span class="tstamp"[^>]+>([^<]*)'

date_format = '%B %d, %Y %I:%M%p' # might break if the log format changes. Keeping it here for ease of future access.

def create_dataframe_from_roll20_chat(raw_text_path, is_pseudonomized=False):
    roll_data = {'playerid': [], 'date': [], 'dice_type': [], 'dice_roll': []}
    player_name_map = {} # maps playerid to the player name as it would be displayed in the chat log.
    date = None
    
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
            date_match = re.search(date_pattern, message)
            if date_match:
                date = date_match.group(1)
            diceroll_matches = re.finditer(dice_pattern, message)
            for diceroll_match in diceroll_matches:
                size, roll = diceroll_match.groups()
                roll_data['playerid'].append(playerid)
                roll_data['date'].append(datetime.strptime(date, date_format))
                roll_data['dice_type'].append(int(size))
                roll_data['dice_roll'].append(int(roll))
    
    roll_df = pd.DataFrame.from_dict(roll_data, orient='columns') # create df from dict
    roll_df['display_name'] = roll_df['playerid'] if is_pseudonomized else roll_df.apply(lambda row: player_name_map.get(row['playerid'], None), axis=1)
    return roll_df[['display_name', 'date', 'dice_type', 'dice_roll']]

def filter_dice_type(df, dice_type):
    filtered_df = df.loc[df['dice_type'] == dice_type]
    return filtered_df

def normalize_dice_df(df):
    norm_df = df.copy()
    norm_df['dice_roll'] = (norm_df['dice_roll']-1)/(norm_df['dice_type']-1)
    return norm_df

def filter_df(df, only_include_players, exclude_players, before_date, after_date, exact_date, dice_type):
    filtered_df = df.copy()
    if only_include_players:
        filtered_df = filtered_df.loc[filtered_df['display_name'].isin(only_include_players)]
    elif exclude_players:
        filtered_df = filtered_df.drop(filtered_df.loc[filtered_df['display_name'].isin(exclude_players)].index)
    
    if exact_date:
        before_date = after_date = exact_date
    if before_date:
        filtered_df = filtered_df.loc[filtered_df['date'] < before_date + timedelta(days=1)]
    if after_date:
        filtered_df = filtered_df.loc[filtered_df['date'] >= after_date]
    
    if dice_type: # if specified dice type, then we don't need to normalize x to [0, 1]
        filtered_df = filter_dice_type(filtered_df, args.dice_type)
    else:
        filtered_df = normalize_dice_df(filtered_df)
    return filtered_df

def generate_plot_args(df, dice_type=None, show_count=False):
    total_rolls = len(df)
    player_total_rolls = df.groupby(['display_name']).count()['dice_type']
    df['display_name_and_rolls'] = df['display_name'].apply(lambda name: f'{name} ({player_total_rolls[name]} rolls)')
    
    dice_result_dist_string = f'd{dice_type}' if dice_type else 'Dice'
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
        'xlabel': 'Dice Result' if dice_type else 'Dice Result Percentile',
        'xticks': np.arange(1, dice_type + 1, 1) if dice_type else np.arange(0, 1.1, .1)
    }
    histplot_args = {
        'x': 'dice_roll',
        'hue': None if is_single_player else 'display_name_and_rolls',
        'stat': 'count' if show_count else 'probability',
        'bins': dice_type if dice_type else 6,
        'discrete': True if dice_type else False,
        'common_norm': True if is_single_player else False,
        'element': 'bars' if is_single_player else 'step', # for visibility
        'fill': True if is_single_player else False
    }
    return figure_args, histplot_args

def plot_dice_results(df, title, xlabel, xticks, save_path=False, **histplot_args):
    sns.set_style('whitegrid')
    sns.set_palette('muted')
    
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
    if save_path:
        plt.savefig(f'{save_path}/{title}.svg')
    else:
        plt.show()

def main(args):
    roll_df = create_dataframe_from_roll20_chat(args.file_path, args.pseudonymized)
    roll_df = filter_df(roll_df, args.players, args.exclude, args.before, args.after, args.exact, args.dice_type)
    figure_args, histplot_args = generate_plot_args(roll_df, args.dice_type, args.absolute)
    plot_dice_results(**figure_args, **histplot_args, save_path=args.save_path)

if __name__ == '__main__':
    args = parser.parse_args()
    if (args.before or args.after) and args.exact:
        parser.error(f'argument -b/--before and -a/--after: not allowed with argument -e/--exact')
    main(args)
