import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages


class Bracket:
    """
    Plotter object used to draw a bracket in a few easy steps.
    """

    def __init__(self, n_teams, team_names=None, winners=None, win_prob_teams=None, win_probabilities=None):
        """
        Parameters
        ----------
        n_teams : int
            Number of teams in the tournament (32 or 64).
        team_names : list of str
            Optional. List of team names used to label the bracket.
            Assumes team names are in the order of top left, bottom left,
               top right, to bottom right.
        winners : list of lists of str
            Optional. A list of lists containing winners from each round.
            (i.e. 1st list is winners from rd. 1, 2nd is winners from rd. 2, etc.)
        win_prob_teams : list of str
            Optional. A list of team names to highlight with win probability lines.
        win_probabilities : list of lists of float
            Optional. A list of lists containing probabilities that each team makes each round
            e.g. if we're labeling 2 teams in a 4 round tournament, it may look like this:
               [ [0.9, 0.7, 0.3, 0.1], [0.8, 0.6, 0.25, 0.05] ]

        """
        self.n_teams = n_teams
        self.team_names = team_names
        self.winners = winners
        self.win_prob_teams = win_prob_teams
        self.win_probabilities = win_probabilities

    def draw_bracket(self):
        """
        Draw a blank tournament bracket based on the number of teams.
        """

        ## Set the height/width to be pretty large
        fig, ax = plt.subplots()
        fig.set_figheight(10)
        fig.set_figwidth(13)

        ## Limits are based on number of teams
        if self.n_teams == 32:
            ## 32 team bracket limits (smaller)
            plt.xlim([19, 201])
            plt.ylim([-1, 67])

        if self.n_teams == 64:
            ## 64 team bracket limits (larger)
            plt.xlim([-1, 221])
            plt.ylim([-1, 67])

            ## Get collection of lines unique to the 64 team version ("round 1")
            ### X values for left/right side
            ## Left side of bracket
            r1_left_x = np.array([0, 20])  # 1st round range x from 0 to 20
            r1_left_connect_x = np.array([20, 20])  # Xs to connect the two teams playing each other

            ## Right side of bracket
            r1_right_x = np.array([200, 220])  # 1st round range x from 0 to 20
            r1_right_connect_x = np.array([200, 200])  # Xs to connect the two teams playing each other

            ### Y values can correspond to either side (reuse these arrays)
            r1_ys = [np.repeat(np.array([x]), 2) for x in np.arange(0, 32, 2)]
            r1_ys.extend([np.repeat(np.array([x]), 2) for x in np.arange(36, 67, 2)])
            r1_connect_ys = [np.arange(x, x + 3, 2) for x in np.arange(0, 29, 4)]
            r1_connect_ys.extend([np.arange(x, x + 3, 2) for x in np.arange(36, 65, 4)])

            ## Make sequences of x,y pairs
            left_r1_lines = collect_lines(r1_left_x, r1_ys)
            left_r1_connect_lines = collect_lines(r1_left_connect_x, r1_connect_ys)

            right_r1_lines = collect_lines(r1_right_x, r1_ys)
            right_r1_connect_lines = collect_lines(r1_right_connect_x, r1_connect_ys)

            ### Add the line segments to the plot
            ## Left side
            ax.add_collection(left_r1_lines)
            ax.add_collection(left_r1_connect_lines)

            ## Right side
            ax.add_collection(right_r1_lines)
            ax.add_collection(right_r1_connect_lines)

        ### Collection of X values for the left side of the bracket
        r2_left_x = np.array([20, 40])
        r2_left_connect_x = np.array([40, 40])
        r3_left_x = np.array([40, 60])
        r3_left_connect_x = np.array([60, 60])
        r4_left_x = np.array([60, 80])
        r4_left_connect_x = np.array([80, 80])
        r5_left_x = np.array([80, 100])
        r5_left_connect_x = np.array([100, 100])
        final_left_x = np.array([115, 100])

        ### Collection of X values for the right side of the bracket
        r2_right_x = np.array([180, 200])
        r2_right_connect_x = np.array([180, 180])
        r3_right_x = np.array([160, 180])
        r3_right_connect_x = np.array([160, 160])
        r4_right_x = np.array([140, 160])
        r4_right_connect_x = np.array([140, 140])
        r5_right_x = np.array([120, 140])
        r5_right_connect_x = np.array([120, 120])
        final_right_x = np.array([120, 105])

        ## Champion Xs
        champ_x = np.array([90, 130])

        ### Y values for each round
        r2_ys = [np.repeat(np.array([x]), 2) for x in np.arange(1, 30, 4)]
        r2_ys.extend([np.repeat(np.array([x]), 2) for x in np.arange(37, 68, 4)])
        r2_connect_ys = [np.arange(x, x + 5, 4) for x in np.arange(1, 28, 8)]
        r2_connect_ys.extend([np.arange(x, x + 5, 4) for x in np.arange(37, 63, 8)])

        r3_ys = [np.repeat(np.array([x]), 2) for x in np.arange(3, 28, 8)]
        r3_ys.extend([np.repeat(np.array([x]), 2) for x in np.arange(38, 64, 8)])
        r3_connect_ys = [np.array([3, 11]), np.array([19, 27]),
                         np.array([38, 46]), np.array([54, 62])]

        r4_ys = [np.repeat(np.array([x]), 2) for x in np.arange(7, 24, 16)]
        r4_ys.extend([np.repeat(np.array([x]), 2) for x in np.arange(42, 59, 16)])
        r4_connect_ys = [np.array([7, 23]), np.array([42, 58])]

        r5_ys = [np.array([15, 15]), np.array([50, 50])]
        r5_connect_ys = [np.array([15, 50]), np.array([15, 50])]

        final_left_y = [np.array([35, 35])]
        final_right_y = [np.array([30, 30])]
        champ_ys = [np.array([60, 60])]

        ## Make sequences of x,y pairs (collections of lines)
        # left side
        left_r2_lines = collect_lines(r2_left_x, r2_ys)
        left_r2_connect_lines = collect_lines(r2_left_connect_x, r2_connect_ys)
        left_r3_lines = collect_lines(r3_left_x, r3_ys)
        left_r3_connect_lines = collect_lines(r3_left_connect_x, r3_connect_ys)
        left_r4_lines = collect_lines(r4_left_x, r4_ys)
        left_r4_connect_lines = collect_lines(r4_left_connect_x, r4_connect_ys)
        left_r5_lines = collect_lines(r5_left_x, r5_ys)
        left_r5_connect_lines = collect_lines(r5_left_connect_x, r5_connect_ys)

        # right side
        right_r2_lines = collect_lines(r2_right_x, r2_ys)
        right_r2_connect_lines = collect_lines(r2_right_connect_x, r2_connect_ys)
        right_r3_lines = collect_lines(r3_right_x, r3_ys)
        right_r3_connect_lines = collect_lines(r3_right_connect_x, r3_connect_ys)
        right_r4_lines = collect_lines(r4_right_x, r4_ys)
        right_r4_connect_lines = collect_lines(r4_right_connect_x, r4_connect_ys)
        right_r5_lines = collect_lines(r5_right_x, r5_ys)
        right_r5_connect_lines = collect_lines(r5_right_connect_x, r5_connect_ys)

        final_left_line = collect_lines(final_left_x, final_left_y)
        final_right_line = collect_lines(final_right_x, final_right_y)

        champ_line = collect_lines(champ_x, champ_ys)

        ### Add the lines common to both numbers of teams
        ## Left side
        ax.add_collection(left_r2_lines)
        ax.add_collection(left_r2_connect_lines)
        ax.add_collection(left_r3_lines)
        ax.add_collection(left_r3_connect_lines)
        ax.add_collection(left_r4_lines)
        ax.add_collection(left_r4_connect_lines)
        ax.add_collection(left_r5_lines)
        ax.add_collection(left_r5_connect_lines)

        ## Right side
        ax.add_collection(right_r2_lines)
        ax.add_collection(right_r2_connect_lines)
        ax.add_collection(right_r3_lines)
        ax.add_collection(right_r3_connect_lines)
        ax.add_collection(right_r4_lines)
        ax.add_collection(right_r4_connect_lines)
        ax.add_collection(right_r5_lines)
        ax.add_collection(right_r5_connect_lines)

        ## Final lines
        ax.add_collection(final_left_line)
        ax.add_collection(final_right_line)
        ax.add_collection(champ_line)

        ## Turn off axes and show it
        plt.axis('off')
        plt.draw()

    def label_teams(self):
        """
        Use a list of team names to label teams on the bracket.

        Assumes team names are sorted from top-left to bottom-right.
        """

        ## Redraw the plot
        plt.draw()

        ## Iterate through the names and annotate the plot in order
        t_names_series = pd.Series(self.team_names)
        if self.n_teams == 64:
            ## Set starting x/y position
            xpos = 1
            ypos = 66.5
            for index, name in t_names_series.iteritems():
                if (index < 16) or (index > 48) or \
                        ((index < 32) and (index > 16)) or \
                        ((index > 32) and (index < 48)):
                    plt.annotate(name, xy=(xpos, ypos), size=6.5)
                    ypos -= 2
                elif (index == 16) or (index == 48):
                    ypos -= 4
                    plt.annotate(name, xy=(xpos, ypos), size=6.5)
                    ypos -= 2
                elif index == 32:
                    ypos = 66.5
                    xpos = 200.8
                    plt.annotate(name, xy=(xpos, ypos), size=6.5)
                    ypos -= 2
        else:
            ## Set starting x/y position
            xpos = 21
            ypos = 65.5
            for index, name in t_names_series.iteritems():
                if (index < 8) or (index > 24) or \
                        ((index < 16) and (index > 8)) or \
                        ((index > 16) and (index < 24)):
                    plt.annotate(name, xy=(xpos, ypos), size=7.5)
                    ypos -= 4
                elif (index == 8) or (index == 24):
                    ypos -= 4
                    plt.annotate(name, xy=(xpos, ypos), size=7.5)
                    ypos -= 4
                elif index == 16:
                    ypos = 65.5
                    xpos = 180.8
                    plt.annotate(name, xy=(xpos, ypos), size=7.5)
                    ypos -= 4

        plt.draw()

    def label_winners(self, actual=False):
        """
        Label the winners of each round (whether actual or projected).

        Parameters
        ----------
        winners : list(list(str))
            A list of lists containing winners from each round
            (i.e. 1st list is winners from rd. 1, 2nd is winners from rd. 2, etc.)

        Note: assumes same ordering of teams as label_teams
              (top left, bottom left, top right, bottom right)

        actual : bool
            Default False, simply controls the color of the text
            (black for True and gray for False/projected currently)

        """

        ## Color of text set by "type" (actual/projected)
        if actual:
            color = 'black'
        else:
            color = 'gray'

        ## Make dictionaries with parameters depending on the round
        ## for the 1st 4 rounds, set the size of steps (within quadrant and extra outside quad)
        #   and y starting position
        y_steps = {1:4, 2:8, 3:16, 4:35, 5:None, 6:None}
        y_init = {1:65.5, 2:62.5, 3:58.5, 4:50.5, 5:None, 6:None}
        quad_y_step = {1:4, 2:3, 3:3, 4:0, 5:None, 6:None}

        ## Params used in all rounds
        rd_teams_left = {1:32, 2:16, 3:8, 4:4, 5:2, 6:1}
        sizes = {1:7.5, 2:7.5, 3:8, 4:8.5, 5:9, 6:14}

        ## Redraw the plot
        plt.draw()

        ## For 32 teams, pretend that it's just 1 round later of 64
        if self.n_teams == 64:
            round = 1
        elif self.n_teams == 32:
            round = 2

        ## Iterate through the list of lists, making each list a series and set params based on the round
        for names in self.winners:
            w_names_series = pd.Series(names)

            ## Set parameters:
            # starting x/y position, y-increments (within quad and out of quad), and # of teams left
            xpos = 1 + 20 * round
            ypos = y_init[round]
            step = y_steps[round]
            q_step = quad_y_step[round]
            teams_left = rd_teams_left[round]
            size = sizes[round]

            ### Up to the final 4 follows a pattern
            if round < 5:

                ## Iterating through team names and writing them on the bracket
                for index, name in w_names_series.iteritems():
                    if (index < (teams_left/4)) or (index > 3*(teams_left/4)) or \
                            (index < (teams_left/2)) and (index > (teams_left/4)) or \
                            (index > (teams_left/2)) and (index < 3*(teams_left/4)):
                        ## When staying within a quadrant, drop y by the standard amount
                        plt.annotate(name, xy=(xpos, ypos), size=size, color=color)
                        ypos -= step

                    elif (index == (teams_left/4)) or (index == 3*(teams_left/4)):
                        ## When going from a top quadrant to a lower quadrant, subtract an extra time
                        ypos -= q_step
                        plt.annotate(name, xy=(xpos, ypos), size=size, color=color)
                        ypos -= step

                    elif index == (teams_left/2):
                        ## Jumping from bottom left to top right, reset the initial X/Y position
                        ypos = y_init[round]
                        xpos = 200.8 - 20*round
                        plt.annotate(name, xy=(xpos, ypos), size=size, color=color)
                        ypos -= step

            ## For the Championship
            # assume 1st team is from left side, 2nd is from right
            elif round == 5:
                plt.annotate(names[0], xy=(101.5, 35.5), size=size, color=color)
                plt.annotate(names[1], xy=(105.5, 30.5), size=size, color=color)

            ## For the last round, write the champion in the top spot
            else:
                plt.annotate(names[0], xy=(109, 60.25), size=size, color=color, ha='center')

            round += 1

        plt.draw()

    def draw_weighted_lines(self, colors=None):
        if self.win_probabilities is None:
            pass
        else:
            plt.draw()

            ## If colors is left blank, just use black for all of the teams
            if colors is None:
                colors = ["Black"]*len(self.win_prob_teams)

            ## Dictionary with starting positions based on index number
            start_dict = {0:[0, 66]}

            ## Find the indices of the teams to label
            slots = [self.team_names.index(team) for team in self.win_prob_teams]

            ## Draw weighted lines based on the index, probability, and color
            for slot, color, win_prob in zip(slots, colors, self.win_probabilities):
                xpos = start_dict[slot][0]
                ypos = start_dict[slot][1]
                plt.plot([xpos,xpos+20],[ypos,ypos], color=color, linewidth=5*win_prob)

    def export_bracket(self, type='png', filename="bracket"):
        ## Save to image
        if type == 'png':
            plt.draw()
            plt.savefig(f"{filename}.{type}")

        ##  Save as PDF
        elif type == 'pdf':
            pp = PdfPages(f"{filename}.{type}")
            pp.savefig()
            pp.close()


def collect_lines(x_array, y_arrays, colors='black',
                  linewidths=1, linestyles='solid'):
    """ Simple helper function to create a matplotlib line collection
        based on an array of xs and many arrays of y.

        Parameters
        ----------
        x_array : np.array
            Array of x-coordinates to align with one or more sets of y-coordinates.
        y-arrays : list(np.array)
            List of one or more arrays of y-coordinates to combine with the x-coordinates.

        Returns
        -------
        matplotlib.collections.LineCollection object
            Group of lines based on the combinations of the x-array and y-arrays.
    """
    return LineCollection([np.column_stack([x_array, y])
                           for y in y_arrays],
                          linewidths=linewidths,
                          colors=colors,
                          linestyles=linestyles)
