import pandas as pd
import requests
import json
import time
import datetime
from dateutil import parser
import pytz
import math
# new
import webbrowser

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress
from sklearn import linear_model

class StravaAnalyzer :
    """
    This class defines various methods for accessing and analyzing a user's data on Strava.

    Attributes
    ----------
    strava_tokens : dictionary
        The access and refresh tokens necessary to use the Strava API

    Methods
    -------
    __init__(offline=False)
        Create a new StravaAnalyzer object while updating the Strava data stored locally (unless offline is True) and creating summary stats for each activity type.
    get_strava_profile(create_csv=True, profile_csv_path='profile.csv')
        Retrieve the Strava user's profile and print to a CSV file.
    get_all_activities(start_date='January 1, 1970', end_date=None, csv_file='strava-actvities.csv', overwrite=False)
        Retrieve the Strava user's activities from start_date to end_date (or all activities if the default parameters are used) and write them to a csv file.
    separate_activity_types()
        Create a CSV file for each activity type, a csv file with summary statistics for all activities, and a file for each activity type that shows a year by year summary for that type.
    predict_avg_speed(*, elev_gain, distance, lower_speed_filter=0, upper_speed_filter=float("inf"), lower_distance_filter=0, upper_distance_filter=float("inf"), model_start_year=1970, dist_fudge=0.1, elev_fudge=0.1, metric=False, activity_type="Ride")
        Predict the expected average speed of a proposed route given elevation gain and distance.
    suggest_similar_activities(*, elev_gain=None, distance=None, dist_fudge=0.1, elev_fudge=0.1, metric=False, activity_type="Ride", open_web=False)
        Returns a list of URLs of your previously uploaded activities with similar distance and elevation gain to those specified.

    """

    # Constructor
    def __init__(self, offline=False) :
        """
        Create a new StravaAnalyzer object.

        When the object is created, the constructor makes sure it has the
        the tokens necessary to connect to Strava and that they
        are up to date. After updating as necessary, it stores the tokens
        in a file called strava_tokens.json.
        This file must already exist since the constructor first reads the
        tokens from it and checks if they have expired. If the file does not yet exist,
        the program displays a message to that effect and exits.
        Run get_strava_tokens.py and try again. For more details see the Readme and
        associated blog post.

        Parameters
        ----------
        offline : bool, optional
            A flag that specifies whether there is no Internet connection (default is False)
        """

        # First check to see if the offline flag is True. If so, don't try to go online.
        if not offline :
            # We think we have Internet.
            # Check if the tokens have expired.
            print("Checking if tokens have expired.")
            try :
                with open('strava_tokens.json') as json_file:
                    self.strava_tokens = json.load(json_file)
            except FileNotFoundError:
                print('strava_tokens.json does not exist yet. Consult the Readme and run get_strava_tokens.py before using this class. You only need to run get_strava_tokens.py once. Exiting.')
                quit()

            # If access_token has expired, use the refresh_token to get the new access_token
            if self.strava_tokens['expires_at'] < time.time():
                print("They expired. Updating...")
                # Make Strava auth API call with current refresh token
                # Your client ID and secret are stored in
                # secret_stuff.json that was created when you ran get_strava_tokens.py.
                try:
                    with open('secret_stuff.json') as secret_file:
                        secret_dict = json.load(secret_file)
                except FileNotFoundError:
                    print("The client ID and client secret haven't been saved to disk yet. Consult the Readme and run get_strava_tokens.py before using this class. You only need to run get_strava_tokens.py once. Exiting.")
                    quit()

                # Make the request and get the response..
                try :
                    response = requests.post(
                        url = 'https://www.strava.com/oauth/token',
                        data = {
                            'client_id': secret_dict['client_id'],
                            'client_secret': secret_dict['client_secret'],
                            'grant_type': 'refresh_token',
                            'refresh_token': self.strava_tokens['refresh_token']
                        }
                    )
                except requests.exceptions.RequestException as e:
                    # There's a problem with the Internet connection so behave as if offline is True
                    # and stop trying to go online.
                    print ("Could not make the API request when refreshing tokens. Received this exception:", e)
                    print("Data will not be updated. Working offline.")
                    self.separate_activity_types()
                    return

                # Replace the old strava_tokens with the response.
                self.strava_tokens = response.json()
                # Save new tokens to file
                with open('strava_tokens.json', 'w') as outfile:
                    json.dump(self.strava_tokens, outfile)
            else:
                print("No need to update tokens.")

            # Now that that's done, go and fetch all of your activities from Strava.
            # If you've done this before, get_all_activities will just update the relevant CSV
            # file with new activities. It doesn't download everything again.
            self.get_all_activities()
        # And now separate out the different activity types and get summary info for each,
        # doing some feature engineering along the way.
        self.add_features()
        self.separate_activity_types()




    def get_strava_profile(self, create_csv = True, profile_csv_path =
                     'profile.csv'):
        """Returns a pandas DataFrame that contains your profile information.

        It also prints the profile information to a csv file unless create_csv = False.

        Parameters
        ----------
        create_csv : bool, optional
            A flag that specifies whether to print the profile information to a CSV file (default is True)
        profile_csv_path: str, optional
            The path of the CSV file to create (default is 'profile.csv')

        Returns
        -------
        DataFrame
            a pandas DataFrame that contains the profile information

        """
        # Get profile info. You need the URL, access_token and header information.
        access_token = self.strava_tokens['access_token']
        url = 'https://www.strava.com/api/v3/athlete'
        hdrs = {'Authorization': 'Bearer ' + access_token}

        # Make a request object.
        try :
            r = requests.get(url, headers=hdrs)
        except requests.exceptions.RequestException as e:
            print ("Failed to make the API request when attempting to get profile. Received this exception:", e)
            return None

        # Create a dataframe from the json content. This is just the profile info.
        # You have to orient by index because some of the fields may be empty
        # which causes a ValueError("arrays must all be same length") otherwise.
        dataframe = pd.DataFrame.from_dict(r.json(), orient="index")
        # Transpose it if you prefer to see it in the opposite orientation.
        #dataframe = dataframe.transpose()

        # Now convert the dataframe to a csv file if desired.
        if create_csv:
            dataframe.to_csv(profile_csv_path)

        return dataframe

    def get_all_activities(self, start_date="January 1, 1970", end_date = None, csv_file="strava-activities.csv", overwrite=False):
        """
        Retrieves activities from Strava and outputs them to a csv file.

        As long as the defaults are left as they are, the first time you run this
        method, it downloads all of your activities on Strava and saves them to a
        csv file. On subsequent runs, it updates the csv file with any new activities
        rather than download everything again. The parameters allow you to download
        activities within a time window and save them to a file other than the default.
        If no end_date is specified, the method assumes from start_date to the present.
        If no start_date is specified, it assumes from the start of time.
        Dates are assumed to be in the UTC timezone and can be represented in human
        readable formats thanks to the dateutil module. If overwrite is False and
        the csv_file already exists, it tries to update the csv_file by changing
        the start_date to right after the most recent activity already present in the
        file. Note that the code does not check if the start_date is earlier than the
        earliest date already in the file. If you want to go back further in time than
        you already have, you will need to make sure that overwrite is True. If the
        start_date is after the latest date in the file, it assumes you want to delete
        everything before the start_date. This is to avoid gaps between the latest date
        in the file and the new start_date.

        Parameters
        ----------
        start_date : str, optional
            The earliest date in human readable form from which to start downloading
            activities (the default is 'January 1, 1970')
        end_date : str, optional
            The latest date from which to retrieve activities (the default is None which
            means up to the present moment)
        csv_file : str, optional
            The path name of the csv file to use (the default is strava-activities.csv)
        overwrite : bool, optional
            flag that specifies whether the method should overwrite the csv file
            (the default is False which means to update the csv file with new activities)

        Returns
        -------
        DataFrame
            a pandas DataFrame containing the activities downloaded from Strava
        """

        # Parse the start date by parsing it into a Datetime object.
        # Set the timezone to UTC
        startDT = parser.parse(start_date)
        timezone = pytz.timezone("UTC")
        startDT = timezone.localize(startDT)
        # Parse the end_date, but only if one is provided.
        # Otherwise, set it to the current time.
        if end_date != None:
            endDT = parser.parse(end_date)
            endDT = timezone.localize(endDT)
        else:
            endDT = datetime.datetime.now(timezone)
            # Dummy check
            if startDT >= endDT :
                print("The start time is later than the end time. Returning an empty DataFrame.")
                return pd.DataFrame()

        # If overwrite is False, check to see if the csv file already exists.
        # If so, check the latest date in the file. If the latest date is
        # after the start date, change the start date to the latest date.
        file_exists = False
        if not overwrite:
            try:
                existingDF = pd.read_csv(csv_file)
            except FileNotFoundError:
                # The file does not exist.
                print("The csv file does not already exist.")
            else :
                # If we made it here, clearly the file exists.
                existingDF.set_index("id", inplace=True)
                file_exists = True

                # Get the latest date which is just the first date
                # in the dataframe.
                latest = existingDF["start_date"].iloc[0]
                latestDT = parser.parse(latest)
                # The earliest date is the last date in the dataframe.
                earliestDT = parser.parse(existingDF["start_date"].iloc[-1])
                print("Earliest date already in the CSV file is ", earliestDT)
                print("Latest date already in the CSV file is ", latestDT)
                print("start date = ", startDT)
                print("end date = ", endDT)
                if endDT < latestDT :
                    # Assume we are looking for a subset of data that we already have
                    # and just start over.
                    print("Overriding overwrite.")
                    overwrite = True
                elif latestDT > startDT :
                    print("Changing the start date to be the latest date already in the file.")
                    startDT = latestDT
                elif startDT > latestDT :
                    # Even though overwrite is false, in this situation
                    # we're going to overwrite anyway to avoid gaps in the data.
                    print("Need to override overwrite.")
                    overwrite = True

        # Now that we have the start and end times, get the timestamps and convert them to strings
        # for inclusion in the URL.
        start_stamp = str(int(startDT.timestamp()))
        print("start_stamp = " + start_stamp)

        end_stamp = str(int(endDT.timestamp()))

        print("end_stamp = " + end_stamp)

        # Now make the API request.
        page = 1
        url = "https://www.strava.com/api/v3/activities"
        access_token = self.strava_tokens['access_token']

        # Create a dataframe to hold the results of the calls.
        # Also get the time so that we can make sure we don't go over
        # the 15 minute limit. (This is unlikely to happen and probably completely
        # unnecessary.)
        resultsDF = pd.DataFrame()
        req_start_time = time.perf_counter()
        curr_time = time.perf_counter()
        elapsed_time = 0
        # Now loop until there are no more results to get.
        while True:
            print('page = ' + str(page))
            # get page of activities from Strava
            # We're going to get 200 at a time.
            payload = {'access_token': access_token, 'after': start_stamp, 'before': end_stamp, 'per_page' : '200', 'page': str(page)}
            try:
                r = requests.get(url, params = payload)
            except requests.exceptions.RequestException as e:
                print ("Could not make the API request when requesting Strava activities. Received this exception:", e)
                print("Data will not be updated. Working offline.")
                return None
            r = r.json()
            # If no results, then exit loop
            if (not r):
                print('Breaking when page = ' + str(page))
                break

            # Otherwise create a dataframe from the json. This holds
            # only the results from this particular page of results.
            try :
                actsDF = pd.DataFrame.from_dict(r, orient='columns')
            except ValueError:
                print("Ack: Value Error: Here's the dict ", r)
            actsDF.set_index("id", inplace=True)
            # Now put all of those into the main storage DF.
            resultsDF = pd.concat([resultsDF, actsDF])

            # increment page.
            # Also check the time and number of calls.
            # Again this is almost certainly unnecessary.
            page += 1
            curr_time = time.perf_counter()
            elapsed_time = curr_time - req_start_time
            if (page == 101) and (elapsed_time/60.0 < 15) :
                print("Reached rate limit. Wait 15 minutes and then restart.")

        print("elapsed time =", elapsed_time, "seconds or ", elapsed_time/60, "minutes.")

        # At this point we've broken out of the loop and resultsDF holds all of our activities.
        # Do some processing and print to a csv file if we actually have results.
        if not resultsDF.empty :
            # Customize the following lines to produce the output you are interested in.
            # What you see below are the metrics that I like. You may want different ones.
            # BUT if you want to use the predict_avg_speed() method as
            # I've written it, you need to keep "distance(miles)", "distance(km)",
            # "Feet per mile", and "Meters per km" which make it possible to represent
            # the average elevation gain per unit of distance in Imperial and metric
            # of an activity. You also need to keep "average_speed(mph)" and "average_speed(kph)".
            # In a nut shell, the lines below adds columns to resultsDF.

            resultsDF["elapsed time(min)"] = resultsDF["elapsed_time"] / 60
            # Convert Elevation info from meters to feet and distance to miles
            # and add columns for them.
            resultsDF["elev_high(ft)"] = resultsDF['elev_high'] * 3.28084
            resultsDF["elev_low(ft)"] =resultsDF['elev_low'] * 3.28084
            resultsDF["elevation_gain(ft)"] = resultsDF['total_elevation_gain'] * 3.28084
            # Keep the next 6 columns if you want to use predict_avg_speed().
            resultsDF["distance(miles)"] = resultsDF['distance'] * 0.000621371
            resultsDF["distance(km)"] = resultsDF['distance']/1000
            resultsDF["Feet per mile"] = resultsDF["elevation_gain(ft)"] / resultsDF["distance(miles)"]
            resultsDF["Meters per km"] = resultsDF["total_elevation_gain"] / resultsDF["distance(km)"]
            # Convert meters per second to mph and to kph.
            resultsDF["average_speed(mph)"] = resultsDF['average_speed'] * 2.23694
            resultsDF["average_speed(kph)"] = resultsDF['average_speed'] * 3.6

            resultsDF["max_speed(mph)"] = resultsDF['max_speed'] * 2.23694
            resultsDF["max_speed(kph)"] = resultsDF['max_speed'] * 3.6
            # Write the dataframe to a csv and return the dataframe.
            # Convert the dates to datetimes. We need to set the timezone on
            # start_date_local. It's specified in the next column. The helper function
            # below takes care of that.
            def correct_tz(d, tzstring) :
                #tzstring = '(GMT-05:00) America/New_York'
                return d.replace(tzinfo=pytz.timezone(tzstring[tzstring.index(' ')+1:]))
            resultsDF['start_date'] = pd.to_datetime(resultsDF['start_date'])
            #resultsDF['start_date_local'] = pd.to_datetime(resultsDF['start_date_local']).apply(correct_tz)
            resultsDF['start_date_local'] = pd.to_datetime(resultsDF['start_date_local'])
            resultsDF['start_date_local'] = resultsDF[['start_date_local', 'timezone']].apply(lambda x: correct_tz(x['start_date_local'], x['timezone']), axis=1)
            if overwrite or (not file_exists) :
                print("Writing to", csv_file)
                resultsDF.to_csv(csv_file)
            else:
                print("Appending to", csv_file)
                # To be totally honest, we're not actually appending to the CSV.
                # We're appending all of the old activities to  all of the new ones
                # and then recreating the CSV file. It amounts to the same thing.
                resultsDF = pd.concat([resultsDF, existingDF])
                resultsDF.to_csv(csv_file)
        else:
            print("There were no new results. Not writing CSV.")

        return resultsDF

    def add_features(self) :
        # Read in the CSV file and make a DataFrame.
        try :
            all_actsDF = pd.read_csv('strava-activities.csv', index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError :
            print("add_features couldn't find strava-activities.csv.")
        else :
            # Add a month column to the DF.
            all_actsDF['start_date_local'] = pd.to_datetime(all_actsDF['start_date_local'], utc=True)
            # Need to localize start_date_local again.
            def localize_again(d, tzstring):
                tz = pytz.timezone(tzstring[tzstring.index(' ')+1:])
                return d.astimezone(tz)
            all_actsDF['start_date_local'] = all_actsDF[['start_date_local', 'timezone']].apply(lambda x: localize_again(x['start_date_local'], x['timezone']), axis=1)
            all_actsDF['month'] = all_actsDF['start_date_local'].dt.strftime('%b')
            all_actsDF['year'] = all_actsDF['start_date_local'].dt.year
            all_actsDF['month_num'] = all_actsDF['start_date_local'].dt.month
            def find_season(dt) :
                year = dt.year
                spr_start = datetime.datetime(year, 3, 21, tzinfo=dt.tzinfo)
                spr_end = datetime.datetime(year, 6, 20, hour=23, minute=59, second=59, tzinfo=dt.tzinfo)
                sum_start = datetime.datetime(year, 6, 21, tzinfo=dt.tzinfo)
                sum_end = datetime.datetime(year, 9, 20, hour=23, minute=59, second=59, tzinfo=dt.tzinfo)
                fall_start = datetime.datetime(year, 9, 21, tzinfo=dt.tzinfo)
                fall_end = datetime.datetime(year, 12, 20, hour=23, minute=59, second=59, tzinfo=dt.tzinfo)
                if spr_start <= dt <= spr_end :
                    return 'spring'
                elif sum_start <= dt <= sum_end :
                    return 'summer'
                elif fall_start <= dt <= fall_end :
                    return 'fall'
                else :
                    return 'winter'
                
            all_actsDF['season'] = all_actsDF['start_date_local'].apply(find_season)
            def get_season_num(season) :
                if season == 'winter' :
                    return 1
                elif season == 'spring' :
                    return 2
                elif season == 'summer' :
                    return 3
                else :
                    return 4
            all_actsDF['season_num'] = all_actsDF['season'].apply(get_season_num)
            all_actsDF.to_csv("strava-activities.csv")


    def separate_activity_types(self):
        """
        Creates a number of CSV files with summary and activity specific information.

        Reads in strava-activities.csv containing downloaded Strava activities and
        creates a CSV file for each activity type, a file with summary statistics for all
        activities, and a file for each activity type that shows a year by year summary
        for that type.
        """
        # Read in the CSV file and make a DataFrame.
        try :
            all_actsDF = pd.read_csv('strava-activities.csv', index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError :
            print("separate_activity_types couldn't find strava-activities.csv.")
        else :
            # We need to make sure that all_actsDF has all of the columns that are referenced
            # in the loop below. Otherwise, the code might throw a key error. For example, if someone
            # has no heart rate data at all, stava-activities.csv won't have a max_heartrate column,
            # causing the code to blow up when it looks for that column. So just add empty columns
            # as needed.
            necessary_columns = ["distance", "total_elevation_gain", "elapsed_time", "moving_time", "max_speed(mph)", "max_speed(kph)", "start_date", "elevation_gain(ft)", "max_heartrate"]
            for col in necessary_columns :
                if not col in all_actsDF.columns :
                    all_actsDF[col] = np.nan

            # Get the list of unique activity types (Ride, Hike, Kayak, etc.)
            act_types = all_actsDF["sport_type"].unique()
            # Get the list of unique years in the data.
            # Extract each year out of the data and sort them.
            years = pd.Series(d.year for d in all_actsDF["start_date"]).unique()
            years.sort()

            # Create a dataframe that will hold summary statistics for each activity.
            # The index or the set of rows is the activity types. The columns are the stats
            # we are interested in.
            stats = ["Total Distance (miles)", "Total Distance (km)", "Total Elev. Gain (meters)", "Total Elev. Gain (ft)", "Total Elev. Gain (miles)", "Total Elev. Gain (km)", "Total Duration (hours)", "Total Duration (days)", "Average Duration (min)", "Total Moving Time (hours)", "Total Moving Time (days)", "Average Moving Time (min)", "Average Speed (mph)", "Average Speed (kph)", "Max Speed (mph)", "Max Speed (kph)", "Max Speed Date", "Max Elevation Gain(ft)", "Max Elevation Gain(m)", "Max Elevation Gain Date", "Max Heart Rate", "Max HR Date"]
            summaryDF = pd.DataFrame(index=act_types, columns=stats)
            # Loop through all of the activity types and add info into the summary file.
            # Also create a csv for each activity that has the Strava info for that activity only.
            for act in act_types:
                actDF = all_actsDF[all_actsDF["sport_type"] == act]
                actDF.to_csv(act + ".csv")
                # Add the summary stats
                summaryDF.loc[act, "Total Distance (miles)"] = actDF["distance"].sum() * 0.000621371
                summaryDF.loc[act, "Total Distance (km)"] = actDF["distance"].sum() / 1000
                summaryDF.loc[act, "Total Elev. Gain (meters)"] = actDF["total_elevation_gain"].sum()
                summaryDF.loc[act, "Total Elev. Gain (ft)"] = actDF["total_elevation_gain"].sum() * 3.28084
                summaryDF.loc[act, "Total Elev. Gain (miles)"] = actDF["total_elevation_gain"].sum() * 3.28084/5280
                summaryDF.loc[act, "Total Elev. Gain (km)"] = actDF["total_elevation_gain"].sum() / 1000
                summaryDF.loc[act, "Total Duration (hours)"] = actDF["elapsed_time"].sum() / 3600
                summaryDF.loc[act, "Total Duration (days)"] = actDF["elapsed_time"].sum() / (3600*24)
                summaryDF.loc[act, "Average Duration (min)"] = actDF["elapsed_time"].mean() / 60
                summaryDF.loc[act, "Total Moving Time (hours)"] = actDF["moving_time"].sum() / 3600
                summaryDF.loc[act, "Total Moving Time (days)"] = actDF["moving_time"].sum() / (3600*24)
                summaryDF.loc[act, "Average Moving Time (min)"] = actDF["moving_time"].mean() / 60
                summaryDF.loc[act, "Average Speed (mph)"] = (actDF["distance"].sum() / actDF["moving_time"].sum()) * 2.23694
                summaryDF.loc[act, "Average Speed (kph)"] = (actDF["distance"].sum() / actDF["moving_time"].sum()) * 3.6
                summaryDF.loc[act, "Max Speed (mph)"] = actDF["max_speed(mph)"].max()
                summaryDF.loc[act, "Max Speed (kph)"] = actDF["max_speed(kph)"].max()
                # We have to be careful anytime we want a specific date that something occured because
                # it may never have occurred and the result may be empty. That's why we do the following
                # five lines.
                s = actDF.loc[actDF["max_speed(mph)"] == actDF["max_speed(mph)"].max(), "start_date"]
                if not s.empty :
                    summaryDF.loc[act, "Max Speed Date"] = s.iloc[0].date()
                else :
                    summaryDF.loc[act, "Max Speed Date"] = None
                summaryDF.loc[act, "Max Elevation Gain(ft)"] = actDF["elevation_gain(ft)"].max()
                summaryDF.loc[act, "Max Elevation Gain(m)"] = actDF["total_elevation_gain"].max()
                s = actDF.loc[actDF["elevation_gain(ft)"] == actDF["elevation_gain(ft)"].max(), "start_date"]
                if not s.empty :
                    summaryDF.loc[act, "Max Elevation Gain Date"] = s.iloc[0].date()
                else :
                    summaryDF.loc[act, "Max Elevation Gain Date"] = None
                summaryDF.loc[act, "Max Heart Rate"] = actDF["max_heartrate"].max()
                # We have to be careful with max heart rate because not all activities will have HR data.
                # The following code makes sure there is HR data before trying to access it.
                s = actDF.loc[actDF["max_heartrate"] == actDF["max_heartrate"].max(), "start_date"]
                if not s.empty :
                    summaryDF.loc[act, "Max HR Date"] = s.iloc[0].date()
                else:
                    summaryDF.loc[act, "Max HR Date"] = None

                # Summarize each activity by year
                act_summaryDF = pd.DataFrame(index=stats, columns = years)
                for y in years :
                    subDF = actDF[(actDF["start_date"] >= datetime.datetime(year = y, month = 1, day = 1, tzinfo=pytz.utc)) & (actDF["start_date"] < datetime.datetime(year = y+1, month = 1, day = 1, tzinfo=pytz.utc))]
                    # Need to check that we had any of this activity in the year.
                    if not subDF.empty :
                        act_summaryDF.loc["Total Distance (miles)", y] = subDF["distance"].sum() * 0.000621371
                        act_summaryDF.loc["Total Distance (km)", y] = subDF["distance"].sum() / 1000
                        act_summaryDF.loc["Total Elev. Gain (meters)", y] = subDF["total_elevation_gain"].sum()
                        act_summaryDF.loc["Total Elev. Gain (ft)", y] = subDF["total_elevation_gain"].sum() * 3.28084
                        act_summaryDF.loc["Total Elev. Gain (miles)", y] = subDF["total_elevation_gain"].sum() * 3.28084/5280
                        act_summaryDF.loc["Total Elev. Gain (km)", y] = subDF["total_elevation_gain"].sum() / 1000
                        act_summaryDF.loc["Total Duration (hours)", y] = subDF["elapsed_time"].sum() / 3600
                        act_summaryDF.loc["Total Duration (days)", y] = subDF["elapsed_time"].sum() / (3600*24)
                        act_summaryDF.loc["Average Duration (min)", y] = subDF["elapsed_time"].mean() / 60
                        act_summaryDF.loc["Total Moving Time (hours)", y] = subDF["moving_time"].sum() / 3600
                        act_summaryDF.loc["Total Moving Time (days)", y] = subDF["moving_time"].sum() / (3600*24)
                        act_summaryDF.loc["Average Moving Time (min)", y] = subDF["moving_time"].mean() / 60
                        act_summaryDF.loc["Average Speed (mph)", y] = (subDF["distance"].sum() / subDF["moving_time"].sum()) * 2.23694
                        act_summaryDF.loc["Average Speed (kph)", y] = (subDF["distance"].sum() / subDF["moving_time"].sum()) * 3.6
                        act_summaryDF.loc["Max Speed (mph)", y] = subDF["max_speed(mph)"].max()
                        act_summaryDF.loc["Max Speed (kph)", y] = subDF["max_speed(kph)"].max()
                        s = subDF.loc[subDF["max_speed(mph)"] == subDF["max_speed(mph)"].max(), "start_date"]
                        if not s.empty:
                            act_summaryDF.loc["Max Speed Date", y] = s.iloc[0].date()
                        else :
                            act_summaryDF.loc["Max Speed Date", y] = None

                        act_summaryDF.loc["Max Elevation Gain(ft)", y] = subDF["elevation_gain(ft)"].max()
                        act_summaryDF.loc["Max Elevation Gain(m)", y] = subDF["total_elevation_gain"].max()
                        s = subDF.loc[subDF["elevation_gain(ft)"] == subDF["elevation_gain(ft)"].max(), "start_date"]
                        if not s.empty :
                            act_summaryDF.loc["Max Elevation Gain Date", y] = s.iloc[0].date()
                        else :
                            act_summaryDF.loc["Max Elevation Gain Date", y] = None
                        act_summaryDF.loc["Max Heart Rate", y] = subDF["max_heartrate"].max()
                        s = subDF.loc[subDF["max_heartrate"] == subDF["max_heartrate"].max(), "start_date"]
                        if not s.empty :
                            act_summaryDF.loc["Max HR Date", y] = s.iloc[0].date()
                        else:
                            act_summaryDF.loc["Max HR Date", y] = None
                # Add a few totals
                act_summaryDF['Total'] = act_summaryDF[years[-1]]
                act_summaryDF.loc["Total Distance (miles)", "Total"] = act_summaryDF.loc["Total Distance (miles)"].sum()
                act_summaryDF.loc["Total Distance (km)", "Total"] = act_summaryDF.loc["Total Distance (km)"].sum()
                act_summaryDF.loc["Total Elev. Gain (meters)", "Total"] = act_summaryDF.loc["Total Elev. Gain (meters)"].sum()
                act_summaryDF.loc["Total Elev. Gain (ft)", "Total"] = act_summaryDF.loc["Total Elev. Gain (ft)"].sum()
                act_summaryDF.loc["Total Elev. Gain (miles)", "Total"] = act_summaryDF.loc["Total Elev. Gain (miles)"].sum()
                act_summaryDF.loc["Total Elev. Gain (km)", "Total"] = act_summaryDF.loc["Total Elev. Gain (km)"].sum()
                act_summaryDF.loc["Total Duration (hours)", "Total"] = act_summaryDF.loc["Total Duration (hours)"].sum()
                act_summaryDF.loc["Total Duration (days)", "Total"] = act_summaryDF.loc["Total Duration (days)"].sum()

                act_summaryDF.loc["Average Duration (min)", "Total"] = summaryDF.loc[act, "Average Duration (min)"]
                act_summaryDF.loc["Total Moving Time (hours)", "Total"] = act_summaryDF.loc["Total Moving Time (hours)"].sum()
                act_summaryDF.loc["Total Moving Time (days)", "Total"] = act_summaryDF.loc["Total Moving Time (days)"].sum()
                act_summaryDF.loc["Average Moving Time (min)", "Total"] = summaryDF.loc[act, "Average Moving Time (min)"]
                act_summaryDF.loc["Average Speed (mph)", "Total"] = summaryDF.loc[act, "Average Speed (mph)"]
                act_summaryDF.loc["Average Speed (kph)", "Total"] = summaryDF.loc[act, "Average Speed (kph)"]
                act_summaryDF.loc["Max Speed (mph)", "Total"] = act_summaryDF.loc["Max Speed (mph)"].max()
                act_summaryDF.loc["Max Speed (kph)", "Total"] = act_summaryDF.loc["Max Speed (kph)"].max()
                act_summaryDF.loc["Max Speed Date", "Total"] = summaryDF.loc[act, "Max Speed Date"]
                act_summaryDF.loc["Max Elevation Gain(ft)", "Total"] = summaryDF.loc[act, "Max Elevation Gain(ft)"]
                act_summaryDF.loc["Max Elevation Gain(m)", "Total"] = summaryDF.loc[act, "Max Elevation Gain(m)"]
                act_summaryDF.loc["Max Elevation Gain Date", "Total"] = summaryDF.loc[act, "Max Elevation Gain Date"]
                act_summaryDF.loc["Max Heart Rate", "Total"] = summaryDF.loc[act, "Max Heart Rate"]
                act_summaryDF.loc["Max HR Date", "Total"] = summaryDF.loc[act, "Max HR Date"]

                # Print the annual summary
                act_summaryDF.to_csv(act + "-by-year.csv")

            # Print the summary to a csv

            summaryDF.to_csv("strava-summary.csv")


    def predict_avg_speed(self, *, elev_gain, distance, lower_speed_filter=0.0, upper_speed_filter=float("inf"), lower_distance_filter=0.0, upper_distance_filter=float("inf"), model_start_year=1970, dist_fudge=0.1, elev_fudge=0.1, metric=False, activity_type="Ride", fig_flag=True) :
        """
        Uses past performance to predict the average speed of a route given its elevation
        gain and distance in either Imperial or metric units.

        It generates three different models and makes a prediction from each.
        The first is based on linear regression where the independent variable is
        average elevation gain of an activity (total elevation gain divided by total
        distance). The dependent variable is average speed. The second model is based
        on a multivariate linear regression using elevation gain and distance. The third
        model is the average of the average speeds of activities with a similar
        distance and elevation profiles.  The filter parameters allow you to specify
        activities to ignore. For example, if you don't want to include slow bike rides
        in your model, set the lower_speed_filter to a speed lower than your usual
        average speed. model_start_year allows you to specify what data you want to
        include in your modeling. If you only want to include bike rides from 2020 on,
        for example, set model_start_year to 2020. The fudge factors let you define what
        it means for an activity to be similar to another in the third model. The defaults
        are both 0.1. This means that an activity is considered similar if it has a distance
        and average elevation gain within plus or minus 10% of the those of the activity you
        are interested in. The metric parameter lets you toggle between Imperial and metric.
        The default is Imperial. activity_type let's you set what type of activity you are
        modeling. The default is Ride. If you want to model a different activity, set the
        activity_type to be exactly as Strava names it: Ride, Hike, Kayaking, Canoeing, etc.
        The method also creates a plot of average speed versus average elevation gain and saves
        it as a png file in the current directory if fig_flag is True.

        Parameters
        ----------
        elev_gain : float
            The elevation gain of the proposed route
        distance: float
            The distance of the proposed route
        lower_speed_filter : float, optional
            Allows you to specify that activities with speeds below the filter should not be
            included in the modeling (default is 0)
        upper_speed_filter : float, optional
            Allows you to specify that activities with speeds above the filter should not be
            included in the modeling (default is infinity)
        lower_distance_filter : float, optional
            Allows you to specify that activities with distances below the filter should not be
            included in the modeling (default is 0)
        upper_distance_filter : float, optional
            Allows you to specify that activities with distances above the filter should not be
            included in the modeling (default is infinity)
        model_start_year : int, optional
            Allows you to indicate that activities that occurred before a certain year should
            not be included in the modeling (default is 1970)
        dist_fudge : float, optional
            Defines what it means for a ride to have a similar distance to another (default is 0.1)
        elev_fudge : float, optional
            Defines what it means for a ride to have a similar elevation gain to another (default is 0.1)
        metric : bool, optional
            Flag that specifies whether to use the metric system (default is False)
        activity_type : str, optional
            The activity type that you are interested in (default is "Ride")
        fig_flag : bool, optional
            Flag that specifies whether to generate a plot showing average speed versus average
            elevation gain (default is True)

        Returns
        -------
        list
            a list of three floats that are the predicted average speeds of the proposed route
            using the three different models
        """

        # Set the name of the csv file to use.
        csv_file = activity_type + ".csv"

        # First read the Ride data from the appropriate csv file and build a pandas DataFrame.
        # Put it in a try statement to catch the situation where the csv file does not exist.
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some of that activity and try again.")
            return None
        else :
            # The file exists.
            # Make the appropriate adjustments for Imperial or metric.
            if not metric :
                speed_key = "average_speed(mph)"
                dist_key = "distance(miles)"
                avg_elev_gain_key = "Feet per mile"
                elev_gain_key = "elevation_gain(ft)"
                speed_unit = "mph"
            else:
                speed_key = "average_speed(kph)"
                dist_key = "distance(km)"
                avg_elev_gain_key = "Meters per km"
                elev_gain_key = "total_elevation_gain"
                speed_unit = "kph"

            # Clean up the data to filter out irrelevant rides. I've put in lower and upper
            # speed filters and lower and upper distance filters because I want these models
            # to ignore certain activities (for example, rides that I have taken with my family
            # which are slower than my solo rides
            # and short rides in which I have not yet fully warmed up).
            filteredDF = actDF[ (actDF[speed_key] > lower_speed_filter) &
                                 (actDF[speed_key] < upper_speed_filter) &
                                 (actDF[dist_key] > lower_distance_filter) &
                                 (actDF[dist_key] < upper_distance_filter) &
                                 (actDF["start_date"] >= datetime.datetime(year = model_start_year, month = 1, day=1, tzinfo=pytz.utc))]
            avg_elev_gain = elev_gain/distance
            # Models 1 and 2 only work for activity types in which there is elevation gain.
            # So check for that before proceeding.
            pred1 = 0.0
            pred2 = [0.0]
            if filteredDF[avg_elev_gain_key].sum() > 0:
                # We have an activity with elevation gain.
                # Do linear regression on avg  elevation gain and speed
                lin = linregress(filteredDF[avg_elev_gain_key], filteredDF[speed_key])
                max_avg_elev_gain = filteredDF[avg_elev_gain_key].max()
                # For kicks let's plot average speed versus average elevation gain and
                # add in a line showing the linear regression.
                if fig_flag :
                    x = np.arange(0,max_avg_elev_gain)
                    y = x*lin.slope + lin.intercept
                    fig, ax = plt.subplots()
                    ax.scatter(filteredDF[avg_elev_gain_key], filteredDF[speed_key])
                    ax.plot(x,y, color='red')
                    ax.set_title("Average Speed vs. Average Elevation Gain - " + activity_type)
                    if not metric:
                        ax.set_xlabel("Average elevation gain (ft) per mile")
                        ax.set_ylabel("Average Speed (mph)")
                        fig.savefig(activity_type+"-AvgSpeedVsAvgElevGainImperial.png")
                    else:
                        ax.set_xlabel("Average elevation gain (meters) per km")
                        ax.set_ylabel("Average Speed (kph)")
                        fig.savefig(activity_type+"-AvgSpeedVsAvgElevGainMetric.png")
                # Now do the multivariate regression.
                Indep = filteredDF[[elev_gain_key, dist_key]]
                dep = filteredDF[speed_key]
                regr = linear_model.LinearRegression()
                regr.fit(Indep.to_numpy(),dep.to_numpy())
                # Now get the predictions.
                pred1 = avg_elev_gain*lin.slope + lin.intercept
                pred2 = regr.predict([[elev_gain, distance]])
                print("Average elevation gain per distance =", f'{avg_elev_gain:.2f}', avg_elev_gain_key)
                print("******** Model 1 ********")
                print("Predicted average speed based on average elevation gain per distance =", f'{pred1:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred1), "hours and", round(((distance%pred1)/pred1)*60), "minutes.")
                print("******** Model 2 ********")
                print("Predicted average speed based on multivariate regression =", f'{pred2[0]:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred2[0]), "hours and", round(((distance%pred2[0])/pred2[0])*60), "minutes.")
            else:
                print("Sorry. Can't use Models 1 and 2 if there is no elevation gain.")

            # To make the prediction based on similar activities we need to filter the data more
            # to include only activities that are similar to the one we're asking about in terms
            # of distance and average elevation gain. Here's where we use the fudge factors.
            similarDF = filteredDF[ (filteredDF[dist_key] >= distance - dist_fudge*distance) & (filteredDF[dist_key] <= distance + dist_fudge*distance) & (filteredDF[avg_elev_gain_key] >= avg_elev_gain - elev_fudge*avg_elev_gain) & (filteredDF[avg_elev_gain_key] <= avg_elev_gain + elev_fudge*avg_elev_gain)  ]
            similar_acts= similarDF[[speed_key, avg_elev_gain_key, dist_key]]
            pred3 = 0
            print("******** Model 3 ********")
            if len(similar_acts) > 0 :
                pred3 = similarDF[speed_key].mean()
                print("Predicted average speed based on", len(similar_acts), "similar activities =", f'{pred3:.2f}', speed_unit)
                print("Moving time would equal", math.floor(distance/pred3), "hours and", round(((distance%pred3)/pred3)*60), "minutes.")
            else :
                print("There were no similar activities. Perhaps try more generous fudge factors.")

            return [pred1, pred2[0], pred3]



    def suggest_similar_activities(self, *, elev_gain=None, distance=None, dist_fudge=0.1, elev_fudge=0.1, metric=False, activity_type="Ride", open_web=False) :
        """
        Takes an elevation gain and a distance and returns a list of URLs of your past Strava activities
        that have similar elevation gain and distance.

        The fudge factors let you define the degree of similarity. The defaults are 0.1.
        The default units are feet of elevation gain and miles of distance, but if you set
        the metric flag to True, the units are meters of elevation gain and kilometers of
        distance. Finally, the default activity is Ride. To specify a different activity type,
        set the activity_type parameter to it. Just make sure you name it the same as Strava does.

        Parameters
        ----------
        elev_gain : float, optional
            The elevation gain of the activity you want a suggestion for (default is None, but
            elev_gain and distance can't both be None)
        distance: float, optional
            The distance of the activity you want a suggestion for (default is None, but
            elev_gain and distance can't both be None)
        dist_fudge : float, optional
            Defines what it means for a ride to have a similar distance to another (default is 0.1)
        elev_fudge : float, optional
            Defines what it means for a ride to have a similar elevation gain to another (default is 0.1)
        metric : bool, optional
            Flag that specifies whether to use the metric system (default is False)
        activity_type : str, optional
            The activity type that you are interested in (default is "Ride")
        open_web : Boolean, optional
            Flag that specifies whether to open the similar activities in new tabs of your default browser.

        Returns
        -------
        list
            list of URLs of your past Strava activites that have similar
            elevation gain and distance.
        """

        # You need to have at least one of elevation gain and distance.
        if elev_gain == None and distance == None :
            raise ValueError("elev_gain and distance can't both equal None.")

        csv_file = activity_type + ".csv"
        try:
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some of that activity and try again.")
        else:
            # Sort out what units we're using.
            if not metric :
                dist_key = "distance(miles)"
                elev_gain_key = "elevation_gain(ft)"
                dist_units = "miles"
                elev_units = "feet"
            else:
                dist_key = "distance(km)"
                elev_gain_key = "total_elevation_gain"
                dist_units = "km"
                elev_units = "meters"

            # Deal with three situations: We have elevation gain and distance. We have
            # elevation but not distance. We have distance but not elevation.
            # We don't need to test if both are None because
            # we did that above.
            if elev_gain and distance :
                similarDF = actDF[ (actDF[dist_key] >= distance - dist_fudge*distance) & (actDF[dist_key] <= distance + dist_fudge*distance) & (actDF[elev_gain_key] >= elev_gain - elev_fudge*elev_gain) & (actDF[elev_gain_key] <= elev_gain + elev_fudge*elev_gain)  ]
                print("\nActivities with elevation gain similar to", elev_gain, elev_units, "and distance similar to", distance, dist_units, ":\n")
            elif not distance :
                similarDF = actDF[ (actDF[elev_gain_key] >= elev_gain - elev_fudge*elev_gain) & (actDF[elev_gain_key] <= elev_gain + elev_fudge*elev_gain)  ]
                print("Activities with elevation gain similar to", elev_gain, elev_units, ":")
            elif not elev_gain:
                similarDF = actDF[ (actDF[dist_key] >= distance - dist_fudge*distance) & (actDF[dist_key] <= distance + dist_fudge*distance) ]
                print("Activities with distance similar to", distance, dist_units,":")

            list_of_urls = []
            if len(similarDF) > 0 :
                for i in similarDF.index:
                    act_url = "https://www.strava.com/activities/" + str(i)
                    # new
                    if open_web :
                        webbrowser.open_new_tab(act_url)
                    list_of_urls.append(act_url)
                    print(similarDF.loc[i, "name"], f'{similarDF.loc[i,dist_key]:.2f}', dist_units,  f'{similarDF.loc[i,elev_gain_key]:.2f}', elev_units)
                    print(act_url + "\n")
            else:
                print("No activities with elevation gain similar to ", elev_gain, "and distance similar to", distance)
            return list_of_urls
        
    
    def make_activity_figures(self, activity_list=['Ride'], metric=False, year_list=None) :
        """
        Produce bar charts comparing activity performance across years, months, and seasons.

        Given a set of Strava sport types, for each type, produce mileage and elevation gain
        comparison bar charts. If metric is False (the default) it assumes miles for distance
        feet for elevation gain. If metric is True, it assumes kilometers and meters.
        
        If a list of years is provided, it limits the charts to those years only.

        Parameters
        ----------
        activity_list : list or String, optional
            The Strava sport types you are interested in. If it's String, the method will
            create a single element list with it in it. Default = ['Ride']
        metric: bool, optional
            If False (default), assumes miles for distance and feet for elevation gain. If
            True, assumes kilometers and meters.
        year_list : list, optional
            List of years to consider when making the charts. If None (default), it uses all years in
            your Strava data.
        """

        # Check if activity_list is a list. If not, turn it into a list.        
        if not isinstance(activity_list, list) :
            activity_list = [activity_list]
            
        # For each activity in the list call make_single_act_figures.
        for act in activity_list :
            self.make_single_act_figures(activity_type=act, metric=metric, year_list=year_list)
        
    def make_single_act_figures(self, activity_type="Ride", metric=False, year_list=None) :
        """
        Produce bar charts comparing activity performance across years, months, and seasons.

        Given a Strava sport type, produce mileage and elevation gain
        comparison bar charts. If metric is False (the default) it assumes miles for distance
        feet for elevation gain. If metric is True, it assumes kilometers and meters.
        
        If a list of years is provided, it limits the charts to those years only.
        
        Saves charts with file names starting with the activity_type and ending .png.
        Ride_distance(miles)_bar_by_year.png, for example.

        Parameters
        ----------
        activity_type : String, optional
            The Strava sport type you are interested in. If it's String, the method will
            create a single element list with it in it. Default = Ride.
        metric: bool, optional
            If False (default), assumes miles for distance and feet for elevation gain. If
            True, assumes kilometers and meters.
        year_list : list, optional
            List of years to consider when making the charts. If None (default), it uses all years in
            your Strava data.
        """
        # Read the csv_file.
        csv_file = activity_type + ".csv"
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some of that activity and try again.")
            return None
        else :
            # The file exists.
            # Let's make some figures
            
            # Ultimately, we'll save the figures and append the years in year_list to the
            # file names, so create a string to be that suffix.
            fn_suffix = '_' + '_'.join([str(y) for y in year_list]) if year_list else ''

            # Each feature tuple below gives a column name from actDF paired with a more readable description.
            if not metric :
                features = [('distance(miles)', 'Distance in Miles'), ('elevation_gain(ft)', 'Total Elevation Gain in Feet')]
            else :
                features = [('distance(km)', 'Distance in Km'), ('total_elevation_gain', 'Total Elevation Gain in Meters')]
            
            # Make some barplots
            # Get just the info we're interested in.
            actDF_subset = actDF[['year', 'month', 'month_num', 'distance(miles)', 'distance(km)', 'total_elevation_gain', 'elevation_gain(ft)', 'season', 'season_num']]
            # If we're only looking at a subset of years, get just the data from those years.
            if year_list :
                actDF_subset = actDF_subset[actDF_subset['year'].isin(year_list)]
                
            # The figures are going to show data by year, season and month.
            # Create dfs for years, months, and seasons.
            df_by_year = actDF_subset.groupby(by='year').sum().reset_index()
            
            # A list of month numbers and a dataframe of months will come in handy later.
            all_month_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            month_df = pd.DataFrame({'month':['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 
                                     'month_num': all_month_nums})
            df_by_year_month = actDF_subset.groupby(by=['year', 'month_num']).sum().reset_index()
            # Make sure that df_by_year_month has a value for every month_num. If not, add a row with 0s all columns
            # except for the month_num and year. If you don't do this, the charts won't display 12 months.
            month_check = df_by_year_month['month_num'].unique()
            for m in all_month_nums :
                if m not in month_check :
                    # Add a dummy row to df_by_year_month
                    # The dummy year is going to be the same as the year in the current
                    # last row. The year doesn't really matter. This is just easy.
                    dum_year = df_by_year_month.loc[len(df_by_year_month)-1, 'year']
                    # Set everything in the new row to 0.
                    df_by_year_month.loc[len(df_by_year_month)] = 0
                    # Set the year and month.
                    df_by_year_month.loc[len(df_by_year_month)-1, 'year'] = dum_year
                    df_by_year_month.loc[len(df_by_year_month)-1, 'month_num'] = m
            
            # As with months, it's useful later to have a list of season_nums and a dataframe of seasons.
            all_season_nums = [1,2,3,4]
            seasons_df = pd.DataFrame({'season':['winter', 'spring', 'summer', 'fall'], 'season_num':all_season_nums})
            df_by_year_season = actDF_subset.groupby(by=['year', 'season_num']).sum().reset_index()
            # As with months, fill in any missing seasons.
            season_check = df_by_year_season['season_num'].unique()
            for s in all_season_nums :
                if s not in season_check :
                    dum_year = df_by_year_season.loc[len(df_by_year_season)-1, 'year']
                    df_by_year_season.loc[len(df_by_year_season)] = 0
                    df_by_year_season.loc[len(df_by_year_season)-1, 'year'] = dum_year
                    df_by_year_season.loc[len(df_by_year_season)-1, 'season_num'] = s
    
            # Helper function for rounding values to the nearest multiple of a base number.
            # For example, if a == 80 and base == 25, it returns 75.
            # We can run into trouble if a < base, hence the while loop.
            # This function and the next will be helpful for formatting the chart axes.
            def my_round(a, base) :
                if a == 0 :
                    return a
                while a < base :
                    base = base // 2
                if base == 0 :
                    base = 1
                x = (a // base) * base
                if x == 0 :
                    x = 1
                y = float(a % base)
                y = int(round(y/base) * base)
                return x+y
            
            # Helper function for calculating various formatting values:
            # distance between ticks on the y-axis, and the order of magnitude
            # at which we'll be displaying numbers.
            # It takes a max y-axis value and scales accordingly.
            def get_formatting_values(max_val) :
                if 10000 <= max_val < 1000000 :
                    y_tick_divisor = 1000
                    order_mag = 'K'
                else :
                    y_tick_divisor = 1
                    order_mag = ''
                num_y_ticks = 20
                y_tick_step = max_val / num_y_ticks
                if y_tick_step == 0 :
                    y_tick_step = 1
                # Now round this to the nearest multiple of 5*y_tick_divisor
                # unless y_tick_divisor is one, in which case use 50.
                y_tick_step = my_round(y_tick_step, 50) if y_tick_divisor == 1 else my_round(y_tick_step, 5*y_tick_divisor)
                return (y_tick_step, y_tick_divisor, order_mag)
            
            # Each figure is going to have the same width and height.
            imwidth = 30
            imheight = 20
            # Loop through the features (distance, elevation gain, etc.)
            for feature in features :
                # Set the style for all of the figures
                sns.set_style("whitegrid", {"grid.linestyle": ""})
                
                # Create barplots for each feature for year, month, season.
                max_in_year = df_by_year[feature[0]].max() # Will use later to set the max value of the y-axis.
                # If the max is zero, it means we didn't have any of that feature at all.
                # Just go on to the next one.
                if max_in_year == 0 :
                    print(f"Skipping bar charts for {feature[0]} in {activity_type} because there wasn't any.")
                    continue
                # Start with comparisons across years.
                fig, ax = plt.subplots(figsize=(imwidth,imheight))
                # Use the feature tuples. feature[0] is the column; feature[1]
                # is the readable version.
                sns.barplot(data=df_by_year, x='year', y=feature[0], ax=ax)
                ax.set_ylabel(feature[1], fontsize=25)
                ax.set_title(feature[1] + " ("+activity_type+")", fontsize=30)
                # Use the helper function above to get the figure formatting values.
                tick_step, y_tick_divisor, order_mag = get_formatting_values(max_in_year)
                # Set up the y axis, giving a little extra room at the top edge.
                y_lim_max = max_in_year + tick_step // 2
                ax.set_ylim((0,y_lim_max))
                yticks = np.arange(0,y_lim_max, tick_step)
                ax.set_yticks(ticks=yticks, labels=[f'{round(y/y_tick_divisor):,}{order_mag}' for y in yticks], fontsize=16)
                for y in yticks :
                    ax.axhline(y, xmin=0, xmax=1, linewidth=0.25, dashes=(1, 1), color='black')
                # Set up the x-axis.
                ax.set_xlabel("Year", fontsize=25)
                ax.set_xticks(ticks=df_by_year.index, labels=df_by_year['year'], fontsize=25)
                for container in ax.containers :
                    ax.bar_label(container, fmt=lambda x: f'{x/y_tick_divisor:,.0f}{order_mag}', fontsize=16)
                plt.tight_layout()
                fig.savefig(activity_type+"_"+feature[0] + "_bar_by_year"+fn_suffix+".png")
                plt.close()

                # Compare total feature value per month per year - 
                # number of miles per month each year for example.
                #sns.set_style("white")
                fig, ax = plt.subplots(figsize=(30,20))
                max_in_month = df_by_year_month[feature[0]].max()
                sns.barplot(data=df_by_year_month, x='month_num', y=feature[0], hue='year', ax=ax, palette='deep')
                ax.set_ylabel(feature[1], fontsize=25)
                ax.set_title(feature[1] + " per Month per Year ("+activity_type+")", fontsize=30)
                ax.set_xticks(ticks=month_df.index, labels=month_df['month'], fontsize=25)
                # Add vertical lines between month groups to make it easier to see where each starts
                # and ends.
                for x in month_df.index :
                    ax.axvline(x=x+0.5, ymin=0, ymax=1, linewidth=3, dashes=(1, 1), color='black')
                    if x % 2 == 1 :
                        ax.axvspan(x-0.5, x+0.5, facecolor='gray', alpha=0.2)
                tick_step, y_tick_divisor, order_mag = get_formatting_values(max_in_month)
                y_lim_max = max_in_month+tick_step // 2
                yticks=np.arange(0,y_lim_max, tick_step)
                ax.set_yticks(ticks=yticks, labels=[f'{round(y/y_tick_divisor):,.0f}{order_mag}' for y in yticks], fontsize=16)
                for y in yticks :
                    ax.axhline(y, xmin=0, xmax=1, linewidth=0.25, dashes=(1, 1), color='black')
                ax.set_xlabel("Month", fontsize=25)
                ax.set_ylim((0,y_lim_max))
                for container in ax.containers :
                    ax.bar_label(container, fmt=lambda x: f'{x/y_tick_divisor:,.0f}{order_mag}' if x>0 else '', fontsize=16)
                ax.legend(fontsize=20, loc='upper left')
                plt.tight_layout()
                plt.grid()
                fig.savefig(activity_type+"_" + feature[0]+"_bar_by_year_month"+fn_suffix+".png")
                plt.close()
            
                # Barplots comparing feature values per season per year
                #sns.set_style("white")
                fig, ax = plt.subplots(figsize=(30,20))
                max_in_season = df_by_year_season[feature[0]].max()
                sns.barplot(data=df_by_year_season, x='season_num', y=feature[0], hue='year', ax=ax, palette='deep')
                ax.set_ylabel(feature[1], fontsize=25)
                ax.set_title(feature[1] + " per Season per Year ("+activity_type+")", fontsize=30)
                ax.set_xticks(ticks=seasons_df.index, labels=seasons_df['season'], fontsize=25)
                for x in seasons_df.index :
                    ax.axvline(x=x+0.5, ymin=0, ymax=1, linewidth=3, dashes=(1, 1), color='black')
                    if x % 2 == 1 :
                        ax.axvspan(x-0.5, x+0.5, facecolor='gray', alpha=0.2)
                tick_step, y_tick_divisor, order_mag = get_formatting_values(max_in_season)
                y_lim_max = max_in_season + tick_step // 2
                yticks=np.arange(0,y_lim_max, tick_step)
                ax.set_yticks(ticks=yticks, labels=[f'{round(y/y_tick_divisor):,.0f}{order_mag}' for y in yticks], fontsize=16)
                for y in yticks :
                    ax.axhline(y, xmin=0, xmax=1, linewidth=0.25, dashes=(1, 1), color='black')
                ax.set_xlabel("Season", fontsize=25)
                ax.set_ylim((0,y_lim_max))
                for container in ax.containers :
                    ax.bar_label(container, fmt=lambda x: f'{x/y_tick_divisor:,.0f}{order_mag}' if x>0 else '', fontsize=16)
                ax.legend(fontsize=20, loc='upper left')
                plt.tight_layout()
                plt.grid()
                fig.savefig(activity_type+"_" + feature[0]+"_bar_by_year_season"+fn_suffix+".png")
                plt.close()
             
    def make_combined_pie_chart(self, year=None, other_threshold=0.01) :
        """
        Produce pie chart illustrating percent of moving time of each Strava activity type.

        If no year is given, the pie chart includes all recorded activities. If a year is given,
        it only includes activities from that year. The parameter other_threshold is used to
        lump together activites that had accounted for very small percentages of the total move time.
        This is used to keep the pie chart from looking messy.
        
        Saves chart with file name all_acts_pie.png or with the year added such as
        all_acts_pie_2023.png, for example.

        Parameters
        ----------
        year : int, optional
            If given, the year of the data we're interested in.
        other_threshold: float, option
            used to lump together activites that had accounted for very small percentages of the total move time.
            0.01 is the default.
        """
        # Get the stored stava data.
        csv_file = "strava-activities.csv"
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some activities and try again.")
            return None
        else :
            # The file exists.
            # Let's make some figures
            
            # Pie chart prep
            # Get the columns we're interested in.
            actDF_subset = actDF[['sport_type', 'moving_time', 'elapsed_time', 'distance', 'total_elevation_gain', 'year']]
            # If a year is given, get the data from that year only.
            if year :
                actDF_subset = actDF_subset[actDF_subset['year'] == year]
            else :
                # If we don't have a year given, get the first and last year so that we can
                # add them to the chart title later.
                # Note that years below will be sorted in reverse chronological order because
                # unique() returns values in order to appearance and Strava results are
                # returned newest to oldest.
                years = actDF_subset['year'].unique()
                start_year = years[-1]
                end_year = years[0]
            # Now create a df for the pie chart, adding a column for fraction of moving time.
            # Group by sport_type and then sum the columns so that we can have the total moving time for each
            # sport_type.
            # Sort pie_df by moving_time so that the type with the largest total moving time is first.
            pie_df = actDF_subset.groupby('sport_type').sum().reset_index().sort_values(by='moving_time', ascending=False).reset_index()
            # Create a column for the fraction of the total moving time. So first get the total moving time across all sport types.
            total_moving_time = actDF_subset['moving_time'].sum()
            pie_df['moving_time_fraction'] = pie_df['moving_time']/total_moving_time
            # Group activities that make up less than other_threshold of the moving time
            # so that the pie doesn't look too crowded.
            wedges = []
            labels = []
            # This works because pie_df is sorted.
            # boolean to mark the first time we pass the threshold. Once this is set to False,
            # all remaining will be less than the threshold and so we'll lump them in with the first.
            first_crossing = True
            for fraction, label in zip(pie_df['moving_time_fraction'], pie_df['sport_type']) :
                # Append fractions and labels until we cross the threshold.
                if fraction >= other_threshold :
                    wedges.append(fraction)
                    labels.append(label)
                else :
                    # The first time we cross, append. Subsequent times
                    # lump them into the last elements.
                    if first_crossing :
                        wedges.append(fraction)
                        labels.append('Other')
                        first_crossing = False
                    else :
                        wedges[-1] += fraction
                        labels[-1] = 'Other'
            # Finally create the plot.
            fig, ax = plt.subplots(figsize=(20,20))
            # For fun, let's explode the biggest wedge out of the pie a bit.
            explode = [0 for i in range(0, len(wedges))]
            explode[0] = 0.1
            # Helper function for labeling wedges.
            # Given a percent figure out how many hours that amounts to.
            def label_func(pct) :
                hours = int(np.round((total_moving_time * pct/100.0)/3600.0))
                return f"{pct:.1f}%\n({hours:,} hours)" if pct >= 2 else f"{pct:.1f}% - ({hours:,} hours)"
            wdgs, texts, autotexts = ax.pie(wedges, labels=labels, autopct=lambda pct: label_func(pct), textprops={'size': 'xx-large'}, shadow=True, explode=explode)
            plt.setp(autotexts, size=20, weight='bold')
            plt.setp(texts, size=20, weight='bold')
            # Set the title
            if year :
                plt.suptitle("Moving Time Breakdown by Activity Type in " + str(year), fontsize=30, weight='bold')
                ax.set_title(f'\nTotal moving time across all activities was {int(np.round(total_moving_time/3600.0)):,} hours\n', fontsize=25, weight='bold')
            else :
                plt.suptitle(f"Moving Time Breakdown by Activity Type from {start_year} to {end_year} ", fontsize=30, weight='bold')
                ax.set_title(f'\nTotal moving time across all activities was {int(np.round(total_moving_time/3600.0)):,} hours\n', fontsize=25, weight='bold')
                #ax.set_title("Percent of Moving Time by Activity Type", fontsize=30, weight='bold')
            plt.tight_layout()
            # Save it.
            if year :
                fig.savefig('all_acts_pie_'+str(year)+'.png')
            else :
                fig.savefig('all_acts_pie.png')
            plt.close()
            
    def make_seasonal_pie_charts(self, activity_type='Ride', year=None, metric=False) :
        """
        Produce two pie chart showing breakdown of distance and elevation gain by season.

        If no year is given, the pie chart includes all recorded activities of the given activity_type.
        If a year is given, it only includes activities from that year. 
        
        Saves charts with file names [activity_type]_distance_pie.png and [activity_type]_elevation_pie.png.
        If the year parameter is included, it appends the year before the file extension. If the metric param
        is False, it assumes miles for distance and elevation gain for feet. If True, it assumes kilometers and
        meters.

        Parameters
        ----------
        activity_type : String, optional
            The Strava sport type we're interested in. Default is Ride.
        year : int, optional
            If given, the year of the data we're interested in.
        metric : bool, optional
            Flag to specifiy metric or imperial units. Default is False (imperial)
        """
        # Get the stored stava data.
        csv_file = activity_type + ".csv"
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some activities of that type and try again.")
            return None
        else :
            # The file exists.
            
            # Pie chart prep
            if year :
                actDF= actDF[actDF['year'] == year]
            else :
                # If we don't have a year given, get the first and last year so that we can
                # add them to the chart title later.
                # Note that years below will be sorted in reverse chronological order because
                # unique() returns values in order to appearance and Strava results are
                # returned newest to oldest.
                years = actDF['year'].unique()
                start_year = years[-1]
                end_year = years[0]
            # Now create a df for the pie charts.
            # Group by season and then sum the columns so that we can have the total distance and
            # elevation gain by sport_type.
            # First subset the data to get just the columns that we want since Pandas will complain
            # about not being able to sum certain data types, like datetimes.
            actDF = actDF[['distance', 'total_elevation_gain', 'season']]
            pie_df = actDF.groupby('season').sum().reset_index()
            # Create a columns for the fractions of the total distance and elevation gain.
            # So first get total distance and total elevation gain across all seasons.
            total_distance = actDF['distance'].sum()
            total_elev = actDF['total_elevation_gain'].sum()
            pie_df['distance_fraction'] = pie_df['distance']/total_distance
            pie_df['elev_fraction'] = pie_df['total_elevation_gain']/total_elev
            dist_wedges = []
            dist_labels = []
            for fraction, label in zip(pie_df['distance_fraction'], pie_df['season']) :
                # Append fractions and labels.
                dist_wedges.append(fraction)
                dist_labels.append(label)
            # Finally create the plots.
            # Distance
            fig, ax = plt.subplots(figsize=(20,20))

            # Helper function for labeling distance wedges.
            # Given a percent figure out how many units of distance that amounts to.
            def label_func_dist(pct) :
                dist = int(np.round((total_distance * pct/100.0)))
                dist = int(np.round(dist * 0.000621371)) if not metric else int(np.round(dist / 1000.0))
                return f"{pct:.1f}%\n({dist:,} miles)" if not metric else f"{pct:.1f}%\n({dist:,} km)"
            # Only make the figure if there is actually any distance.
            if total_distance > 0 :
                wdgs, texts, autotexts = ax.pie(dist_wedges, labels=dist_labels, autopct=lambda pct: label_func_dist(pct), textprops={'size': 'xx-large'}, shadow=True)
                plt.setp(autotexts, size=20, weight='bold')
                plt.setp(texts, size=20, weight='bold')
                # Set the title
                if year :
                    plt.suptitle(f"{activity_type} Distance Breakdown by Season in {str(year)}", fontsize=30, weight='bold')
                else :
                    plt.suptitle(f"{activity_type} Distance Breakdown by Season from {start_year} to {end_year} ", fontsize=30, weight='bold')
                if not metric :
                    ax.set_title(f'\nTotal {activity_type} Distance was {int(np.round(total_distance * 0.000621371)):,} miles\n', fontsize=25, weight='bold')
                else :
                    ax.set_title(f'\nTotal {activity_type} Distance was {int(np.round(total_distance / 1000.0)):,} km\n', fontsize=25, weight='bold')
                plt.tight_layout()
                # Save it.
                if year :
                    fig.savefig(activity_type+'_distance_pie_'+str(year)+'.png')
                else :
                    fig.savefig(activity_type+'_distance_pie.png')
                plt.close()
            else :
                print(f'{activity_type} has no distance. Did not produce pie chart.')
            
            # Elevation
            elev_wedges = []
            elev_labels = []
            for fraction, label in zip(pie_df['elev_fraction'], pie_df['season']) :
                # Append fractions and labels.
                elev_wedges.append(fraction)
                elev_labels.append(label)
            fig, ax = plt.subplots(figsize=(20,20))

            # Helper function for labeling elevation wedges.
            # Given a percent figure out how many units of elevation that amounts to.
            def label_func_elev(pct) :
                elev = int(np.round((total_elev * pct/100.0)))
                elev = int(np.round(elev * 3.28084)) if not metric else int(np.round(elev))
                return f"{pct:.1f}%\n({elev:,} feet)" if not metric else f"{pct:.1f}%\n({elev:,}m)"
            if total_elev > 0 :
                wdgs, texts, autotexts = ax.pie(elev_wedges, labels=elev_labels, autopct=lambda pct: label_func_elev(pct), textprops={'size': 'xx-large'}, shadow=True)
                plt.setp(autotexts, size=20, weight='bold')
                plt.setp(texts, size=20, weight='bold')
                # Set the title
                if year :
                    plt.suptitle(f"{activity_type} Elevation Gain Breakdown by Season in {str(year)}", fontsize=30, weight='bold')
                else :
                    plt.suptitle(f"{activity_type} Elevation Gain Breakdown by Season from {start_year} to {end_year} ", fontsize=30, weight='bold')
                if not metric :
                    ax.set_title(f'\nTotal {activity_type} Elevation Gain was {int(np.round(total_elev * 3.28084)):,} feet\n', fontsize=25, weight='bold')
                else :
                    ax.set_title(f'\nTotal {activity_type} Elevation Gain was {int(np.round(total_elev)):,} m\n', fontsize=25, weight='bold')
                plt.tight_layout()
                # Save it.
                if year :
                    fig.savefig(activity_type+'_elevation_pie_'+str(year)+'.png')
                else :
                    fig.savefig(activity_type+'_elevation_pie.png')
                plt.close()
            else :
                print(f'{activity_type} has no elevation gain. Did not produce pie chart.')
            
    def get_activity_list(self) :
        """
        Return a list of all recorded Strava sport_types.

        The list is NOT sortable because pd.Series.unique() is not sortable.

        Returns
        -------
        list
            list of Strava sport types.
        """
        csv_file = "strava-activities.csv"
        try :
            actDF = pd.read_csv(csv_file, index_col="id", parse_dates=["start_date", "start_date_local"])
        except FileNotFoundError:
            print(csv_file, "file does not exist. Record some activities and try again.")
            return None
        else :
            # The file exists.
            return actDF['sport_type'].unique()
                
