import strava

# First create a StravaAnalyzer object
sa = strava.StravaAnalyzer()

# If you want a list of all of Strava sport types you have
# recorded, use the next line.
print(sa.get_activity_list())

# Make some summary bar charts that compare distance and elevation gain
# across years, months, and seasons for certain sport types that you've
# recorded. The metric flag is False. Set it to True if you want kilometers
# for distance and meters for elevation gain.
sa.make_activity_figures(activity_list=['Ride', 'VirtualRide', 'Walk', 'Hike'], metric=False)

# If you don't want the summary bar charts to include
# every year you've been on Strava, use a list of years as below.
sa.make_activity_figures(activity_list=['Ride'], year_list=[2019, 2020, 2021, 2022, 2023, 2024], metric=False)

# Make a pie chart that shows the breakdown of moving time by sport type
# for one year.
sa.make_combined_pie_chart(2023)

# If you want the breakdown of moving time by sport type for all years
# you've been on Strava, leave the year out.
sa.make_combined_pie_chart()

# Make a pie chart that illustrate the breakdown of distance and elevation gain
# by season for a particular sport type and year.
sa.make_seasonal_pie_charts(activity_type='Ride', metric=False, year=2023)

# Make the same pie chart but for all the years you've been on Strava.
sa.make_seasonal_pie_charts(activity_type='Ride', metric=False)

# Uncomment the line below if you want to retrieve your Strava profile data.
#sa.get_strava_profile()

# Get the predicted average speed for a route with a particular
# elevation gain and distance. This also includes a lower bound on speed,
# a lower bound on distance, and a year from which to start basing your model.
# It will use data from the start year to the present. If you want all of your data included,
# omit everything except elevation gain and distance. They are the only two required parameters.

sa.predict_avg_speed(elev_gain=2000, distance=30, lower_speed_filter=15, lower_distance_filter=20,
                     model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1, metric=False)

# This is the same as the above just with the metric flag set to true.
#sa.predict_avg_speed(elev_gain=609.6, distance=48.2803, lower_speed_filter=22.5308, lower_distance_filter=32.1869, model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1, metric=True)

# Examples of using suggest_similar_activities
sa.suggest_similar_activities(elev_gain=3000, distance=30, elev_fudge=0.1, dist_fudge=0.1, activity_type="Ride", metric=False, open_web=False)

#sa.suggest_similar_activities(distance=5, activity_type="Walk")


