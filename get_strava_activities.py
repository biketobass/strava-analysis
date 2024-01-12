import strava

# First create a StravaAnalyzer object
sa = strava.StravaAnalyzer()

# If you want a list of all of Strava sport types you have
# recorded, uncomment the next line.
print(sa.get_activity_list().sort())

# Make some summary bar charts that compare distance and elevation gain
# across years, months, and seasons.
sa.make_activity_figures(activity_list=['Ride', 'VirtualRide', 'Walk', 'Hike'])

# Make a pie chart that shows the breakdown of moving time by sport type
# for the year 2023
sa.make_combined_pie_chart(2023)

sa.make_seasonal_pie_charts()
sa.make_seasonal_pie_charts(metric=False, year=2023)

# Uncomment the line below if you want to retrieve your Strava profile data.
#sa.get_strava_profile()

# Get the predicted average speed for a route with a particular
# elevation gain and distance. This also includes a lower bound on speed,
# a lower bound on distance, and a year from which to start basing your model.
# It will use data from the start year to the present. If you want all of your data included,
# omit everything except elevation gain and distance. They are the only two required parameters.

#sa.predict_avg_speed(elev_gain=2000, distance=30, lower_speed_filter=15, lower_distance_filter=20, model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1)

# This is the same as the above just with the metric flag set to true.
#sa.predict_avg_speed(elev_gain=609.6, distance=48.2803, lower_speed_filter=22.5308, lower_distance_filter=32.1869, model_start_year=2020, dist_fudge=0.1, elev_fudge=0.1, metric=True)

# Examples of using suggest_similar_activities
#sa.suggest_similar_activities(elev_gain=200, distance=50, elev_fudge=0.1, dist_fudge=0.1, activity_type="Ride")

#sa.suggest_similar_activities(distance=5, activity_type="Walk")

#sa.predict_avg_speed(elev_gain=364, distance=10, activity_type="Hike")
