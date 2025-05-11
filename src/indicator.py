import webbrowser
import easygui
from geopy.geocoders import Nominatim
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
import os
import subprocess
import osmnx as ox
import folium
import tkinter as tk
from tkinter import messagebox, ttk
import shapely
import json

"""
This program takes existing survey data or lets you choose a person group for which to calculate and visualize the x-min-city. The survey data needs to be
pre-processed to be handled correctly. If you choose to calculate the accessibility map for a person group, an origin point needs to be specified. Firstly,
the data is processed into a geo-dataframe which is then iterated over for each amenity that is specified for this specific person or group. The program then
uses the OSMnx module for creating the network graph, finding the nearest nodes of both origin and destination, and finding the shortest paths. For
visualization it uses a folium map that is then saved and can be displayed in any html capable viewer. 
Have fun!
"""


# Load destination mapping from JSON file
def load_destination_mapping():
    try:
        with open('destination_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        tk.messagebox.showerror(message=f'Could not load destination mapping: {e}. The program will end now.')
        quit()


# Apply destination mapping to survey data
def apply_destination_mapping(data, mapping):
    # Create a copy of the data to avoid modifying the original
    mapped_data = data.copy()

    # For each row in the data
    for i, row in mapped_data.iterrows():
        # Get the destinations and split them
        destinations = [d.strip() for d in row['destinations original'].split(',')]
        mapped_destinations = []
        mapped_tags = []

        # Map each destination to its corresponding OSM tag(s)
        for dest in destinations:
            if dest in mapping:
                tags_dict = mapping[dest]["tags"]
                mapped_destinations.append(dest)
                mapped_tags.append(json.dumps(tags_dict))  # Store as JSON string for later parsing
            else:
                print(f"Warning: No mapping found for destination '{dest}'. Using original value.")
                mapped_destinations.append(dest)
                mapped_tags.append(json.dumps({}))

        # Update the row with mapped destinations and tags
        mapped_data.at[i, 'destinations original'] = ', '.join(mapped_destinations)
        mapped_data.at[i, 'destinations mapped'] = ', '.join(mapped_destinations)
        mapped_data.at[i, 'destinations tags'] = '|'.join(mapped_tags)

    return mapped_data


# Splits the address data of the survey into separate columns for further processing to find the coordinates
### DONE
def extract_addressdata(row, col_name):
    return pd.Series(row[col_name].split(','))


# Finds the corresponding coordinates of the location. Currently only locations in Germany are possible
### DONE
def find_coordinates(row, col_name):
    geolocator = Nominatim(user_agent="myGeocoder", timeout=10)
    try:
        # Geocode the address
        location = geolocator.geocode(row[col_name])
        return pd.Series([location.latitude, location.longitude])
    except Exception as e:
        tk.messagebox.showerror(message=f'Could not geocode origin address: {e}. The program will end now.')
        quit()


# Import survey data and add coordinates of home address
### DONE
def read_data():
    try:
        path = easygui.enterbox(title='Survey path',
                                msg='Please enter the path to the survey file your data is stored in. '
                                    'You can later choose to read individual entries or form aggregates based on demographics.',
                                default='../Survey/Data/Survey_data.csv')
    except Exception as e:
        tk.messagebox.showerror(message=f'The path is incorrect: {e}. The program will end now.')
        quit()
    points_data = pd.read_csv(path, delimiter=';')

    # Preprocess dataframe for csv input file
    points_data[['home_street', 'home_city', 'home_country']] = points_data.apply(extract_addressdata, args=('home',),
                                                                                  axis=1)
    points_data[['lat', 'lon']] = points_data.apply(find_coordinates, args=('home',), axis=1)

    # Load and apply destination mapping
    destination_mapping = load_destination_mapping()
    points_data = apply_destination_mapping(points_data, destination_mapping)

    return points_data


# Define function to get coordinates for the amenities, based on geometry column
### DONE
def get_lon_lat(row):
    geom = row['geometry']  # Extract geometry
    if geom.geom_type == "Point":
        return geom.x, geom.y
    else:  # For Polygon, LineString, etc., use centroid
        centroid = geom.centroid
        return centroid.x, centroid.y


# Create a list with all possible routes
### DONE
def generate_multindex(route_nodes):
    multiindex_list = []
    # append the index to list
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        multiindex_list.append((u, v, 0))
    return multiindex_list


# Find the closest location for each of the amenities specified in the destinations column of the survey
### NOT DONE
def find_closest_amenity(survey_line, speed, nearest_node_home, mode):
    # Get the amenities as a list to iterate over
    destination_list = survey_line.loc['destinations mapped'].split(',')
    destination_amounts = [int(amount.strip()) for amount in survey_line['destination amount'].split(',')]
    destination_tags_list = survey_line.get('destinations tags', '').split('|')

    if mode == 'wheelchair':
        mode = 'walk'

    all_relevant_amenities = pd.DataFrame()
    for idx, (destination, required_amount) in enumerate(zip(destination_list, destination_amounts)):
        destination = destination.strip()
        print(destination, required_amount)

        # Get the tags for this destination
        tags_dict = {}
        if idx < len(destination_tags_list):
            try:
                tags_dict = json.loads(destination_tags_list[idx])
            except Exception as e:
                print(f"Error parsing tags for {destination}: {e}")

        # Initialize search distance and create empty GeoDataFrame
        search_dist = 1000
        all_amenities = gpd.GeoDataFrame()

        # Use all key-value pairs for the OSM query
        if tags_dict:
            tags = tags_dict
        else:
            # Load mapping and use the correct tag for this destination
            destination_mapping = load_destination_mapping()
            if destination in destination_mapping:
                tags = destination_mapping[destination]["tags"]
            else:
                tags = {}
        print(tags)
        # Special handling for schools: query all schools, then filter by isced:level
        if tags.get("amenity") == "school":
            # Query all schools
            while len(all_amenities) < required_amount:
                try:
                    new_amenities = ox.features_from_point(
                        center_point=(survey_line['lat'], survey_line['lon']),
                        tags={"amenity": "school"},
                        dist=search_dist
                    )
                    if not new_amenities.empty:
                        new_amenities[['lon', 'lat']] = new_amenities.apply(get_lon_lat, axis=1).apply(pd.Series)
                        new_amenities['destination_type'] = destination
                        # Post-filter by isced:level if specified
                        isced_level = str(tags.get("isced:level", "")).strip()
                        if isced_level:
                            # Accept matches like '1', '01', '1;2', etc.
                            new_amenities = new_amenities[
                                new_amenities['isced:level'].astype(str).str.contains(rf'(?:^|;)0?{isced_level}(?:;|$)', na=False)
                            ]
                        all_amenities = pd.concat([all_amenities, new_amenities]).drop_duplicates()
                    if len(all_amenities) >= required_amount:
                        break
                except Exception as e:
                    print(f"Error searching for {destination} within {search_dist} meters: {str(e)}")
                if search_dist < 5000:
                    search_dist += 1000
                elif search_dist < 15000:
                    search_dist += 2500
                else:
                    search_dist += 5000
        else:
            # Keep searching until we find enough amenities
            while len(all_amenities) < required_amount:
                try:
                    new_amenities = ox.features_from_point(
                        center_point=(survey_line['lat'], survey_line['lon']),
                        tags=tags,
                        dist=search_dist
                    )
                    if not new_amenities.empty:
                        new_amenities[['lon', 'lat']] = new_amenities.apply(get_lon_lat, axis=1).apply(pd.Series)
                        new_amenities['destination_type'] = destination
                        all_amenities = pd.concat([all_amenities, new_amenities]).drop_duplicates()
                    if len(all_amenities) >= required_amount:
                        break
                except Exception as e:
                    print(f"Error searching for {destination} within {search_dist} meters: {str(e)}")
                if search_dist < 5000:
                    search_dist += 1000
                elif search_dist < 15000:
                    search_dist += 2500
                else:
                    search_dist += 5000
        network_home = ox.graph_from_point(
            center_point=(survey_line['lat'], survey_line['lon']),
            dist=search_dist,
            network_type=f'{mode}',
            simplify=False
        )
        if not all_amenities.empty:
            for i, row in all_amenities.iterrows():
                nearest_node_dest = ox.nearest_nodes(network_home, row['lon'], row['lat'])
                try:
                    route = ox.shortest_path(network_home, nearest_node_home, nearest_node_dest, weight='length')
                    gdf_nodes, gdf_edges = ox.graph_to_gdfs(network_home)
                    multiindex_list = generate_multindex(route)
                    shrt_gdf_edges = gdf_edges[gdf_edges.index.isin(multiindex_list)]
                    all_amenities.at[i, 'dist_to'] = shrt_gdf_edges['length'].sum()
                    all_amenities.at[i, 'time_to'] = shrt_gdf_edges['length'].sum() / speed
                except Exception as e:
                    print(f"Error calculating path to amenity: {str(e)}")
                    all_amenities.at[i, 'dist_to'] = float('inf')
            closest_amenities = all_amenities.nsmallest(required_amount, 'dist_to')
            all_relevant_amenities = pd.concat([all_relevant_amenities, closest_amenities], ignore_index=True)
    return all_relevant_amenities


def create_accessibility_polygon(edges_gdf):
    # Project the edges to a local UTM projection for accurate calculations
    local_utm_crs = ox.projection.project_gdf(edges_gdf).crs
    edges_projected = edges_gdf.to_crs(local_utm_crs)

    # Create a small buffer to connect the network
    buffer_size = 1  # meters
    edges_buffered = edges_projected.geometry.buffer(buffer_size)

    # Merge all buffered edges
    unified = shapely.ops.unary_union(edges_buffered)

    # Get the convex hull to create a smooth outer boundary
    if isinstance(unified, (Polygon, MultiPolygon)):
        boundary = unified.convex_hull
    else:
        # If somehow we don't have a polygon, create one from the bounds
        boundary = unified.envelope

    # Convert back to GeoSeries in EPSG:4326 for folium
    return gpd.GeoSeries([boundary], crs=local_utm_crs).to_crs(epsg=4326)


def find_nearest_pt_stop(survey_line, speed, network_home, nearest_node_home):
    """Find the nearest public transport stop and add it to the map with the path to reach it."""
    # Define tags for different types of public transport stops
    pt_tags = {
        'train': {'railway': ['station', 'halt']},
        'subway': {'railway': ['subway_entrance', 'station'], 'station': 'subway'},
        'tram': {'railway': 'tram_stop'},
        'bus': {'highway': 'bus_stop'}
    }

    # Initialize variables to track the closest stop
    min_distance = float('inf')
    closest_stop = None
    closest_stop_type = None
    closest_route = None

    # Search for each type of public transport stop
    search_dist = 1000
    max_search_dist = 5000

    while closest_stop is None and search_dist <= max_search_dist:
        for pt_type, tags in pt_tags.items():
            try:
                # Search for stops
                stops = ox.features_from_point(
                    center_point=(survey_line['lat'], survey_line['lon']),
                    tags=tags,
                    dist=search_dist
                )

                if not stops.empty:
                    # Add coordinates to stops
                    stops[['lon', 'lat']] = stops.apply(get_lon_lat, axis=1).apply(pd.Series)

                    # Find the closest stop
                    for i, stop in stops.iterrows():
                        nearest_node_stop = ox.nearest_nodes(network_home, stop['lon'], stop['lat'])
                        try:
                            route = ox.shortest_path(network_home, nearest_node_home, nearest_node_stop,
                                                     weight='length')
                            if route:
                                gdf_nodes, gdf_edges = ox.graph_to_gdfs(network_home)
                                multiindex_list = generate_multindex(route)
                                route_edges = gdf_edges[gdf_edges.index.isin(multiindex_list)]
                                distance = route_edges['length'].sum()
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_stop = stop
                                    closest_stop_type = pt_type
                                    closest_route = route
                        except Exception as e:
                            print(f"Error calculating path to {pt_type} stop: {str(e)}")
            except Exception as e:
                print(f"Error searching for {pt_type} stops: {str(e)}")
        if closest_stop is None:
            search_dist += 1000

    time = round((min_distance / speed), 1)
    print(f'Time to PT: {time} min')

    return closest_stop_type, closest_stop, closest_route, time


# The function creates the map based on the current line and the mode
def mode_map(start_point, accessibility_map, mode, survey_line, legend_input=None):
    network_home = ox.graph_from_point(center_point=(survey_line['lat'], survey_line['lon']),
                                       dist=500,
                                       network_type='walk',
                                       simplify=False)

    if mode == 'bike':
        network_home = ox.graph_from_point(center_point=(survey_line['lat'], survey_line['lon']),
                                           dist=500,
                                           network_type='bike',
                                           simplify=False)

    # Find the nearest node with Euclidean distance to home according to the current network type
    nearest_node_home = ox.nearest_nodes(network_home, survey_line['lon'], survey_line['lat'])

    # Find speed for age and gender group
    speed = demographic_speed(survey_line, mode)  # speed in m/min

    # Find and display nearest public transport stop
    closest_stop_type, closest_stop, closest_route, pt_time = find_nearest_pt_stop(survey_line,
                                                                                   speed,
                                                                                   network_home,
                                                                                   nearest_node_home)

    # Call the closest amenity finder for the current survey entry
    all_relevant_amenities = find_closest_amenity(survey_line, speed, nearest_node_home, mode)

    # Calculate distance thresholds based on different destination categories
    if not all_relevant_amenities.empty:
        # Get all destinations and their distances
        all_distances = all_relevant_amenities['dist_to'].tolist()
        max_dist_all = max(all_distances)

        ranks = [int(r.strip()) for r in survey_line['destinations rank'].split(',')]
        frequencies = [int(f.strip()) for f in survey_line['destinations frequency'].split(',')]
        destinations = [d.strip() for d in survey_line['destinations mapped'].split(',')]
        dest_amounts = [int(a.strip()) for a in survey_line['destination amount'].split(',')]
        dest_info = list(zip(destinations, ranks, frequencies, dest_amounts))
        dest_info.sort(key=lambda x: x[1], reverse=True)

        # Get top 3 most important destinations
        top_3_destinations = [d[0] for d in dest_info[:3]]
        top_3_dists = []
        for dest in top_3_destinations:
            required_amount = next((d[3] for d in dest_info if d[0] == dest), 1)
            dest_dists = all_relevant_amenities[all_relevant_amenities['destination_type'] == dest]['dist_to'].nlargest(
                required_amount)
            top_3_dists.extend(dest_dists)
        max_dist_top3 = max(top_3_dists) if top_3_dists else 0

        # Get frequently visited destinations (weekly or more)
        frequent_destinations = [d[0] for d in dest_info if d[2] in [1, 2, 3]]
        frequent_dists = []
        for dest in frequent_destinations:
            required_amount = next((d[3] for d in dest_info if d[0] == dest), 1)
            dest_dists = all_relevant_amenities[all_relevant_amenities['destination_type'] == dest]['dist_to'].nlargest(
                required_amount)
            frequent_dists.extend(dest_dists)
        max_dist_frequent = max(frequent_dists) if frequent_dists else 0

        # Create distance intervals based on the calculated thresholds
        dist_intervals = [max_dist_all, max_dist_top3, max_dist_frequent]
        # Remove duplicates and sort
        dist_intervals = sorted(list(dist_intervals))
    else:
        # If no amenities found, return empty map
        return accessibility_map

    # Use the maximum distance found in the amenities
    dist = max(all_relevant_amenities['dist_to'].max(), dist_intervals[-1])

    network_home = ox.graph_from_point(center_point=(survey_line['lat'], survey_line['lon']),
                                       dist=dist,
                                       network_type='walk',
                                       simplify=False)

    if mode == 'bike':
        network_home = ox.graph_from_point(center_point=(survey_line['lat'], survey_line['lon']),
                                           dist=dist,
                                           network_type='bike',
                                           simplify=False)

    # Assign numbers to each unique destination type
    destination_types = list(all_relevant_amenities['destination_type'].unique())
    type_to_number = {dtype: str(i + 1) for i, dtype in enumerate(destination_types)}

    # Draw isochrone polygons from largest to smallest, with higher opacity for outermost
    opacities = [0.1, 0.1, 0.1]
    isochrone_colors = {
        'All needs': '#0096c7',
        'Needs frequented at least once a week': '#0a9548',
        '3 most important needs': '#e6b400'
    }
    isochrone_data = [
        ('All needs', max_dist_all, isochrone_colors['All needs']),
        ('Needs frequented at least once a week', max_dist_frequent, isochrone_colors['Needs frequented at least once a week']),
        ('3 most important needs', max_dist_top3, isochrone_colors['3 most important needs'])
    ]

    combined_color = '#ff7f00'  # Orange color for combined category

    # Sort by distance descending (largest first)
    isochrone_data = sorted(isochrone_data, key=lambda x: x[1], reverse=True)
    time_intervals = [d[1] / speed for d in isochrone_data]

    # Draw polygons
    for (label, dist_radius, _), opacity in zip(isochrone_data, opacities):
        color = isochrone_colors[label]
        subgraph = nx.ego_graph(network_home, nearest_node_home, radius=dist_radius, distance='length')
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(subgraph)
        if len(edges_gdf) > 0:
            accessibility_area = create_accessibility_polygon(edges_gdf)
            for geom in accessibility_area:
                if isinstance(geom, (Polygon, MultiPolygon)):
                    geojson_data = gpd.GeoSeries([geom]).__geo_interface__
                    folium.GeoJson(
                        geojson_data,
                        style_function=lambda x, color=color, opacity=opacity: {
                            'fillColor': color,
                            'color': color,
                            'weight': 2,  # Border width
                            'fillOpacity': opacity,
                            'opacity': 1  # Border opacity
                        }
                    ).add_to(accessibility_map)
        else:
            print(f'Isochrone {dist_radius} m: No edges found!')

    # Add PT stop marker BEFORE amenity markers
    if closest_stop is not None:
        # Create a custom pane for PT stop marker with lower z-index
        folium.map.CustomPane("lower_pane", z_index=450).add_to(accessibility_map)
        # Downward-pointing red triangle for PT stop, in custom pane
        pt_marker_html = '''
            <div style="
                width: 0;
                height: 0;
                border-left: 7px solid transparent;
                border-right: 7px solid transparent;
                border-top: 14px solid #e53935;
                margin-top: 0px;
                margin-left: 0px;
            "></div>
        '''
        folium.Marker(
            location=[closest_stop['lat'], closest_stop['lon']],
            icon=folium.DivIcon(html=pt_marker_html),
            pane="lower_pane"
        ).add_to(accessibility_map)

        if closest_route:
            route_coords = []
            for node in closest_route:
                node_data = network_home.nodes[node]
                route_coords.append([node_data['y'], node_data['x']])

            folium.PolyLine(
                route_coords,
                weight=4,
                color='#e53935',
                opacity=0.8,
                popup=f"Path to nearest PT stop ({closest_stop_type})"
            ).add_to(accessibility_map)

    # Add markers for amenities with different colors based on categories and travel time
    if not all_relevant_amenities.empty:
        top_3_set = set(top_3_destinations)
        frequent_set = set(frequent_destinations)
        for i, row in all_relevant_amenities.iterrows():
            dest_type = row['destination_type']
            is_top_3 = dest_type in top_3_set
            is_frequent = dest_type in frequent_set
            if is_top_3 and is_frequent:
                color = combined_color
            elif is_top_3:
                color = isochrone_colors['3 most important needs']
            elif is_frequent:
                color = isochrone_colors['Needs frequented at least once a week']
            else:
                color = isochrone_colors['All needs']
            if color is not None:
                number_icon = type_to_number.get(dest_type, '?')
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    icon=folium.DivIcon(html=f'''
                        <div style="
                            background-color: {color};
                            color: black;
                            border-radius: 50%;
                            width: 12px;
                            height: 12px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 9px;
                            font-weight: bold;
                        ">{number_icon}</div>
                    '''),
                ).add_to(accessibility_map)

    # Always add origin marker as a triangle
    origin_marker_html = '''
        <div style="
            width: 0;
            height: 0;
            border-left: 7px solid transparent;
            border-right: 7px solid transparent;
            border-top: 14px solid #8BC34A;
            margin-top: 0px;
            margin-left: 0px;
        "></div>
    '''
    folium.Marker(
        location=[start_point[1], start_point[0]],
        icon=folium.DivIcon(html=origin_marker_html),
        pane="lower_pane"
    ).add_to(accessibility_map)

    # Add north arrow and coordinate system information in a single container
    map_info_html = '''
        <div style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: white;
            padding: 5px;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0,0,0,0.2);
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 15px;
        ">
            <div style="
                width: 0;
                height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 20px solid black;
                margin: 0 auto;
            "></div>
            <div style="text-align: center; font-size: 12px; margin-top: 5px;">N</div>
            <div style="
                border-left: 1px solid #ccc;
                height: 20px;
                margin: 0 5px;
            "></div>
            <div style="font-size: 12px;">
                <div>Coordinate System: WGS84 (EPSG:4326)</div>
            </div>
        </div>
    '''
    accessibility_map.get_root().html.add_child(folium.Element(map_info_html))

    # Add a legend with smaller text and reduced spacing
    legend_html = f'''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 200px; height: 500px; 
                        border:2px solid grey; z-index:9999; 
                        background-color:white;
                        padding: 5px;
                        font-size: 11px;
                        ">
            <p style="margin-bottom: 4px; font-size: 11px; display: flex; align-items: center;">
                <span style="display: inline-block; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 14px solid #8BC34A; margin-right: 7px;"></span>
                Origin
            </p>
            <p style="margin-bottom: 4px; font-size: 11px; display: flex; align-items: center;">
                <span style="display: inline-block; width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-top: 14px solid #e53935; margin-right: 7px;"></span>
                Nearest Public Transport Stop ({closest_stop_type}): {pt_time} min
            </p>
            <p style="margin-bottom: 4px; font-weight: bold; font-size: 12px;">Isochrones:</p>
    '''

    # Add legend entries for each time interval
    for (label, _, _), time in zip(isochrone_data, time_intervals):
        legend_html += f'''
            <p style="margin-bottom: 2px; font-size: 11px;">
                <span style="background-color: {isochrone_colors[label]}; padding: 0 2px; margin-right: 5px;"></span>{label}: {round(time, 1)} min
            </p>
        '''

    # Add legend for amenity numbers and colors
    legend_html += '''<p style="margin-bottom: 4px; font-weight: bold; font-size: 12px;">Amenity Types:</p>'''
    for dtype, num in type_to_number.items():
        legend_html += f'<p style="font-size: 11px; margin-bottom: 2px;"><b>{num}</b>: {dtype}</p>'

    # Add legend for amenity categories
    legend_html += '''<p style="margin-bottom: 4px; font-weight: bold; font-size: 12px;">Amenity Categories:</p>'''
    for label in ['All needs', '3 most important needs', 'Needs frequented at least once a week']:
        legend_html += f'''<p style="font-size: 11px; margin-bottom: 2px;"><span style="background-color: {isochrone_colors[label]}; border-radius: 50%; width: 11px; height: 11px; display: inline-block; margin-right: 5px;"></span>{label}</p>'''
    legend_html += f'''<p style="font-size: 11px; margin-bottom: 2px;"><span style="background-color: {combined_color}; border-radius: 50%; width: 11px; height: 11px; display: inline-block; margin-right: 5px;"></span>Top 3 & Frequent</p>'''

    # Add demographic constraints from legend_input if provided
    if legend_input is not None:
        demo_lines = []
        for field, value in legend_input.items():
            if str(value).lower() != 'all':
                demo_lines.append(f"<b>{field.capitalize()}:</b> {value}")
        if demo_lines:
            legend_html += '<div style="margin-bottom: 6px; font-size: 11px;"><b>Demographic Constraints:</b><br>' + '<br>'.join(
                demo_lines) + '</div>'

    legend_html += '''</div>'''

    accessibility_map.get_root().html.add_child(folium.Element(legend_html))

    return accessibility_map


# The function calculates the accessibility for each origin or group
def accessibility_indicator(data, legend_input=None):
    # For each survey line or group, create accessibility maps
    for i, row in data.iterrows():
        orig_home = row['lon'], row['lat']
        modes = ['walk', 'bike', 'wheelchair']

        # Parse destination ranks and frequencies for individuals
        ranks = [int(r.strip()) for r in row['destinations rank'].split(',')]
        frequencies = [int(f.strip()) for f in row['destinations frequency'].split(',')]
        destinations = [d.strip() for d in row['destinations mapped'].split(',')]
        dest_info = list(zip(destinations, ranks, frequencies))
        dest_info.sort(key=lambda x: x[1], reverse=True)

        for j, mode_name in enumerate(modes):
            # Create a single map for all destination types
            accessibility_map = folium.Map(location=[orig_home[1], orig_home[0]],
                                           zoom_start=15,
                                           tiles='cartodbpositron',
                                           control_scale=True)

            # Add title to map
            title_html = f'''
                <div style="position: fixed; 
                            top: 10px; left: 50px; width: 270px; height: 35px; 
                            border:2px solid grey; z-index:9999; 
                            background-color:white;
                            padding: 5px;
                            font-size: 14px;
                            font-weight: bold;
                            text-align: center;
                            ">
                Accessibility Map ({mode_name.capitalize()})
                </div>
            '''
            accessibility_map.get_root().html.add_child(folium.Element(title_html))

            # Create the map with all isochrones
            accessibility_map = mode_map(orig_home, accessibility_map, mode_name, row, legend_input)

            # Save the single map
            filename = f'{i}_{mode_name}_accessibility_map.html'
            accessibility_map.save(filename)
            webbrowser.open(filename)


# Define movement speeds for demographics
### NOT DONE
def demographic_speed(survey_line, mode):
    # Determine age group
    age = survey_line['age']

    match age:
        case 1:
            age_group = 'child'
        case 2:
            age_group = 'young adult'
        case 3:
            age_group = 'adult'
        case 4:
            age_group = 'senior'
        case _:
            age_group = 'standard'

    # Determine gender group
    gender = survey_line['gender']

    match gender:
        case 1:
            gender_group = 'male'
        case 2:
            gender_group = 'female'
        case 3:
            gender_group = 'non-binary'
        case _:
            gender_group = 'standard'

    # Assign respective movement speed for age and gender
    if mode == 'walk':
        # Define walking speeds by gender and age group
        walking_speeds = {
            'standard': {  # gender
                'standard': 4.5 * (1000 / 60),
                'child': 5.3 * (1000 / 60),
                'young adult': 4.9 * (1000 / 60),
                'adult': 5.0 * (1000 / 60),
                'senior': 4.0 * (1000 / 60)
            },
            'male': {
                'standard': 4.6 * (1000 / 60),
                'child': 5.3 * (1000 / 60),
                'young adult': 4.9 * (1000 / 60),
                'adult': 5.2 * (1000 / 60),
                'senior': 4.1 * (1000 / 60)
            },
            'female': {
                'standard': 4.3 * (1000 / 60),
                'child': 5.4 * (1000 / 60),
                'young adult': 4.8 * (1000 / 60),
                'adult': 4.8 * (1000 / 60),
                'senior': 3.8 * (1000 / 60)
            },
            'non-binary': {
                'standard': 4.5 * (1000 / 60),
                'child': 5.3 * (1000 / 60),
                'young adult': 4.9 * (1000 / 60),
                'adult': 5.0 * (1000 / 60),
                'senior': 4.0 * (1000 / 60)
            }
        }

        if gender_group in walking_speeds and age_group:
            speed = walking_speeds[gender_group][age_group]

    elif mode == 'bike':
        # Define walking speeds by gender and age group
        cycling_speeds = {
            'standard': {  # gender
                'standard': 15.3 * (1000 / 60),
                'child': 13.0 * (1000 / 60),
                'young adult': 16.6 * (1000 / 60),
                'adult': 15.8 * (1000 / 60),
                'senior': 13.9 * (1000 / 60)
            },
            'male': {
                'standard': 15.3 * (1000 / 60),
                'child': 13.6 * (1000 / 60),
                'young adult': 16.6 * (1000 / 60),
                'adult': 15.8 * (1000 / 60),
                'senior': 13.9 * (1000 / 60)
            },
            'female': {
                'standard': 15.3 * (1000 / 60),
                'child': 12.5 * (1000 / 60),
                'young adult': 16.6 * (1000 / 60),
                'adult': 15.8 * (1000 / 60),
                'senior': 13.9 * (1000 / 60)
            },
            'non-binary': {
                'standard': 15.3 * (1000 / 60),
                'child': 13.0 * (1000 / 60),
                'young adult': 16.6 * (1000 / 60),
                'adult': 15.8 * (1000 / 60),
                'senior': 13.9 * (1000 / 60)
            }
        }

        if gender_group in cycling_speeds and age_group:
            speed = cycling_speeds[gender_group][age_group]

    elif mode == 'wheelchair':
        # Define walking speeds by gender and age group
        wheelchair_speeds = {
            'standard': {
                'standard': 2.2 * (1000 / 60),
                'child': 2.5 * (1000 / 60),
                'young adult': 2.2 * (1000 / 60),
                'adult': 2.2 * (1000 / 60),
                'senior': 2.2 * (1000 / 60)
            },
            'male': {
                'standard': 2.3 * (1000 / 60),
                'child': 2.7 * (1000 / 60),
                'young adult': 2.3 * (1000 / 60),
                'adult': 2.3 * (1000 / 60),
                'senior': 2.3 * (1000 / 60)
            },
            'female': {
                'standard': 2.1 * (1000 / 60),
                'child': 2.2 * (1000 / 60),
                'young adult': 2.1 * (1000 / 60),
                'adult': 2.1 * (1000 / 60),
                'senior': 2.1 * (1000 / 60)
            },
            'non-binary': {
                'standard': 2.2 * (1000 / 60),
                'child': 2.5 * (1000 / 60),
                'young adult': 2.2 * (1000 / 60),
                'adult': 2.2 * (1000 / 60),
                'senior': 2.2 * (1000 / 60)
            }
        }

        if gender_group in wheelchair_speeds and age_group:
            speed = wheelchair_speeds[gender_group][age_group]

    return speed


# Choose the demographic attributes that you want to consider
def person_group(data):
    # Create the main window
    root = tk.Tk()
    root.title("Person Group Selection")

    # Set window size (width x height)
    window_width = 600
    window_height = 500  # Increased height to accommodate new fields
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width / 2 - window_width / 2)
    center_y = int(screen_height / 2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    # Create a frame to hold the dropdown menus with scrollbar
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    # Add a canvas
    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Create a frame to hold the content
    frame = ttk.Frame(canvas, padding="20")
    canvas.create_window((0, 0), window=frame, anchor='nw')

    # Add a title label
    title_label = ttk.Label(frame, text="Select Person Group Characteristics", font=("Arial", 14, "bold"))
    title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

    # 1. Age Group
    ttk.Label(frame, text="Age Group:", font=("Arial", 11)).grid(row=1, column=0, sticky=tk.W, pady=10)
    age_var = tk.StringVar()
    age_dropdown = ttk.Combobox(frame, textvariable=age_var, width=30)
    age_dropdown['values'] = ('All', 'younger than 18', '18-29', '30-59', '60 or older')
    age_dropdown.grid(row=1, column=1, sticky=tk.W, pady=10)
    age_dropdown.current(0)

    # 2. Gender
    ttk.Label(frame, text="Gender:", font=("Arial", 11)).grid(row=2, column=0, sticky=tk.W, pady=10)
    gender_var = tk.StringVar()
    gender_dropdown = ttk.Combobox(frame, textvariable=gender_var, width=30)
    gender_dropdown['values'] = ('All', 'male', 'female', 'non-binary')
    gender_dropdown.grid(row=2, column=1, sticky=tk.W, pady=10)
    gender_dropdown.current(0)

    # 3. Occupation
    ttk.Label(frame, text="Primary Occupation:", font=("Arial", 11)).grid(row=3, column=0, sticky=tk.W, pady=10)
    occupation_var = tk.StringVar()
    occupation_dropdown = ttk.Combobox(frame, textvariable=occupation_var, width=30)
    occupation_dropdown['values'] = ('All', 'employed (full/part time)', 'self-employed (full/part time)',
                                     'student', 'retired', 'unemployed', 'household work or similar',
                                     'unable to work', 'other')
    occupation_dropdown.grid(row=3, column=1, sticky=tk.W, pady=10)
    occupation_dropdown.current(0)

    # 4. Household Composition
    ttk.Label(frame, text="Household Composition:", font=("Arial", 11)).grid(row=4, column=0, sticky=tk.W, pady=10)
    household_var = tk.StringVar()
    household_dropdown = ttk.Combobox(frame, textvariable=household_var, width=30)
    household_dropdown['values'] = ('All', 'living alone', 'living with a partner', 'living in a shared flat',
                                    'other')
    household_dropdown.grid(row=4, column=1, sticky=tk.W, pady=10)
    household_dropdown.current(0)

    # 4a. Children in Household
    ttk.Label(frame, text="Children under 18 in household:", font=("Arial", 11)).grid(row=5, column=0, sticky=tk.W,
                                                                                      pady=10)
    children_var = tk.StringVar()
    children_dropdown = ttk.Combobox(frame, textvariable=children_var, width=30)
    children_dropdown['values'] = ('All', 'yes', 'no')
    children_dropdown.grid(row=5, column=1, sticky=tk.W, pady=10)
    children_dropdown.current(0)

    # 4b. Adults needing care
    ttk.Label(frame, text="Adults needing care:", font=("Arial", 11)).grid(row=6, column=0, sticky=tk.W, pady=10)
    adults_var = tk.StringVar()
    adults_dropdown = ttk.Combobox(frame, textvariable=adults_var, width=30)
    adults_dropdown['values'] = ('All', 'yes', 'no')
    adults_dropdown.grid(row=6, column=1, sticky=tk.W, pady=10)
    adults_dropdown.current(0)

    # 5. Income Range
    ttk.Label(frame, text="Monthly Net Income:", font=("Arial", 11)).grid(row=7, column=0, sticky=tk.W, pady=10)
    income_var = tk.StringVar()
    income_dropdown = ttk.Combobox(frame, textvariable=income_var, width=30)
    income_dropdown['values'] = ('All', '0-1999€', '2000-3499€', '3500-5999€', '6000€ or more')
    income_dropdown.grid(row=7, column=1, sticky=tk.W, pady=10)
    income_dropdown.current(0)

    # 6. Education
    ttk.Label(frame, text="Highest Education:", font=("Arial", 11)).grid(row=8, column=0, sticky=tk.W, pady=10)
    education_var = tk.StringVar()
    education_dropdown = ttk.Combobox(frame, textvariable=education_var, width=30)
    education_dropdown['values'] = ('All', 'primary school', 'secondary school', "bachelor's degree",
                                    "master's/doctoral degree", 'vocational degree', 'no degree',
                                    'other')
    education_dropdown.grid(row=8, column=1, sticky=tk.W, pady=10)
    education_dropdown.current(0)

    # 7. Physical Aid
    ttk.Label(frame, text="Use of Physical Aid:", font=("Arial", 11)).grid(row=9, column=0, sticky=tk.W, pady=10)
    aid_var = tk.StringVar()
    aid_dropdown = ttk.Combobox(frame, textvariable=aid_var, width=30)
    aid_dropdown['values'] = ('All', 'yes', 'no')
    aid_dropdown.grid(row=9, column=1, sticky=tk.W, pady=10)
    aid_dropdown.current(0)

    # Create a custom style for the button
    style = ttk.Style()
    style.configure('Accent.TButton', font=('Arial', 11, 'bold'))

    # Add a submit button with styling
    submit_button = ttk.Button(
        frame,
        text="Generate Accessibility Map",
        style='Accent.TButton',
        command=lambda: process_selections(
            age_var.get(), gender_var.get(), occupation_var.get(),
            household_var.get(), children_var.get(), adults_var.get(),
            income_var.get(), education_var.get(), aid_var.get(), root, data
        )
    )
    submit_button.grid(row=10, column=0, columnspan=2, pady=30)

    # Run the application
    root.mainloop()


# Run the accessibility indicator with the selected group attributes
def process_selections(age, gender, occupation, household, children, adults,
                       income, education, aid, root, data):
    """
    Process the selected values from the dropdown menus.
    """
    # Print the selections (for debugging)
    print(f"""
    Age: {age}
    Gender: {gender}
    Occupation: {occupation}
    Household: {household}
    Children under 18: {children}
    Adults needing care: {adults}
    Income: {income}
    Education: {education}
    Physical Aid: {aid}
    """)

    # Close the window
    root.destroy()

    # Map user-friendly input to codes for group filtering
    user_input = {
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'household': household,
        'children': children,
        'adults': adults,
        'income': income,
        'education': education,
        'aid': aid
    }
    mapped_input = map_demographic_input_to_codes(user_input)

    # Filter the data based on mapped codes
    filtered_data = aggregate_data(
        mapped_input['age'], mapped_input['gender'], mapped_input['occupation'],
        mapped_input['household'], mapped_input['children'], mapped_input['adults'],
        mapped_input['income'], mapped_input['education'], mapped_input['aid'], data
    )

    filtered_data['age'] = mapped_input['age']
    filtered_data['gender'] = mapped_input['gender']
    filtered_data['aid'] = mapped_input['aid']

    origin = choose_origin()
    # Get coordinates from geocoding
    geolocator = Nominatim(user_agent="myGeocoder", timeout=10)
    try:
        location = geolocator.geocode(f"{origin[0]}, {origin[1]} {origin[2]}, {origin[3]}")
        lat, lon = location.latitude, location.longitude
    except Exception as e:
        tk.messagebox.showerror(message=f'Could not geocode origin address: {e}')
        return

    filtered_data['lat'] = lat
    filtered_data['lon'] = lon

    # If data was found, process it with the accessibility indicator
    if filtered_data is not None:
        accessibility_indicator(filtered_data, user_input)


def map_demographic_input_to_codes(user_input):
    # Define mappings for each demographic field
    mappings = {
        'age': {
            'younger than 18': 1,
            '18-29': 2,
            '30-59': 3,
            '60 or older': 4,
            'prefer not to say': 5
        },
        'gender': {
            'male': 1,
            'female': 2,
            'non-binary': 3,
            'prefer not to say': 4
        },
        'occupation': {
            'employed (full/part time)': 1,
            'self-employed (full/part time)': 2,
            'student': 3,
            'retired': 4,
            'unemployed': 5,
            'household work or similar': 6,
            'unable to work': 7,
            'other': 8,
            'prefer not to say': 9
        },
        'household': {
            'living alone': 1,
            'living with a partner': 2,
            'living in a shared flat': 3,
            'other': 4,
            'prefer not to say': 5
        },
        'children': {
            'yes': 1,
            'no': 2,
            'prefer not to say': 3
        },
        'adults': {
            'yes': 1,
            'no': 2,
            'prefer not to say': 3
        },
        'income': {
            '0-1999€': 1,
            '2000-3499€': 2,
            '3500-5999€': 3,
            '6000€ or more': 4,
            'prefer not to say': 5
        },
        'education': {
            'primary school': 1,
            'secondary school': 2,
            "bachelor's degree": 3,
            "master's/doctoral degree": 4,
            'vocational degree': 5,
            'no degree': 6,
            'other': 7,
            'prefer not to say': 8
        },
        'aid': {
            'yes': 1,
            'no': 2,
            'prefer not to say': 3
        }
    }
    mapped = {}
    for key, value in user_input.items():
        if key in mappings and value in mappings[key]:
            mapped[key] = mappings[key][value]
        else:
            mapped[key] = 0  # Default to 0 if not found
    return mapped


# Find all the datasets from the survey that fulfill the chosen group's attributes
def aggregate_data(age, gender, occupation, household, children, adults,
                   income, education, aid, data):
    try:
        # Create a copy of the data to filter
        filtered_data = data.copy()

        # Apply filters based on selections
        if age != 0:
            filtered_data = filtered_data[filtered_data['age'] == age]

        if gender != 0:
            filtered_data = filtered_data[filtered_data['gender'] == gender]

        if occupation != 0:
            filtered_data = filtered_data[filtered_data['occupation'] == occupation]

        if household != 0:
            filtered_data = filtered_data[filtered_data['household'] == household]

        if children != 0:
            filtered_data = filtered_data[filtered_data['children'] == children]

        if adults != 0:
            filtered_data = filtered_data[filtered_data['adults'] == adults]

        if income != 0:
            filtered_data = filtered_data[filtered_data['income'] == income]

        if education != 0:
            filtered_data = filtered_data[filtered_data['education'] == education]

        if aid != 0:
            filtered_data = filtered_data[filtered_data['aid'] == aid]

        # Check if any data matches the criteria
        if len(filtered_data) == 0:
            tk.messagebox.showwarning(
                title="No Matching Data",
                message="No data matches the selected criteria. Please try different selections."
            )
            return None

        # Identify common amenities (named by more than half of the people in the group)
        common_amenities = find_common_amenities(filtered_data)

        # Add the common amenities to the filtered data
        if common_amenities.empty:
            tk.messagebox.showinfo(
                title="No Common Amenities",
                message="No common amenities found for this group. The program will end now."
            )
            quit()
        else:
            # Show the common amenities in a message
            tk.messagebox.showinfo(
                title="Common Amenities",
                message=f"Common amenities for this group:\n{common_amenities['destinations original'].values[0]}"
            )

        # Show how many entries match the criteria
        tk.messagebox.showinfo(
            title="Data Filtered",
            message=f"Found {len(filtered_data)} entries matching your criteria."
        )

        return common_amenities

    except Exception as e:
        tk.messagebox.showerror(
            title="Error",
            message=f"An error occurred while filtering the data: {str(e)}"
        )
        return None


# Find amenities that are common among the chosen demographic constraints
def find_common_amenities(filtered_data):
    # Get the total number of people in the group
    total_people = len(filtered_data)

    # Create a dictionary to count occurrences and collect stats for each amenity (by tags)
    dest_stats = {}

    for i, row in filtered_data.iterrows():
        # Split the needs string into individual amenities
        dests = [amenity.strip() for amenity in row['destinations original'].split(',')]
        tags_list = row.get('destinations tags', '').split('|')
        ranks = [int(r.strip()) for r in row['destinations rank'].split(',')]
        freqs = [int(f.strip()) for f in row['destinations frequency'].split(',')]
        amounts = [int(a.strip()) for a in row['destination amount'].split(',')]

        for j, tags_json in enumerate(tags_list):
            if not tags_json:
                continue
            # Use tags_json as the key for grouping
            if tags_json not in dest_stats:
                dest_stats[tags_json] = {
                    'count': 0,
                    'ranks': [],
                    'freqs': [],
                    'amounts': [],
                    'names': []
                }
            dest_stats[tags_json]['count'] += 1
            if j < len(ranks):
                dest_stats[tags_json]['ranks'].append(ranks[j])
            if j < len(freqs):
                dest_stats[tags_json]['freqs'].append(freqs[j])
            if j < len(amounts):
                dest_stats[tags_json]['amounts'].append(amounts[j])
            if j < len(dests):
                dest_stats[tags_json]['names'].append(dests[j])

    # Find amenities that appear in more than half of the people
    threshold = total_people / 2
    common_amenities = []
    avg_ranks = []
    avg_freqs = []
    avg_amounts = []
    display_names = []
    for tags_json, stats in dest_stats.items():
        if stats['count'] > threshold:
            common_amenities.append(tags_json)
            avg_ranks.append(str(int(round(sum(stats['ranks']) / len(stats['ranks'])))))

            # Check if 50% or more people visit the destination at least once a week and set default value of 3 otherwise 4
            frequent_count = sum(1 for f in stats['freqs'] if f in [1, 2, 3])
            if frequent_count >= len(stats['freqs']) / 2:
                avg_freqs.append('3')  # Set to 3 if at least half are frequent
            else:
                avg_freqs.append('4')  # Set to 4 if not frequent enough

            avg_amounts.append(str(int(round(sum(stats['amounts']) / len(stats['amounts'])))))
            # Use the most common name for display
            if stats['names']:
                name_counts = pd.Series(stats['names']).value_counts()
                display_names.append(name_counts.idxmax())
            else:
                display_names.append('Unknown')

    # Prepare a single-row DataFrame with the same columns as single entry processing
    if common_amenities:
        data = {
            'destinations original': ', '.join(display_names),
            'destinations mapped': ', '.join(display_names),
            'destinations tags': '|'.join(common_amenities),
            'destinations rank': ', '.join(avg_ranks),
            'destinations frequency': ', '.join(avg_freqs),
            'destination amount': ', '.join(avg_amounts),
        }
        return pd.DataFrame([data])
    else:
        return pd.DataFrame([])


# Choose the origin for the person group analysis
def choose_origin():
    address_fields = ['Street Address', 'ZipCode', 'City', 'Country']
    def_values = ['Arcisstraße 21', '80333', 'Munich', 'Germany']
    origin = easygui.multenterbox(title='Origin address',
                                  msg='Please fill in the address information for your starting point',
                                  fields=address_fields,
                                  values=def_values)
    if origin:
        check = easygui.buttonbox(title='Check address information',
                                  msg=f'Is the information you entered correct?: {origin[0]}, {origin[1] + ' ' + origin[2]}, {origin[3]}',
                                  choices=['Yes', 'No', 'Cancel'])
        if check == 'Yes':
            return origin
        elif check == 'No':
            choose_origin()
        elif check == 'Cancel':
            quit()
        else:
            tk.messagebox.showerror(message='An error has occurred. Please try again')
    else:
        quit()


# The main script that starts the process of loading the survey and creating maps for every entry
def main_script(data):
    actions = ['Demographic group', 'Individuals', 'Cancel']
    answer = easygui.buttonbox(title='Action choice',
                               msg='Would you like to create maps for a demographic group or individuals?',
                               choices=actions)
    if answer == 'Demographic group':
        person_group(data)
    elif answer == 'Individuals':
        if not data.empty:
            # Prepare options for the user
            options = ['All'] + [
                f"{i}: {row['home_street']}, {row['home_city']}" if 'home_street' in row and 'home_city' in row else str(
                    i) for i, row in data.iterrows()]
            choice = easygui.choicebox(
                msg='Which survey line should be processed? (Default: All)',
                title='Survey Line Selection',
                choices=options,
                preselect=0
            )
            if choice and choice != 'All':
                # Extract the index from the choice string
                idx = int(choice.split(':')[0])
                data = data.iloc[[idx]]
        accessibility_indicator(data)
        easygui.msgbox(msg='All maps were successfully created.')
    elif answer == 'Cancel':
        quit()
    else:
        tk.messagebox.showerror(message='An error has occurred. Please try again')
        main_script(data)


# Prompt user for which survey line to process
survey_data = read_data()
main_script(survey_data)