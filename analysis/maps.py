import numpy as np
import pandas as pd
from analysis.utils import *
import sys
from shapely import wkt
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.express as px


class GetMaps:
    """
        Produce geo-maps included in the manuscript.
    """
    def __init__(self,
                 df,
                 ct_all,
                 random_seed = 2025
        ):
        self.random_seed = random_seed
        self.percentiles = [0, 20, 40, 60, 80, 100]
        self.metrics = ['FEND', 'RND STDEV', 'CDC mRFEI', 'USDA %LowAccess', 'LSR Density', '#Restaurants']
        self.df = rename(df) # Rename columns
        # To gpd
        self.df = self.to_gpd(self.df)
        self.ct_all = ct_all

    @staticmethod
    def to_gpd(df):
        # To gpd using geomoetry
        df['geometry'] = df['geometry'].apply(wkt.loads)
        df = gpd.GeoDataFrame(df, crs="EPSG:4326", geometry='geometry')

        return df

    def county_maps(self):
        # County level maps
        for metric in self.metrics:
            bins = np.percentile(self.df[metric], self.percentiles)
            colors = get_map_colors(metric)
            # Create labels for the legend
            labels = [f"{self.percentiles[i]}-{self.percentiles[i+1]}th percentile" for i in range(len(self.percentiles)-1)]
            # Assign percentile group
            self.df[f'{metric}_percentile'] = pd.cut(
                self.df[metric], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
            # Rename labels adding average for each percentile group
            for i in set(self.df[f'{metric}_percentile']):
                self.df[f'{metric}_percentile'].replace(i, f"{i} ({round(self.df[self.df[f'{metric}_percentile']==i][metric].mean(), 2)})", inplace=True)
                
            # Create the choropleth map
            fig = px.choropleth(
                self.df[[f'{metric}_percentile', metric, 'CountyName', 'StateAbbr']], 
                geojson = self.df.geometry,
                locations = self.df.index,
                color = f'{metric}_percentile',
                color_discrete_sequence = colors,
                scope = "usa",
                hover_data = [metric, 'CountyName', 'StateAbbr'],
                category_orders = {f'{metric}_percentile': sorted(list(set(self.df[f'{metric}_percentile'])))}
            )

            # Update layout
            fig.update_layout(
                margin = {"r":0, "l":0, "b":0},
                geo = dict(bgcolor = 'rgba(0,0,0,0)',
                            lakecolor = 'rgba(51,17,0)',
                            landcolor = 'grey',
                            subunitcolor = 'rgba(51,17,0)'
                )
            )
            # Customize legend
            fig.update_layout(
                legend_title_text='',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                )
            )
            # Export map
            fig.write_html(f"{metric}_county_map.html")

    @staticmethod
    def regular_choropleth(city, df, metric, column, color_map, boundary_df, legend):
        # Create a figure, axis and plot
        fig, ax = plt.subplots(figsize=(40, 50)) 
        df.plot(
            column = column,
            cmap = color_map,
            linewidth = 0.8, 
            edgecolor = '0.8', 
            ax = ax,
            legend = legend
        )
        # Plot the boundaries on top
        boundary_df.boundary.plot(ax=ax, color='black', linewidth=1)
        # Remove axis and adjust plot
        ax.axis('off')
        ax.autoscale()
        # Show plot
        fig.savefig(f"{city}_{metric}_map.pdf", bbox_inches='tight')

    @staticmethod
    def bubble_map(city, df, metric, color_map, boundary_df):
        # Create a figure, axis and plot
        fig, ax = plt.subplots(figsize=(40, 50)) 
        # Calculate centroids for each census tracts
        df['centroid'] = df.geometry.centroid
        # Plot the boundaries first in grey color
        boundary_df.plot(ax=ax, color='grey', edgecolor='white')
        # Add bubble map to the base map
        ax.scatter(df.centroid.x, 
                   df.centroid.y,
                   s = df['#Restaurants'] * 15,
                   c = df[f'{metric}_percentile'],
                   cmap = color_map,
                   alpha = 1.0,
                   edgecolors = 'black',
                   linewidth = 0.5)
        # Remove axis and adjust plot
        ax.axis('off')
        ax.autoscale()
        # Show plot
        fig.savefig(f"{city}_{metric}_bubblemap.pdf", bbox_inches='tight')
        
    def city_maps(self, fend_df, boundary_df, city):
        # To gpd
        fend_df = self.to_gpd(fend_df)
        # NYC and LA maps
        for metric in ['FEND', 'RND STDEV', 'CDC mRFEI', '#Restaurants']:
            if metric == 'CDC mRFEI':
                ct_df = remove_na(self.ct_all, metric)
            else:
                ct_df = self.ct_all
            # Percentile based on national census tracts
            bins = np.percentile(ct_df[metric], self.percentiles)
            colors = get_map_colors(metric)
            custom_cmap = ListedColormap(colors)

            # Assign percentile group
            fend_df[f'{metric}_percentile'] = pd.cut(
                fend_df[metric],
                bins=bins, 
                labels=range(len(bins)-1), 
                include_lowest=True
            )
            # Regular choropleth colored by the percentiles
            self.regular_choropleth(city, fend_df, metric, f'{metric}_percentile', custom_cmap, boundary_df, False)
            # Bubble map based on number of restaurant # De-emphasizing tracts with less restaurants
            self.bubble_map(city, fend_df, metric, custom_cmap, boundary_df)

    @staticmethod
    def get_policy_df(df):
        # Policy implication based on FEND and mRFEI# Divide in 10 group by percentiles
        df['decile_fend'] = pd.qcut(df['FEND'], 10, labels=range(1, 11))
        df['decile_mrfei'] = pd.qcut(df['CDC mRFEI'], 10, labels=range(1, 11))
        # Get High FEND (>80%) and Low mRFEI (<20%) counties
        hflm = df[(df['decile_fend'] > 8) & (df['decile_mrfei'] < 3)].copy()
        hflm['policy'] = 'Above 80th percentile FEND & Below 20th percentile mRFEI'
        # Get Low FEND (<20%) and High mRFEI (>80%) counties
        lfhm = df[(df['decile_fend'] < 3) & (df['decile_mrfei'] > 8)].copy()
        lfhm['policy'] = 'Below 20th percentile FEND & Above 80th percentile mRFEI'
        # df for policy map
        policy_df = pd.concat([hflm, lfhm], ignore_index=True)
        
        return policy_df
    
    def county_policy_map(self):
        # Maps for guiding policy implications
        # Locating county with disagreement in healthiness based on different FE metrics
        policy_df = self.get_policy_df(self.df)
        # Create map
        fig = px.choropleth(policy_df[['policy', 'CountyName', 'StateAbbr', 'FEND', 'CDC mRFEI']],
                            geojson = policy_df.geometry,
                            locations = policy_df.index,
                            color = 'policy',
                            color_discrete_map = {"Below 20th percentile FEND & Above 80th percentile mRFEI": "#ff0000",
                                                  "Above 80th percentile FEND & Below 20th percentile mRFEI": "#54ff00"},
                            hover_data = ['CountyName', 'StateAbbr', 'FEND', 'CDC mRFEI'],
                            scope="usa"
                            )
        # Update layout
        fig.update_layout(
            margin = {"r":0, "l":0, "b":0},
            geo = dict(bgcolor = 'rgba(0,0,0,0)',
                        lakecolor = 'rgba(51,17,0)',
                        landcolor = 'grey',
                        subunitcolor = 'rgba(51,17,0)'
            )
        )
        # fig.update_layout(showlegend=False)
        # Export map
        fig.write_html(f"county_policy_map.html")
    
    def city_policy_map(self, fend_df, boundary_df, city):
        # To gpd
        fend_df = self.to_gpd(fend_df)
        # Policy implication for each region # Local policy
        # Locating census tract with disagreement in healthiness based on different FE metrics
        policy_df = self.get_policy_df(fend_df) 
        self.regular_choropleth(city, policy_df, 'policy', 'policy', 'prism', boundary_df, True)


if __name__ == "__main__":
    # Get arguments for score
    nds = sys.argv[1]  # nutrition desnsity score # ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA
    # All (>0), 30% (>3), 50% (>15), and 70% (>80) for sensitivity analysis
    threshold = sys.argv[2] # ex) 0, 30, 50, 70
    county_df = pd.read.csv(f"../data/files/FEND_{nds}.csv", dtype={'CountyFIPS':str})
    ct_df = pd.read.csv(f"../data/files/FEND_{nds}_ct.csv", dtype={'TractFIPS':str})
    # 50% thresholding (>15): main result 
    county_df_main = thresholding(threshold, county_df)
    # NYC FEND df and boundary outline
    nyc_ct_fend = pd.read_csv('../data/files/nyc_ct_fend.csv')
    nyc_ct = gpd.read_file('../data/files/nyct2020_23c/nyct2020.shp')
    nyc_ct = nyc_ct.to_crs({'init': 'EPSG:4326'}) # Init to epsg 4326
    nyc_ct = nyc_ct.rename(columns={"GEOID":"TractFIPS"})
    # LA FEND df and boundary outline
    la_ct_fend = pd.read_csv('../data/files/la_ct_fend.csv')
    la_ct = gpd.read_file('../data/files/LA_Census_Tracts_2020/Census_Tracts_2020.shp')
    la_ct = la_ct.to_crs({'init': 'EPSG:4326'}) # Init to epsg 4326
    la_ct['TractFIPS'] = '06037' + la_ct['CT20']
    la_ct = la_ct.drop(la_ct[la_ct['CT20'].isin(['599000','599100'])].index) # Drop Catalina Islands
    # Plot maps
    maps = GetMaps(df=county_df_main, ct_all=ct_df)
    maps.county_maps()
    maps.city_maps(fend_df=nyc_ct_fend, boundary_df=nyc_ct, city='NYC') # NYC
    maps.city_maps(fend_df=la_ct_fend, boundary_df=la_ct, city='LA') # LA
    maps.county_policy_map()
    maps.city_policy_map(fend_df=nyc_ct_fend, boundary_df=nyc_ct, city='NYC') # NYC
    maps.city_policy_map(fend_df=la_ct_fend, boundary_df=la_ct, city='LA') # LA
