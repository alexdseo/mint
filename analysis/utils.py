import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def thresholding(p, df_all):
    # Get the threshold based on pecentile
    threshold = round(df_all['restaurant_count'].quantile(p/100))
    df = df_all[df_all['restaurant_count']>=threshold]
    df = df.reset_index(drop=True)

    return df

def rename(df):
    df = df.rename(columns={"pp_lowAccess": "USDA %LowAccess", "lsr_density": "LSR Density", "mrfei":"CDC mRFEI",
                            "num_limited_service":"#Limited-Service", "restaurant_count": "#Restaurants",
                            "pp_black":"%Black", "pp_white":"%White", "pp_asian":"%Asian", "pp_hispanic":"%Hispanic",
                            "median_age":"Median Age", "median_income":"Median Income",
                            "employment_rate":"Employment Rate", "gini_index":"Gini Index",
                            "pp_publicTP":"%PublicTransportation", "pp_longcomute":"%LongComute",
                            "pp_lowskillJob":"%LowSkillJob", "pp_collegeEd":"%CollegeEdu",
                            "log_total_pop":"Total Population (log-scaled)",
                            "OBESITY_AdjPrev":"Obesity Prevalence", "DIABETES_AdjPrev":"Diabetes Prevalence",
                            "CHD_AdjPrev":"CHD Prevalence"})

    return df

def metro_rural_ct(df, ruca):
        # Renam columns
        ruca.columns = ['CountyFIPS','State','CountyName','TractFIPS','ruca']
        # Get metro code (1-3), rural coode (7-10)
        metro_code, rural_code = list(range(1,4)), list(range(7,11))
        metro_ruca = ruca[ruca['ruca'].isin(metro_code)]
        rural_ruca = ruca[ruca['ruca'].isin(rural_code)]
        # Filter census tracts
        metro_ct = df[df['TractFIPS'].isin(metro_ruca['TractFIPS'])].reset_index(drop=True)
        rural_ct = df[df['TractFIPS'].isin(rural_ruca['TractFIPS'])].reset_index(drop=True)

        return metro_ct, rural_ct

def normalize(df, features):
    scaler = StandardScaler()
    normalized_df = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

    return normalized_df

def remove_na(df, metric):
    # When performing analysis on 'USDA %LowAccess' or 'CDC mRFEI' on census tract-level and city-level
    df = df[df[metric].notna()].reset_index(drop=True)

    return df

def linear_model(df, target, variables):
    # Independent variables
    X = df[variables]
    X = sm.add_constant(X)
    # Target variablees
    y = df[target]
    # Model
    model = sm.OLS(y, X).fit()
    
    return model

def get_map_colors(metric):
    # Color schemes
    color_schemes = {
        'FEND': ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'],  # RdYlBu
        'RND_STDEV': ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],  # Plasma
        'CDC mRFEI': ['#00224e', '#304f7a', '#636675', '#a6977b', '#ffe345'],  # Cividis
        'USDA %LowAccess': ['#deebf7', '#9ecae1', '#6baed6', '#3182bd', '#08519c'],  # Blues
        'LSR Density': ['#fde725', '#90d743', '#35b779', '#31688e', '#440154'],  # Viridis_r
        '#Restaurant': ['#000004', '#3b0f70', '#8c2981', '#de636f', '#fcfdbf']  # Magma
    }
    # Default to an empty list if metric is not found
    colors = color_schemes.get(metric, [])  

    return colors