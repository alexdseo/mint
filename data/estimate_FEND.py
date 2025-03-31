import pandas as pd
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from census import Census
import sys

CENSUS_API_KEY = ''

def county_data(county_ho_df, county_boundary_df):
    """
    Merge health outcome data and geo data

    Args:
        county_ho_df: Community health outcome including diet-related diseases in county-level
        county_boundary_df: Shape boundary data for county-level

    Returns:
        county_ho_df: Merged df
    """
    # Structure boundary data
    county_boundary_df['CountyFIPS'] = county_boundary_df['STATEFP'] + county_boundary_df['COUNTYFP']
    county_boundary_df = county_boundary_df.to_crs({'init': 'EPSG:4326'})  
    # Merge
    county_ho_df = county_ho_df.merge(county_boundary_df, how='left', on=['CountyFIPS'])
    # To gpd
    county_ho_df = gpd.GeoDataFrame(county_ho_df, crs="EPSG:4326", geometry='geometry')

    return county_ho_df


def locate_df(RND_df, gpd_df):
    """
    Locate USA restaurants

    Args:
        RND_df: Restaurants with RND score
        gpd_df: USA county geo-pandas df 

    Returns:
        RND_df_located: All available restauratns in the USA counties with RND score; filtering out Cananda data
    """
    # Make RND df gpd
    valid_points = gpd.GeoDataFrame(RND_df, geometry=gpd.points_from_xy(RND_df.location_longitude, RND_df.location_latitude))
    #ct.crs
    valid_points.crs = 'EPSG:4326'
    # Join with envionment gpd to locate them
    RND_df_located = gpd.sjoin(valid_points, gpd_df, predicate = 'within')

    return RND_df_located


def estimate_FEND(RND_df, meal_type, county_ho, nds):
    """
    Estimate Food Enivronemnt Nutrient Density (FEND) on county-level using estimated RND score on ~600k restaurants in the US

    Args:
        RND_df: df with RND score
        meal_type: Meal type of appetizers, main dish, dessert, drink
        county_ho: county level health outcome data to merge with
        nds: nutrient density score

    Returns:
        FEND_df: df with FEND score
    """
    # FEND type
    metric = {
        'APP': f"RND_{nds}_APP",
        'MAIN': f"RND_{nds}_MAIN",
        'DSRT': f"RND_{nds}_DSRT",
        'DRNK': f"RND_{nds}_DRNK"
    }.get(meal_type, f"RND_{nds}")
    # FEND
    FEND_df = RND_df.groupby(['CountyFIPS']).median(numeric_only=True)[[metric,'lower_ci','upper_ci',
                                                                          'location_latitude','location_longitude']]
    FEND_df['CountyFIPS'] = FEND_df.index
    FEND_df = FEND_df.rename(columns={f"RND_{nds}":f"FEND_{nds}"}).reset_index(drop=True)
    # RND STDEV
    RND_stdev = RND_df.groupby(['CountyFIPS']).std(numeric_only=True)[[metric]]
    RND_stdev['CountyFIPS'] = RND_stdev.index
    RND_stdev = RND_stdev.rename(columns={metric:'RND_STDEV'}).reset_index(drop=True)
    RND_stdev['RND_STDEV'].fillna(0, inplace = True) 
    FEND_df = FEND_df.merge(RND_stdev[['CountyFIPS','RND_STDEV']], how='left', on=['CountyFIPS'])
    # restaurant count in the environment
    restaurant_count = RND_df.groupby(['CountyFIPS']).count()[[metric]]
    restaurant_count['CountyFIPS'] = restaurant_count.index
    restaurant_count = restaurant_count.rename(columns={metric:'restaurant_count'}).reset_index(drop=True)
    FEND_df = FEND_df.merge(restaurant_count[['CountyFIPS','restaurant_count']], how='left', on=['CountyFIPS'])
    # Most healthiest restaurant in the environemnt
    max_RND = RND_df.groupby(['CountyFIPS']).max(numeric_only=True)[[metric]]
    max_RND['CountyFIPS'] = max_RND.index
    max_RND = max_RND.rename(columns={metric:'max_RND'}).reset_index(drop=True)
    FEND_df = FEND_df.merge(max_RND[['CountyFIPS','max_RND']], how='left', on=['CountyFIPS'])
    # Least healthiest restaurant in the environemnt
    min_RND = RND_df.groupby(['CountyFIPS']).min(numeric_only=True)[[metric]]
    min_RND['CountyFIPS'] = min_RND.index
    min_RND = min_RND.rename(columns={metric:'min_RND'}).reset_index(drop=True)
    FEND_df = FEND_df.merge(min_RND[['CountyFIPS','min_RND']], how='left', on=['CountyFIPS'])

    # merge with gpd # Additional Diet-related diseases: 'HIGHCHOL_AdjPrev', 'BPHIGH_AdjPrev',
    FEND_df = FEND_df.merge(county_ho[['CountyFIPS', 'StateAbbr', 'CountyName', 'TotalPopulation',
                                       'OBESITY_AdjPrev', 'DIABETES_AdjPrev', 'CHD_AdjPrev',
                                       'geometry']], how='left', on=['CountyFIPS'])
    
    return FEND_df

def merge_ses(state_code, FEND_df):
    """
    Merge FEND dataset with Socioeconomice spatial differences (SES) factors retrieved from Census API

    Args:
        state_code: state FIPS code
        FEND_df: df with FEND score on counties

    Returns:
        FEND_df: Merged df
    """
    # Census API
    c = Census(f"{CENSUS_API_KEY}", year=2019)
    # Define list placeholder
    total_pop, pp_male, pp_female, pp_black, pp_white, pp_asian, pp_hispanic, median_income, pp_ptp, pp_comute45,\
          pp_lsjF, pp_lsjM, pp_ceF, pp_ceM, pp_emp, gini_index, median_age =[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    for i in tqdm(range(len(state_code)), desc='Getting SES variables...'):
        # Proportion of Black population
        population = c.acs5.state_county(('C02003_001E','C02003_003E','C02003_004E','C02003_006E','B01001I_001E',
                                        'B01001_002E','B01001_026E'),
                                        state_code[i][1], Census.ALL)
        for j in range(len(population)):
            if population[j]['C02003_001E'] == 0:
                proportion_black = 0
                proportion_white = 0
                proportion_asian = 0
                proportion_hispanic = 0
                proportion_male = 0
                proportion_female = 0
            else:
                proportion_black = population[j]['C02003_004E']/population[j]['C02003_001E']
                proportion_white = population[j]['C02003_003E']/population[j]['C02003_001E']
                proportion_asian = population[j]['C02003_006E']/population[j]['C02003_001E']
                proportion_hispanic = population[j]['B01001I_001E']/population[j]['C02003_001E']
                proportion_male = population[j]['B01001_002E']/population[j]['C02003_001E']
                proportion_female = population[j]['B01001_026E']/population[j]['C02003_001E']
            countyfips = population[j]['state'] + population[j]['county']
            # Save
            total_pop.append([countyfips, population[j]['C02003_001E']])
            pp_black.append([countyfips, proportion_black])
            pp_white.append([countyfips, proportion_white])
            pp_asian.append([countyfips, proportion_asian])
            pp_hispanic.append([countyfips, proportion_hispanic])
            pp_male.append([countyfips, proportion_male])
            pp_female.append([countyfips, proportion_female])
        # Median income of each household
        income = c.acs5.state_county(('B19013_001E'), state_code[i][1], Census.ALL)
        for j in range(len(income)):
            countyfips = income[j]['state'] + income[j]['county']
            median_income.append([countyfips, income[j]['B19013_001E']])
        # Proportion of people that use public transportation for that commuting
        transportation = c.acs5.state_county(('B08301_001E','B08301_010E'), state_code[i][1], Census.ALL)
        for j in range(len(transportation)):
            if transportation[j]['B08301_001E'] == 0:
                proportion_tp = 0
            else:
                proportion_tp = transportation[j]['B08301_010E']/transportation[j]['B08301_001E']
            countyfips = transportation[j]['state'] + transportation[j]['county'] 
            pp_ptp.append([countyfips, proportion_tp])
        # Proportion of people commuting longer than 45 minutes
        comute = c.acs5.state_county(('B08303_001E','B08303_011E','B08303_012E','B08303_013E'), state_code[i][1], Census.ALL)
        for j in range(len(comute)):
            if comute[j]['B08303_001E'] == 0:
                proportion_c45 = 0
            else:
                proportion_c45 = (comute[j]['B08303_011E']+comute[j]['B08303_012E']+comute[j]['B08303_013E'])/comute[j]['B08303_001E']
            countyfips = comute[j]['state'] + comute[j]['county']
            pp_comute45.append([countyfips, proportion_c45])
        # Proportion of people working in low-skill jobs - female
        lsj_f = c.acs5.state_county(('C24030_029E','C24030_030E','C24030_033E','C24030_034E','C24030_035E',
                                        'C24030_036E','C24030_037E'), state_code[i][1], Census.ALL)
        for j in range(len(lsj_f)):
            if lsj_f[j]['C24030_029E'] == 0:
                proportion_lsj_f = 0
            else:
                proportion_lsj_f = (lsj_f[j]['C24030_030E']+lsj_f[j]['C24030_033E']+lsj_f[j]['C24030_034E']+lsj_f[j]['C24030_035E']+lsj_f[j]['C24030_036E']+lsj_f[j]['C24030_037E'])/lsj_f[j]['C24030_029E']
            countyfips = lsj_f[j]['state'] + lsj_f[j]['county'] 
            pp_lsjF.append([countyfips, proportion_lsj_f])
        # Proportion of people working in low-skill jobs - male
        lsj_m = c.acs5.state_county(('C24030_002E','C24030_003E','C24030_006E','C24030_007E','C24030_008E',
                                        'C24030_009E','C24030_010E'), state_code[i][1], Census.ALL)
        for j in range(len(lsj_m)):
            if lsj_m[j]['C24030_002E'] == 0:
                proportion_lsj_m = 0
            else:
                proportion_lsj_m = (lsj_m[j]['C24030_003E']+lsj_m[j]['C24030_006E']+lsj_m[j]['C24030_007E']+lsj_m[j]['C24030_008E']+lsj_m[j]['C24030_009E']+lsj_m[j]['C24030_010E'])/lsj_m[j]['C24030_002E']
            countyfips = lsj_m[j]['state'] + lsj_m[j]['county'] 
            pp_lsjM.append([countyfips, proportion_lsj_m])
        # Proportion of people with a college education - female
        ce_f = c.acs5.state_county(('B15002_019E','B15002_031E','B15002_032E','B15002_033E','B15002_034E',
                                        'B15002_035E'), state_code[i][1], Census.ALL)
        for j in range(len(ce_f)):
            if ce_f[j]['B15002_019E'] == 0:
                proportion_ce_f = 0
            else:
                proportion_ce_f = (ce_f[j]['B15002_031E']+ce_f[j]['B15002_032E']+ce_f[j]['B15002_033E']+ce_f[j]['B15002_034E']+ce_f[j]['B15002_035E'])/ce_f[j]['B15002_019E']
            countyfips = ce_f[j]['state'] + ce_f[j]['county']
            pp_ceF.append([countyfips, proportion_ce_f])
        # Proportion of people with a college education - male
        ce_m = c.acs5.state_county(('B15002_002E','B15002_014E','B15002_015E','B15002_016E','B15002_017E',
                                        'B15002_018E'), state_code[i][1], Census.ALL)
        for j in range(len(ce_m)):
            if ce_m[j]['B15002_002E'] == 0:
                proportion_ce_m = 0
            else:
                proportion_ce_m = (ce_m[j]['B15002_014E']+ce_m[j]['B15002_015E']+ce_m[j]['B15002_016E']+ce_m[j]['B15002_017E']+ce_m[j]['B15002_018E'])/ce_m[j]['B15002_002E']
            countyfips = ce_m[j]['state'] + ce_m[j]['county']
            pp_ceM.append([countyfips, proportion_ce_m])
        # Proportion of employed (civilian population)
        employed = c.acs5.state_county(('B23025_003E','B23025_004E'), state_code[i][1], Census.ALL)
        for j in range(len(employed)):
            if employed[j]['B23025_003E'] == 0:
                proportion_employed = 0
            else:
                proportion_employed = employed[j]['B23025_004E']/employed[j]['B23025_003E']
            countyfips = employed[j]['state'] + employed[j]['county'] 
            pp_emp.append([countyfips, proportion_employed])
        # Gini Index
        gini = c.acs5.state_county(('B19083_001E'), state_code[i][1], Census.ALL)
        for j in range(len(gini)):
            countyfips = gini[j]['state'] + gini[j]['county']
            gini_index.append([countyfips, gini[j]['B19083_001E']])
        # Median Age
        age = c.acs5.state_county(('B01002_001E'), state_code[i][1], Census.ALL)
        for j in range(len(age)):
            countyfips = age[j]['state'] + age[j]['county']
            median_age.append([countyfips, age[j]['B01002_001E']])
    # to df
    total_pop = pd.DataFrame(total_pop, columns=['CountyFIPS','total_pop'])
    pp_black = pd.DataFrame(pp_black, columns=['CountyFIPS','pp_black'])
    pp_white = pd.DataFrame(pp_white, columns=['CountyFIPS','pp_white'])
    pp_asian = pd.DataFrame(pp_asian, columns=['CountyFIPS','pp_asian'])
    pp_hispanic = pd.DataFrame(pp_hispanic, columns=['CountyFIPS','pp_hispanic'])
    pp_male = pd.DataFrame(pp_male, columns=['CountyFIPS','pp_male'])
    pp_female = pd.DataFrame(pp_female, columns=['CountyFIPS','pp_female'])
    median_income = pd.DataFrame(median_income, columns=['CountyFIPS','median_income'])
    pp_ptp = pd.DataFrame(pp_ptp, columns=['CountyFIPS','pp_publicTP'])
    pp_comute45 = pd.DataFrame(pp_comute45, columns=['CountyFIPS','pp_longcomute'])
    pp_lsjF = pd.DataFrame(pp_lsjF, columns=['CountyFIPS','pp_lowskillJob_F'])
    pp_lsjM = pd.DataFrame(pp_lsjM, columns=['CountyFIPS','pp_lowskillJob_M'])
    pp_ceF = pd.DataFrame(pp_ceF, columns=['CountyFIPS','pp_collegeEd_F'])
    pp_ceM = pd.DataFrame(pp_ceM, columns=['CountyFIPS','pp_collegeEd_M'])
    pp_emp = pd.DataFrame(pp_emp, columns=['CountyFIPS','employment_rate'])
    gini_index = pd.DataFrame(gini_index, columns=['CountyFIPS','gini_index'])
    median_age = pd.DataFrame(median_age, columns=['CountyFIPS','median_age'])
    # Merge SES
    ses = total_pop
    for x in [pp_black, pp_white, pp_asian, pp_hispanic, pp_male, pp_female, median_income,
            pp_ptp, pp_comute45, pp_lsjF, pp_lsjM, pp_ceF, pp_ceM, pp_emp, gini_index, median_age]:
        ses = ses.merge(x, how='left', on=['CountyFIPS'])
    
    # Merge FEND and SES
    FEND_df = FEND_df.merge(ses, how='left', on=['CountyFIPS'])
    # Aggregate for gender divided variables
    FEND_df['pp_lowskillJob'] = (FEND_df['pp_lowskillJob_M'] + FEND_df['pp_lowskillJob_M']) / 2
    FEND_df['pp_collegeEd'] = (FEND_df['pp_collegeEd_M'] + FEND_df['pp_collegeEd_F']) / 2
    # log-scaled varibales
    FEND_df['log_total_pop'] = np.log(FEND_df['total_pop'])
    FEND_df['log_restaurant_count'] = np.log(FEND_df['restaurant_count'])

    return FEND_df

def check_fips(df):
    """
    Check for incomplete census tract fips, 11 numbers (2 state, 3 county, 6 tract)

    Args:
        df: df with tractfips

    Returns:
        df: checked df
    """
    # Check for incomplete census tract fips # If 10 numbers add 0 infront
    for i in tqdm(range(len(df)), desc='Checking FIPS...' ):
        if len(df['TractFIPS'][i])!=11:
            df['TractFIPS'][i] = '0' + df['TractFIPS'][i]
    
    return df

def get_lsr(df):
    """
    Retrieve information regarding limited service restaurants from County Business Patterns (CBP) data 

    Args:
        df: CBP data

    Returns:
        lsr_df: limited service restauratn data wtih County FIPS
    """
    # Retrieve metada and stats for limited service restaurant
    lsr_df = df[df['NAICS2017']=='722513']
    lsr_df = lsr_df[lsr_df['EMPSZES_LABEL']=='All establishments'][['NAME','ESTAB']]
    lsr_df = lsr_df.rename(columns={"ESTAB":"num_limited_service"})
    # Get GEO ID
    lsr_df = lsr_df.merge(df[['GEO_ID','NAME']][1:].drop_duplicates(), how='left', on=['NAME'])
    for i in range(len(lsr_df)):
        lsr_df['GEO_ID'][i] = lsr_df['GEO_ID'][i][-5:]
    lsr_df = lsr_df.rename(columns={'GEO_ID':'CountyFIPS'})

    return lsr_df


def merge_traditional_indicators(FEND_df, mrfei_df, lsr_df, lsr_df2, usda_df):
    """
    Merge FEND dataset with traditional indicators from multiple data sources

    Args:
        FEND_df: County level data with FEND score and SES factors
        mrfei_df: modified Food Environment Indicator (mRFEI) data from CDC
        lsr_df: Limited-service restaurant 2020v
        lsr_df2: Limited-service restaurant 2021v
        usda_df: Low Access data from USDA

    Returns:
        FEND_df: FEND df merged with traditional indicators variables
    """
    # mRFEI
    mrfei_df = mrfei_df.rename(columns={'fips':'TractFIPS'})
    mrfei_df = check_fips(mrfei_df)
    # Initialize count and state id
    mrfei_df['CountyFIPS'] = '00000'
    mrfei_df['state_id'] = '00'
    # Get county fips and state id from tractfips
    for i in range(len(mrfei_df)):
        mrfei_df['CountyFIPS'][i] = mrfei_df['TractFIPS'][i][:5]
        mrfei_df['state_id'][i] = mrfei_df['TractFIPS'][i][:2]
    # Drop NaN/empty values
    mrfei_nan_ind = list(mrfei_df[mrfei_df['mrfei']==' '].index.values) + list(mrfei_df[mrfei_df['mrfei'].isna()].index.values)
    mrfei_df = mrfei_df.drop(mrfei_nan_ind).reset_index(drop=True)
    # to numeric
    mrfei_df[['mrfei']] = mrfei_df[['mrfei']].apply(pd.to_numeric)
    # median for county-level mRFEI
    mrfei_county = mrfei_df.groupby(['CountyFIPS']).median(numeric_only=True)
    mrfei_county['CountyFIPS'] = mrfei_county.index
    mrfei_county = mrfei_county.reset_index(drop=True)
    # Merge mRFEI
    FEND_df = FEND_df.merge(mrfei_county, how='left', on=['CountyFIPS'])
    # Exclude counties with no stats
    FEND_df = FEND_df.drop(FEND_df[FEND_df['mrfei'].isna()].index.values).reset_index(drop=True)

    # LSR Density
    lsr_df_2020 = get_lsr(lsr_df)
    lsr_df_2021 = get_lsr(lsr_df2)
    # Merge Num LSR using 2020v
    FEND_df = FEND_df.merge(lsr_df_2020[['CountyFIPS','num_limited_service']], on='CountyFIPS', how='left')
    # Fill NaN values for Num LSR using 2021v
    FEND_df = FEND_df.merge(lsr_df_2021[['CountyFIPS', 'num_limited_service']], on='CountyFIPS', how='left', suffixes=('', '_tmp'))
    FEND_df['num_limited_service'] = FEND_df['num_limited_service'].fillna(FEND_df['num_limited_service_tmp'])
    # Drop temporary column
    FEND_df.drop(columns=['num_limited_service_tmp'], inplace=True)
    # Calculate Density LSR: Number of LSR per 1k population
    FEND_df['num_limited_service'] = pd.to_numeric(FEND_df['num_limited_service'], errors='coerce')
    FEND_df['lsr_density'] = FEND_df['num_limited_service'] *1000 / FEND_df['total_pop']
    # Exclude counties with no stats
    FEND_df = FEND_df.drop(FEND_df[FEND_df['lsr_density'].isna()].index.values).reset_index(drop=True)

    # USDA %LowAccess
    usda_df = usda_df.rename(columns={'CensusTract': 'TractFIPS'})
    usda_df = check_fips(usda_df)
    # Initialize county fips
    usda_df['CountyFIPS'] = '00000'
    # Get county fips
    for i in range(len(usda_df)):
        usda_df['CountyFIPS'][i] = usda_df['TractFIPS'][i][:5]
    # lapop1share: Share of tract population that are beyond 1 mile from supermarket
    usda_county = usda_df[['CountyFIPS', 'Pop2010','lapop1share']].groupby(['CountyFIPS']).sum(numeric_only=True)
    usda_county['CountyFIPS'] = usda_county.index
    usda_county = usda_county.reset_index(drop=True)
    # Get LowAccess stats
    # Proportion of population with low access
    usda_county['pp_lowAccess'] = usda_county['lapop1share'] / usda_county['Pop2010']
    # Merge USDA %LowAccess
    FEND_df = FEND_df.merge(usda_county[['CountyFIPS', 'lapop1share', 'pp_lowAccess']], how='left', on=['CountyFIPS'])
    # Exclude counties with no stats
    FEND_df = FEND_df.drop(FEND_df[FEND_df['pp_lowAccess'].isna()].index.values).reset_index(drop=True)
    
    return FEND_df


if __name__ == "__main__":
    # Get arguments for score
    nds = sys.argv[1]  # nutrition desnsity score # ex) RRR, NRF9.3, NRF6.3, LIM, WHO, FSA
    # CDC PLACES Community Health Outcomes # County-level # 2020 release
    county_ho = pd.read_csv('PLACES__County_Data__GIS_Friendly_Format___2020_release.csv', dtype={'CountyFIPS':str}) 
    # County boundaries
    county_boundary = gpd.read_file('./files/cb_2015_us_county_500k/cb_2015_us_county_500k.shp')
    county_df = county_data(county_ho, county_boundary)
    # Read RND data
    RND = pd.read_csv(f"./files/RND_{nds}.csv")
    RND_APP = pd.read_csv(f"./files/RND_{nds}_APP.csv")
    RND_MAIN = pd.read_csv(f"./files/RND_{nds}_MAIN.csv")
    RND_DSRT = pd.read_csv(f"./files/RND_{nds}_DSRT.csv")
    RND_DRNK = pd.read_csv(f"./files/RND_{nds}_DRNK.csv")
    # Locate df by joining the environment df with RND df
    RND, RND_APP, RND_MAIN, RND_DSRT, RND_DRNK = locate_df(RND, county_df), locate_df(RND_APP, county_df),\
          locate_df(RND_MAIN, county_df), locate_df(RND_DSRT, county_df), locate_df(RND_DRNK, county_df)
    # Estimate FEND
    FEND, FEND_APP, FEND_MAIN, FEND_DSRT, FEND_DRNK = estimate_FEND(RND, 'ALL', county_df, nds=nds),\
        estimate_FEND(RND_APP, 'APP', county_df, nds=nds), estimate_FEND(RND_MAIN, 'MAIN', county_df, nds=nds),\
        estimate_FEND(RND_DSRT, 'DSRT', county_df, nds=nds), estimate_FEND(RND_DRNK, 'DRNK', county_df, nds=nds)
    
    # Get state codes
    state_code = county_df[['StateAbbr', 'STATEFP']].drop_duplicates().values.tolist()

    # Merge additional variables
    # Socioeconomic and Spatial Differences (SES) # Race/Ethnicity, Demographics, and etc.
    FEND = merge_ses(state_code, FEND)
    # Traditional Indicators (CDC_mRFEI, Limited-Service Restaurant (LSR) Density, USDA_%LowAccess)
    mrfei = pd.read_excel(open('cdc_61367_DS2.xlsx', 'rb'), dtype={'fips':str})# mRFEI
    lsr_2020 = pd.read_csv('CBP2020.CB2000CBP-Data.csv', low_memory=False) # Number of LSR -2020v
    lsr_2021 = pd.read_csv('CBP2021.CB2100CBP-Data.csv',low_memory=False) # Number of LSR -2021v
    food_access = pd.read_excel(open('FoodAccessResearchAtlasData2019.xlsx', 'rb'), sheet_name='Food Access Research Atlas', dtype={'CensusTract':str}) # USDA_%LowAccess
    FEND = merge_traditional_indicators(FEND, mrfei, lsr_2020, lsr_2021, food_access)
    
    # Export FEND data as csv
    FEND.to_csv(f"./files/FEND_{nds}.csv", index=False)
    FEND_APP.to_csv(f"./files/FEND_{nds}_APP.csv", index=False)
    FEND_MAIN.to_csv(f"./files/FEND_{nds}_MAIN.csv", index=False)
    FEND_DSRT.to_csv(f"./files/FEND_{nds}_DSRT.csv", index=False)
    FEND_DRNK.to_csv(f"./files/FEND_{nds}_DRNK.csv", index=False)
