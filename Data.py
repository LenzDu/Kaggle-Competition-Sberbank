# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:36:14 2017

@author: vrtjso
"""
import numpy as np
import pandas as pd
from datetime import datetime, date
from operator import le, eq
from Utils import sample_vals, FeatureCombination
import gc
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

####Data Cleaning####
print('Data Cleaning...')

#Data importing
trainDf = pd.read_csv('train.csv').set_index('id')
testDf = pd.read_csv('test.csv').set_index('id')
fix = pd.read_excel('BAD_ADDRESS_FIX.xlsx').set_index('id')
testDf['isTrain'] = 0
trainDf['isTrain'] = 1
allDf = pd.concat([trainDf,testDf])
allDf.update(fix, filter_func = lambda x:np.array([True]*x.shape[0])) #update fix data
macro = pd.read_csv('macro.csv')

#Join division and macro
divisions = pd.read_csv('divisions.csv')
allDf = allDf.join(divisions.set_index('sub_area'), on='sub_area')
# macro = pd.read_csv('macro.csv')
# allDf = allDf.join(macro[['timestamp','macro_combined_index']].set_index('timestamp'), on='timestamp')

# macro = macro.loc[365:2343,:] #drop data before 2011 and after 2016.6
# macro_full = macro.loc[:,macro.count()==1979] # drop nan columns
# macro_missing = macro.loc[:2190,macro.count()==1826]
# allDf = allDf.join(macro_full.set_index('timestamp'), on='timestamp')
# FeatureCombination(macro_full.drop('timestamp',1),'',10)

# Drop variable with no use (actuallly they are useful :)
# allDf = allDf.drop(['16_29_male','cafe_count_5000_price_1500','market_count_1000',
#       '0_6_male','young_male','build_count_before_1920','market_count_1500',
#       'trc_count_500','church_count_3000','cafe_count_2000_na_price',
#       'mosque_count_3000','leisure_count_2000','build_count_slag',
#       "oil_chemistry_raion","railroad_terminal_raion","mosque_count_500",
#       "nuclear_reactor_raion", "build_count_foam", "big_road1_1line",
#       "trc_sqm_500", "cafe_count_500_price_high","mosque_count_1000", "mosque_count_1500"],1)
# Drop no use macro
# allDf = allDf.drop(["real_dispos_income_per_cap_growth","profitable_enterpr_share",
#          "unprofitable_enterpr_share","share_own_revenues","overdue_wages_per_cap",
#          "fin_res_per_cap","marriages_per_1000_cap","divorce_rate","construction_value",
#          "invest_fixed_assets_phys","pop_migration","pop_total_inc","housing_fund_sqm",
#          "lodging_sqm_per_cap","water_pipes_share","baths_share","sewerage_share","gas_share",
#          "hot_water_share","electric_stove_share","heating_share","old_house_share",
#          "infant_mortarity_per_1000_cap", "perinatal_mort_per_1000_cap", "incidence_population",
#          "load_of_teachers_preschool_per_teacher","provision_doctors","power_clinics","hospital_beds_available_per_cap",
#          "hospital_bed_occupancy_per_year","provision_retail_space_sqm","provision_retail_space_sqm",
#          "theaters_viewers_per_1000_cap","museum_visitis_per_100_cap","population_reg_sports_share",
#          "students_reg_sports_share","apartment_build",
#          'gdp_annual_growth','old_education_build_share','provision_nurse','employment', #这行开始是importance为0的feature
#          'apartment_fund_sqm','invest_fixed_capital_per_cap'],1)

### Change price by rate ###
allDf['timestamp'] = pd.to_datetime(allDf['timestamp'])
# price_q_rate = [0,1.1,1,2.36,7.6,2.79,2.79,2.77,-1.68,1.04,.44,.41,-.98,1.26,.86,1.69,1.12,-.68,-1.85,-1.66,-1.69,-.097]
# price_rate = [1]
# for i in range(1,len(price_q_rate)):
#     price_rate.append(price_rate[i-1] * (1 + price_q_rate[i] * 0.01))
# year_quarter = np.array((allDf.timestamp.dt.year - 2011) * 4 + allDf.timestamp.dt.quarter - 1)
# p = np.ones(allDf.shape[0])
# for i in range(0,allDf.shape[0]):
#     p[i] = price_rate[year_quarter[i]]
# allDf['price_rate'] = p
# allDf['price_doc'] = allDf.price_doc / allDf.price_rate
# time = np.array([])
# for i in allDf.index:
#     time = np.append(time, datetime.strptime(allDf['timestamp'][i], '%Y-%m-%d').timestamp())
# allDf['time'] = time
# allDf.drop('timestamp', 1, inplace=True)

allDf['apartment_name'] = allDf.sub_area + allDf['metro_km_avto'].astype(str)
eco_map = {'excellent':4, 'good':3, 'satisfactory':2, 'poor':1, 'no data':0}
allDf['ecology'] = allDf['ecology'].map(eco_map)
#encode subarea in order
# price_by_area = allDf['price_doc'].groupby(allDf.sub_area).mean().sort_values()
# area_dict = {}
# for i in range(0,price_by_area.shape[0]):
#    area_dict[price_by_area.index[i]] = i
# allDf['sub_area'] = allDf['sub_area'].map(area_dict)
for c in allDf.columns:
    if allDf[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(allDf[c].values))
        allDf[c] = lbl.transform(list(allDf[c].values))

# PCA on area feature
# area_feature = []
# for i in allDf.columns:
#    if allDf[i].groupby(allDf.sub_area).var().mean()==0 and i != 'sub_area':
#        area_feature.append(i)
# areaDf = allDf[area_feature]
# nonareaDf = allDf.drop(area_feature,1)
# areaDf = FeatureCombination(areaDf,'',10)
# allDf = pd.concat([nonareaDf,areaDf],1)

# allDf = FeatureCombination(allDf,'cafe_count',7)
#FeatureCombination(allDf,'sport_count',5)
#FeatureCombination(allDf,'market_count',3)
#FeatureCombination(allDf,'leisure_count',5)
#FeatureCombination(allDf,'church_count',5)
#FeatureCombination(allDf,'big_church_count',5)
#FeatureCombination(allDf,'trc_count',5)
#FeatureCombination(allDf,'office_sqm',5)
#FeatureCombination(allDf,'trc_sqm',3)
#FeatureCombination(allDf,'railroad_station',2)
#FeatureCombination(allDf,'metro',2)

#Transform price to log price
#allDf['log_price'] = np.log1p(allDf.price_doc)
#Drop all training samples with strange price.
#allDf = allDf[~((allDf.price_doc==1000000) & (allDf.product_type_Investment==1))]
#allDf = allDf[~((allDf.price_doc==2000000) & (allDf.product_type_Investment==1))]

#allDf.ix[allDf.price_doc==2000000,'w'] = 0.7
#Undersample strange price
#            allDf = sample_vals(allDf, 1000000, 1/8, le)
#            allDf = sample_vals(allDf, 2000000, 1/4, eq)
#            allDf = sample_vals(allDf, 3000000, 1/2, eq)
#allDf = allDf.reset_index(drop=True)
#allDf.drop('price_doc',1,inplace=True)

###Dealing with Outlier###
allDf.loc[allDf.full_sq>2000,'full_sq'] = np.nan
allDf.loc[allDf.full_sq<3,'full_sq'] = np.nan
allDf.loc[allDf.life_sq>500,'life_sq'] = np.nan
allDf.loc[allDf.life_sq<3,'life_sq'] = np.nan
# allDf['lifesq_to_fullsq'] = 0 # 0 for normal, 1 for close,2 for outlier
allDf.loc[allDf.life_sq>0.8*allDf.full_sq,'life_sq'] = np.nan
# allDf.ix[allDf.life_sq>allDf.full_sq,['life_sq','lifesq_to_fullsq']] = np.nan, 2
allDf.loc[allDf.kitch_sq>=allDf.life_sq,'kitch_sq'] = np.nan
allDf.loc[allDf.kitch_sq>500,'kitch_sq'] = np.nan
allDf.loc[allDf.kitch_sq<2,'kitch_sq'] = np.nan
allDf.loc[allDf.state>30,'state'] = np.nan
allDf.loc[allDf.build_year<1800,'build_year'] = np.nan
allDf.loc[allDf.build_year==20052009,'build_year'] = 2005
allDf.loc[allDf.build_year==4965,'build_year'] = np.nan
allDf.loc[allDf.build_year>2021,'build_year'] = np.nan
allDf.loc[allDf.num_room>15,'num_room'] = np.nan
allDf.loc[allDf.num_room==0,'num_room'] = np.nan
allDf.loc[allDf.floor==0,'floor'] = np.nan
allDf.loc[allDf.max_floor==0,'max_floor'] = np.nan
allDf.loc[allDf.floor>allDf.max_floor,'max_floor'] = np.nan
#allDf.ix[allDf.full_sq>300,'full_sq'] = np.nan
#allDf.ix[allDf.life_sq>250,'life_sq'] = np.nan

# brings error down a lot by removing extreme price per sqm
bad_index = allDf[allDf.price_doc/allDf.full_sq > 600000].index
bad_index = bad_index.append(allDf[allDf.price_doc/allDf.full_sq < 10000].index)
allDf.drop(bad_index,0,inplace=True)

####Feature Engineering####
print('Feature Engineering...')
gc.collect()

##Time
# isWeekend = []
# month = []
# year = []
# weekday = []
# week_of_year = []
# year_month = []
# for i in allDf.index:
#     dateS = date.fromtimestamp(allDf.time[i]) #timestamp
#     isWeekend.append(1 if dateS.isoweekday() == 6 or dateS.isoweekday() == 7 else 0)
#     month.append(dateS.month)
#     year.append(dateS.year)
#     year_month.append(dateS.year*100 + dateS.month)
#     weekday.append(dateS.weekday())
#     week_of_year.append(dateS.isocalendar()[1])
##allDf['is_weekend'] = pd.Series(isWeekend) #seems to be of no use
# allDf['month'] = np.array(month)
allDf['year'] = allDf.timestamp.dt.year  #may be no use because test data is out of range
allDf['weekday'] = allDf.timestamp.dt.weekday

#allDf['week_of_year'] = np.array(week_of_year)
##allDf['year_month'] = np.array(year_month)

#w_map = {2011:0.8, 2012:0.8, 2013:0.9, 2014:1, 2015:1, 2016:0}
#allDf['w'] = [w_map[i] for i in year]

# Assign weight
allDf['w'] = 1
allDf.loc[allDf.price_doc==1000000,'w'] *= 0.5
allDf.loc[allDf.year==2015,'w'] *= 1.5

#May lead to overfitting
#Change timestamp to accumulated days.
#accum_day = np.array([])
#day0 = date(2011,8,20)
#for i in range(0,allDf.shape[0]):
#    accum_day = np.append(accum_day, (date.fromtimestamp(allDf.time[allDf.index[i]]) - day0).days)
#allDf['accum_day'] = pd.Series(accum_day) #试试把时间去掉

# Sale count
# mon_to_sale = allDf.groupby('month')['month'].count().to_dict()
# allDf['sale_cnt_mon'] = allDf['month'].map(mon_to_sale)
# week_to_sale = allDf.groupby('week_of_year')['week_of_year'].count().to_dict()
# allDf['sale_cnt_week'] = allDf['week_of_year'].map(week_to_sale)
# allDf = allDf.drop('week_of_year',1)
# allDf = allDf.drop('month',1)
# weekday_to_sale = allDf.groupby('weekday')['weekday'].count().to_dict()
# allDf['sale_cnt_weekday'] = allDf['weekday'].map(weekday_to_sale)
# area_to_sale = allDf.groupby('sub_area')['sub_area'].count().to_dict()
# allDf['sale_cnt_area'] = allDf['sub_area'].map(area_to_sale)
# OKRUGS_to_sale = allDf.groupby('OKRUGS')['OKRUGS'].count().to_dict()
# allDf['sale_cnt_OKRUGS'] = allDf['OKRUGS'].map(OKRUGS_to_sale)
# allDf['year_month'] = (allDf.timestamp.dt.year - 2011) * 12 + allDf.timestamp.dt.month
# year_mon_to_sale = allDf.groupby('year_month')['year_month'].count().to_dict()
# allDf['sale_cnt_year_mon'] = allDf['year_month'].map(year_mon_to_sale)
# allDf.drop('year_month',1,inplace=True)

#Location
#center_OKRUGS_lon = allDf.groupby('OKRUGS')['lon'].mean().to_dict()
#center_OKRUGS_lat = allDf.groupby('OKRUGS')['lat'].mean().to_dict()
#allDf['dist_to_OKRUGS_center'] = np.sqrt((allDf['lon'] - allDf['OKRUGS'].map(center_OKRUGS_lon)) ** 2 +
#                                         (allDf['lat'] - allDf['OKRUGS'].map(center_OKRUGS_lat)) ** 2)

#Floor
allDf['floor_by_max_floor'] = allDf.floor / allDf.max_floor
#allDf['floor_to_top'] = allDf.max_floor - allDf.floor

#Room
allDf['avg_room_size'] = (allDf.life_sq - allDf.kitch_sq) / allDf.num_room
allDf['life_sq_prop'] = allDf.life_sq / allDf.full_sq
allDf['kitch_sq_prop'] = allDf.kitch_sq / allDf.full_sq

#Calculate age of building
allDf['build_age'] = allDf.year - allDf.build_year
allDf = allDf.drop('build_year', 1)

#Population
allDf['popu_den'] = allDf.raion_popul / allDf.area_m
allDf['gender_rate'] = allDf.male_f / allDf.female_f
allDf['working_rate'] = allDf.work_all / allDf.full_all

#Education
allDf.loc[allDf.preschool_quota==0,'preschool_quota'] = np.nan
allDf['preschool_ratio'] =  allDf.children_preschool / allDf.preschool_quota
allDf['school_ratio'] = allDf.children_school / allDf.school_quota

## Group statistics
# avg_yearbuilt_area = allDf.groupby('sub_area')['build_age'].mean().to_dict()
# allDf['avg_yearbuilt_area'] = allDf['sub_area'].map(avg_yearbuilt_area)
# avg_yearbuilt_OKRUGS = allDf.groupby('OKRUGS')['build_age'].mean().to_dict()
# allDf['avg_yearbuilt_OKRUGS'] = allDf['OKRUGS'].map(avg_yearbuilt_OKRUGS)

# Mathematical features
# polyf = ['full_sq','build_age','life_sq','floor','max_floor','num_room']
# for i in range(0,len(polyf)):
#     for j in range(i,len(polyf)):
#         allDf[polyf[i]+'*'+polyf[j]] = allDf[polyf[i]] * allDf[polyf[j]]
allDf['square_full_sq'] = (allDf.full_sq - allDf.full_sq.mean()) ** 2
allDf['square_build_age'] = (allDf.build_age - allDf.build_age.mean()) ** 2
allDf['nan_count'] = allDf[['full_sq','build_age','life_sq','floor','max_floor','num_room']].isnull().sum(axis=1)
allDf['full*maxfloor'] = allDf.max_floor * allDf.full_sq
allDf['full*floor'] = allDf.floor * allDf.full_sq

allDf['full/age'] = allDf.full_sq / (allDf.build_age + 0.5)
allDf['age*state'] = allDf.build_age * allDf.state

# new trial
allDf['main_road_diff'] = allDf['big_road2_km'] - allDf['big_road1_km']
allDf['rate_metro_km'] = allDf['metro_km_walk'] / allDf['ID_metro'].map(allDf.metro_km_walk.groupby(allDf.ID_metro).mean().to_dict())
allDf['rate_road1_km'] = allDf['big_road1_km'] / allDf['ID_big_road1'].map(allDf.big_road1_km.groupby(allDf.ID_big_road1).mean().to_dict())
# best on LB with weekday

allDf['rate_road2_km'] = allDf['big_road2_km'] / allDf['ID_big_road2'].map(allDf.big_road2_km.groupby(allDf.ID_big_road2).mean().to_dict())
allDf['rate_railroad_km'] = allDf['railroad_station_walk_km'] / allDf['ID_railroad_station_walk'].map(allDf.railroad_station_walk_km.groupby(allDf.ID_railroad_station_walk).mean().to_dict())
# increase CV from 2.35 to 2.33 but lower LB a little bit (with month)

# allDf['additional_edu_index'] = allDf.additional_education_km / allDf.additional_education_raion
# allDf['rate_edu_km'] = (allDf['additional_education_km']
#                        / allDf['sub_area'].map(allDf.additional_education_km.groupby(allDf.sub_area).mean().to_dict())) / (allDf.additional_education_raion+0.5)
# allDf['num_house_metro'] = allDf['ID_metro'].map(allDf['full_sq'].groupby(allDf.ID_metro).count().to_dict())
# allDf['num_house_road'] = allDf['ID_big_road1'].map(allDf['full_sq'].groupby(allDf.ID_big_road1).count().to_dict())
# do not improve both CV and LB

allDf.drop(['year','timestamp'], 1, inplace = True)

#Separate train and test again
trainDf = allDf[allDf.isTrain==1].drop(['isTrain'],1)
testDf = allDf[allDf.isTrain==0].drop(['isTrain','price_doc', 'w'],1)

outputFile = 'train_featured.csv'
trainDf.to_csv(outputFile,index=False)
outputFile = 'test_featured.csv'
testDf.to_csv(outputFile,index=False)

# Xgboost handles nan itself
'''
### Dealing with NA ###
#num_room, filled by linear regression of full_sq
if filename == 'train_encoded.csv': #na in num_room only appear in training set
    LR = LinearRegression()
    X = allDf.full_sq[~(np.isnan(allDf.num_room) | np.isnan(allDf.full_sq))].values.reshape(-1, 1)
    y = np.array(allDf.num_room[~(np.isnan(allDf.num_room) | np.isnan(allDf.full_sq))])
    LR.fit(X,y)
    newX = allDf.full_sq[np.isnan(allDf.num_room)].values.reshape(-1, 1)
    newX[np.isnan(newX)] = newX[~np.isnan(newX)].mean() #Special cases (na in full_sq) in test data
    yfit = LR.predict(newX)
    allDf.ix[np.isnan(allDf.num_room),'num_room'] = yfit
#max_floor, twice as the floor
allDf.ix[np.isnan(allDf.max_floor),'max_floor'] = allDf.ix[np.isnan(allDf.max_floor),'floor'] * 2
'''



