import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols

#null hypothese->The parameters are not related
#If p value > level of significance -> accept the null hypothesis
#if p value < level of significance -> reject the null hypotesis
#level of significance -> 0.05
#The null hypothesis is a characteristic arithmetic theory suggesting that no statistical relationship and significance exists in a set of given, single, observed variables between two sets of observed data and measured phenomena. 


# # Cleaning the Candidate Detals CSV File----------------------------------------------------------------------------------------------------
# missing_values = ['NA','NAN','','Not Available']
# data = pd.read_csv('../Data_Analytics_Project-2/lok-sabha-candidate-details-2019.csv',na_values=missing_values)
# data.dropna()
# data = data.fillna(0)
# data.to_csv('candidates_details.csv')


# To remove skewness and normalise the parameters---------------------------------------------------------------------------------------------
# def normalize(column):
#     upper = column.max()
#     lower = column.min()
#     y = (column - lower)/(upper-lower)
#     return y
#x = np.log(x+1)
#-------------------------------------------------------------------------------------------------------------------------------------------

raw_data = pd.read_csv('candidates_details.csv') #cleaning Education Column
raw_data["EDUCATION"] = raw_data['EDUCATION'].replace(['0','Post Graduate\n'],['Illiterate','Post Graduate'])

def conversion_money(x):
    try:
        temp1 = (x.split('Rs')[1].split('\n')[0].strip())
        temp2 = ''
        for i in temp1.split(","):
            temp2 = temp2+i
        return int(temp2)
    except:
        return 0
raw_data['ASSETS'] = raw_data['ASSETS'].apply(conversion_money) #cleaning assets column
raw_data['LIABILITIES'] = raw_data['LIABILITIES'].apply(conversion_money) #cleaning liabilities column
#-------------------------------------------------------------------------------------------------------------------------------------------


# #ASSETS Wise-------------------------------------------------------------------------------------------------------------------------------
crore = raw_data.loc[(raw_data['ASSETS']>=10000000)]['ASSETS'].count()
lakh = raw_data.loc[(raw_data['ASSETS']>=100000)&(raw_data['ASSETS']<10000000)]['ASSETS'].count()
thousand = raw_data.loc[(raw_data['ASSETS']>1000)&(raw_data['ASSETS']<100000)]['ASSETS'].count()
fig15 = plt.figure(15)
ax15 = fig15.add_subplot(1,1,1)
ax15.pie([crore,lakh,thousand],labels=['Crore+','Lakh+','Thousand+'],autopct='%0.2f%%')
ax15.set_title('Candidates percentage Assets wise')
#--------------------------------------------------------------------------------------------------------------------------------------------

# #Liability Wise-----------------------------------------------------------------------------------------------------------------------------
crore = raw_data.loc[(raw_data['LIABILITIES']>=10000000)]['LIABILITIES'].count()
lakh = raw_data.loc[(raw_data['LIABILITIES']>=100000)&(raw_data['LIABILITIES']<10000000)]['LIABILITIES'].count()
thousand = raw_data.loc[(raw_data['LIABILITIES']>1000)&(raw_data['LIABILITIES']<100000)]['LIABILITIES'].count()
fig16 = plt.figure(16)
ax16= fig16.add_subplot(1,1,1)
ax16.pie([crore,lakh,thousand],labels=['Crore+','Lakh+','Thousand+'],autopct='%0.2f%%')
ax16.set_title('Candidates percentage Liabilites wise')
#---------------------------------------------------------------------------------------------------------------------------------------------

# #Party Wise Percentage--------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[raw_data['PARTY']!='NOTA']
x = data['PARTY'].value_counts()
y = x.head()
z = pd.Series([sum(x[6:])],index=['Others'])
y = y.append(z)
# print(y)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1,1,1)
ax1.pie(y,labels=y.index,autopct='%0.2f%%')
ax1.set_title("Election 2019 participation proportion party wise")
# # plt.show()

# #Party Wins
x1 = raw_data[raw_data["WINNER"]==1]['PARTY'].value_counts()
x1 = x1.rename('WINS')
x2 = raw_data[raw_data["WINNER"]==0]['PARTY'].value_counts()
x2 = x2.rename('LOSS')
data = pd.concat([x1,x2],axis=1).reset_index()
data = data.fillna(0)
data = data.rename(columns={'index':'PARTY'})
y = data.head(20)
add_win = data['WINS'][20:].sum()
add_loss = data['LOSS'][20:].sum()
temp_df = {'PARTY':'OTHERS','WINS':add_win,'LOSS':add_loss}
df = pd.DataFrame(temp_df, index=[0])
y = y.append(df,ignore_index=True)
# print(y)
ind = np.arange(21)
y1 = y['WINS']
y2 = y['LOSS']
fig6 = plt.figure(6)
ax6 = fig6.add_subplot(1,1,1)
ax6.bar(ind, y1, label='WINS')
ax6.bar(ind, y2,  bottom=y1, label='LOSS')
ax6.set_xticks(ind)
ax6.set_xticklabels(y["PARTY"],rotation=45)
ax6.legend()
ax6.set_xlabel('Party')
ax6.set_ylabel('Count')
ax6.set_title('Party Win Loss Count')
# # plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------

# #Based on Age-----------------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[raw_data['PARTY']!='NOTA']['AGE']
data1 = data.value_counts()
data1.sort_index(ascending=True,inplace=True)
# print(data1)
fig7 = plt.figure(7)
ax7 = fig7.add_subplot(1,1,1)
ax7.hist(data,bins=[20,25,30,35,40,45,50,55,60,65,70,75,80,85,90], edgecolor='black')
ax7.set_xticks([20,25,30,35,40,45,50,55,60,65,70,75,80,85,90])
ax7.set_ylabel('Count')
ax7.set_xlabel('Age Range')
ax7.set_title('Age Distribution')
# plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------

# #Gender Wise-------------------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[raw_data['GENDER']!='0']
x = data['GENDER'].value_counts()
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1,1,1)
ax2.pie(x,labels=x.index,autopct='%0.2f%%')
ax2.set_title("Gender proportion in Election 2019")
# plt.show()

# # Gender Wise Winners
# # data = raw_data.loc[raw_data["PARTY"]!='NOTA']['GENDER'].value_counts()
data1 = raw_data.loc[raw_data["PARTY"]!='NOTA'].groupby('WINNER')['GENDER'].value_counts()
print(data1)
y1 = [data1[(0,'FEMALE')]+data1[(1,'FEMALE')],data1[(0,'MALE')]+data1[(1,'MALE')]]
y2 = [data1[(1,'FEMALE')],data1[(1,'MALE')]]
fig12 = plt.figure(12)
ax12 = fig12.add_subplot(1,1,1)
ind = np.arange(2)
width=0.35
rect1 = ax12.bar(ind, y1, width, color='royalblue')
rect2 = ax12.bar(ind+width, y2, width, color='seagreen',)
ax12.set_xticks(ind + width / 2)
ax12.set_xticklabels(('FEMALE','MALE'))
ax12.legend((rect1[0], rect2[0]),('Overall', 'Winning'))
ax12.set_title("Participation vs Win count Gender Wise")
# # plt.show()

# #ANOVA GENDER
x1 = raw_data.loc[raw_data["PARTY"]!='NOTA']["GENDER"]
# print(x1.value_counts())
x2 = raw_data.loc[raw_data["PARTY"]!='NOTA']["WINNER"]
data = pd.concat([x1, x2], axis=1)
mod = ols('WINNER~GENDER',data=data).fit()
aov = sm.stats.anova_lm(mod,type=2)
print(aov)
# #null hypothesis -> Gender is not related to Winning
# #alternate hypotheses -> Gender plays a important role in winning

# # x = raw_data[raw_data['PARTY']!='NOTA'].groupby('PARTY')['GENDER'].value_counts()
# # x.sort_values(ascending=False,inplace=True)
# # print(x)
#---------------------------------------------------------------------------------------------------------------------------------------------

# #Crimal Cases wise---------------------------------------------------------------------------------------------------------------------------
x1 = raw_data.loc[pd.Series(map(int,raw_data['CRIMINALCASES']))>0]['CRIMINALCASES'].count() #count where criminal cases are >0
x2 = raw_data.loc[raw_data['PARTY']!='NOTA'][pd.Series(map(int,raw_data['CRIMINALCASES']))==0]['CRIMINALCASES'].count()#count where criminal case  is 0
y = [x1,x2]
x = ['Canidates with Criminal Cases','Candidates without Criminal Cases']
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(1,1,1)
ax3.pie(y,labels=x,autopct='%0.2f%%')
ax3.set_title('Proportion of Candidates having criminal charges against them')
# plt.show()

#Criminal Cases party wise
data = raw_data.loc[raw_data['PARTY']!='NOTA'].groupby('PARTY')['CRIMINALCASES'].sum()
data.sort_values(ascending=False,inplace=True)
y = data.head()
z = pd.Series([sum(data[6:])],index=['Others'])
y = y.append(z)
fig9 = plt.figure(9)
ax9 = fig9.add_subplot(1,1,1)
ax9.bar(y.index,y)
ax9.set_ylabel('Cases')
ax9.set_xlabel('Party')
ax9.set_title('Parties with number of candidates having criminal cases against them')
# plt.show()

# #wininig Party criminal Cases
data = raw_data.loc[raw_data["PARTY"]!='NOTA'][raw_data['WINNER']==1]
data = data.groupby('PARTY')['CRIMINALCASES'].sum()
data.sort_values(ascending=False,inplace=True)
y = data.head()
z = pd.Series([sum(data[6:])],index=['Others'])
y = y.append(z)
fig10 = plt.figure()
ax10 = fig10.add_subplot(1,1,1)
ax10.bar(y.index,y)
ax10.set_ylabel('Cases')
ax10.set_xlabel('Party')
ax10.set_title('Winning Parties with number of criminal cases against them')
# plt.show()

#Criminal vs non criminal
# data = raw_data.loc[raw_data["PARTY"]!='NOTA']['WINNER'].value_counts()
data1 = raw_data.loc[pd.Series(map(int,raw_data['CRIMINALCASES']))>0].groupby('WINNER')['CRIMINALCASES'].count()
print(data1)
data2 = raw_data.loc[raw_data['PARTY']!='NOTA'][pd.Series(map(int,raw_data['CRIMINALCASES']))==0].groupby('WINNER')['CRIMINALCASES'].count()
print(data2)
ind = np.arange(2)
y1 = [data1[0],data2[0]]
y2 = [data1[1],data2[1]]
fig11 = plt.figure(11)
ax11 = fig11.add_subplot(1,1,1)
ax11.bar(ind, y1, label='Loss')
ax11.bar(ind, y2,  bottom=y1, label='Win')
ax11.set_xticks(ind)
ax11.set_xticklabels(('With Criminal Cases','Without Criminal Cases'))
ax11.set_xlabel('Candidate')
ax11.set_ylabel('Count')
ax11.set_title('Candidates with number of crimnal cases and wins & losses')
ax11.legend()
# plt.show()

# #Anova Criminal
def conversion(value):
    if(int(value)>0):
        return 'yes'
    else:
        return 'no'
x1 = raw_data.loc[raw_data["PARTY"]!='NOTA']["WINNER"]
x2 = raw_data.loc[raw_data["PARTY"]!='NOTA']["CRIMINALCASES"].apply(conversion)
data = pd.concat([x1, x2], axis=1)
mod = ols('WINNER~CRIMINALCASES',data=data).fit()
aov = sm.stats.anova_lm(mod,type=2)
print(aov)
# #null hypothesis -> Criminals does not play an important role in election
# #alternate hypothesis -> Criminals does play an important role in election
#----------------------------------------------------------------------------------------------------------------------------------------------

# #Category Wise-------------------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[raw_data['CATEGORY']!='0']['CATEGORY'].value_counts()
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(1,1,1)
ax4.pie(data,labels=data.index,autopct='%0.2f%%')
ax4.set_title('Percentages of candidates category wise')
# plt.show()

#Category & Wins
data = raw_data.loc[raw_data['CATEGORY']!='0']['CATEGORY'].value_counts()
data2 = raw_data.loc[raw_data['PARTY']!='NOTA'][raw_data['WINNER']==1]
data2 = data2['CATEGORY'].value_counts()
# print(data)
# print(data2)
fig8 = plt.figure(8)
ax8 = fig8.add_subplot(1,1,1)
ind = np.arange(3)
width=0.35
rect1 = ax8.bar(ind, data, width, color='royalblue')
rect2 = ax8.bar(ind+width, data2, width, color='seagreen',)
ax8.set_xticks(ind + width / 2)
ax8.set_xticklabels(('General', 'SC', 'ST'))
ax8.legend((rect1[0], rect2[0]),('Overall Category Count', 'Winning Category Count'))
ax8.set_ylabel('Count')
ax8.set_xlabel('Category')
ax8.set_title('Category wise winnings')
# plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------

# #State---------------------------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[pd.Series(map(int,raw_data['CRIMINALCASES']))>0].groupby('STATE')['CRIMINALCASES'].count()
data.sort_values(ascending=False,inplace=True)
# print(data)
y = data.head()
fig13 = plt.figure()
ax13 = fig13.add_subplot(1,1,1)
ax13.bar(y.index,y)
# plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------

# #Education Wise-----------------------------------------------------------------------------------------------------------------------------
data = raw_data.loc[raw_data['EDUCATION']!='0']['EDUCATION'].value_counts()
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(1,1,1)
ax5.pie(data,labels=data.index,autopct='%0.2f%%',explode =[0.1,0.1, 0.1, 0.1, 0.1,0.2, 0.2, 0.2,0.2, 0.2, 0.2])
ax5.set_title('Porportion of different education levels of candidates')
# plt.show()

# #Winning candidate Education
# data = raw_data.loc[raw_data['PARTY']!='NOTA'].groupby('WINNER')['EDUCATION'].value_counts()
data1 = raw_data.loc[raw_data['PARTY']!='NOTA'][raw_data['WINNER']==1]['EDUCATION'].value_counts()
fig14 = plt.figure(14)
ax14 = fig14.add_subplot(1,1,1)
ax14.bar(data1.index,data1)
# print(data)
print(data1)
ax14.set_xticklabels(data1.index, rotation = 45)
ax14.set_ylabel('Count')
ax14.set_xlabel('Education')
ax14.set_title('Winning Candidates Education qualification')
# # plt.show()

# #ANOVA Education
def conversion(value):
    if (value=='Post Graduate') or (value=='Graduate') or (value=='Graduate Professional') or (value=='12th Pass') or (value=='Doctorate'):
        return 'yes'
    else:
        return 'no'
x1 = raw_data.loc[raw_data["PARTY"]!='NOTA']["EDUCATION"].apply(conversion)
print(x1.value_counts())
x2 = raw_data.loc[raw_data["PARTY"]!='NOTA']["WINNER"]
data = pd.concat([x1, x2], axis=1)
mod = ols('WINNER~EDUCATION',data=data).fit()
aov = sm.stats.anova_lm(mod,type=2)
print(aov)
# #null hypothesis -> Education does not play an important role in election
# #alternate hypothesis -> Education does play an important role in election
#------------------------------------------------------------------------------------------------------------------------------------------

plt.show()

